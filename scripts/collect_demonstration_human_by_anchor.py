import argparse
import datetime
import h5py
import init_path
import json
import numpy as np
import os
import robosuite as suite
import time
from glob import glob
from pathlib import Path
from robosuite import load_controller_config
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from robosuite.utils.input_utils import input2action

import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import *
from standalone.utils.bddl_path_utils import canonicalize_bddl_file_name


def sanitize_ranges(raw_ranges):
    cleaned = []
    for entry in raw_ranges:
        if len(entry) != 4:
            raise ValueError(f"Expected 4 values per range, received {entry}")
        x0, y0, x1, y1 = entry
        cleaned.append(
            (
                min(float(x0), float(x1)),
                min(float(y0), float(y1)),
                max(float(x0), float(x1)),
                max(float(y0), float(y1)),
            )
        )
    if not cleaned:
        raise ValueError("No ranges defined for the requested region.")
    return cleaned


def match_range(xy, ranges, tol):
    for idx, (xmin, ymin, xmax, ymax) in enumerate(ranges):
        if (
            xmin - tol <= xy[0] <= xmax + tol
            and ymin - tol <= xy[1] <= ymax + tol
        ):
            return idx
    return None


def get_base_env(env):
    base_env = env
    while hasattr(base_env, "env"):
        base_env = base_env.env
    return base_env


def get_anchor_idx(env, obs, ranges, tol):
    base_env = get_base_env(env)
    obj_of_interest = base_env.obj_of_interest.copy()
    target_object_name = obj_of_interest[0]
    obs_key = f"{target_object_name.replace('_main', '')}_pos"
    if obs_key not in obs:
        print(f"[warning] missing {obs_key} in observations; skip episode")
        return None
    target_pos = obs[obs_key].copy()
    anchor_idx = match_range(target_pos[:2], ranges, tol)
    if anchor_idx is None:
        print(f"[warning] target position {target_pos[:2]} not in any anchor range")
    return anchor_idx


def collect_human_trajectory(env, device, arm, env_configuration):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    """
    # ID = 2 always corresponds to agentview
    env.render()

    task_completion_hold_count = -1
    device.start_control()

    saving = True
    count = 0

    while True:
        count += 1
        active_robot = (
            env.robots[0]
            if env_configuration == "bimanual"
            else env.robots[arm == "left"]
        )

        action, _ = input2action(
            device=device,
            robot=active_robot,
            active_arm=arm,
            env_configuration=env_configuration,
        )

        if action is None:
            print("Break")
            saving = False
            break

        env.step(action)
        env.render()
        if task_completion_hold_count == 0:
            break

        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1
            else:
                task_completion_hold_count = 10
        else:
            task_completion_hold_count = -1

    print(count)
    return saving


def unwrap_reset_obs(obs):
    if isinstance(obs, tuple) and len(obs) > 0:
        return obs[0]
    return obs


def get_next_demo_index(grp):
    demo_keys = [k for k in grp.keys() if k.startswith("demo_")]
    if not demo_keys:
        return 1
    indices = []
    for key in demo_keys:
        try:
            indices.append(int(key.split("_")[1]))
        except Exception:
            continue
    return max(indices) + 1 if indices else 1


def ensure_hdf5_group(f, env_info, args, env_name):
    bddl_ref = canonicalize_bddl_file_name(args.bddl_file)
    bddl_path = Path(args.bddl_file).expanduser().resolve()
    if "data" in f:
        grp = f["data"]
    else:
        grp = f.create_group("data")

    if "date" not in grp.attrs:
        now = datetime.datetime.now()
        grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
        grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
        grp.attrs["repository_version"] = suite.__version__
    if "env_info" not in grp.attrs:
        grp.attrs["env_info"] = env_info
    if "problem_info" not in grp.attrs:
        grp.attrs["problem_info"] = json.dumps(problem_info)
    if "bddl_file_name" not in grp.attrs:
        grp.attrs["bddl_file_name"] = bddl_ref
    if "bddl_file_content" not in grp.attrs:
        grp.attrs["bddl_file_content"] = bddl_path.read_text(encoding="utf-8")
    if "env" not in grp.attrs and env_name is not None:
        grp.attrs["env"] = env_name

    return grp


def append_demo_to_hdf5(directory, ep_directory, out_dir, env_info, args, anchor_idx=None):
    state_paths = os.path.join(directory, ep_directory, "state_*.npz")
    states = []
    actions = []
    env_name = None

    for state_file in sorted(glob(state_paths)):
        dic = np.load(state_file, allow_pickle=True)
        env_name = str(dic["env"])

        states.extend(dic["states"])
        for ai in dic["action_infos"]:
            actions.append(ai["actions"])

    if len(states) == 0:
        return False

    del states[-1]
    assert len(states) == len(actions)

    xml_path = os.path.join(directory, ep_directory, "model.xml")
    with open(xml_path, "r") as f_xml:
        xml_str = f_xml.read()

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    with h5py.File(hdf5_path, "a") as f:
        grp = ensure_hdf5_group(f, env_info, args, env_name)
        demo_idx = get_next_demo_index(grp)
        ep_data_grp = grp.create_group("demo_{}".format(demo_idx))
        ep_data_grp.attrs["model_file"] = xml_str
        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("actions", data=np.array(actions))
        if anchor_idx is not None:
            ep_data_grp.attrs["anchor_idx"] = int(anchor_idx)
    return True


def load_anchor_counts(hdf5_path, num_anchors):
    counts = {idx: 0 for idx in range(num_anchors)}
    if not os.path.exists(hdf5_path):
        return counts
    with h5py.File(hdf5_path, "r") as f:
        data = f.get("data", None)
        if data is None:
            return counts
        for key in data.keys():
            if not key.startswith("demo_"):
                continue
            anchor_idx = data[key].attrs.get("anchor_idx", None)
            if anchor_idx is None:
                continue
            try:
                anchor_idx = int(anchor_idx)
            except Exception:
                continue
            if anchor_idx in counts:
                counts[anchor_idx] += 1
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, default="demonstration_data")
    parser.add_argument("--robots", nargs="+", type=str, default=["Panda"])
    parser.add_argument("--config", type=str, default="single-arm-opposed")
    parser.add_argument("--arm", type=str, default="right")
    parser.add_argument("--camera", type=str, default="agentview")
    parser.add_argument("--controller", type=str, default="OSC_POSE")
    parser.add_argument("--device", type=str, default="spacemouse")
    parser.add_argument("--pos-sensitivity", type=float, default=1.5)
    parser.add_argument("--rot-sensitivity", type=float, default=1.0)
    parser.add_argument("--bddl-file", type=str, required=True)
    parser.add_argument(
        "--per-anchor",
        type=int,
        default=1,
        help="Desired number of demos per anchor point.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Tolerance when matching target position to anchor ranges.",
    )
    parser.add_argument(
        "--resume-dir",
        type=str,
        default=None,
        help="Existing output directory to append new demos.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--vendor-id", type=int, default=9583)
    parser.add_argument("--product-id", type=int, default=50741)
    args = parser.parse_args()

    controller_config = load_controller_config(default_controller=args.controller)
    config = {
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    assert os.path.exists(args.bddl_file)
    problem_info = BDDLUtils.get_problem_info(args.bddl_file)

    parsed = BDDLUtils.robosuite_parse_problem(args.bddl_file)
    target_object = parsed["obj_of_interest"][0]
    region_key = [st[2] for st in parsed["initial_state"] if st[1] == target_object][0]
    anchor_ranges = sanitize_ranges(parsed["regions"][region_key]["ranges"])

    problem_name = problem_info["problem_name"]
    domain_name = problem_info["domain_name"]
    language_instruction = problem_info["language_instruction"]
    if "TwoArm" in problem_name:
        config["env_configuration"] = args.config
    print(language_instruction)

    env = TASK_MAPPING[problem_name](
        bddl_file_name=args.bddl_file,
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
        region_sampling_strategy="round_robin",
        region_sampling_quota=args.per_anchor,
    )
    if args.seed is not None:
        env.seed(args.seed)

    env = VisualizationWrapper(env)

    env_info = json.dumps(config)
    tmp_directory = "demonstration_data/tmp/{}_ln_{}/{}".format(
        problem_name,
        language_instruction.replace(" ", "_").strip('""'),
        str(time.time()).replace(".", "_"),
    )
    env = DataCollectionWrapper(env, tmp_directory)

    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(
            pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity
        )
        env.viewer.add_keypress_callback(device.on_press)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(
            args.vendor_id,
            args.product_id,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
    else:
        raise Exception(
            "Invalid device choice: choose either 'keyboard' or 'spacemouse'."
        )

    t1, t2 = str(time.time()).split(".")
    default_dir = os.path.join(
        args.directory,
        f"{domain_name}_ln_{problem_name}_{t1}_{t2}_"
        + language_instruction.replace(" ", "_").strip('""'),
    )
    new_dir = os.path.abspath(args.resume_dir) if args.resume_dir else default_dir
    os.makedirs(new_dir, exist_ok=True)

    anchor_counts = {idx: 0 for idx in range(len(anchor_ranges))}
    total_needed = args.per_anchor * len(anchor_ranges)
    if args.resume_dir:
        hdf5_path = os.path.join(new_dir, "demo.hdf5")
        anchor_counts = load_anchor_counts(hdf5_path, len(anchor_ranges))
        collected = sum(anchor_counts.values())
        print(f"[info] resuming from {hdf5_path} with {collected} demos")
    else:
        collected = 0

    while collected < total_needed:
        reset_success = False
        obs = None
        while not reset_success:
            try:
                obs = unwrap_reset_obs(env.reset())
                reset_success = True
            except Exception:
                continue

        if obs is None:
            env.close()
            continue

        anchor_idx = get_anchor_idx(env, obs, anchor_ranges, args.tolerance)

        if anchor_idx is None:
            env.close()
            continue

        if anchor_counts[anchor_idx] >= args.per_anchor:
            env.close()
            continue

        saving = collect_human_trajectory(env, device, args.arm, args.config)
        ep_dir = getattr(env, "ep_directory", None)
        if ep_dir:
            ep_dir = os.path.basename(ep_dir)
        if not saving:
            env.close()
            continue
        if not ep_dir:
            print("[warning] ep_directory not set after interaction; skipping episode")
            env.close()
            continue

        if not append_demo_to_hdf5(
            tmp_directory, ep_dir, new_dir, env_info, args, anchor_idx
        ):
            print("[warning] failed to write episode data; skipping")
            env.close()
            continue
        anchor_counts[anchor_idx] += 1
        collected += 1
        print(
            f"Collected demo {collected}/{total_needed} "
            f"(anchor {anchor_idx + 1}, count {anchor_counts[anchor_idx]})"
        )
        env.close()

    print("Per-anchor human data collection completed.")

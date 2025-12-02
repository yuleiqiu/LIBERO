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

from robosuite import load_controller_config
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper

import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import *


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


class BasePolicy:
    def __init__(self, inject_noise=False, debug=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.trajectory = None
        self.debug = debug

    def generate_trajectory(self, ts_first):
        raise NotImplementedError

    def _log_step(self, curr_wp, next_wp, target_xyz, gripper, current_eef_pos):
        if not self.debug:
            return
        next_t = next_wp["t"] if next_wp is not None else None
        print(
            f"[debug] step={self.step_count} curr_t={curr_wp['t']} next_t={next_t} "
            f"target_xyz={target_xyz} grip={gripper} eef={current_eef_pos}"
        )

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        denom = next_waypoint["t"] - curr_waypoint["t"]
        if abs(denom) < 1e-6:
            t_frac = 1.0
        else:
            t_frac = (t - curr_waypoint["t"]) / denom
            t_frac = np.clip(t_frac, 0.0, 1.0)
        curr_xyz = curr_waypoint["xyz"]
        curr_grip = curr_waypoint["gripper"]
        next_xyz = next_waypoint["xyz"]
        next_grip = next_waypoint["gripper"]
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        # Use step change for gripper to respect open/close commands at the waypoint.
        gripper = curr_grip if t_frac < 1.0 else next_grip
        return xyz, gripper

    def __call__(self, target_pos, destination_pos, current_eef_pos):
        if self.step_count == 0:
            self.generate_trajectory(target_pos, destination_pos)
            if self.debug:
                print("[debug] Generated trajectory:")
                for wp in self.trajectory:
                    print(f"  t={wp['t']}, xyz={wp['xyz']}, grip={wp['gripper']}")

        while self.trajectory and self.trajectory[0]["t"] <= self.step_count:
            self.curr_waypoint = self.trajectory.pop(0)
            if self.debug:
                next_wp = self.trajectory[0] if self.trajectory else None
                print(
                    f"[debug] step={self.step_count} -> curr_wp t={self.curr_waypoint['t']} "
                    f"xyz={self.curr_waypoint['xyz']} grip={self.curr_waypoint['gripper']} "
                    f"next_wp={next_wp}"
                )

        if self.trajectory:
            next_waypoint = self.trajectory[0]
            target_xyz, gripper = self.interpolate(
                self.curr_waypoint, next_waypoint, self.step_count
            )
            # Avoid spatial noise exactly at grasp/place depth to reduce hovering jitter.
            if self.inject_noise and not np.isclose(target_xyz[2], next_waypoint["xyz"][2], atol=1e-3):
                scale = 0.02
                target_xyz += np.random.uniform(-scale, scale, target_xyz.shape)
            xyz = target_xyz - current_eef_pos
            self._log_step(self.curr_waypoint, next_waypoint, target_xyz, gripper, current_eef_pos)
        else:
            target_xyz = self.curr_waypoint["xyz"]
            gripper = self.curr_waypoint["gripper"]
            xyz = target_xyz - current_eef_pos
            self._log_step(self.curr_waypoint, None, target_xyz, gripper, current_eef_pos)

        self.step_count += 1
        return xyz, gripper


def random_lateral_offset(start_pos, goal_pos, max_offset=0.08):
    """
    Sample a lateral offset in the XY-plane perpendicular to the straight-line direction.
    """
    direction = goal_pos[:2] - start_pos[:2]
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        # Fallback to an arbitrary axis if start and goal are almost identical.
        direction = np.array([1.0, 0.0])
        norm = 1.0
    tangent = direction / norm
    lateral_dir = np.array([-tangent[1], tangent[0]])
    magnitude = np.random.uniform(0.0, max_offset) * np.random.choice([-1.0, 1.0])
    return np.array([lateral_dir[0] * magnitude, lateral_dir[1] * magnitude, 0.0])


class RandomizedPickAndPlacePolicy(BasePolicy):
    """
    Pick-place policy that samples mid-waypoints and timing so the trajectory family is multi-modal.
    """

    def __init__(self, inject_noise=True, lock_keypoints=True):
        super().__init__(inject_noise, debug=os.environ.get("DEBUG_TRAJ", "0") == "1")
        # Raise hover heights to avoid staying too close to the object.
        self.hover_range = (0.22, 0.30)
        self.transfer_height_range = (0.26, 0.38)
        self.lock_keypoints = lock_keypoints
        # If we want key nodes not to drift, disable lateral offsets.
        self.max_lateral_offset = 0.0 if self.lock_keypoints else 0.07
        # Lean timing windows (controller steps) to avoid停在物体上方太久.
        self.dt_approach = (60, 80)
        # Close almost立即: keep预抓窗口很小.
        self.dt_pregrasp = (30, 50)
        self.dt_lift = (12, 28)
        self.dt_transfer = (25, 55)
        # Hover over basket before descending to avoid early drop.
        self.dt_preplace_hover = (12, 22)
        self.dt_place = (12, 28)
        self.dt_release = (5, 10)
        self.dt_retreat = (18, 32)
        # Hold durations to let gripper actually close on the object.
        self.dt_grasp_hold = (12, 20)
        # Optional open hold at grasp depth before closing to satisfy "hover then close".
        self.dt_grasp_hover = (4, 8)

    def _dt(self, low, high):
        return int(np.random.randint(low, high))

    def _transfer_midpoints(self, start, goal):
        """
        Build 0/1/2 mid-waypoints with different arc styles to induce multi-modal paths.
        """
        mode = np.random.choice(["direct", "arc", "double_arc"])
        midpoints = []
        if self.max_lateral_offset <= 1e-6:
            # Force straight-line midpoints when locking keypoints.
            mode = "direct"
        if mode == "direct":
            center = (start + goal) / 2.0
            center[2] = np.random.uniform(*self.transfer_height_range)
            midpoints.append(center)
        elif mode == "arc":
            offset = random_lateral_offset(start, goal, self.max_lateral_offset)
            center = (start + goal) / 2.0 + offset
            center[2] = np.random.uniform(*self.transfer_height_range)
            midpoints.append(center)
        else:  # double_arc
            offset1 = random_lateral_offset(start, goal, self.max_lateral_offset)
            offset2 = random_lateral_offset(goal, start, self.max_lateral_offset)
            mid1 = start + 0.35 * (goal - start) + offset1
            mid2 = start + 0.7 * (goal - start) + offset2
            mid1[2] = np.random.uniform(*self.transfer_height_range)
            mid2[2] = np.random.uniform(*self.transfer_height_range)
            midpoints.extend([mid1, mid2])
        return midpoints

    def generate_trajectory(self, target_pos, destination_pos):
        waypoints = []
        t = 0

        def add_wp(pos, grip, dt):
            nonlocal t
            if len(waypoints) == 0:
                # allow first waypoint at t=0
                dt_eff = 0
            else:
                dt_eff = max(1, int(dt))
            waypoints.append({"t": t, "xyz": pos, "gripper": grip})
            t += dt_eff

        hover_pick = np.random.uniform(*self.hover_range)
        pre_grasp_height = np.random.uniform(0.06, 0.10)
        grasp_depth = np.random.uniform(-0.025, -0.010)
        post_grasp_lift = np.random.uniform(0.16, 0.20)

        hover_place = np.random.uniform(*self.hover_range)
        # Keep higher above the basket; avoid early deep descent.
        place_depth = np.random.uniform(0.04, 0.09)
        retreat_height = np.random.uniform(0.22, 0.32)

        # Approach and grasp
        add_wp(target_pos + np.array([0, 0, hover_pick]), -1, 0)
        add_wp(
            target_pos + np.array([0, 0, pre_grasp_height]),
            -1,
            self._dt(*self.dt_approach),
        )
        # Hover at grasp plane while still open.
        add_wp(
            target_pos + np.array([0, 0, grasp_depth]),
            -1,
            self._dt(*self.dt_grasp_hover),
        )
        add_wp(
            target_pos + np.array([0, 0, grasp_depth]),
            1,
            self._dt(*self.dt_pregrasp),
        )
        # Hold at grasp depth to let gripper close.
        add_wp(
            target_pos + np.array([0, 0, grasp_depth]),
            1,
            self._dt(*self.dt_grasp_hold),
        )
        add_wp(
            target_pos + np.array([0, 0, post_grasp_lift]),
            1,
            self._dt(*self.dt_lift),
        )

        # Transfer with sampled arcs
        transfer_start = target_pos + np.array([0, 0, post_grasp_lift])
        transfer_goal = destination_pos + np.array([0, 0, hover_place])
        for mid in self._transfer_midpoints(transfer_start, transfer_goal):
            add_wp(mid, 1, self._dt(*self.dt_transfer))
        add_wp(transfer_goal, 1, self._dt(*self.dt_transfer))
        # Hold over basket at hover height before descending.
        add_wp(
            transfer_goal,
            1,
            self._dt(*self.dt_preplace_hover),
        )

        # Place and retreat
        add_wp(
            destination_pos + np.array([0, 0, place_depth]),
            1,
            self._dt(*self.dt_place),
        )
        add_wp(
            destination_pos + np.array([0, 0, place_depth]),
            -1,
            self._dt(*self.dt_release),
        )
        add_wp(
            destination_pos + np.array([0, 0, retreat_height]),
            -1,
            self._dt(*self.dt_retreat),
        )
        add_wp(
            destination_pos + np.array([0, 0, retreat_height]),
            -1,
            self._dt(8, 16),
        )

        self.trajectory = waypoints


class FixedPickAndPlacePolicy(BasePolicy):
    """
    Deterministic pick-and-place trajectory (mirrors the original _from_script behavior)
    that reliably reaches the target without lateral offsets or noise.
    """

    def __init__(self):
        super().__init__(inject_noise=False, debug=os.environ.get("DEBUG_TRAJ", "0") == "1")

    def generate_trajectory(self, target_pos, destination_pos):
        self.trajectory = [
            {"t": 0, "xyz": target_pos + np.array([0, 0, 0.3]), "gripper": -1},
            {"t": 200, "xyz": target_pos + np.array([0, 0, 0]), "gripper": -1},
            {"t": 300, "xyz": target_pos + np.array([0, 0, 0]), "gripper": 1},
            {"t": 380, "xyz": target_pos + np.array([0, 0, 0.2]), "gripper": 1},
            {"t": 470, "xyz": destination_pos + np.array([0, 0, 0.25]), "gripper": 1},
            {"t": 540, "xyz": destination_pos + np.array([0, 0, 0.1]), "gripper": 1},
            {"t": 570, "xyz": destination_pos + np.array([0, 0, 0.1]), "gripper": -1},
            {"t": 590, "xyz": destination_pos + np.array([0, 0, 0.3]), "gripper": -1},
            {"t": 600, "xyz": destination_pos + np.array([0, 0, 0.3]), "gripper": -1},
        ]


def collect_scripted_trajectory(env, tol, ranges, centers, remove_directory=[], debug=False):
    reset_success = False
    obs = None
    while not reset_success:
        try:
            obs = env.reset()
            reset_success = True
        except Exception:
            continue

    env.render()

    obj_of_interest = env.env.obj_of_interest.copy()
    target_object_name = obj_of_interest[0]
    destination_name = obj_of_interest[-1]

    target_pos = obs[f"{target_object_name.replace('_main', '')}_pos"].copy()
    destination_pos = obs[f"{destination_name.replace('_main', '')}_pos"].copy()

    # Use deterministic policy matching the working _from_script trajectory; no noise.
    policy = FixedPickAndPlacePolicy()
    # policy = RandomizedPickAndPlacePolicy(inject_noise=True, lock_keypoints=False)
    if debug:
        policy.debug = True

    success = False
    task_completed = False
    task_completion_hold_count = -1

    for _ in range(600):
        current_eef_pos = obs["robot0_eef_pos"].copy()

        if not success:
            xyz_delta, gripper = policy(target_pos, destination_pos, current_eef_pos)
            action = np.zeros(7)
            action[:3] = xyz_delta * 1.5
            action[3:6] = 0
            action[6] = gripper
        else:
            action = np.zeros(7)
            action[2] = 0.1
            action[6] = -1

        obs, _, success, _ = env.step(action)
        env.render()

        if task_completion_hold_count == 0:
            task_completed = True
            break

        if success:
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1
            else:
                task_completion_hold_count = 40
        else:
            task_completion_hold_count = -1
    else:
        print("Trajectory collection timed out, retrying...")

    if not task_completed:
        remove_directory.append(env.ep_directory.split("/")[-1])
        env.close()
        return None, False

    target_xy = target_pos[:2]
    anchor_idx = match_range(target_xy, ranges, tol)
    if anchor_idx is None:
        print(f"Warning: target position {target_xy} not in any anchor range.")
    env.close()
    return anchor_idx, True


def gather_demonstrations_as_hdf5(
    directory, out_dir, env_info, args, remove_directory=[]
):
    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")

    grp = f.create_group("data")

    num_eps = 0
    env_name = None

    for ep_directory in os.listdir(directory):
        if ep_directory in remove_directory:
            continue
        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])

        if len(states) == 0:
            continue

        del states[-1]
        assert len(states) == len(actions)

        num_eps += 1
        ep_data_grp = grp.create_group("demo_{}".format(num_eps))

        xml_path = os.path.join(directory, ep_directory, "model.xml")
        with open(xml_path, "r") as f_xml:
            xml_str = f_xml.read()
        ep_data_grp.attrs["model_file"] = xml_str

        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("actions", data=np.array(actions))

    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    grp.attrs["problem_info"] = json.dumps(problem_info)
    grp.attrs["bddl_file_name"] = args.bddl_file
    grp.attrs["bddl_file_content"] = str(open(args.bddl_file, "r", encoding="utf-8"))

    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, default="demonstration_data")
    parser.add_argument("--robots", nargs="+", type=str, default=["Panda"])
    parser.add_argument("--config", type=str, default="single-arm-opposed")
    parser.add_argument("--arm", type=str, default="right")
    parser.add_argument("--camera", type=str, default="agentview")
    parser.add_argument("--controller", type=str, default="OSC_POSE")
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
        "--debug-trajectory",
        action="store_true",
        help="Print generated waypoint list and current/next waypoint during rollout.",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    controller_config = load_controller_config(default_controller=args.controller)
    config = {
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    assert os.path.exists(args.bddl_file)
    problem_info = BDDLUtils.get_problem_info(args.bddl_file)
    task_name = args.bddl_file.split("/")[-1].split(".")[0]

    parsed = BDDLUtils.robosuite_parse_problem(args.bddl_file)
    target_object = parsed["obj_of_interest"][0]
    region_key = [st[2] for st in parsed["initial_state"] if st[1] == target_object][0]
    anchor_ranges = sanitize_ranges(parsed["regions"][region_key]["ranges"])
    anchor_centers = [
        (np.mean([r[0], r[2]]), np.mean([r[1], r[3]])) for r in anchor_ranges
    ]

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

    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join(
        args.directory,
        f"{domain_name}_ln_{problem_name}_{t1}_{t2}_"
        + language_instruction.replace(" ", "_").strip('""'),
    )
    os.makedirs(new_dir, exist_ok=True)

    remove_directory = []
    anchor_counts = {idx: 0 for idx in range(len(anchor_ranges))}
    total_needed = args.per_anchor * len(anchor_ranges)
    collected = 0

    while collected < total_needed:
        current_anchor, saving = collect_scripted_trajectory(
            env,
            args.tolerance,
            anchor_ranges,
            anchor_centers,
            remove_directory,
            debug=args.debug_trajectory,
        )
        if not saving or current_anchor is None:
            continue

        if anchor_counts[current_anchor] >= args.per_anchor:
            remove_directory.append(env.ep_directory.split("/")[-1])
            continue

        anchor_counts[current_anchor] += 1
        gather_demonstrations_as_hdf5(
            tmp_directory, new_dir, env_info, args, remove_directory
        )
        collected += 1
        print(
            f"Collected demo {collected}/{total_needed} "
            f"(anchor {current_anchor + 1}, count {anchor_counts[current_anchor]})"
        )

    print("Per-anchor data collection completed.")

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import robosuite.macros as macros
import robosuite.utils.transform_utils as T

import init_path
import libero.libero.utils.utils as libero_utils
from libero.libero import get_libero_path
from libero.libero.envs import TASK_MAPPING
from standalone.utils.bddl_path_utils import (
    canonicalize_bddl_file_name,
    resolve_bddl_path,
)


CAMERA_SPECS = (
    {
        "dataset_prefix": "agentview",
        "image_key": "agentview_image",
        "depth_key": "agentview_depth",
        "segmentation_key": "agentview_segmentation_instance",
    },
    {
        "dataset_prefix": "eye_in_hand",
        "image_key": "robot0_eye_in_hand_image",
        "depth_key": "robot0_eye_in_hand_depth",
        "segmentation_key": "robot0_eye_in_hand_segmentation_instance",
    },
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-file", default="demo.hdf5")
    parser.add_argument("--use-camera-obs", action="store_true")
    parser.add_argument("--no-proprio", action="store_true")
    parser.add_argument("--use-depth", action="store_true")
    parser.add_argument("--use-segmentation", action="store_true")
    return parser.parse_args()


def _build_output_path(bddl_file_name, use_segmentation):
    bddl_basename = os.path.basename(bddl_file_name).replace(".bddl", "")
    suffix = "_seg" if use_segmentation else ""
    output_filename = f"{bddl_basename}_demo{suffix}.hdf5"

    datasets_dir = get_libero_path("datasets")
    output_dir = os.path.join(datasets_dir, "processed")
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, output_filename)


def _refresh_segmentation_mapping(env):
    segmentation_id_mapping = {}
    segmentation_robot_id = None

    instances = list(env.model.instances_to_ids.keys())
    for i, name in enumerate(instances):
        if name in ("Panda0", "OnTheGroundPanda0"):
            segmentation_robot_id = i
            break

    for i, name in enumerate(instances):
        if name not in ("Panda0", "RethinkMount0", "PandaGripper0"):
            segmentation_id_mapping[i] = name

    instance_to_id = {name: seg_id + 1 for seg_id, name in segmentation_id_mapping.items()}
    return instance_to_id, segmentation_robot_id


def _get_segmentation_of_interest(segmentation_image, obj_of_interest, instance_to_id):
    ret_seg = np.zeros_like(segmentation_image, dtype=np.int8)
    for obj in obj_of_interest:
        seg_id = instance_to_id.get(obj)
        if seg_id is not None:
            ret_seg[segmentation_image == seg_id] = 1
    ret_seg[segmentation_image == 0] = -1
    return ret_seg


def main():
    args = parse_args()

    if args.use_segmentation and not args.use_camera_obs:
        raise ValueError("--use-segmentation requires --use-camera-obs")

    replay_hdf5_path = args.demo_file
    with h5py.File(replay_hdf5_path, "r") as f:
        demo_path = Path(replay_hdf5_path).expanduser().resolve()
        env_name = f["data"].attrs["env"]
        env_kwargs = json.loads(f["data"].attrs["env_info"])

        problem_info = json.loads(f["data"].attrs["problem_info"])
        problem_name = problem_info["problem_name"]

        demos = list(f["data"].keys())
        bddl_file_name = canonicalize_bddl_file_name(f["data"].attrs["bddl_file_name"])
        resolved_bddl_path = resolve_bddl_path(bddl_file_name, demo_path)
        if resolved_bddl_path is None:
            raise FileNotFoundError(f"bddl file not found: {bddl_file_name}")

        hdf5_path = _build_output_path(
            bddl_file_name=bddl_file_name,
            use_segmentation=args.use_segmentation,
        )
        print(f"Output dataset will be saved to: {hdf5_path}")

        output_parent_dir = Path(hdf5_path).parent
        output_parent_dir.mkdir(parents=True, exist_ok=True)

        with h5py.File(hdf5_path, "w") as h5py_f:
            grp = h5py_f.create_group("data")

            grp.attrs["env_name"] = env_name
            grp.attrs["problem_info"] = f["data"].attrs["problem_info"]
            grp.attrs["macros_image_convention"] = macros.IMAGE_CONVENTION

            libero_utils.update_env_kwargs(
                env_kwargs,
                bddl_file_name=resolved_bddl_path,
                has_renderer=not args.use_camera_obs,
                has_offscreen_renderer=args.use_camera_obs,
                ignore_done=True,
                use_camera_obs=args.use_camera_obs,
                camera_depths=args.use_depth,
                camera_names=["robot0_eye_in_hand", "agentview"],
                reward_shaping=True,
                control_freq=20,
                camera_heights=128,
                camera_widths=128,
                camera_segmentations="instance" if args.use_segmentation else None,
            )

            with open(resolved_bddl_path, "r") as bddl_file:
                bddl_file_content = bddl_file.read()

            grp.attrs["bddl_file_name"] = bddl_file_name
            grp.attrs["bddl_file_content"] = bddl_file_content
            grp.attrs["segmentation_of_interest"] = args.use_segmentation
            print(grp.attrs["bddl_file_content"])

            env = TASK_MAPPING[problem_name](**env_kwargs)
            try:
                env_args = {
                    "type": 1,
                    "env_name": env_name,
                    "problem_name": problem_name,
                    "bddl_file": bddl_file_name,
                    "env_kwargs": {
                        **env_kwargs,
                        "bddl_file_name": bddl_file_name,
                    },
                }

                grp.attrs["env_args"] = json.dumps(env_args)
                print(grp.attrs["env_args"])
                total_len = 0
                cap_index = 5

                for i, ep in enumerate(demos):
                    print("Playing back episodes... (press ESC to quit)")
                    print(f"Processing episode {i + 1}/{len(demos)}: {ep}")

                    model_xml = f[f"data/{ep}"].attrs["model_file"]
                    reset_success = False
                    while not reset_success:
                        try:
                            env.reset()
                            reset_success = True
                        except Exception:
                            continue

                    model_xml = libero_utils.postprocess_model_xml(model_xml, {})

                    if not args.use_camera_obs:
                        env.viewer.set_camera(0)

                    states = f[f"data/{ep}/states"][()]
                    actions = np.array(f[f"data/{ep}/actions"][()])
                    num_actions = actions.shape[0]

                    init_idx = 0
                    env.reset_from_xml_string(model_xml)
                    env.sim.reset()
                    env.sim.set_state_from_flattened(states[init_idx])
                    env.sim.forward()
                    model_xml = env.sim.model.get_xml()

                    instance_to_id = {}
                    if args.use_segmentation:
                        instance_to_id, _ = _refresh_segmentation_mapping(env)

                    ee_states = []
                    gripper_states = []
                    joint_states = []
                    robot_states = []

                    agentview_images = []
                    eye_in_hand_images = []

                    agentview_depths = []
                    eye_in_hand_depths = []

                    agentview_segmentation_masks = []
                    eye_in_hand_segmentation_masks = []

                    valid_index = []

                    for j, action in enumerate(actions):
                        obs, reward, done, info = env.step(action)

                        if j < num_actions - 1:
                            state_playback = env.sim.get_state().flatten()
                            err = np.linalg.norm(states[j + 1] - state_playback)
                            if err > 0.01:
                                print(
                                    f"[warning] playback diverged by {err:.2f} "
                                    f"for ep {ep} at step {j}"
                                )

                        if j < cap_index:
                            continue

                        valid_index.append(j)

                        if not args.no_proprio:
                            if "robot0_gripper_qpos" in obs:
                                gripper_states.append(obs["robot0_gripper_qpos"])

                            joint_states.append(obs["robot0_joint_pos"])
                            ee_states.append(
                                np.hstack(
                                    (
                                        obs["robot0_eef_pos"],
                                        T.quat2axisangle(obs["robot0_eef_quat"]),
                                    )
                                )
                            )

                        robot_states.append(env.get_robot_state_vector(obs))

                        if args.use_camera_obs:
                            if args.use_depth:
                                agentview_depths.append(obs["agentview_depth"])
                                eye_in_hand_depths.append(obs["robot0_eye_in_hand_depth"])

                            agentview_images.append(obs["agentview_image"])
                            eye_in_hand_images.append(obs["robot0_eye_in_hand_image"])

                            if args.use_segmentation:
                                for spec, frames in (
                                    (CAMERA_SPECS[0], agentview_segmentation_masks),
                                    (CAMERA_SPECS[1], eye_in_hand_segmentation_masks),
                                ):
                                    seg_key = spec["segmentation_key"]
                                    if seg_key not in obs:
                                        raise KeyError(f"missing segmentation key: {seg_key}")
                                    seg_img = np.squeeze(obs[seg_key])
                                    seg_mask = _get_segmentation_of_interest(
                                        segmentation_image=seg_img,
                                        obj_of_interest=env.obj_of_interest,
                                        instance_to_id=instance_to_id,
                                    )
                                    frames.append(seg_mask)
                        else:
                            env.render()

                    states = states[valid_index]
                    actions = actions[valid_index]
                    dones = np.zeros(len(actions), dtype=np.uint8)
                    dones[-1] = 1
                    rewards = np.zeros(len(actions), dtype=np.uint8)
                    rewards[-1] = 1

                    print(
                        f"Episode {i} has {len(actions)} actions, "
                        f"{len(agentview_images)} agentview images"
                    )
                    assert len(actions) == len(agentview_images)

                    ep_data_grp = grp.create_group(f"demo_{i}")
                    obs_grp = ep_data_grp.create_group("obs")

                    if not args.no_proprio:
                        ee_states_np = np.stack(ee_states, axis=0)
                        obs_grp.create_dataset(
                            "gripper_states", data=np.stack(gripper_states, axis=0)
                        )
                        obs_grp.create_dataset(
                            "joint_states", data=np.stack(joint_states, axis=0)
                        )
                        obs_grp.create_dataset("ee_states", data=ee_states_np)
                        obs_grp.create_dataset("ee_pos", data=ee_states_np[:, :3])
                        obs_grp.create_dataset("ee_ori", data=ee_states_np[:, 3:])

                    obs_grp.create_dataset(
                        "agentview_rgb", data=np.stack(agentview_images, axis=0)
                    )
                    obs_grp.create_dataset(
                        "eye_in_hand_rgb", data=np.stack(eye_in_hand_images, axis=0)
                    )

                    if args.use_depth:
                        obs_grp.create_dataset(
                            "agentview_depth", data=np.stack(agentview_depths, axis=0)
                        )
                        obs_grp.create_dataset(
                            "eye_in_hand_depth", data=np.stack(eye_in_hand_depths, axis=0)
                        )

                    if args.use_segmentation:
                        obs_grp.create_dataset(
                            "agentview_segmentation_of_interest",
                            data=np.stack(agentview_segmentation_masks, axis=0),
                        )
                        obs_grp.create_dataset(
                            "eye_in_hand_segmentation_of_interest",
                            data=np.stack(eye_in_hand_segmentation_masks, axis=0),
                        )

                    ep_data_grp.create_dataset("actions", data=actions)
                    ep_data_grp.create_dataset("states", data=states)
                    ep_data_grp.create_dataset(
                        "robot_states", data=np.stack(robot_states, axis=0)
                    )
                    ep_data_grp.create_dataset("rewards", data=rewards)
                    ep_data_grp.create_dataset("dones", data=dones)
                    ep_data_grp.attrs["num_samples"] = len(agentview_images)
                    ep_data_grp.attrs["model_file"] = model_xml
                    ep_data_grp.attrs["init_state"] = states[init_idx]

                    src_attrs = f[f"data/{ep}"].attrs
                    for key in ("anchor_id", "anchor_idx"):
                        if key in src_attrs:
                            ep_data_grp.attrs[key] = src_attrs[key]

                    total_len += len(agentview_images)

                grp.attrs["num_demos"] = len(demos)
                grp.attrs["total"] = total_len
            finally:
                env.close()

    print("The created dataset is saved in the following path: ")
    print(hdf5_path)


if __name__ == "__main__":
    main()

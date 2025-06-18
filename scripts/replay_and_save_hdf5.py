#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
replay_and_save_hdf5.py
-----------------------
Replay LIBERO demonstrations and save them in the same format as collect_demonstration.py
"""
import argparse
import copy
import datetime
import h5py
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import robosuite as suite
from robosuite import load_controller_config

from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import OffScreenRenderEnv


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Replay LIBERO demonstrations and save as HDF5")
    parser.add_argument(
        "--task-id", 
        type=int, 
        default=0,
        help="Task ID to replay (default: 0)"
    )
    parser.add_argument(
        "--task-suite-multiple",
        type=str,
        default="libero_object",
        help="Multiple task suite name (default: libero_object)"
    )
    parser.add_argument(
        "--task-suite-single",
        type=str,
        default="libero_object_single",
        help="Single task suite name (default: libero_object_single)"
    )
    parser.add_argument(
        "--max-demos",
        type=int,
        default=50,
        help="Maximum number of demonstrations to process (default: 50)"
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=128,
        help="Camera height (default: 128)"
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=128,
        help="Camera width (default: 128)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for HDF5 and JSON files"
    )
    return parser.parse_args()


def main():
    """Main function to run the replay process."""
    args = parse_args()
    
    # Configuration constants
    TASK_SUITE_NAME_MULTIPLE = args.task_suite_multiple
    TASK_SUITE_NAME_SINGLE = args.task_suite_single
    TASK_ID = args.task_id
    MAX_DEMOS = args.max_demos
    CAMERA_HEIGHT = args.camera_height
    CAMERA_WIDTH = args.camera_width
    OUTPUT_DIR = args.output_dir

    # Initialize benchmark and task objects
    try:
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite_multiple = benchmark_dict[TASK_SUITE_NAME_MULTIPLE]()
        task_multiple = task_suite_multiple.get_task(TASK_ID)
        task_suite_single = benchmark_dict[TASK_SUITE_NAME_SINGLE]()
        task_single = task_suite_single.get_task(TASK_ID)
    except KeyError as e:
        print(f"[ERROR] Task suite not found: {e}")
        sys.exit(1)
    except IndexError as e:
        print(f"[ERROR] Task ID {TASK_ID} not found in task suite")
        sys.exit(1)

    # Set up file paths
    bddl_dir = get_libero_path("bddl_files")
    bddl_multiple = os.path.join(bddl_dir, task_multiple.problem_folder, task_multiple.bddl_file)
    bddl_single = os.path.join(bddl_dir, task_single.problem_folder, task_single.bddl_file)

    demo_dir = get_libero_path("datasets") # Still needed for input demo file path
    demo_file = os.path.join(demo_dir, task_suite_multiple.get_task_demonstration(TASK_ID))

    # Use the output_dir passed as an argument
    output_dir = OUTPUT_DIR 
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if demo file exists
    if not os.path.exists(demo_file):
        print(f"[ERROR] Demo file not found: {demo_file}")
        sys.exit(1)
        
    demo_h5 = h5py.File(demo_file, "r")["data"]

    # Create environments
    cam_args = dict(camera_heights=CAMERA_HEIGHT, camera_widths=CAMERA_WIDTH)
    env_multiple = OffScreenRenderEnv(bddl_file_name=bddl_multiple, **cam_args)
    env_single = OffScreenRenderEnv(bddl_file_name=bddl_single, **cam_args)

    print(f"[INFO] Using single task suite: {TASK_SUITE_NAME_SINGLE}, "
          f"task ID: {TASK_ID}, task name: {task_single.name}")
    
    # Prepare environment information
    # Simulate config information from collect_demonstration.py
    controller_config = load_controller_config(default_controller="OSC_POSE")
    config = {
        "robots": ["Panda"],  # Adjust according to actual robot
        "controller_configs": controller_config,  # Adjust according to actual controller
    }
    env_info = json.dumps(config)

    # Get problem information (requires importing BDDLUtils)
    try:
        import libero.libero.envs.bddl_utils as BDDLUtils
        problem_info = BDDLUtils.get_problem_info(bddl_single)
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve problem information from {bddl_single}: {str(e)}")

    # Collect replay data
    collected_demos = []
    success_cnt = 0
    demo_results = []  # Record results for each demo

    for demo_idx in range(MAX_DEMOS):
        demo_key = f"demo_{demo_idx}"
        if demo_key not in demo_h5:
            print(f"[!] Dataset does not contain {demo_key}, ending early")
            break

        demo = demo_h5[demo_key]
        states = demo["states"]
        actions = demo["actions"]

        # Reset environments
        env_multiple.reset()
        env_multiple.set_init_state(states[0])

        env_single.reset()
        copy_overlap(env_multiple, env_single)

        # Collect states and actions
        replay_states = []
        replay_actions = []
        
        # Get initial state
        obs, _, _, _ = env_single.step([0.] * 7)  # Don't advance physics, just get state
        initial_state = env_single.env.sim.get_state().flatten()
        replay_states.append(initial_state)

        # Replay actions and collect states
        done, info = False, {}
        for step_id, action in enumerate(actions):
            # Execute action
            obs, _, done, info = env_single.step(action)
            
            # Record action and new state
            replay_actions.append(action.copy())
            current_state = env_single.env.sim.get_state().flatten()
            replay_states.append(current_state)

            if done:
                break

        # Remove the last state (consistent with collect_demonstration.py)
        if len(replay_states) > 0:
            replay_states = replay_states[:-1]
        
        # Ensure states and actions have consistent length
        min_len = min(len(replay_states), len(replay_actions))
        replay_states = replay_states[:min_len]
        replay_actions = replay_actions[:min_len]
        
        if len(replay_states) > 0 and len(replay_actions) > 0:
            collected_demos.append({
                "states": replay_states,
                "actions": replay_actions
            })
            
            success = info.get("success", done)
            # Convert numpy boolean to Python boolean for JSON serialization
            success = bool(success)
            demo_results.append({
                "demo_id": demo_idx,
                "demo_key": demo_key,
                "success": success,
                "num_states": len(replay_states),
                "num_actions": len(replay_actions),
                "status": "SUCCESS" if success else "FAIL"
            })
            
            print(f"Demo {demo_idx:02d} – {'SUCCESS' if success else 'FAIL'} "
                  f"(states: {len(replay_states)}, actions: {len(replay_actions)})")
            if success:
                success_cnt += 1
        else:
            collected_demos.append(None)
            demo_results.append({
                "demo_id": demo_idx,
                "demo_key": demo_key,
                "success": False,  # Explicitly use Python bool
                "num_states": 0,
                "num_actions": 0,
                "status": "EMPTY"
            })
            print(f"Demo {demo_idx:02d} – EMPTY (skipped)")

    # Save to HDF5
    # Use task name as filename prefix
    output_path = os.path.join(output_dir, f"{task_single.name}_replay.hdf5")
    create_hdf5_from_replays(
        output_path, 
        collected_demos, 
        env_info, 
        problem_info, 
        bddl_single,
        env_single
    )
    
    # Save results to JSON
    json_output_path = os.path.join(output_dir, f"{task_single.name}_replay_results.json")
    total_demos = len(demo_results)
    valid_demos = len([d for d in collected_demos if d is not None])
    
    results_summary = {
        "task_name": task_single.name,
        "task_id": TASK_ID,
        "task_suite_single": TASK_SUITE_NAME_SINGLE,
        "task_suite_multiple": TASK_SUITE_NAME_MULTIPLE,
        "timestamp": datetime.datetime.now().isoformat(),
        "total_demos_processed": total_demos,
        "valid_demos_collected": valid_demos,
        "success_count": success_cnt,
        "success_rate": success_cnt / valid_demos if valid_demos > 0 else 0.0,
        "demo_details": demo_results
    }
    
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    # Statistics and summary
    total_demos = len([d for d in collected_demos if d is not None])
    if total_demos > 0:
        success_rate = success_cnt / total_demos
        print(f"\nSUMMARY: {success_cnt}/{total_demos} replays succeed "
              f"({success_rate:.2%})")
    else:
        print(f"\nSUMMARY: No valid demonstrations were collected")
    print(f"HDF5 file saved to: {output_path}")
    print(f"Results JSON saved to: {json_output_path}")

    # Verify saved file
    with h5py.File(output_path, "r") as f:
        demo_keys = [key for key in f["data"].keys() if key.startswith("demo_")]
        # Sort demo keys by numerical order
        demo_keys.sort(key=lambda x: int(x.split("_")[1]))
        
        print(f"Saved {len(demo_keys)} demonstrations")
        for key in demo_keys:
            demo_grp = f["data"][key]
            print(f"  {key}: {len(demo_grp['states'])} states, {len(demo_grp['actions'])} actions")

    # Clean up
    env_multiple.close()
    env_single.close()


def joint_sizes(jt: int) -> Tuple[int, int]:
    """Get joint sizes for different MuJoCo joint types.
    
    This function maps MuJoCo joint type identifiers to their corresponding
    position and velocity dimensions. These dimensions are used to properly
    copy joint states between different simulation environments.
    
    Args:
        jt: Joint type identifier from MuJoCo simulation model
        
    Returns:
        Tuple of (position dimensions, velocity dimensions)
        
    Joint Type Mappings:
        0 - Free Joint (mjJNT_FREE):
            - Position dims (7): 3D position (x,y,z) + quaternion (qw,qx,qy,qz)
            - Velocity dims (6): 3D linear velocity + 3D angular velocity
            - Used for objects with complete 6-DOF freedom in 3D space
            
        1 - Ball Joint (mjJNT_BALL):
            - Position dims (4): quaternion representing 3D rotation (qw,qx,qy,qz)
            - Velocity dims (3): 3D angular velocity
            - Allows rotation around all axes but no translation
            
        2 - Slide Joint (mjJNT_SLIDE):
            - Position dims (1): displacement along single axis
            - Velocity dims (1): velocity along that axis
            - Allows linear motion along one axis only (e.g., drawer sliding)
            
        3 - Hinge Joint (mjJNT_HINGE):
            - Position dims (1): angle around single axis
            - Velocity dims (1): angular velocity around that axis
            - Allows rotation around one axis only (e.g., door opening)
    """
    return {0: (7, 6), 1: (4, 3), 2: (1, 1), 3: (1, 1)}[jt]

def copy_overlap(src_env: OffScreenRenderEnv, dst_env: OffScreenRenderEnv) -> None:
    """Copy overlapping joint and mocap states from source to destination environment.
    
    This function transfers the physical state of shared objects between two different
    MuJoCo simulation environments. It's used to initialize the destination environment
    with the same object positions and orientations as the source environment, enabling
    consistent replay across different task configurations.
    
    The function handles:
    1. Joint states (positions and velocities) for all matching joints
    2. Motion capture (mocap) data for cameras and other tracked objects
    3. Different joint types with appropriate dimensional copying

    Note that mocap is important. If not copied, the new environment will not have the same object positions and orientations.

    Args:
        src_env: Source environment to copy state from (typically multi-task env)
        dst_env: Destination environment to copy state to (typically single-task env)
    """
    # Get model and data references for both environments
    m_s, d_s = src_env.env.sim.model, src_env.env.sim.data  # source model & data
    m_d, d_d = dst_env.env.sim.model, dst_env.env.sim.data  # destination model & data

    # Copy joint states for all joints that exist in both environments
    for jid in range(m_s.njnt):  # iterate through all joints in source model
        # Get joint name and handle byte string conversion if needed
        name = m_s.joint_id2name(jid)
        if isinstance(name, bytes):
            name = name.decode()
            
        # Find corresponding joint in destination model
        try:
            jid_d = m_d.joint_name2id(name)
        except ValueError:
            # Joint doesn't exist in destination model, skip it
            continue
        if jid_d < 0:
            # Invalid joint ID, skip it
            continue

        # Get position and velocity dimensions for this joint type
        nq, nv = joint_sizes(m_s.jnt_type[jid])
        
        # Get array indices for joint data in both models
        # MuJoCo stores all joint data in continuous arrays (qpos for positions, qvel for velocities)
        # qposadr and dofadr provide the starting indices for each joint's data within these arrays
        qs, vs = m_s.jnt_qposadr[jid], m_s.jnt_dofadr[jid]      # source starting indices  
        qd, vd = m_d.jnt_qposadr[jid_d], m_d.jnt_dofadr[jid_d]  # destination starting indices
        
        # Copy position and velocity data with appropriate dimensions
        d_d.qpos[qd:qd+nq] = d_s.qpos[qs:qs+nq]  # copy joint positions
        d_d.qvel[vd:vd+nv] = d_s.qvel[vs:vs+nv]  # copy joint velocities

    # Copy motion capture data (for cameras, props, etc.) if both models use it
    nmc = min(m_s.nmocap, m_d.nmocap)  # use minimum number of mocap objects
    if nmc:
        d_d.mocap_pos[:nmc] = d_s.mocap_pos[:nmc]    # copy mocap positions
        d_d.mocap_quat[:nmc] = d_s.mocap_quat[:nmc]  # copy mocap orientations

    # Update the destination simulation to reflect the copied state
    dst_env.env.sim.forward()

def create_hdf5_from_replays(
    output_path: str, 
    collected_demos: List[Any], 
    env_info: str, 
    problem_info: Dict[str, Any], 
    bddl_file_path: str,
    env_single: OffScreenRenderEnv
) -> None:
    """Create an HDF5 file from collected demonstrations.
    
    Args:
        output_path: Path to save the HDF5 file
        collected_demos: List of dictionaries containing states and actions, or None for failed demos
        env_info: Environment configuration information in JSON format
        problem_info: Problem information extracted from BDDL
        bddl_file_path: Path to the BDDL file used for the task
        env_single: Environment instance for extracting model XML
    """
    f = h5py.File(output_path, "w")
    grp = f.create_group("data")
    
    # Store each demonstration
    for demo_idx, demo_data in enumerate(collected_demos):
        if demo_data is None:
            continue
            
        ep_data_grp = grp.create_group("demo_{}".format(demo_idx + 1))
        
        # Store model XML
        model_xml = env_single.env.sim.model.get_xml()
        ep_data_grp.attrs["model_file"] = model_xml
        
        # Write states and actions datasets
        ep_data_grp.create_dataset("states", data=np.array(demo_data["states"]))
        ep_data_grp.create_dataset("actions", data=np.array(demo_data["actions"]))
    
    # Write metadata attributes
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_single.env.__class__.__name__
    grp.attrs["env_info"] = env_info
    grp.attrs["problem_info"] = json.dumps(problem_info)
    grp.attrs["bddl_file_name"] = bddl_file_path
    
    # Read and store BDDL file content
    with open(bddl_file_path, "r", encoding="utf-8") as f:
        bddl_content = f.read()
    grp.attrs["bddl_file_content"] = bddl_content
    
    f.close()


if __name__ == "__main__":
    main()
import os
import argparse
from typing import Any, Dict, List, Optional, Tuple, Union
import re

import pprint
import robosuite.utils.binding_utils
import numpy as np
import importlib
# Dynamically import tqdm for progress bar; fallback to identity
try:
    tqdm = importlib.import_module('tqdm').tqdm
except ImportError:
    tqdm = lambda x, desc=None: x
import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs.env_wrapper import OffScreenRenderEnv

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Parse a BDDL file and display objects in the task."
    )
    parser.add_argument(
        "--benchmark-name", 
        type=str, 
        default="libero_object", 
        help="Benchmark name to use (default: libero_object)"
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=10,
        help="Number of tasks in the benchmark (default: 10)"
    )
    return parser.parse_args()

def joint_sizes(jt: int) -> Tuple[int, int]:
    """
    Get joint sizes for different MuJoCo joint types.
    
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

def find_corresponding_jid(
        model: robosuite.utils.binding_utils.MjModel,
        object_names: List[str]
    ) -> Dict[str, int]:
    """
    Find the corresponding joint ID for a given object name in the MuJoCo model.
    
    Args:
        model: MuJoCo simulation model
        object_names: List of object names to find the corresponding joint for
        
    Returns:
        A dictionary mapping object names to their corresponding joint IDs.
    """
    corresponding_jids = {}
    for jid in range(model.njnt):
        name = model.joint_id2name(jid)
        # match base object name to joint naming pattern
        for obj in object_names:
            pattern = rf'^{re.escape(obj)}(_\d+)?_joint\d+$'
            if re.match(pattern, name):
                corresponding_jids[obj] = jid
                break

    return corresponding_jids

def main():
    args = parse_args()
    benchmark_name = args.benchmark_name
    num_tasks = args.num_tasks

    # Initialize a benchmark (e.g., LIBERO_OBJECT)
    benchmark = get_benchmark(benchmark_name)()
    print(f"Examining {benchmark_name} benchmark with {benchmark.get_num_tasks()} tasks")
    print("-" * 80)
    bddl_files_default_path = get_libero_path("bddl_files")

    for task_id in range(num_tasks):
        print(f"Task ID: {task_id}")
        task = benchmark.get_task(task_id)
        init_states = benchmark.get_task_init_states(task_id)
        bddl_file = os.path.join(bddl_files_default_path, task.problem_folder, task.bddl_file)

        parsed_problem = BDDLUtils.robosuite_parse_problem(bddl_file)
        objects_info = parsed_problem.get("objects", {})
        object_list = list(objects_info.keys())
        print(f"Objects in the problem: {object_list}")
        print("-" * 80)

        env_args = {
            "bddl_file_name": bddl_file,
            "camera_heights": 128,
            "camera_widths": 128
        }
        env = OffScreenRenderEnv(**env_args)

        object_dict = env.env.objects_dict

        qpos = {name: [] for name in object_dict.keys()}
        for idx, init_state in enumerate(tqdm(init_states, desc="Loop over init states")):
            # print(f"Init State {idx}:")
            env.set_init_state(init_state)
            env.reset()
            obs, _, _, _ = env.step([0.] * 7)

            sim = env.env.sim
            model = sim.model
            data = sim.data

            # Dict that stores joint ids for objects in the scene
            object2id = find_corresponding_jid(model, list(object_dict.keys()))

            for name, jid in object2id.items():
                nq, _ = joint_sizes(model.jnt_type[jid])
                qs = model.jnt_qposadr[jid]
                qpos_values = data.qpos[qs:qs+nq].copy()
                qpos[name].append(qpos_values)
            # break

        # Compute mean and std for qpos across init states
        for name, pos_list in qpos.items():
            # Stack sliced first 2 dims (x, y) of qpos values
            arr = np.stack([p[:2] for p in pos_list], axis=0)
            # mean = arr.mean(axis=0)
            # std = arr.std(axis=0)

            arr_min = arr.min(axis=0)
            arr_max = arr.max(axis=0)
            center = (arr_max + arr_min) / 2
            half_range = (arr_max - arr_min) / 2

            center = [f"{v:.6f}" for v in center]
            half_range = [f"{v:.6f}" for v in half_range]
            pm_str = [f"{c} +/- {hr}" for c, hr in zip(center, half_range)]

            print(f"Object: {name}")
            # print(f"  mean: {[f'{v:.6f}' for v in mean]}")
            # print(f"  std: {[f'{v:.6f}' for v in std]}")
            print(f"  center ± half_range: {pm_str}")
            print("-" * 80)

        env.close()

if __name__ == "__main__":
    main()

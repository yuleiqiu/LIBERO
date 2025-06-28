import os
import argparse
from typing import Any, Dict, List, Optional, Tuple, Union
import re

import h5py
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
    parser.add_argument(
        "--object-list",
        type=str,
        nargs='+',
        default=["alphabet_soup"],
        help="Name(s) of the object(s) to inspect (default: [\"alphabet_soup\"])"
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
    object_list = args.object_list

    # Set numpy print options to display 3 decimal places
    np.set_printoptions(formatter={'float_kind': '{:.3f}'.format})

    # Initialize a benchmark (e.g., LIBERO_OBJECT)
    benchmark = get_benchmark(benchmark_name)()
    print(f"Examining {benchmark_name} benchmark with {benchmark.get_num_tasks()} tasks")
    print("-" * 80)
    bddl_files_default_path = get_libero_path("bddl_files")
    demo_files_default_path = get_libero_path("datasets")

    for task_id in range(num_tasks):
        print(f"Task ID: {task_id}")
        task = benchmark.get_task(task_id)
        init_states = benchmark.get_task_init_states(task_id)
        bddl_file = os.path.join(bddl_files_default_path, task.problem_folder, task.bddl_file)
        demo_file = os.path.join(demo_files_default_path, benchmark.get_task_demonstration(task_id))

        # Check if bddl and demo file exist
        assert os.path.exists(bddl_file), f"BDDL file not found: {bddl_file}"
        assert os.path.exists(demo_file), f"Demo file not found: {demo_file}"

        demo_hdf5 = h5py.File(demo_file, "r")["data"]

        env_args = {
            "bddl_file_name": bddl_file,
            "camera_heights": 128,
            "camera_widths": 128
        }
        env = OffScreenRenderEnv(**env_args)

        for demo_idx in range(len(demo_hdf5)):
            if demo_idx > 4:
                print(f"Reached demo_idx {demo_idx}, stopping early")
                env.close()
                break
            demo_key = f"demo_{demo_idx}"
            if demo_key not in demo_hdf5:
                print(f"[!] Dataset does not contain {demo_key}, ending early")
                break
            
            demo = demo_hdf5[demo_key]
            states = demo["states"]

            # states[0] is the initial state, which should match init_states[0]
            assert states[0].shape == init_states[0].shape, \
                f"States[0] shape {states[0].shape} does not match init states[0] shape {init_states[0].shape}"
            
            env.seed(demo_idx+100)
            env.reset()
            obs, _, _, _ = env.step([0.] * 7)

            sim = env.env.sim
            model = sim.model
            data = sim.data

            # Dict that stores joint ids for objects in the scene
            object2id = find_corresponding_jid(model, object_list)

            for name, jid in object2id.items():
                nq, _ = joint_sizes(model.jnt_type[jid])
                qs = model.jnt_qposadr[jid]
                init_state_in_file = init_states[demo_idx][1+qs:1+qs+nq].copy()
                init_state_in_demo = states[0][1+qs:1+qs+nq].copy()
                init_state_in_sim = sim.get_state().flatten()[1+qs:1+qs+nq].copy()

                print(f"Object: {name}")
                print(f"  Initial state in file: {init_state_in_file}")
                print(f"  Initial state in demo: {init_state_in_demo}")
                print(f"  Initial state in sim: {init_state_in_sim}")
        break

if __name__ == "__main__":
    main()

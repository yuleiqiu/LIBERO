import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import numpy as np
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs.env_wrapper import OffScreenRenderEnv

def inspect_init_states_from_benchmark():
    """Load init states through benchmark API and examine their shapes."""
    # Initialize a benchmark (e.g., LIBERO_OBJECT)
    benchmark_name = "libero_object"  # Change to any benchmark you want to examine
    benchmark = get_benchmark(benchmark_name)()

    print(f"Examining {benchmark_name} benchmark with {benchmark.get_num_tasks()} tasks")
    print("-" * 80)

    task_id = 0
    bddl_files_default_path = get_libero_path("bddl_files")
    task = benchmark.get_task(task_id)
    init_states = benchmark.get_task_init_states(task_id)
    bddl_file = os.path.join(bddl_files_default_path, task.problem_folder, task.bddl_file)
    print(f"BDDL file: {bddl_file}")

    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": 128,
        "camera_widths": 128
    }
    env = OffScreenRenderEnv(**env_args)

    for idx, state in enumerate(init_states):
        if idx >= 10:  # Limit to first 10 init states for brevity
            break
        print(f"Init state {idx}:")
        env.set_init_state(state)
        env.reset()
        obs, _, _, _ = env.step([0.] * 7)  # Don't advance physics, just get state

        sim = env.env.sim
        model = sim.model
        data = sim.data

        for jid in range(model.njnt):
            name = model.joint_id2name(jid)
            if isinstance(name, bytes):
                name = name.decode()
            # print(f"Joint {jid}: {name}, type={model.jnt_type[jid]}")
            # if "basket" in name or "soup" in name:
            if jid >= 9 and jid <= 13:
                print(f"Joint {jid}: {name}, type={model.jnt_type[jid]}")
                nq, _ = joint_sizes(model.jnt_type[jid])
                qs = model.jnt_qposadr[jid]
                qpos_values = data.qpos[qs:qs+nq].copy()
                qpos_formatted = [f"{val:.6f}" for val in qpos_values]
                print(f"qpos={qpos_formatted}.")
        # break

        print("-" * 80)

    env.close()

    # # Iterate through all tasks in the benchmark
    # for i in range(benchmark.get_num_tasks()):
    #     task = benchmark.get_task(i)
    #     print(f"Task {i}: {task.name}")
        
    #     # Load init states using the benchmark method
    #     init_states = benchmark.get_task_init_states(i)
        
    #     # Print information about the loaded init states
    #     if isinstance(init_states, dict):
    #         print("  Init states is a dictionary with keys:")
    #         for key, value in init_states.items():
    #             if hasattr(value, "shape"):
    #                 print(f"    {key}: shape={value.shape}, type={type(value)}")
    #             else:
    #                 print(f"    {key}: type={type(value)}")
    #     elif hasattr(init_states, "shape"):
    #         print(f"  Init states shape: {init_states.shape}, type={type(init_states)}")
    #     else:
    #         print(f"  Init states type: {type(init_states)}")
    #     print()

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

if __name__ == "__main__":
    inspect_init_states_from_benchmark()
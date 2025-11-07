#!/usr/bin/env python3

# Standard library imports
import os
import argparse
import pprint

# Third-party imports
import h5py

# LIBERO-specific imports
from libero.libero import benchmark, get_libero_path

def main():
    pp = pprint.PrettyPrinter(indent=2)

    # Allow specifying task_id via command line argument
    parser = argparse.ArgumentParser(description='Visualize init states of a LIBERO task')
    parser.add_argument('--task-id', type=int, default=0, help='Task ID to visualize')
    parser.add_argument('--benchmark', type=str, default="libero_object", help='Benchmark name')
    args = parser.parse_args()

    datasets_default_path = get_libero_path("datasets")
    bddl_files_default_path = get_libero_path("bddl_files")
    benchmark_dict = benchmark.get_benchmark_dict()
    pp.pprint(benchmark_dict)
    print("============================================================")
    benchmark_instance = benchmark_dict[args.benchmark]()
    
    # # Allow specifying task_id via command line argument
    # parser = argparse.ArgumentParser(description='Visualize init states of a LIBERO task')
    # parser.add_argument('--task_id', type=int, default=0, help='Task ID to visualize')
    # args = parser.parse_args()
    task_id = args.task_id
    task = benchmark_instance.get_task(task_id)
    pp.pprint(task)
    print("============================================================")
    # init_states = benchmark_instance.get_task_init_states(task_id)
    # print(f"Number of initial states for task {task_id}: {len(init_states)}")
    demo_file = os.path.join(datasets_default_path, benchmark_instance.get_task_demonstration(task_id))

    with h5py.File(demo_file, "r+") as f:
        demo_keys = [k for k in f["data"].keys() if k.startswith("demo_")]
        # num_demos = len(demo_keys)
        # print(f"Number of demos for task {task_id}: {num_demos}")

        # Process each demo
        for demo_key in demo_keys:
            demo_id = demo_key.split("_")[1]
            # print(f"Processing {demo_key}...")
            actions = f[f"data/{demo_key}/actions"][()]
            print(f"Demo {demo_id} has {len(actions)} actions")
            gripper_actions = actions[:, -1]
            print(f"Gripper actions for demo {demo_id}: {gripper_actions}")

if __name__ == "__main__":
    main()
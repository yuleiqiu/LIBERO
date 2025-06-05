#!/usr/bin/env python3

# Standard library imports
import os
import argparse
import pprint

# Third-party imports
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from termcolor import colored
import h5py

# LIBERO-specific imports
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils.dataset_utils import get_dataset_info

def main():
    pp = pprint.PrettyPrinter(indent=2)

    init_states_default_path = get_libero_path("init_states")
    benchmark_dict = benchmark.get_benchmark_dict()
    pp.pprint(benchmark_dict)
    print("============================================================")
    benchmark_instance = benchmark_dict["libero_goal"]()
    
    # Allow specifying task_id via command line argument
    parser = argparse.ArgumentParser(description='Visualize init states of a LIBERO task')
    parser.add_argument('--task_id', type=int, default=0, help='Task ID to visualize')
    args = parser.parse_args()
    task_id = args.task_id
    task = benchmark_instance.get_task(task_id)
    pp.pprint(task)
    print("============================================================")
    # init_states = benchmark_instance.get_task_init_states(task_id)
    # print(f"Number of initial states for task {task_id} in {task.init_states_file}: {len(init_states)}")
    pruned_init_states_path = os.path.join(init_states_default_path, task.problem_folder, f"{task.name}.pruned_init")
    unpruned_init_states_path = os.path.join(init_states_default_path, task.problem_folder, f"{task.name}.init")
    pruned_init_states = torch.load(pruned_init_states_path)
    print(f"Number of pruned initial states for task {task_id}: {len(pruned_init_states)}")
    print(f"{pruned_init_states.shape=}")
    unpruned_init_states = torch.load(unpruned_init_states_path)
    print(f"{unpruned_init_states.shape=}")

    if np.all(pruned_init_states == unpruned_init_states):
        print("The pruned and unpruned initial states are identical.")

    # # Check if shapes are compatible for comparison
    # if pruned_init_states.shape != unpruned_init_states.shape:
    #     print("Shapes of the two arrays are different, cannot compare row by row.")
    # else:
    #     # Compare row by row
    #     are_rows_equal = np.all(pruned_init_states == unpruned_init_states, axis=1)

    #     # Print the comparison result for each row
    #     for i, equal in enumerate(are_rows_equal):
    #         if equal:
    #             print(f"Row {i}: Pruned == Unpruned")
    #         else:
    #             print(f"Row {i}: Pruned != Unpruned")

    #             # Optional: Find the indices where they differ in this row
    #             # diff_indices = np.where(pruned_init_states[i] != unpruned_init_states[i])[0]
    #             # print(f"  Differences at indices: {diff_indices}")
    #             # print(f"  Pruned values: {pruned_init_states[i, diff_indices]}")
    #             # print(f"  Unpruned values: {unpruned_init_states[i, diff_indices]}")
    

if __name__ == "__main__":
    main()
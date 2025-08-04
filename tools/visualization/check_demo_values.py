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
import imageio

# LIBERO-specific imports
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils.dataset_utils import get_dataset_info

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

            # # 1. 找到第一个1出现的位置
            # ones = np.where(gripper_actions == 1)[0]
            # start_idx = ones[0]
            # # 2. 找到start_idx之后第一个-1的位置
            # minus_ones = np.where((gripper_actions == -1) & (np.arange(len(gripper_actions)) > start_idx))[0]
            # end_idx = minus_ones[0] if len(minus_ones) > 0 else len(gripper_actions)
            # # 3. start_idx到end_idx之间的0变为1
            # gripper_actions[start_idx:end_idx][gripper_actions[start_idx:end_idx] == 0] = 1
            # # 4. start_idx之前全部设为-1
            # gripper_actions[:start_idx] = -1

            # actions[:, -1] = gripper_actions
            # f[f"data/{demo_key}/actions"][...] = actions  # 写回文件

if __name__ == "__main__":
    main()
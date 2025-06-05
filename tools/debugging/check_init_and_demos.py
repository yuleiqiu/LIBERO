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

    datasets_default_path = get_libero_path("datasets")
    bddl_files_default_path = get_libero_path("bddl_files")
    # print("============================================================")
    benchmark_dict = benchmark.get_benchmark_dict()
    pp.pprint(benchmark_dict)
    print("============================================================")
    benchmark_instance = benchmark_dict["libero_object"]()
    
    # Allow specifying task_id via command line argument
    parser = argparse.ArgumentParser(description='Visualize init states of a LIBERO task')
    parser.add_argument('--task_id', type=int, default=0, help='Task ID to visualize')
    args = parser.parse_args()
    task_id = args.task_id
    task = benchmark_instance.get_task(task_id)
    pp.pprint(task)
    print("============================================================")
    init_states = benchmark_instance.get_task_init_states(task_id)
    print(f"Number of initial states for task {task_id}: {len(init_states)}")
    demo_file = os.path.join(datasets_default_path, benchmark_instance.get_task_demonstration(task_id))
    
    with h5py.File(demo_file, "r") as f:
        num_demos = len([k for k in f["data"].keys() if k.startswith("demo_")])
        print(f"Number of demos for task {task_id}: {num_demos}")

if __name__ == "__main__":
    main()
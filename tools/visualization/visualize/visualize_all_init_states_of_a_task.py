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

# LIBERO-specific imports
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils.dataset_utils import get_dataset_info

def main():
    pp = pprint.PrettyPrinter(indent=2)

    # benchmark_root_path = get_libero_path("benchmark_root")
    # init_states_default_path = get_libero_path("init_states")
    datasets_default_path = get_libero_path("datasets")
    bddl_files_default_path = get_libero_path("bddl_files")
    # print("Default benchmark root path: ", benchmark_root_path)
    # print("Default dataset root path: ", datasets_default_path)
    # print("Default bddl files root path: ", bddl_files_default_path)
    # print("============================================================")
    benchmark_dict = benchmark.get_benchmark_dict()
    pp.pprint(benchmark_dict)
    print("============================================================")
    # Allow specifying task_id via command line argument
    parser = argparse.ArgumentParser(description='Visualize init states of a LIBERO task')
    parser.add_argument('--task_id', type=int, default=0, help='Task ID to visualize')
    parser.add_argument('--benchmark', type=str, default="libero_object", help='Benchmark name')
    args = parser.parse_args()
    benchmark_instance = benchmark_dict[args.benchmark]()
    # num_tasks = benchmark_instance.get_num_tasks()
    # print(f"{num_tasks} tasks in the benchmark {benchmark_instance.name}: ")
    # print("============================================================")
    # task_names = benchmark_instance.get_task_names()
    # print("The benchmark contains the following tasks:")
    # for i in range(num_tasks):
    #     task_name = task_names[i]
    #     task = benchmark_instance.get_task(i)
    #     bddl_file = os.path.join(bddl_files_default_path, task.problem_folder, task.bddl_file)
    #     print(f"\t {task_name}")
    #     if not os.path.exists(bddl_file):
    #         print(colored(f"[error] bddl file {bddl_file} cannot be found. Check your paths", "red"))
    # print("============================================================")

    task_id = args.task_id
    task = benchmark_instance.get_task(task_id)
    pp.pprint(task)
    print("============================================================")
    # demo_files_path = os.path.join(datasets_default_path, benchmark_instance.get_task_demonstration(task_id))
    # get_dataset_info(demo_files_path)
    # print("============================================================")
    env_args = {
        "bddl_file_name": os.path.join(bddl_files_default_path, task.problem_folder, task.bddl_file),
        "camera_heights": 128,
        "camera_widths": 128
    }
    env = OffScreenRenderEnv(**env_args)

    init_states = benchmark_instance.get_task_init_states(task_id)
    pp.pprint(f"Init states shape: {init_states.shape}")
    print("============================================================")

    # Fix random seeds for reproducibility
    env.seed(0)

    def make_grid(images, nrow=8, padding=2, normalize=False, pad_value=0):
        """Make a grid of images. Make sure images is a 4D tensor in the shape of (B x C x H x W)) or a list of torch tensors."""
        grid_image = torchvision.utils.make_grid(images, nrow=nrow, padding=padding, normalize=normalize, pad_value=pad_value).permute(1, 2, 0)
        return grid_image

    images = []
    env.reset()
    for eval_index in range(len(init_states)):
        env.set_init_state(init_states[eval_index])

        for _ in range(5):
            obs, _, _, _ = env.step([0.] * 7)
        images.append(torch.from_numpy(obs["agentview_image"]).permute(2, 0, 1))

    grid_image = make_grid(images, nrow=10, padding=2, pad_value=0)
    
    # Convert to numpy and display with matplotlib instead of IPython display
    grid_image_np = grid_image.numpy()[::-1]
    
    plt.figure(figsize=(20, 20))
    plt.imshow(grid_image_np)
    plt.axis('off')
    plt.title(f"Task {task_id}: {task.name} - Initial States")
    
    # Save the figure to a file
    # Create output directory if it doesn't exist
    output_dir = f"visualizations/{benchmark_instance.name}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"task_{task_id}_init_states.png")
    plt.savefig(output_file)
    print(f"Saved visualization to {output_file}")
    
    # Show the figure
    plt.show()
    
    env.close()

if __name__ == "__main__":
    main()
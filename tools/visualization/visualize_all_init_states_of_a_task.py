#!/usr/bin/env python3

"""Visualize initialization states of a LIBERO task and save the visualization without opening a window."""

# Standard library imports
import os
import argparse
import pprint

# Third-party imports
import numpy as np
import torch
import torchvision
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

# LIBERO-specific imports
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

def main():
    pp = pprint.PrettyPrinter(indent=2)

    benchmark_dict = benchmark.get_benchmark_dict()
    bddl_files_default_path = get_libero_path("bddl_files")
    
    # Allow specifying task_id via command line argument
    parser = argparse.ArgumentParser(description='Visualize init states of a LIBERO task')
    parser.add_argument('--task-id', type=int, default=0, help='Task ID to visualize')
    parser.add_argument('--benchmark', type=str, default="libero_object", help='Benchmark name')
    args = parser.parse_args()
    benchmark_instance = benchmark_dict[args.benchmark]()

    task_id = args.task_id
    task = benchmark_instance.get_task(task_id)
    pp.pprint(task)
    print("============================================================")
    env_args = {
        "bddl_file_name": os.path.join(bddl_files_default_path, task.problem_folder, task.bddl_file),
        "camera_heights": 128,
        "camera_widths": 128
    }
    env = OffScreenRenderEnv(**env_args)

    init_states = benchmark_instance.get_task_init_states(task_id)

    # # Debug: print all init states to inspect variations per object
    # print(f"Initial states for task {task_id} ({task.name}):")
    # for idx, state in enumerate(init_states):
    #     print(f"Init state {idx}:")
    #     # If state is a dict mapping object names to their states, print each separately
    #     if isinstance(state, dict):
    #         for obj_key, obj_state in state.items():
    #             print(f"  Object '{obj_key}':")
    #             pp.pprint(obj_state)
    #     else:
    #         with np.printoptions(precision=6, suppress=True):
    #             pp.pprint(state)

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
    output_dir = f"libero/libero/init_files/{benchmark_instance.name}/visualizations"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"task_{task_id}_{task.name}_init_states.png")
    plt.savefig(output_file)
    print(f"Saved visualization to {output_file}")
    
    # Show the figure
    # plt.show()  # disabled interactive display
    plt.close()
    
    env.close()

if __name__ == "__main__":
    main()
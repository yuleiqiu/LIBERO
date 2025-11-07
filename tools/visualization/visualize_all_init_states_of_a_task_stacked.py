#!/usr/bin/env python3

"""Visualize initialization states of a LIBERO task and save the visualization without opening a window."""

# Standard library imports
import os
import argparse
import pprint

# Third-party imports
import numpy as np
import torch
from PIL import Image
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
    parser.add_argument('--render-size', type=int, default=256, help='Square camera render size')
    parser.add_argument('--output-size', type=int, default=640, help='Side length (pixels) of saved visualizations')
    parser.add_argument('--stack-alpha', type=float, default=0.35, help='Transparency for stacked overlay (0-1)')
    args = parser.parse_args()
    benchmark_instance = benchmark_dict[args.benchmark]()

    task_id = args.task_id
    task = benchmark_instance.get_task(task_id)
    pp.pprint(task)
    print("============================================================")
    env_args = {
        "bddl_file_name": os.path.join(bddl_files_default_path, task.problem_folder, task.bddl_file),
        "camera_heights": args.render_size,
        "camera_widths": args.render_size
    }
    env = OffScreenRenderEnv(**env_args)

    init_states = benchmark_instance.get_task_init_states(task_id)

    images = []
    env.reset()
    for eval_index in range(len(init_states)):
        env.set_init_state(init_states[eval_index])

        for _ in range(5):
            obs, _, _, _ = env.step([0.] * 7)
        images.append(torch.from_numpy(obs["agentview_image"]).permute(2, 0, 1))

    def tensor_to_display_np(tensor):
        """Convert CxHxW tensor to HxWxC numpy array (flipped for display)."""
        np_img = tensor.permute(1, 2, 0).numpy()
        return np_img[::-1]

    display_images = [tensor_to_display_np(img) for img in images]

    def resize_image(np_img, size):
        pil_img = Image.fromarray(np_img)
        resized = pil_img.resize((size, size), resample=Image.BILINEAR)
        return np.array(resized)

    resized_images = [resize_image(img, args.output_size) for img in display_images]

    # Save individual enlarged images for close inspection
    output_dir = f"libero/libero/init_files/{benchmark_instance.name}/visualizations"
    os.makedirs(output_dir, exist_ok=True)
    task_dir_name = f"task_{task_id}_{task.name}".replace(" ", "_")
    individual_dir = os.path.join(output_dir, f"{task_dir_name}_individual")
    os.makedirs(individual_dir, exist_ok=True)

    for idx, img in enumerate(resized_images):
        fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
        ax.imshow(img)
        ax.set_title(f"Init State {idx}")
        ax.axis('off')
        individual_path = os.path.join(individual_dir, f"init_state_{idx:03d}.png")
        fig.savefig(individual_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    # Stack translucent images to inspect spatial distribution
    stack_alpha = min(max(args.stack_alpha, 0.05), 1.0)
    stack_fig, stack_ax = plt.subplots(figsize=(7, 7), dpi=200)
    for img in resized_images:
        stack_ax.imshow(img, alpha=stack_alpha)
    stack_ax.set_title(f"Task {task_id}: {task.name}\nStacked init states (alpha={stack_alpha:.2f})")
    stack_ax.axis('off')
    stacked_output = os.path.join(output_dir, f"{task_dir_name}_stacked_overlay.png")
    stack_fig.savefig(stacked_output, bbox_inches='tight', pad_inches=0)
    plt.close(stack_fig)
    print(f"Saved {len(resized_images)} individual images to {individual_dir}")
    print(f"Saved stacked overlay to {stacked_output}")
    
    env.close()

if __name__ == "__main__":
    main()

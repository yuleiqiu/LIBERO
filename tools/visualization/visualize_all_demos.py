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
    init_states = benchmark_instance.get_task_init_states(task_id)
    print(f"Number of initial states for task {task_id}: {len(init_states)}")
    demo_file = os.path.join(datasets_default_path, benchmark_instance.get_task_demonstration(task_id))

    # Create output directory if it doesn't exist
    output_dir = f"libero/datasets/{benchmark_instance.name}/task_{task_id}_demos"
    os.makedirs(output_dir, exist_ok=True)

    # Check if videos already exist in the output directory
    existing_videos = [f for f in os.listdir(output_dir) if f.endswith('.mp4')]
    if existing_videos:
        print(f"Found {len(existing_videos)} existing videos in {output_dir}")
        overwrite = input("Do you want to overwrite existing videos? (y/n): ").lower().strip()
        if overwrite != 'y':
            print("Operation cancelled. Existing videos will not be overwritten.")
            return

    with h5py.File(demo_file, "r") as f:
        demo_keys = [k for k in f["data"].keys() if k.startswith("demo_")]
        num_demos = len(demo_keys)
        print(f"Number of demos for task {task_id}: {num_demos}")

        # Process each demo
        for demo_key in demo_keys:
            demo_id = demo_key.split("_")[1]
            print(f"Processing {demo_key}...")
            
            # Get images for both views
            agentview_images = f[f"data/{demo_key}/obs/agentview_rgb"][()]
            eye_in_hand_images = f[f"data/{demo_key}/obs/eye_in_hand_rgb"][()]
            
            # Create video file with side-by-side views
            output_path = os.path.join(output_dir, f"demo_{demo_id}.mp4")
            video_writer = imageio.get_writer(output_path, fps=60)
            
            for agentview_img, eye_in_hand_img in zip(agentview_images, eye_in_hand_images):
                # Create a side-by-side composite image
                # First, flip the images (the ::-1 operation)
                agentview_img = agentview_img[::-1]
                eye_in_hand_img = eye_in_hand_img[::-1]
                
                # Concatenate the images horizontally
                combined_img = np.hstack((agentview_img, eye_in_hand_img))
                
                # Add to video
                video_writer.append_data(combined_img)
            
            video_writer.close()
            print(f"Saved side-by-side video to {output_path}")
    
    print(f"All {num_demos} demos have been saved as videos in the '{output_dir}' directory")

if __name__ == "__main__":
    main()
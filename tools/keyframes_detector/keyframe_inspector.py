#!/usr/bin/env python3

# Standard library imports
import json
import os
import argparse
import pprint

# Third-party imports
import numpy as np
import tqdm
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
    parser.add_argument('--output-dir', type=str, default="tools/keyframes_detector/tmp", help='Output directory for videos')
    args = parser.parse_args()

    datasets_default_path = get_libero_path("datasets")
    bddl_dir = get_libero_path("bddl_files")
    benchmark_dict = benchmark.get_benchmark_dict()
    pp.pprint(benchmark_dict)
    print("============================================================")
    benchmark_instance = benchmark_dict[args.benchmark]()
    task_id = args.task_id
    task = benchmark_instance.get_task(task_id)
    pp.pprint(task)
    print("============================================================")

    demo_files = os.path.join(datasets_default_path, benchmark_instance.get_task_demonstration(task_id))
    demo_h5 = h5py.File(demo_files, "r")["data"]
    demo_keys = [k for k in demo_h5.keys() if k.startswith("demo_")]
    num_demos = len(demo_keys)
    print(f"Number of demos for task {task_id}: {num_demos}")

    bddl_file_name = os.path.join(bddl_dir, task.problem_folder, task.bddl_file)
    print(f"Using BDDL file: {bddl_file_name}")
    env = OffScreenRenderEnv(bddl_file_name=bddl_file_name)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    for demo_idx in range(num_demos):
        demo_key = demo_keys[demo_idx]
        # print(f"Processing {demo_key}...")
        env.reset()

        states = demo_h5[f"{demo_key}/states"][()]
        agentview_images = []
        eye_in_hand_images = []
        for _, state in enumerate(tqdm.tqdm(states, desc=f"Processing {demo_key}")):
            obs = env.regenerate_obs_from_state(state)
            # print(f"Observation keys: {obs.keys()}")
            agentview_images.append(obs["agentview_image"])
            eye_in_hand_images.append(obs["robot0_eye_in_hand_image"])

        output_path = os.path.join(output_dir, f"{demo_key}.mp4")
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
        print(f"Saved video for demo {demo_idx} at {output_path}")
        video_writer.close()

        env.close()
        break

    # print(f"All {num_demos} demos have been saved as videos in the '{output_dir}' directory")

if __name__ == "__main__":
    main()
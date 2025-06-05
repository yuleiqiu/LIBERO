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

# ---------- Configuration ----------
task_suite_name = "libero_object"
output_dir      = os.path.join("replay", "hdf5_output")

# ---------- Path ----------
# benchmark_dict  = benchmark.get_benchmark_dict()
# task_suite      = benchmark_dict[task_suite_name]()
# task            = task_suite.get_task(task_id)

# bddl_dir        = get_libero_path("bddl_files")
# bddl_full       = os.path.join(bddl_dir, task.problem_folder, task.bddl_file)

# current_dir     = os.path.dirname(os.path.abspath(__file__))
# parent_dir      = os.path.dirname(current_dir)
# tmp_bddl_dir    = os.path.join(parent_dir, "tmp", "pddl_files")
# bddl_simplified = os.path.join(tmp_bddl_dir,
#                                "MY_FLOOR_SCENE_pick_the_alphabet_soup_and_place_it_in_the_basket_2.bddl")

demo_dir        = get_libero_path("datasets")
demo_file       = os.path.join(demo_dir, "processed", "MY_FLOOR_SCENE_pick_the_alphabet_soup_and_place_it_in_the_basket_2_processed_demo.hdf5")

# # ---------- Create environment ----------
# cam_args        = dict(camera_heights=128, camera_widths=128)
# env_full        = OffScreenRenderEnv(bddl_file_name=bddl_full,       **cam_args)
# env_simple      = OffScreenRenderEnv(bddl_file_name=bddl_simplified, **cam_args)


# Create output directory if it doesn't exist
output_dir = f"visualize/videos/demos/custom/floor_single_alphabet_soup_demos"
os.makedirs(output_dir, exist_ok=True)

# Check if videos already exist in the output directory
existing_videos = [f for f in os.listdir(output_dir) if f.endswith('.mp4')]
if existing_videos:
    print(f"Found {len(existing_videos)} existing videos in {output_dir}")
    overwrite = input("Do you want to overwrite existing videos? (y/n): ").lower().strip()
    if overwrite != 'y':
        print("Operation cancelled. Existing videos will not be overwritten.")
        exit(0)

with h5py.File(demo_file, "r") as f:
    demo_keys = [k for k in f["data"].keys() if k.startswith("demo_")]
    num_demos = len(demo_keys)
    print(f"Number of demos for task: {num_demos}")

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
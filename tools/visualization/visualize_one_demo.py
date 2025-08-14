#!/usr/bin/env python3
"""
Visualize a single demo from a hdf5 file.
"""

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
    parser = argparse.ArgumentParser(description='Visualize one demo from a hdf5 file')

    parser.add_argument(
        "--demo_file",
        type=str,
        help="Path to the demo hdf5 file to be visualized",
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save the output videos",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not args.output_dir:
        output_dir = "tmp/visualization/"
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    demo_file = args.demo_file
    if not os.path.exists(demo_file):
        raise FileNotFoundError(f"Demo file {demo_file} does not exist.")
    with h5py.File(demo_file, "r") as f:
        demo_keys = [k for k in f["data"].keys() if k.startswith("demo_")]
        num_demos = len(demo_keys)

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
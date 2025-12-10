#!/usr/bin/env python3
"""Visualize all demos of one task for a given benchmark"""


# Standard library imports
import os
import argparse
import pprint

# Third-party imports
import numpy as np
import h5py
import imageio

# LIBERO-specific imports
from libero.libero import benchmark, get_libero_path

def main():
    pp = pprint.PrettyPrinter(indent=2)

    # Allow specifying task_id via command line argument
    parser = argparse.ArgumentParser(description='Visualize init states of a LIBERO task')
    parser.add_argument('--task-id', type=int, default=0, help='Task ID to visualize')
    parser.add_argument('--benchmark', type=str, default="libero_object", help='Benchmark name')
    parser.add_argument(
        '--hdf5-path',
        type=str,
        default=None,
        help='Optional path to a demonstration hdf5 file or directory containing one. '
             'Overrides benchmark/task-id lookup when provided.',
    )
    args = parser.parse_args()

    task_id = args.task_id

    benchmark_instance = benchmark.get_benchmark(args.benchmark)()
    task = benchmark_instance.get_task(task_id)
    pp.pprint(task)
    print("============================================================")
    if args.hdf5_path:
        input_path = os.path.expanduser(args.hdf5_path)
        if os.path.isdir(input_path):
            hdf5_files = [f for f in os.listdir(input_path) if f.endswith(".hdf5")]
            if not hdf5_files:
                raise FileNotFoundError(f"No .hdf5 files found under {input_path}")
            if len(hdf5_files) > 1:
                raise ValueError(
                    f"Multiple .hdf5 files found under {input_path}: {hdf5_files}. "
                    "Please specify one explicitly."
                )
            demo_file = os.path.join(input_path, hdf5_files[0])
        else:
            demo_file = input_path
        print(f"Using provided hdf5 file: {demo_file}")
    else:
        datasets_default_path = get_libero_path("datasets")
        demo_file = os.path.join(datasets_default_path, benchmark_instance.get_task_demonstration(task_id))
        print(f"Located hdf5 via benchmark/task_id: {demo_file}")

    if not os.path.exists(demo_file):
        raise FileNotFoundError(f"hdf5 file not found: {demo_file}")

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

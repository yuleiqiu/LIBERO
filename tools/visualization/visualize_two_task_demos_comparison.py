#!/usr/bin/env python3
"""Visualize paired demos from two benchmarks in a single four-panel video."""

# Standard library imports
import argparse
import os
import pprint

# Third-party imports
import h5py
import imageio
import numpy as np

# LIBERO-specific imports
from libero.libero import benchmark, get_libero_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare demos from two benchmarks by stacking their views."
    )
    parser.add_argument("--benchmark-a", type=str, required=True, help="First benchmark name")
    parser.add_argument("--task-a-id", type=int, required=True, help="Task id from first benchmark")
    parser.add_argument("--benchmark-b", type=str, required=True, help="Second benchmark name")
    parser.add_argument("--task-b-id", type=int, required=True, help="Task id from second benchmark")
    parser.add_argument("--fps", type=int, default=60, help="FPS for the output videos")
    return parser.parse_args()


def build_task_context(benchmark_name, task_id):
    bench_instance = benchmark.get_benchmark(benchmark_name)()
    task = bench_instance.get_task(task_id)
    datasets_default_path = get_libero_path("datasets")
    demo_file = os.path.join(
        datasets_default_path, bench_instance.get_task_demonstration(task_id)
    )
    return bench_instance, task, demo_file


def pad_image(img, target_height=None, target_width=None):
    """Pad image with zeros to reach the requested shape."""
    height, width = img.shape[:2]
    target_height = target_height or height
    target_width = target_width or width
    pad_bottom = max(target_height - height, 0)
    pad_right = max(target_width - width, 0)
    if pad_bottom == 0 and pad_right == 0:
        return img
    return np.pad(
        img,
        pad_width=((0, pad_bottom), (0, pad_right), (0, 0)),
        mode="constant",
        constant_values=0,
    )


def hstack_with_padding(left_img, right_img):
    target_height = max(left_img.shape[0], right_img.shape[0])
    left = pad_image(left_img, target_height=target_height)
    right = pad_image(right_img, target_height=target_height)
    return np.hstack((left, right))


def vstack_with_padding(top_img, bottom_img):
    target_width = max(top_img.shape[1], bottom_img.shape[1])
    top = pad_image(top_img, target_width=target_width)
    bottom = pad_image(bottom_img, target_width=target_width)
    return np.vstack((top, bottom))


def fetch_demo_frames(h5_file, demo_key):
    agentview_images = h5_file[f"data/{demo_key}/obs/agentview_rgb"][()]
    eye_in_hand_images = h5_file[f"data/{demo_key}/obs/eye_in_hand_rgb"][()]
    return agentview_images, eye_in_hand_images


def flip_image(img):
    """Match the orientation used by the single-task visualizer."""
    return img[::-1]


def main():
    args = parse_args()
    pp = pprint.PrettyPrinter(indent=2)

    bench_a, task_a, demo_file_a = build_task_context(args.benchmark_a, args.task_a_id)
    bench_b, task_b, demo_file_b = build_task_context(args.benchmark_b, args.task_b_id)

    print("Task A:")
    pp.pprint(task_a)
    print("============================================================")
    print("Task B:")
    pp.pprint(task_b)
    print("============================================================")

    comparison_name = (
        f"{bench_a.name}_task_{args.task_a_id}_vs_{bench_b.name}_task_{args.task_b_id}"
    )
    output_dir = os.path.join("libero", "datasets", "comparisons", comparison_name)
    os.makedirs(output_dir, exist_ok=True)

    existing_videos = [f for f in os.listdir(output_dir) if f.endswith(".mp4")]
    if existing_videos:
        print(f"Found {len(existing_videos)} existing videos in {output_dir}")
        overwrite = input("Do you want to overwrite existing videos? (y/n): ").lower().strip()
        if overwrite != "y":
            print("Operation cancelled. Existing videos will not be overwritten.")
            return

    with h5py.File(demo_file_a, "r") as file_a, h5py.File(demo_file_b, "r") as file_b:
        demo_keys_a = sorted(
            (k for k in file_a["data"].keys() if k.startswith("demo_")),
            key=lambda x: int(x.split("_")[1]),
        )
        demo_keys_b = sorted(
            (k for k in file_b["data"].keys() if k.startswith("demo_")),
            key=lambda x: int(x.split("_")[1]),
        )

        num_demos_a = len(demo_keys_a)
        num_demos_b = len(demo_keys_b)
        num_pairs = min(num_demos_a, num_demos_b)

        print(
            f"Task A demos: {num_demos_a}, Task B demos: {num_demos_b}. "
            f"Creating {num_pairs} paired comparison videos."
        )

        if num_pairs == 0:
            print("No demos available in one of the tasks. Nothing to visualize.")
            return

        for idx in range(num_pairs):
            demo_key_a = demo_keys_a[idx]
            demo_key_b = demo_keys_b[idx]
            print(f"Processing {demo_key_a} vs {demo_key_b} ...")

            agent_a, hand_a = fetch_demo_frames(file_a, demo_key_a)
            agent_b, hand_b = fetch_demo_frames(file_b, demo_key_b)

            num_frames_a = agent_a.shape[0]
            num_frames_b = agent_b.shape[0]
            total_frames = max(num_frames_a, num_frames_b)

            output_path = os.path.join(output_dir, f"comparison_{idx:03d}.mp4")
            video_writer = imageio.get_writer(output_path, fps=args.fps)

            for frame_idx in range(total_frames):
                idx_a = min(frame_idx, num_frames_a - 1)
                idx_b = min(frame_idx, num_frames_b - 1)

                frame_agent_a = flip_image(agent_a[idx_a])
                frame_hand_a = flip_image(hand_a[idx_a])
                frame_agent_b = flip_image(agent_b[idx_b])
                frame_hand_b = flip_image(hand_b[idx_b])

                top_row = hstack_with_padding(frame_agent_a, frame_hand_a)
                bottom_row = hstack_with_padding(frame_agent_b, frame_hand_b)
                combined_frame = vstack_with_padding(top_row, bottom_row)

                video_writer.append_data(combined_frame)

            video_writer.close()
            print(f"Saved comparison video to {output_path}")

    print(
        f"Finished saving {num_pairs} comparison videos in the '{output_dir}' directory."
    )


if __name__ == "__main__":
    main()

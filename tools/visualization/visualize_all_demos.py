#!/usr/bin/env python3
"""Visualize LIBERO demos from one unified entrypoint."""


import argparse
import os
import pprint
import sys

import h5py

try:
    from ._demo_video_utils import (
        build_task_context,
        fetch_demo_frames,
        flip_image,
        hstack_with_padding,
        list_demo_keys,
        normalize_demo_key,
        render_demo_video,
        resolve_hdf5_path,
        vstack_with_padding,
        write_video,
    )
except ImportError:
    from _demo_video_utils import (
        build_task_context,
        fetch_demo_frames,
        flip_image,
        hstack_with_padding,
        list_demo_keys,
        normalize_demo_key,
        render_demo_video,
        resolve_hdf5_path,
        vstack_with_padding,
        write_video,
    )


def _confirm_overwrite(output_dir: str) -> bool:
    existing_videos = [f for f in os.listdir(output_dir) if f.endswith(".mp4")]
    if not existing_videos:
        return True
    print(f"Found {len(existing_videos)} existing videos in {output_dir}")
    overwrite = input("Do you want to overwrite existing videos? (y/n): ").lower().strip()
    if overwrite != "y":
        print("Operation cancelled. Existing videos will not be overwritten.")
        return False
    return True


def _resolve_demo_file(benchmark_name: str, task_id: int, hdf5_path: str = None):
    benchmark_instance, task, benchmark_demo_file = build_task_context(benchmark_name, task_id)
    if hdf5_path:
        demo_file = resolve_hdf5_path(hdf5_path)
        source_msg = f"Using provided hdf5 file: {demo_file}"
    else:
        demo_file = benchmark_demo_file
        source_msg = f"Located hdf5 via benchmark/task_id: {demo_file}"
    return benchmark_instance, task, demo_file, source_msg


def _build_parser():
    parser = argparse.ArgumentParser(description="Visualize LIBERO demos.")
    subparsers = parser.add_subparsers(dest="command")

    all_parser = subparsers.add_parser("all", help="Export all demos from one task.")
    all_parser.add_argument("--task-id", type=int, default=0, help="Task ID to visualize")
    all_parser.add_argument("--benchmark", type=str, default="libero_object", help="Benchmark name")
    all_parser.add_argument(
        "--hdf5-path",
        type=str,
        default=None,
        help="Optional path to a demonstration hdf5 file or directory containing one. Overrides benchmark/task-id lookup when provided.",
    )
    all_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save output videos. Defaults to libero/datasets/<benchmark>/task_<id>_demos.",
    )
    all_parser.add_argument("--fps", type=int, default=60, help="FPS for the output videos.")

    one_parser = subparsers.add_parser("one", help="Export one demo from a dataset.")
    one_parser.add_argument("--demo-file", type=str, required=True, help="Path to the demo hdf5 file.")
    one_parser.add_argument(
        "--demo-id",
        type=str,
        default=None,
        help="Demo id to visualize (for example: 0 or demo_0). Defaults to the first demo.",
    )
    one_parser.add_argument("--output-dir", type=str, default=None, help="Directory to save the output video.")
    one_parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional full path for the output video. Overrides --output-dir when set.",
    )
    one_parser.add_argument("--fps", type=int, default=60, help="FPS for the output video.")

    compare_parser = subparsers.add_parser("compare", help="Compare paired demos across two tasks.")
    compare_parser.add_argument("--benchmark-a", type=str, required=True, help="First benchmark name")
    compare_parser.add_argument("--task-a-id", type=int, required=True, help="Task id from first benchmark")
    compare_parser.add_argument("--benchmark-b", type=str, required=True, help="Second benchmark name")
    compare_parser.add_argument("--task-b-id", type=int, required=True, help="Task id from second benchmark")
    compare_parser.add_argument("--fps", type=int, default=60, help="FPS for the output videos")

    return parser


def _parse_args(argv):
    parser = _build_parser()
    known_commands = {"all", "one", "compare"}
    normalized_argv = list(argv)
    if not normalized_argv or normalized_argv[0] not in known_commands:
        normalized_argv.insert(0, "all")
    return parser.parse_args(normalized_argv)


def _run_all(args):
    pp = pprint.PrettyPrinter(indent=2)
    benchmark_instance, task, demo_file, source_msg = _resolve_demo_file(
        args.benchmark,
        args.task_id,
        hdf5_path=args.hdf5_path,
    )
    pp.pprint(task)
    print("============================================================")
    print(source_msg)

    if not os.path.exists(demo_file):
        raise FileNotFoundError(f"hdf5 file not found: {demo_file}")

    output_dir = (
        args.output_dir
        or f"libero/datasets/{benchmark_instance.name}/task_{args.task_id}_demos"
    )
    os.makedirs(output_dir, exist_ok=True)
    if not _confirm_overwrite(output_dir):
        return

    with h5py.File(demo_file, "r") as f:
        demo_keys = list_demo_keys(f["data"])
        num_demos = len(demo_keys)
        print(f"Number of demos for task {args.task_id}: {num_demos}")

        for demo_key in demo_keys:
            print(f"Processing {demo_key}...")
            output_path = os.path.join(output_dir, f"{demo_key}.mp4")
            render_demo_video(f, demo_key, output_path, fps=args.fps)
            print(f"Saved side-by-side video to {output_path}")

    print(f"All {num_demos} demos have been saved as videos in the '{output_dir}' directory")


def _run_one(args):
    demo_file = os.path.expanduser(args.demo_file)
    if not os.path.exists(demo_file):
        raise FileNotFoundError(f"Demo file {demo_file} does not exist.")

    with h5py.File(demo_file, "r") as f:
        demo_key = normalize_demo_key(f["data"], demo_id=args.demo_id)

        if args.output_path:
            output_path = os.path.expanduser(args.output_path)
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        else:
            output_dir = args.output_dir or "tmp/visualization"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{demo_key}.mp4")

        print(f"Processing {demo_key}...")
        render_demo_video(f, demo_key, output_path, fps=args.fps)
        print(f"Saved side-by-side video to {output_path}")


def _run_compare(args):
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
    if not _confirm_overwrite(output_dir):
        return

    with h5py.File(demo_file_a, "r") as file_a, h5py.File(demo_file_b, "r") as file_b:
        demo_keys_a = list_demo_keys(file_a["data"])
        demo_keys_b = list_demo_keys(file_b["data"])

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

            def iter_frames():
                for frame_idx in range(total_frames):
                    idx_a = min(frame_idx, num_frames_a - 1)
                    idx_b = min(frame_idx, num_frames_b - 1)

                    frame_agent_a = flip_image(agent_a[idx_a])
                    frame_hand_a = flip_image(hand_a[idx_a])
                    frame_agent_b = flip_image(agent_b[idx_b])
                    frame_hand_b = flip_image(hand_b[idx_b])

                    top_row = hstack_with_padding(frame_agent_a, frame_hand_a)
                    bottom_row = hstack_with_padding(frame_agent_b, frame_hand_b)
                    yield vstack_with_padding(top_row, bottom_row)

            write_video(output_path, iter_frames(), fps=args.fps)
            print(f"Saved comparison video to {output_path}")

    print(
        f"Finished saving {num_pairs} comparison videos in the '{output_dir}' directory."
    )


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "one":
        _run_one(args)
    elif args.command == "compare":
        _run_compare(args)
    else:
        _run_all(args)


if __name__ == "__main__":
    main()

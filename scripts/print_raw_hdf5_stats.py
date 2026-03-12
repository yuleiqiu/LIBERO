#!/usr/bin/env python3
import argparse

import h5py
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print trajectory counts and lengths for raw demo.hdf5 files."
    )
    parser.add_argument(
        "--demo-file",
        required=True,
        help="Path to a raw HDF5 file produced by collect_human_demos_by_anchor.py",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print aggregate statistics, not per-trajectory lengths.",
    )
    return parser.parse_args()


def demo_sort_key(name):
    try:
        return int(name.split("_")[1])
    except (IndexError, ValueError):
        return name


def main():
    args = parse_args()

    with h5py.File(args.demo_file, "r") as f:
        if "data" not in f:
            raise ValueError(f"Missing 'data' group in {args.demo_file}")

        demos = sorted(
            [key for key in f["data"].keys() if key.startswith("demo_")],
            key=demo_sort_key,
        )
        if not demos:
            raise ValueError(f"No demo_* groups found in {args.demo_file}")

        lengths = np.array([f[f"data/{demo}/actions"].shape[0] for demo in demos], dtype=int)

        print(f"dataset: {args.demo_file}")
        print(f"num_trajectories: {len(demos)}")
        print(f"total_transitions: {int(lengths.sum())}")
        print(f"traj_length_mean: {lengths.mean():.2f}")
        print(f"traj_length_std: {lengths.std():.2f}")
        print(f"traj_length_min: {int(lengths.min())}")
        print(f"traj_length_max: {int(lengths.max())}")

        if not args.summary_only:
            print("trajectory_lengths:")
            for demo, length in zip(demos, lengths.tolist()):
                print(f"  {demo}: {length}")


if __name__ == "__main__":
    main()

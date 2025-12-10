#!/usr/bin/env python3

import argparse
import os
import sys
from typing import Iterable, List, Tuple

import h5py
import numpy as np


def describe(values: np.ndarray) -> Tuple[float, float, float, float]:
    """Return (mean, std, min, max) for a 1D array."""
    return (
        float(np.mean(values)),
        float(np.std(values)),
        float(np.min(values)),
        float(np.max(values)),
    )


def format_stats(label: str, stats: Tuple[float, float, float, float]) -> str:
    mean, std, min_v, max_v = stats
    return (
        f"{label:>10s}: mean={mean:.6f}  std={std:.6f}  "
        f"min={min_v:.6f}  max={max_v:.6f}"
    )


def resolve_demo_keys(data_group, demo_id: str = None) -> List[str]:
    demo_keys = [k for k in data_group.keys() if k.startswith("demo_")]
    if not demo_keys:
        raise ValueError("No demo_* groups found under the HDF5 'data' group.")

    if demo_id is None:
        return sorted(demo_keys)

    # Allow passing either an integer (e.g., 0) or full key name (e.g., demo_0)
    target_key = demo_id if demo_id.startswith("demo_") else f"demo_{demo_id}"
    if target_key not in data_group:
        raise ValueError(f"Requested demo '{target_key}' not found. Available: {demo_keys}")
    return [target_key]


def summarize_demo_actions(demo_key: str, actions: np.ndarray) -> None:
    if actions.ndim != 2:
        raise ValueError(f"Expected actions to be 2D (T x D), got shape {actions.shape}")

    print(f"\n{demo_key}: {actions.shape[0]} timesteps, {actions.shape[1]} dims")
    overall_stats = describe(actions.reshape(-1))
    print(format_stats("overall", overall_stats))

    # Per-dimension stats
    for dim in range(actions.shape[1]):
        dim_stats = describe(actions[:, dim])
        print(format_stats(f"dim_{dim}", dim_stats))


def summarize_dim_across_demos(actions_list: List[np.ndarray], dim: int) -> None:
    """Aggregate stats for a specific dimension across multiple demos."""
    if not actions_list:
        return
    dim_size = actions_list[0].shape[1]
    if dim < 0 or dim >= dim_size:
        raise ValueError(f"dim={dim} is out of bounds for actions with {dim_size} dims.")
    # Ensure consistent shape across demos
    for arr in actions_list:
        if arr.shape[1] != dim_size:
            raise ValueError("Inconsistent action dimensions across demos; cannot aggregate.")
    stacked = np.concatenate([arr[:, dim] for arr in actions_list], axis=0)
    print(f"\nAcross demos dim_{dim}:")
    print(format_stats(f"dim_{dim}", describe(stacked)))


def summarize_file(path: str, demo_id: str = None, aggregate: bool = False, dim: int = None) -> None:
    with h5py.File(path, "r") as f:
        if "data" not in f:
            raise ValueError("HDF5 file missing top-level 'data' group.")

        data_group = f["data"]
        selected_keys = resolve_demo_keys(data_group, demo_id)
        collected: List[np.ndarray] = []

        for key in selected_keys:
            if "actions" not in data_group[key]:
                print(f"Skipping {key}: no 'actions' dataset found.")
                continue
            actions = np.asarray(data_group[key]["actions"])
            summarize_demo_actions(key, actions)
            collected.append(actions)

        if aggregate and collected:
            combined = np.concatenate(collected, axis=0)
            print("\nAggregate over selected demos:")
            summarize_demo_actions("combined", combined)

        if dim is not None:
            summarize_dim_across_demos(collected, dim)


def parse_args(args: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display basic statistics (mean/std/min/max) of actions in a demo file."
    )
    parser.add_argument(
        "demo_path",
        type=str,
        help="Path to the HDF5 demo file (e.g., ~/datasets/libero/.../demo.hdf5).",
    )
    parser.add_argument(
        "--demo-id",
        type=str,
        default=None,
        help="Specific demo id to inspect (e.g., 0 or demo_0). Defaults to all demos.",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Also print stats aggregated across the selected demos.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=None,
        help="Compute aggregated stats for a specific action dimension across all selected demos.",
    )
    return parser.parse_args(args)


def main(argv: Iterable[str]) -> None:
    args = parse_args(argv)
    demo_path = os.path.expanduser(args.demo_path)
    if not os.path.isfile(demo_path):
        raise FileNotFoundError(f"Demo file not found: {demo_path}")

    summarize_file(demo_path, demo_id=args.demo_id, aggregate=args.aggregate, dim=args.dim)


if __name__ == "__main__":
    main(sys.argv[1:])

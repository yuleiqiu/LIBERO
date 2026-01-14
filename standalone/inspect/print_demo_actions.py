#!/usr/bin/env python3
"""
Print action values (and gripper values) from LIBERO demo HDF5 files.
"""

import argparse
import os
import sys
from typing import Iterable, List, Optional

import h5py
import numpy as np


def parse_demo_ids(raw: str) -> Optional[List[str]]:
    if not raw:
        return None
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if not tokens:
        return None
    demo_ids: List[str] = []
    for token in tokens:
        if token.startswith("demo_"):
            demo_ids.append(token)
        elif token.isdigit():
            demo_ids.append(f"demo_{int(token)}")
        else:
            raise ValueError(
                f"Invalid demo id '{token}'. Use integers (e.g. 0) or demo_* keys."
            )
    return demo_ids


def demo_sort_key(key: str):
    suffix = key.split("_", 1)[-1]
    return int(suffix) if suffix.isdigit() else suffix


def resolve_demo_keys(data_group, demo_ids: Optional[List[str]]) -> List[str]:
    demo_keys = [key for key in data_group.keys() if key.startswith("demo_")]
    if not demo_keys:
        raise ValueError("No demo_* groups found under the HDF5 'data' group.")
    if demo_ids is None:
        return sorted(demo_keys, key=demo_sort_key)
    missing = [key for key in demo_ids if key not in demo_keys]
    if missing:
        raise ValueError(f"Requested demos not found: {missing}")
    return demo_ids


def format_unique(values: np.ndarray, max_unique: int = 10) -> str:
    rounded = np.round(values, 6)
    unique_vals = np.unique(rounded)
    if unique_vals.size <= max_unique:
        items = ", ".join(f"{val:.6f}" for val in unique_vals.tolist())
        return f"[{items}]"
    return f"{unique_vals.size} unique values"


def print_demo_actions(
    demo_key: str,
    actions: np.ndarray,
    max_steps: int,
    gripper_only: bool,
) -> None:
    if actions.ndim != 2:
        raise ValueError(f"{demo_key}: expected 2D actions, got shape {actions.shape}")

    total_steps, dims = actions.shape
    gripper = actions[:, -1]
    gripper_summary = (
        f"min={float(np.min(gripper)):.6f}, "
        f"max={float(np.max(gripper)):.6f}, "
        f"mean={float(np.mean(gripper)):.6f}, "
        f"unique={format_unique(gripper)}"
    )

    print(f"{demo_key}: {total_steps} steps, {dims} dims")
    print(f"  gripper (last dim) stats: {gripper_summary}")

    if max_steps == 0:
        return

    if max_steps < 0:
        slice_actions = actions
    else:
        slice_actions = actions[:max_steps]

    if gripper_only:
        for idx, value in enumerate(slice_actions[:, -1]):
            print(f"  {idx:04d}: {float(value):.6f}")
    else:
        for idx, action in enumerate(slice_actions):
            formatted = np.array2string(action, precision=6, separator=", ")
            print(f"  {idx:04d}: {formatted}")

    if 0 <= max_steps < total_steps:
        print(f"  ... truncated {total_steps - max_steps} steps "
              f"(use --max-steps -1 to print all)")


def summarize_file(
        path: str,
        demo_ids: Optional[List[str]],
        max_steps: int,
        gripper_only: bool,
) -> None:
    with h5py.File(path, "r") as f:
        if "data" not in f:
            raise ValueError(f"{path}: missing top-level 'data' group")
        data_group = f["data"]
        selected_keys = resolve_demo_keys(data_group, demo_ids)
        for key in selected_keys:
            if "actions" not in data_group[key]:
                print(f"{key}: no actions dataset found, skipping")
                continue
            actions = np.asarray(data_group[key]["actions"])
            print_demo_actions(key, actions, max_steps, gripper_only)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print action values from LIBERO demo HDF5 files."
    )
    parser.add_argument(
        "demo_files",
        nargs="+",
        help="Path(s) to demo HDF5 files.",
    )
    parser.add_argument(
        "--demo-ids",
        type=str,
        default="",
        help="Comma-separated demo ids/keys (e.g., 0,1 or demo_0,demo_1).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Max timesteps to print per demo. Use -1 to print all.",
    )
    parser.add_argument(
        "--gripper-only",
        action="store_true",
        help="Only print the gripper action (last dimension).",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str]) -> None:
    args = parse_args(argv)
    demo_ids = parse_demo_ids(args.demo_ids)
    for demo_file in args.demo_files:
        path = os.path.expanduser(demo_file)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Demo file not found: {path}")
        print(f"\n=== {path} ===")
        summarize_file(path, demo_ids, args.max_steps, args.gripper_only)


if __name__ == "__main__":
    main(sys.argv[1:])

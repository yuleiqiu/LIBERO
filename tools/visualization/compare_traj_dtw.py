#!/usr/bin/env python3
"""Compute DTW similarity between two LIBERO end-effector trajectories."""

import argparse
import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import h5py
import numpy as np

from libero.libero import benchmark, get_libero_path


@dataclass
class TrajectorySpec:
    benchmark: str
    task_id: int
    demo_index: int
    label: str


def _load_benchmark(benchmark_name: str):
    benchmark_dict = benchmark.get_benchmark_dict()
    if benchmark_name not in benchmark_dict:
        known = ", ".join(sorted(benchmark_dict.keys()))
        raise ValueError(f"Unknown benchmark '{benchmark_name}'. Available: {known}")
    return benchmark_dict[benchmark_name]()


def _load_positions(spec: TrajectorySpec) -> np.ndarray:
    bench = _load_benchmark(spec.benchmark)
    datasets_root = get_libero_path("datasets")
    demo_path = os.path.join(datasets_root, bench.get_task_demonstration(spec.task_id))
    with h5py.File(demo_path, "r") as handle:
        demo_groups = sorted(k for k in handle["data"].keys() if k.startswith("demo_"))
        if not demo_groups:
            raise RuntimeError(f"No demonstrations found for task {spec.task_id} in {demo_path}")
        if spec.demo_index < 0 or spec.demo_index >= len(demo_groups):
            raise IndexError(
                f"demo_index {spec.demo_index} out of range (0-{len(demo_groups) - 1}) for '{demo_path}'."
            )
        demo_key = demo_groups[spec.demo_index]
        obs_group = handle[f"data/{demo_key}/obs"]
        if "ee_pos" in obs_group:
            positions = np.asarray(obs_group["ee_pos"][()])
        elif "ee_states" in obs_group:
            ee_states = np.asarray(obs_group["ee_states"][()])
            positions = ee_states[:, :3]
        else:
            raise KeyError(f"{demo_key} lacks 'ee_pos' or 'ee_states' observations.")
    if positions.ndim != 2 or positions.shape[1] < 3:
        raise ValueError(
            f"Expected EE positions with shape (T, 3+) for '{demo_key}'. Got {positions.shape}."
        )
    return positions


def _parse_spec(spec_str: str) -> TrajectorySpec:
    parts = spec_str.split(":")
    if len(parts) not in (3, 4):
        raise ValueError(
            f"Invalid --traj '{spec_str}'. Expected benchmark:task_id:demo_idx[:label]."
        )
    try:
        task_id = int(parts[1])
        demo_idx = int(parts[2])
    except ValueError as exc:
        raise ValueError(f"Task id and demo index must be integers in '{spec_str}'.") from exc
    label = parts[3] if len(parts) == 4 else f"{parts[0]}:task{task_id}:{demo_idx}"
    return TrajectorySpec(parts[0], task_id, demo_idx, label)


def _compute_dtw(
    seq_a: np.ndarray, seq_b: np.ndarray, norm: str = "l2"
) -> Tuple[float, float, List[Tuple[int, int]]]:
    if seq_a.size == 0 or seq_b.size == 0:
        raise ValueError("DTW requires non-empty trajectories.")
    if seq_a.shape[1] != seq_b.shape[1]:
        raise ValueError(
            f"DTW sequences must share dimensionality. Got {seq_a.shape[1]} and {seq_b.shape[1]}."
        )

    if norm not in ("l1", "l2"):
        raise ValueError("--norm must be either 'l1' or 'l2'.")

    len_a, len_b = seq_a.shape[0], seq_b.shape[0]
    cost = np.full((len_a + 1, len_b + 1), np.inf, dtype=np.float64)
    cost[0, 0] = 0.0

    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            delta = seq_a[i - 1] - seq_b[j - 1]
            if norm == "l2":
                step = float(np.linalg.norm(delta))
            else:
                step = float(np.linalg.norm(delta, ord=1))
            cost[i, j] = step + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    i, j = len_a, len_b
    path: List[Tuple[int, int]] = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        prev_vals = (
            cost[i - 1, j],  # insertion
            cost[i, j - 1],  # deletion
            cost[i - 1, j - 1],  # match
        )
        move = int(np.argmin(prev_vals))
        if move == 0:
            i -= 1
        elif move == 1:
            j -= 1
        else:
            i -= 1
            j -= 1

    while i > 0:
        i -= 1
        path.append((i, 0))
    while j > 0:
        j -= 1
        path.append((0, j))

    path.reverse()
    total_cost = float(cost[len_a, len_b])
    normalized = total_cost / len(path) if path else float("nan")
    return total_cost, normalized, path


def _warp_sequence(
    reference: np.ndarray, target: np.ndarray, path: Sequence[Tuple[int, int]]
) -> np.ndarray:
    warped = np.zeros_like(reference)
    counts = np.zeros(reference.shape[0], dtype=np.int32)
    for ref_idx, target_idx in path:
        if ref_idx < 0 or target_idx < 0:
            continue
        warped[ref_idx] += target[target_idx]
        counts[ref_idx] += 1

    missing = counts == 0
    if np.any(missing):
        warped[missing] = reference[missing]
        counts[missing] = 1

    warped /= counts[:, None]
    return warped


def save_alignment(path: Sequence[Tuple[int, int]], save_path: str):
    np.savetxt(save_path, np.asarray(path, dtype=np.int32), fmt="%d", delimiter=",")
    print(f"Saved DTW alignment path with {len(path)} steps to {save_path}")


def save_warped(reference_label: str, cmp_label: str, warped: np.ndarray, save_path: str):
    np.savez_compressed(save_path, reference_label=reference_label, warped=warped)
    print(f"Saved warped trajectory for '{cmp_label}' (aligned to '{reference_label}') at {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute DTW between two LIBERO trajectories identified by benchmark/task/demo."
    )
    parser.add_argument(
        "--traj",
        action="append",
        metavar="SPEC",
        help="Trajectory spec benchmark:task_id:demo_index[:label]. Provide exactly two.",
    )
    parser.add_argument(
        "--norm", choices=("l1", "l2"), default="l2", help="Distance metric for DTW (default: l2)."
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Optional path to store DTW alignment indices as CSV.",
    )
    parser.add_argument(
        "--save-warped",
        type=str,
        default=None,
        help="Optional path to store the warped version of the second trajectory (npz).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.traj or len(args.traj) != 2:
        raise ValueError("Provide exactly two --traj specifications (benchmark:task_id:demo_idx[:label]).")

    specs = [_parse_spec(spec_str) for spec_str in args.traj]
    trajectories = [_load_positions(spec) for spec in specs]

    distance, normalized_distance, path = _compute_dtw(trajectories[0], trajectories[1], norm=args.norm)

    ref_spec, cmp_spec = specs
    print("DTW comparison")
    print("--------------")
    print(f"Reference: {ref_spec.label} (length {trajectories[0].shape[0]})")
    print(f"Compared : {cmp_spec.label} (length {trajectories[1].shape[0]})")
    print(f"Metric   : {args.norm}")
    print(f"Distance : {distance:.6f}")
    print(f"Normalized distance: {normalized_distance:.6f}")
    print(f"Alignment steps: {len(path)}")

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True) if os.path.dirname(args.save_path) else None
        save_alignment(path, args.save_path)

    if args.save_warped:
        warped = _warp_sequence(trajectories[0], trajectories[1], path)
        os.makedirs(os.path.dirname(args.save_warped), exist_ok=True) if os.path.dirname(args.save_warped) else None
        save_warped(ref_spec.label, cmp_spec.label, warped, args.save_warped)


if __name__ == "__main__":
    main()

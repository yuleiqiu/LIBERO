#!/usr/bin/env python3
"""Plot one or more 3D end-effector trajectories for LIBERO demonstrations."""

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from libero.libero import benchmark, get_libero_path


@dataclass
class TrajectorySpec:
    benchmark: str
    task_id: int
    demo_index: int
    label: str


def _load_benchmark(benchmark_name: str):
    """Return an instantiated benchmark, raising a helpful error otherwise."""
    benchmark_dict = benchmark.get_benchmark_dict()
    if benchmark_name not in benchmark_dict:
        known = ", ".join(sorted(benchmark_dict.keys()))
        raise ValueError(f"Unknown benchmark '{benchmark_name}'. Available: {known}")
    return benchmark_dict[benchmark_name]()


def _get_demo_group(benchmark_name: str, task_id: int, demo_index: int) -> Tuple[np.ndarray, str]:
    """Load a demo group and return its EE positions and label."""
    benchmark_instance = _load_benchmark(benchmark_name)
    datasets_root = get_libero_path("datasets")
    demo_file = os.path.join(datasets_root, benchmark_instance.get_task_demonstration(task_id))

    with h5py.File(demo_file, "r") as handle:
        demo_groups = [k for k in handle["data"].keys() if k.startswith("demo_")]
        demo_groups.sort()
        if not demo_groups:
            raise RuntimeError(f"No demos found for task {task_id} in {demo_file}")
        if demo_index < 0 or demo_index >= len(demo_groups):
            raise IndexError(f"demo_index {demo_index} out of range (0-{len(demo_groups) - 1})")
        demo_key = demo_groups[demo_index]
        obs_group = handle[f"data/{demo_key}/obs"]
        if "ee_pos" in obs_group:
            ee_positions = np.asarray(obs_group["ee_pos"][()])
        elif "ee_states" in obs_group:
            ee_states = np.asarray(obs_group["ee_states"][()])
            ee_positions = ee_states[:, :3]
        else:
            raise KeyError(f"{demo_key} is missing 'ee_pos' and 'ee_states' observations.")

    return ee_positions, demo_key


def plot_trajectories(trajs: List[Tuple[np.ndarray, str]], elev: float, azim: float):
    """Create the 3D trajectory plot for the provided EE positions."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["royalblue"])
    for idx, (positions, label) in enumerate(trajs):
        color = color_cycle[idx % len(color_cycle)]
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        ax.plot(x, y, z, color=color, linewidth=2, label=label)
        ax.scatter(x[0], y[0], z[0], color=color, s=30, marker="o")
        ax.scatter(x[-1], y[-1], z[-1], color=color, s=30, marker="^")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("End-Effector Trajectory Comparison")
    ax.legend(loc="best")
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, linestyle="--", linewidth=0.5)
    return fig


def _parse_traj_spec(spec: str) -> TrajectorySpec:
    """Parse a spec string of the form benchmark:task_id:demo_idx[:label]."""
    parts = spec.split(":")
    if len(parts) not in (3, 4):
        raise ValueError(
            f"Invalid --traj '{spec}'. Expected benchmark:task_id:demo_idx[:label]."
        )
    benchmark_name = parts[0]
    try:
        task_id = int(parts[1])
        demo_index = int(parts[2])
    except ValueError as exc:
        raise ValueError(f"Task id and demo index must be integers in '{spec}'.") from exc
    label = parts[3] if len(parts) == 4 else f"{benchmark_name}:task{task_id}:{demo_index}"
    return TrajectorySpec(benchmark_name, task_id, demo_index, label)


def _collect_specs(args) -> List[TrajectorySpec]:
    """Build trajectory specs either from --traj or legacy single-demo flags."""
    specs: List[TrajectorySpec] = []
    if args.traj:
        specs = [_parse_traj_spec(spec) for spec in args.traj]
    else:
        if args.task_id is None:
            raise ValueError("Provide --task-id/--benchmark/--demo-index or at least one --traj.")
        label = args.label or f"{args.benchmark}:task{args.task_id}:{args.demo_index}"
        specs = [TrajectorySpec(args.benchmark, args.task_id, args.demo_index, label)]
    return specs


def main():
    parser = argparse.ArgumentParser(description="Plot one or more 3D EEF trajectories.")
    parser.add_argument("--task-id", type=int, help="Task ID to visualize (used when --traj is not provided).")
    parser.add_argument("--demo-index", type=int, default=0, help="Which demo within the task to load.")
    parser.add_argument("--benchmark", type=str, default="libero_object", help="Benchmark name.")
    parser.add_argument("--label", type=str, help="Legend label for the single-demo mode.")
    parser.add_argument(
        "--traj",
        action="append",
        metavar="SPEC",
        help="Add a trajectory: benchmark:task_id:demo_index[:label]. Can be repeated.",
    )
    parser.add_argument("--save-path", type=str, default=None, help="Optional output path for the figure (.png, .pdf, etc.).")
    parser.add_argument("--elev", type=float, default=30.0, help="Elevation angle for the 3D view.")
    parser.add_argument("--azim", type=float, default=-60.0, help="Azimuth angle for the 3D view.")
    parser.add_argument("--no-show", action="store_true", help="Skip rendering an interactive window.")
    args = parser.parse_args()

    specs = _collect_specs(args)

    trajectories = []
    for spec in specs:
        positions, demo_key = _get_demo_group(spec.benchmark, spec.task_id, spec.demo_index)
        if positions.ndim != 2 or positions.shape[1] < 3:
            raise ValueError(f"Expected positions with shape (T, 3+) for {demo_key}. Got {positions.shape}.")
        label = spec.label or demo_key
        trajectories.append((positions, label))

    fig = plot_trajectories(trajectories, args.elev, args.azim)

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True) if os.path.dirname(args.save_path) else None
        fig.savefig(args.save_path, bbox_inches="tight", dpi=300)
        print(f"Saved figure to {args.save_path}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()

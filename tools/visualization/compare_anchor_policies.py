#!/usr/bin/env python3
"""
Visualize the difference between the legacy fixed pick-place path and the new randomized policy.
This script does not run the simulator; it just plots the scripted waypoints and interpolated paths.
"""
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def random_lateral_offset(start_pos, goal_pos, max_offset=0.08):
    direction = goal_pos[:2] - start_pos[:2]
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        direction = np.array([1.0, 0.0])
        norm = 1.0
    tangent = direction / norm
    lateral_dir = np.array([-tangent[1], tangent[0]])
    magnitude = np.random.uniform(0.0, max_offset) * np.random.choice([-1.0, 1.0])
    return np.array([lateral_dir[0] * magnitude, lateral_dir[1] * magnitude, 0.0])


def legacy_waypoints(target_pos, destination_pos):
    return [
        {"t": 0, "xyz": target_pos + np.array([0, 0, 0.25])},
        {"t": 120, "xyz": target_pos + np.array([0, 0, 0.08])},
        {"t": 170, "xyz": target_pos + np.array([0, 0, 0.0])},
        {"t": 200, "xyz": target_pos + np.array([0, 0, 0.12])},
        {"t": 250, "xyz": destination_pos + np.array([0, 0, 0.30])},
        {"t": 320, "xyz": destination_pos + np.array([0, 0, 0.12])},
        {"t": 370, "xyz": destination_pos + np.array([0, 0, 0.12])},
        {"t": 380, "xyz": destination_pos + np.array([0, 0, 0.32])},
        {"t": 400, "xyz": destination_pos + np.array([0, 0, 0.32])},
    ]


def randomized_waypoints(target_pos, destination_pos):
    hover_range = (0.18, 0.28)
    transfer_height_range = (0.22, 0.34)
    max_lateral_offset = 0.07

    def dt(low, high):
        return int(np.random.randint(low, high))

    def transfer_midpoints(start, goal):
        mode = np.random.choice(["direct", "arc", "double_arc"])
        mids = []
        if mode == "direct":
            center = (start + goal) / 2.0
            center[2] = np.random.uniform(*transfer_height_range)
            mids.append(center)
        elif mode == "arc":
            offset = random_lateral_offset(start, goal, max_lateral_offset)
            center = (start + goal) / 2.0 + offset
            center[2] = np.random.uniform(*transfer_height_range)
            mids.append(center)
        else:
            offset1 = random_lateral_offset(start, goal, max_lateral_offset)
            offset2 = random_lateral_offset(goal, start, max_lateral_offset)
            mid1 = start + 0.35 * (goal - start) + offset1
            mid2 = start + 0.7 * (goal - start) + offset2
            mid1[2] = np.random.uniform(*transfer_height_range)
            mid2[2] = np.random.uniform(*transfer_height_range)
            mids.extend([mid1, mid2])
        return mids

    waypoints = []
    t = 0
    hover_pick = np.random.uniform(*hover_range)
    pre_grasp_height = np.random.uniform(0.06, 0.10)
    grasp_depth = np.random.uniform(-0.005, 0.005)
    post_grasp_lift = np.random.uniform(0.16, 0.22)

    hover_place = np.random.uniform(*hover_range)
    place_depth = np.random.uniform(-0.005, 0.01)
    retreat_height = np.random.uniform(0.22, 0.32)

    def add_wp(pos, dt_step):
        nonlocal t
        waypoints.append({"t": t, "xyz": pos.copy()})
        t += dt_step

    add_wp(target_pos + np.array([0, 0, hover_pick]), 0)
    add_wp(target_pos + np.array([0, 0, pre_grasp_height]), dt(70, 120))
    add_wp(target_pos + np.array([0, 0, grasp_depth]), dt(35, 60))
    add_wp(target_pos + np.array([0, 0, post_grasp_lift]), dt(40, 70))

    transfer_start = target_pos + np.array([0, 0, post_grasp_lift])
    transfer_goal = destination_pos + np.array([0, 0, hover_place])
    for mid in transfer_midpoints(transfer_start, transfer_goal):
        add_wp(mid, dt(50, 90))
    add_wp(transfer_goal, dt(60, 100))

    add_wp(destination_pos + np.array([0, 0, place_depth]), dt(45, 80))
    add_wp(destination_pos + np.array([0, 0, place_depth]), dt(10, 20))
    add_wp(destination_pos + np.array([0, 0, retreat_height]), dt(30, 60))
    add_wp(destination_pos + np.array([0, 0, retreat_height]), dt(15, 25))
    return waypoints


def interpolate_path(waypoints):
    xs, ys, zs = [], [], []
    ts = []
    for idx in range(len(waypoints) - 1):
        curr = waypoints[idx]
        nxt = waypoints[idx + 1]
        t_curr, t_next = curr["t"], nxt["t"]
        for t in range(t_curr, t_next):
            frac = (t - t_curr) / (t_next - t_curr + 1e-6)
            xyz = curr["xyz"] + (nxt["xyz"] - curr["xyz"]) * frac
            xs.append(xyz[0])
            ys.append(xyz[1])
            zs.append(xyz[2])
            ts.append(t)
    # include final waypoint
    xs.append(waypoints[-1]["xyz"][0])
    ys.append(waypoints[-1]["xyz"][1])
    zs.append(waypoints[-1]["xyz"][2])
    ts.append(waypoints[-1]["t"])
    return np.array(ts), np.stack([xs, ys, zs], axis=1)


def plot_trajectories(target_pos, dest_pos, num_random, out_path, seed=None):
    if seed is not None:
        np.random.seed(seed)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_xy, ax_z = axes

    # Legacy path
    legacy = legacy_waypoints(target_pos, dest_pos)
    t_legacy, xyz_legacy = interpolate_path(legacy)
    ax_xy.plot(xyz_legacy[:, 0], xyz_legacy[:, 1], "-o", label="legacy", alpha=0.9)
    ax_z.plot(t_legacy, xyz_legacy[:, 2], "-o", label="legacy", alpha=0.9)

    # Randomized samples
    for i in range(num_random):
        wps = randomized_waypoints(target_pos, dest_pos)
        t_rand, xyz_rand = interpolate_path(wps)
        alpha = 0.4 + 0.6 * (i == 0)
        ax_xy.plot(xyz_rand[:, 0], xyz_rand[:, 1], "-", label="random" if i == 0 else None, alpha=alpha)
        ax_z.plot(t_rand, xyz_rand[:, 2], "-", label="random" if i == 0 else None, alpha=alpha)

    ax_xy.set_title("Top-down (XY)")
    ax_xy.set_xlabel("x")
    ax_xy.set_ylabel("y")
    ax_xy.scatter([target_pos[0]], [target_pos[1]], c="red", marker="x", s=80, label="target")
    ax_xy.scatter([dest_pos[0]], [dest_pos[1]], c="green", marker="*", s=120, label="destination")
    ax_xy.legend()
    ax_xy.axis("equal")

    ax_z.set_title("Height over time (Z)")
    ax_z.set_xlabel("timestep")
    ax_z.set_ylabel("z")
    ax_z.legend()

    fig.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=200)
        print(f"Saved to {out_path}")
    else:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Compare legacy vs randomized anchor policy trajectories.")
    parser.add_argument("--target", nargs=3, type=float, default=[0.0, 0.0, 0.0], help="Target object position (x y z).")
    parser.add_argument("--destination", nargs=3, type=float, default=[0.25, -0.2, 0.0], help="Destination position (x y z).")
    parser.add_argument("--num-random", type=int, default=5, help="Number of randomized trajectories to plot.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--output", type=str, default="tools/visualization/policy_traj_comparison.png", help="Path to save the figure. Leave empty to show interactively.")
    return parser.parse_args()


def main():
    args = parse_args()
    target = np.array(args.target, dtype=float)
    dest = np.array(args.destination, dtype=float)
    out_path = args.output if args.output.strip() else None
    plot_trajectories(target, dest, args.num_random, out_path, args.seed)


if __name__ == "__main__":
    # Allow running from any working directory when invoked via python -m
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if repo_root not in sys.path:
        sys.path.append(repo_root)
    main()

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs import bddl_utils as BDDLUtils


def sanitize_ranges(raw_ranges):
    cleaned = []
    for entry in raw_ranges:
        if len(entry) != 4:
            raise ValueError(f"Expected 4 values per range, received {entry}")
        x0, y0, x1, y1 = entry
        cleaned.append(
            (
                min(float(x0), float(x1)),
                min(float(y0), float(y1)),
                max(float(x0), float(x1)),
                max(float(y0), float(y1)),
            )
        )
    return cleaned


def infer_tracked_object(parsed_problem):
    movable_objects = []
    for names in parsed_problem["objects"].values():
        movable_objects.extend(names)
    if len(movable_objects) != 1:
        raise ValueError(
            "This script assumes a single movable object environment. "
            f"Found movable objects: {movable_objects}"
        )
    return movable_objects[0]


def infer_region_key(initial_state, object_name):
    for state in initial_state:
        if (
            isinstance(state, list)
            and len(state) >= 3
            and state[0].lower() == "on"
            and state[1] == object_name
        ):
            return state[2]
    raise ValueError(
        f"Could not infer init region for object '{object_name}' from initial_state."
    )


def infer_goal_region_key(goal_state):
    for state in goal_state:
        if (
            isinstance(state, list)
            and len(state) >= 3
            and state[0].lower() in {"in", "on"}
        ):
            return state[2]
    raise ValueError("Could not infer target region from goal_state.")


def convex_hull(points):
    pts = np.unique(np.asarray(points, dtype=float), axis=0)
    if len(pts) <= 2:
        return pts

    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))

    hull = np.array(lower[:-1] + upper[:-1], dtype=float)
    return hull


def rotate_for_grid(points):
    pts = np.asarray(points, dtype=float)
    rotated = np.empty_like(pts)
    rotated[..., 0] = pts[..., 1]
    rotated[..., 1] = pts[..., 0]
    return rotated


def rect_corners_2d(xmin, ymin, xmax, ymax):
    return np.array(
        [
            [xmin, ymin],
            [xmax, ymin],
            [xmax, ymax],
            [xmin, ymax],
        ],
        dtype=float,
    )


def site_box_corners(center, rotation, size):
    center = np.asarray(center, dtype=float)
    rot = np.asarray(rotation, dtype=float).reshape(3, 3)
    half_size = np.asarray(size, dtype=float)
    axis_x = rot[:, 0] * half_size[0]
    axis_y = rot[:, 1] * half_size[1]
    corners = []
    for sign_x, sign_y in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        corners.append(center + sign_x * axis_x + sign_y * axis_y)
    return np.asarray(corners, dtype=float)


def add_polygon_patch(ax, polygon, **kwargs):
    patch = patches.Polygon(np.asarray(polygon, dtype=float), closed=True, **kwargs)
    ax.add_patch(patch)
    return patch


def render_workspace_illustration(
    env,
    defined_ranges,
    sampled_positions,
    illustration_path,
    include_robot=False,
    target_region_key=None,
):
    illustration_path = Path(illustration_path)
    illustration_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 6.5), dpi=200)
    plotted_items = []

    for idx, (xmin, ymin, xmax, ymax) in enumerate(defined_ranges):
        polygon_2d = rotate_for_grid(rect_corners_2d(xmin, ymin, xmax, ymax))
        add_polygon_patch(
            ax,
            polygon_2d,
            fill=False,
            edgecolor="#d94841",
            linestyle="--",
            linewidth=1.5,
            label="init range" if idx == 0 else None,
        )
        plotted_items.append(polygon_2d)

    projected_samples = []
    for pos in sampled_positions:
        projected_samples.append(rotate_for_grid(np.asarray(pos[:2], dtype=float)))
    if projected_samples:
        projected_samples = np.asarray(projected_samples, dtype=float)
        ax.scatter(
            projected_samples[:, 0],
            projected_samples[:, 1],
            s=18,
            color="#2b6cb0",
            alpha=0.8,
            label="sampled init positions",
        )
        plotted_items.append(projected_samples)

    if include_robot:
        robot_plot = rotate_for_grid(np.asarray(env.env.robots[0].base_pos[:2], dtype=float))
        ax.scatter(
            robot_plot[0],
            robot_plot[1],
            marker="s",
            s=90,
            color="#7b2cbf",
            label="robot",
        )
        ax.text(
            robot_plot[0],
            robot_plot[1] + 0.03,
            "",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#4a1d73",
        )
        plotted_items.append(robot_plot[None, :])

    if target_region_key:
        if target_region_key not in env.env.object_sites_dict:
            available = ", ".join(sorted(env.env.object_sites_dict.keys()))
            raise ValueError(
                f"Target region '{target_region_key}' not found in object_sites_dict. "
                f"Available regions: {available}"
            )
        site = env.env.object_sites_dict[target_region_key]
        site_center = env.sim.data.get_site_xpos(target_region_key)
        site_rotation = env.sim.data.get_site_xmat(target_region_key)
        site_polygon = site_box_corners(site_center, site_rotation, site.size)
        site_polygon_2d = rotate_for_grid(np.asarray(site_polygon[:, :2], dtype=float))
        add_polygon_patch(
            ax,
            site_polygon_2d,
            facecolor="#ffb703",
            edgecolor="#c96c00",
            linewidth=1.5,
            alpha=0.35,
            label="target region",
        )
        site_center_plot = rotate_for_grid(np.asarray(site_center[:2], dtype=float))
        ax.text(
            site_center_plot[0],
            site_center_plot[1],
            "",
            ha="center",
            va="center",
            fontsize=8,
            color="#8a4b00",
            bbox={
                "facecolor": "white",
                "alpha": 0.85,
                "edgecolor": "none",
                "pad": 1.5,
            },
        )
        plotted_items.append(site_polygon_2d)

    if not plotted_items:
        raise ValueError("No illustration primitives were available for the top-down plot.")

    bounds = np.vstack(plotted_items)
    min_xy = bounds.min(axis=0)
    max_xy = bounds.max(axis=0)
    padding = 0.1
    ax.set_xlim(min_xy[0] - padding, max_xy[0] + padding)
    ax.set_ylim(min_xy[1] - padding, max_xy[1] + padding)
    ax.invert_yaxis()
    ax.set_aspect("equal", "box")
    ax.set_xlabel("y (m)")
    ax.set_ylabel("x (m)")
    ax.set_title("Illustration")
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(illustration_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot the sampled init distribution of a single-object BDDL task."
    )
    parser.add_argument("--bddl-file", required=True, help="Path to the BDDL file.")
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of environment resets to sample.",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=128,
        help="Render height; only used to construct the environment.",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=128,
        help="Render width; only used to construct the environment.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory. Default: ./tmp/new_scene_overview/<bddl_file_stem>",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for reproducible sampling.",
    )
    parser.add_argument(
        "--no-illustration",
        action="store_true",
        help="Disable the extra illustration output.",
    )
    parser.add_argument(
        "--illustration-path",
        type=str,
        default=None,
        help="Custom output path for the illustration image.",
    )
    parser.add_argument(
        "--include-robot",
        action="store_true",
        help="Show the robot base in the illustration.",
    )
    parser.add_argument(
        "--target-region-key",
        type=str,
        default=None,
        help="Target region to highlight. Use 'goal' to infer from the goal predicate.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive.")

    bddl_path = Path(args.bddl_file).expanduser().resolve()
    if not bddl_path.exists():
        raise FileNotFoundError(f"BDDL file not found: {bddl_path}")

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else Path("./tmp/new_scene_overview") / bddl_path.stem
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    illustration_path = (
        Path(args.illustration_path).expanduser().resolve()
        if args.illustration_path
        else out_dir / "illustration.png"
    )

    parsed = BDDLUtils.robosuite_parse_problem(str(bddl_path))
    tracked_object = infer_tracked_object(parsed)
    region_key = infer_region_key(parsed["initial_state"], tracked_object)
    defined_ranges = sanitize_ranges(parsed["regions"][region_key]["ranges"])
    target_region_key = args.target_region_key
    if target_region_key == "goal":
        target_region_key = infer_goal_region_key(parsed["goal_state"])

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=args.camera_height,
        camera_widths=args.camera_width,
    )
    if args.seed is not None:
        env.seed(args.seed)

    sampled_positions = []
    for _ in range(args.samples):
        env.reset()
        object_pos = env.env.sim.data.body_xpos[env.env.obj_body_id[tracked_object]].copy()
        sampled_positions.append(object_pos)

    sampled_positions = np.asarray(sampled_positions)
    sampled_xy = sampled_positions[:, :2]
    sampled_xy = np.asarray(sampled_xy)
    hull = convex_hull(sampled_xy)
    xy_min = sampled_xy.min(axis=0)
    xy_max = sampled_xy.max(axis=0)
    rotated_samples = rotate_for_grid(sampled_xy)
    rotated_hull = rotate_for_grid(hull) if len(hull) else hull

    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=200)

    for idx, (xmin, ymin, xmax, ymax) in enumerate(defined_ranges):
        rotated_rect = rotate_for_grid(rect_corners_2d(xmin, ymin, xmax, ymax))
        rect = patches.Rectangle(
            rotated_rect.min(axis=0),
            rotated_rect[:, 0].max() - rotated_rect[:, 0].min(),
            rotated_rect[:, 1].max() - rotated_rect[:, 1].min(),
            fill=False,
            edgecolor="tab:red",
            linestyle="--",
            linewidth=1.5,
            label="BDDL range" if idx == 0 else None,
        )
        ax.add_patch(rect)

    if len(hull) >= 3:
        ax.fill(
            rotated_hull[:, 0],
            rotated_hull[:, 1],
            color="tab:blue",
            alpha=0.15,
            label="sample convex hull",
        )
        closed_hull = np.vstack([rotated_hull, rotated_hull[0]])
        ax.plot(closed_hull[:, 0], closed_hull[:, 1], color="tab:blue", linewidth=1.5)

    ax.scatter(
        rotated_samples[:, 0],
        rotated_samples[:, 1],
        s=18,
        color="tab:blue",
        alpha=0.75,
        label=f"{args.samples} sampled positions",
    )

    rotated_mean = rotate_for_grid(sampled_xy.mean(axis=0))
    ax.scatter(
        rotated_mean[0],
        rotated_mean[1],
        marker="x",
        s=90,
        color="black",
        linewidths=2.0,
        label="sample mean",
    )

    ax.invert_yaxis()
    ax.set_xlabel("y (m)")
    ax.set_ylabel("x (m)")
    ax.set_title(f"Init distribution: {tracked_object}")
    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    padding = 0.03
    rotated_range_points = []
    for rect in defined_ranges:
        rotated_range_points.append(rotate_for_grid(rect_corners_2d(*rect)))
    rotated_range_points = np.vstack(rotated_range_points)
    ax.set_xlim(
        min(rotated_samples[:, 0].min(), rotated_range_points[:, 0].min()) - padding,
        max(rotated_samples[:, 0].max(), rotated_range_points[:, 0].max()) + padding,
    )
    ax.set_ylim(
        min(rotated_samples[:, 1].min(), rotated_range_points[:, 1].min()) - padding,
        max(rotated_samples[:, 1].max(), rotated_range_points[:, 1].max()) + padding,
    )

    summary = (
        f"min=({xy_min[0]:+.3f}, {xy_min[1]:+.3f})\n"
        f"max=({xy_max[0]:+.3f}, {xy_max[1]:+.3f})"
    )
    ax.text(
        0.02,
        0.98,
        summary,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )

    plot_path = out_dir / "init_distribution.png"
    npy_path = out_dir / "init_distribution_xy.npy"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close(fig)
    np.save(npy_path, sampled_xy)

    if not args.no_illustration:
        render_workspace_illustration(
            env=env,
            defined_ranges=defined_ranges,
            sampled_positions=sampled_positions,
            illustration_path=illustration_path,
            include_robot=args.include_robot,
            target_region_key=target_region_key,
        )

    print(f"[info] tracked object: {tracked_object}")
    print(f"[info] init region: {region_key}")
    if target_region_key is not None:
        print(f"[info] target region: {target_region_key}")
    print(f"[info] sampled {args.samples} resets")
    print(f"[info] observed x range: [{xy_min[0]:+.6f}, {xy_max[0]:+.6f}]")
    print(f"[info] observed y range: [{xy_min[1]:+.6f}, {xy_max[1]:+.6f}]")
    print(f"[info] saved plot to {plot_path}")
    if not args.no_illustration:
        print(f"[info] saved illustration to {illustration_path}")
    print(f"[info] saved raw xy samples to {npy_path}")

    env.close()


if __name__ == "__main__":
    main()

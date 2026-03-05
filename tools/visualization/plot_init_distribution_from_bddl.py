#!/usr/bin/env python3
"""Unified BDDL visualization entrypoint.

Subcommands:
- distribution: sampled init distribution and optional illustration.
- verify: check whether samples land on discrete init patches.
- anchors: map anchor indices to discrete init points.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs import bddl_utils as BDDLUtils

try:
    from ._bddl_vis_utils import infer_region_key, sanitize_ranges
except ImportError:
    from _bddl_vis_utils import infer_region_key, sanitize_ranges


def load_anchor_indices(json_path=None, hdf5_path=None, inline=None):
    anchors = []

    def extend_from_value(val):
        if val is None:
            return
        if isinstance(val, (list, tuple)):
            anchors.extend(val)
        elif isinstance(val, np.ndarray):
            anchors.extend(val.tolist())
        else:
            raise ValueError(f"Unsupported anchor container type: {type(val)}")

    if json_path:
        with open(json_path, "r") as file:
            data = json.load(file)
        if isinstance(data, dict):
            for key in ("anchor_idx", "anchor_id", "anchors"):
                if key in data:
                    extend_from_value(data[key])
                    break
            else:
                raise ValueError(f"JSON file {json_path} does not contain anchor indices.")
        elif isinstance(data, list):
            extend_from_value(data)
        else:
            raise ValueError(f"Unsupported JSON structure in {json_path}")

    if hdf5_path:
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("h5py is required to read anchor indices from an HDF5 file.") from exc

        with h5py.File(hdf5_path, "r") as file:

            def try_read(attrs):
                for key in ("anchor_idx", "anchor_id"):
                    if key in attrs:
                        extend_from_value(attrs[key])
                        return True
                return False

            if "data" in file:
                try_read(file["data"].attrs)
                for group in file["data"].values():
                    if isinstance(group, h5py.Group):
                        try_read(group.attrs)
            else:
                try_read(file.attrs)

    if inline:
        extend_from_value(inline)

    return [int(anchor) for anchor in anchors]


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

    def cross(origin, point_a, point_b):
        return (point_a[0] - origin[0]) * (point_b[1] - origin[1]) - (
            point_a[1] - origin[1]
        ) * (point_b[0] - origin[0])

    lower = []
    for point in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(tuple(point))

    upper = []
    for point in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(tuple(point))

    return np.array(lower[:-1] + upper[:-1], dtype=float)


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


def match_range(xy, ranges, tol):
    for idx, (xmin, ymin, xmax, ymax) in enumerate(ranges):
        if xmin - tol <= xy[0] <= xmax + tol and ymin - tol <= xy[1] <= ymax + tol:
            return idx
    return None


def run_distribution(args):
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
    defined_ranges = sanitize_ranges(
        parsed["regions"][region_key]["ranges"], require_non_empty=False
    )
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

    try:
        sampled_positions = []
        for _ in range(args.samples):
            env.reset()
            object_pos = env.env.sim.data.body_xpos[env.env.obj_body_id[tracked_object]].copy()
            sampled_positions.append(object_pos)

        sampled_positions = np.asarray(sampled_positions)
        sampled_xy = sampled_positions[:, :2]
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
        if rotated_range_points:
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
    finally:
        env.close()


def run_verify(args):
    parsed = BDDLUtils.robosuite_parse_problem(args.bddl_file)

    object_name = args.object_name
    if object_name is None:
        if parsed["obj_of_interest"]:
            object_name = parsed["obj_of_interest"][0]
        else:
            raise ValueError("Please provide --object-name (obj_of_interest is empty).")

    region_key = args.region_key or infer_region_key(parsed["initial_state"], object_name)
    if region_key not in parsed["regions"]:
        available = ", ".join(parsed["regions"].keys())
        raise ValueError(f"Region '{region_key}' not present in BDDL. Options: {available}")

    discrete_ranges = sanitize_ranges(parsed["regions"][region_key]["ranges"])
    range_centers = np.array(
        [(np.mean([rng[0], rng[2]]), np.mean([rng[1], rng[3]])) for rng in discrete_ranges]
    )

    env = OffScreenRenderEnv(
        bddl_file_name=args.bddl_file,
        camera_heights=128,
        camera_widths=128,
        region_sampling_strategy=args.region_sampling_strategy,
        region_sampling_quota=args.region_sampling_quota,
    )
    if args.seed is not None:
        env.seed(args.seed)

    key = f"{object_name}_pos"
    hits = Counter()
    unmatched = []
    samples_xy = []
    sample_assignments = []

    for i in range(args.samples):
        obs = env.reset()
        if key in obs:
            pos = np.array(obs[key])
        else:
            pos = np.array(env.env.sim.data.body_xpos[env.env.obj_body_id[object_name]])
        xy = pos[:2]
        samples_xy.append(xy)
        match_idx = match_range(xy, discrete_ranges, args.tolerance)
        sample_assignments.append(match_idx)
        if match_idx is None:
            unmatched.append((i, xy))
            print(
                f"[{i:02d}] sampled XY={xy} -> outside defined patches "
                f"(tolerance={args.tolerance:.4f} m)"
            )
        else:
            hits[match_idx] += 1
            cx, cy = range_centers[match_idx]
            print(
                f"[{i:02d}] sampled XY={xy} -> point#{match_idx + 1} "
                f"center=({cx:.4f}, {cy:.4f})"
            )

    env.close()

    print("\nSummary")
    for idx in range(len(discrete_ranges)):
        cx, cy = range_centers[idx]
        print(f"  point#{idx + 1}: center=({cx:.4f}, {cy:.4f}), hits={hits[idx]}")

    if unmatched:
        print("\nSamples outside allowed regions:")
        for idx, xy in unmatched:
            print(f"  sample {idx}: XY={xy}")
    else:
        print("\nAll samples landed inside the specified discrete patches.")

    samples_xy = np.asarray(samples_xy)
    sample_assignments = np.asarray(sample_assignments, dtype=object)
    if len(samples_xy) == 0:
        print("No samples captured for visualization.")
        return

    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(discrete_ranges))))
    fig, ax = plt.subplots()
    for idx in range(len(discrete_ranges)):
        mask = sample_assignments == idx
        if not np.any(mask):
            continue
        ax.scatter(
            samples_xy[mask, 0],
            samples_xy[mask, 1],
            color=colors[idx],
            label=f"samples near P{idx + 1}",
            s=20,
        )
    if unmatched:
        unmatched_xy = np.array([xy for _, xy in unmatched])
        ax.scatter(
            unmatched_xy[:, 0],
            unmatched_xy[:, 1],
            color="gray",
            marker="x",
            label="outside ranges",
        )
    ax.scatter(
        range_centers[:, 0],
        range_centers[:, 1],
        marker="s",
        s=70,
        facecolors="none",
        edgecolors="red",
        linewidths=1.5,
        label="defined points",
    )
    for idx, (cx, cy) in enumerate(range_centers):
        ax.text(cx, cy, f"P{idx + 1}", fontsize=8, color="red")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Sampled initialization points vs defined targets")
    ax.legend(loc="best")
    ax.set_aspect("equal", "box")
    plt.tight_layout()
    plt.savefig(args.plot_path, dpi=200)
    plt.close(fig)
    print(f"Scatter plot saved to {args.plot_path}")

    if args.illustration_path:

        def rotate_xy(xy):
            return np.array([xy[1], -xy[0]])

        def rotate_rect(xmin, ymin, xmax, ymax):
            corners = np.array(
                [
                    [xmin, ymin],
                    [xmax, ymin],
                    [xmax, ymax],
                    [xmin, ymax],
                ]
            )
            rotated = np.array([rotate_xy(pt) for pt in corners])
            min_corner = rotated.min(axis=0)
            max_corner = rotated.max(axis=0)
            width_height = max_corner - min_corner
            return min_corner, width_height

        rotated_samples = np.array([rotate_xy(pt) for pt in samples_xy])
        rotated_centers = np.array([rotate_xy(pt) for pt in range_centers])

        robot_rot = rotate_xy(np.asarray(args.robot_coords))
        basket_rot = rotate_xy(np.asarray(args.basket_coords))
        rect_origin, rect_size = rotate_rect(*args.workspace_rect)

        fig2, (ax_overview, ax_zoom) = plt.subplots(
            1,
            2,
            figsize=(14, 5.5),
            gridspec_kw={"width_ratios": [1.3, 1.1]},
        )
        plt.subplots_adjust(wspace=0.25)

        rect_patch = patches.Rectangle(
            rect_origin,
            rect_size[0],
            rect_size[1],
            linewidth=1,
            edgecolor="g",
            facecolor="g",
            alpha=0.3,
            label="Region",
        )
        ax_overview.add_patch(rect_patch)
        ax_overview.plot(robot_rot[0], robot_rot[1], "rs", markersize=8, label="Robot")
        ax_overview.text(robot_rot[0], robot_rot[1] + 0.05, "Robot", ha="center")
        ax_overview.plot(basket_rot[0], basket_rot[1], "bo", markersize=8, label="Basket")
        ax_overview.text(basket_rot[0], basket_rot[1] + 0.05, "Basket", ha="center")

        zoom_min = np.array([np.inf, np.inf])
        zoom_max = np.array([-np.inf, -np.inf])
        for xmin, ymin, xmax, ymax in discrete_ranges:
            local_origin, local_size = rotate_rect(xmin, ymin, xmax, ymax)
            zoom_min = np.minimum(zoom_min, local_origin)
            zoom_max = np.maximum(zoom_max, local_origin + local_size)
        zoom_origin = zoom_min
        zoom_size = zoom_max - zoom_min
        zoom_patch_overview = patches.Rectangle(
            zoom_origin,
            zoom_size[0],
            zoom_size[1],
            fill=False,
            edgecolor="orange",
            linewidth=1.5,
            linestyle="--",
            label="Zoomed area",
        )
        ax_overview.add_patch(zoom_patch_overview)

        overview_sample_labelled = False
        for idx in range(len(discrete_ranges)):
            mask = sample_assignments == idx
            if not np.any(mask):
                continue
            ax_overview.scatter(
                rotated_samples[mask, 0],
                rotated_samples[mask, 1],
                color=colors[idx],
                s=10,
                alpha=0.7,
                label="Samples" if not overview_sample_labelled else None,
            )
            if not overview_sample_labelled:
                overview_sample_labelled = True
        if unmatched:
            unmatched_xy = np.array([rotate_xy(xy) for _, xy in unmatched])
            ax_overview.scatter(
                unmatched_xy[:, 0],
                unmatched_xy[:, 1],
                color="gray",
                marker="x",
                label="Outside ranges",
            )

        ax_overview.scatter(
            rotated_centers[:, 0],
            rotated_centers[:, 1],
            marker="s",
            s=60,
            facecolors="none",
            edgecolors="red",
            linewidths=1.2,
            label="Defined P",
        )
        for idx, (cx, cy) in enumerate(rotated_centers):
            ax_overview.text(cx, cy, f"P{idx + 1}", fontsize=8, color="red")

        ax_overview.set_xlim(-0.6, 0.6)
        ax_overview.set_ylim(-0.25, 0.8)
        ax_overview.set_xlabel("Original Y-Coordinate")
        ax_overview.set_ylabel("Original X-Coordinate")
        ax_overview.set_title("Workspace overview")
        ax_overview.grid(True)
        ax_overview.set_aspect("equal", adjustable="box")
        ax_overview.legend(loc="upper right")

        ax_zoom.scatter(
            rotated_centers[:, 0],
            rotated_centers[:, 1],
            marker="s",
            s=60,
            facecolors="none",
            edgecolors="red",
            linewidths=1.2,
            label="Defined P",
        )
        for idx, (cx, cy) in enumerate(rotated_centers):
            ax_zoom.text(cx, cy, f"P{idx + 1}", fontsize=8, color="red")

        for idx in range(len(discrete_ranges)):
            mask = sample_assignments == idx
            if not np.any(mask):
                continue
            ax_zoom.scatter(
                rotated_samples[mask, 0],
                rotated_samples[mask, 1],
                color=colors[idx],
                s=25,
                label=f"P{idx + 1} samples",
            )
        if unmatched:
            unmatched_xy = np.array([rotate_xy(xy) for _, xy in unmatched])
            ax_zoom.scatter(
                unmatched_xy[:, 0],
                unmatched_xy[:, 1],
                color="gray",
                marker="x",
                label="Outside ranges",
            )

        zoom_patch_zoom = patches.Rectangle(
            zoom_origin,
            zoom_size[0],
            zoom_size[1],
            fill=False,
            edgecolor="orange",
            linewidth=1.5,
            linestyle="--",
        )
        ax_zoom.add_patch(zoom_patch_zoom)

        ax_zoom.set_xlabel("Original Y-Coordinate")
        ax_zoom.set_ylabel("Original X-Coordinate")
        ax_zoom.set_title("Zoomed-in initialization points")
        ax_zoom.grid(True)
        ax_zoom.set_aspect("equal", adjustable="box")
        ax_zoom.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))

        margin = args.zoom_margin
        ax_zoom.set_xlim(zoom_origin[0] - margin, zoom_origin[0] + zoom_size[0] + margin)
        ax_zoom.set_ylim(zoom_origin[1] - margin, zoom_origin[1] + zoom_size[1] + margin)

        zoom_corners = [
            zoom_origin,
            zoom_origin + np.array([zoom_size[0], 0.0]),
            zoom_origin + zoom_size,
            zoom_origin + np.array([0.0, zoom_size[1]]),
        ]
        for corner in zoom_corners:
            conn = patches.ConnectionPatch(
                xyA=corner,
                xyB=corner,
                coordsA="data",
                coordsB="data",
                axesA=ax_overview,
                axesB=ax_zoom,
                color="orange",
                linestyle="--",
                linewidth=1,
            )
            fig2.add_artist(conn)

        plt.tight_layout()
        plt.savefig(args.illustration_path, dpi=200)
        plt.close(fig2)
        print(f"Illustration plot saved to {args.illustration_path}")


def run_anchors(args):
    if not any([args.anchor_json, args.anchor_hdf5, args.anchors]):
        raise ValueError("Provide at least one anchor source (JSON, HDF5, or --anchors).")

    bddl_path = Path(args.bddl_file).expanduser().resolve()
    if not bddl_path.exists():
        raise FileNotFoundError(f"BDDL file not found: {bddl_path}")

    parsed = BDDLUtils.robosuite_parse_problem(str(bddl_path))

    object_name = parsed["obj_of_interest"][0] if parsed["obj_of_interest"] else None
    if object_name is None:
        raise ValueError("obj_of_interest is empty; cannot infer region.")
    region_key = infer_region_key(parsed["initial_state"], object_name)
    if region_key not in parsed["regions"]:
        available = ", ".join(parsed["regions"].keys())
        raise ValueError(f"Region '{region_key}' not present. Options: {available}")

    discrete_ranges = sanitize_ranges(parsed["regions"][region_key]["ranges"])
    range_centers = np.array(
        [(np.mean([rng[0], rng[2]]), np.mean([rng[1], rng[3]])) for rng in discrete_ranges]
    )

    anchor_indices = load_anchor_indices(
        json_path=args.anchor_json, hdf5_path=args.anchor_hdf5, inline=args.anchors
    )
    if not anchor_indices:
        raise ValueError("No anchor indices found from provided sources.")

    out_of_range = [a for a in anchor_indices if a < 0 or a >= len(discrete_ranges)]
    if out_of_range:
        print(
            f"Warning: {len(out_of_range)} anchors fall outside the defined ranges "
            f"(min=0, max={len(discrete_ranges) - 1}). They will be ignored."
        )

    counts = Counter([a for a in anchor_indices if 0 <= a < len(discrete_ranges)])
    total = sum(counts.values())

    nearest_dists = []
    for idx in range(len(range_centers)):
        others = np.delete(range_centers, idx, axis=0)
        if len(others) == 0:
            nearest_dists.append(np.nan)
            continue
        dists = np.linalg.norm(others - range_centers[idx], axis=1)
        nearest_dists.append(float(np.min(dists)))

    print("\nAnchor -> point mapping")
    for idx, (cx, cy) in enumerate(range_centers):
        xmin, ymin, xmax, ymax = discrete_ranges[idx]
        spacing = nearest_dists[idx]
        spacing_str = f"nearest_dist={spacing:.4f}" if not np.isnan(spacing) else "nearest_dist=nan"
        print(
            f"  anchor {idx}: center=({cx:.4f}, {cy:.4f}), "
            f"range=x[{xmin:.4f},{xmax:.4f}], y[{ymin:.4f},{ymax:.4f}], "
            f"count={counts.get(idx, 0)}, {spacing_str}"
        )
    if out_of_range:
        bad_preview = ", ".join(map(str, sorted(set(out_of_range)))[:10])
        print(f"Out-of-range anchors (showing up to 10): {bad_preview}")

    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(discrete_ranges))))
    fig, ax = plt.subplots()

    for idx, (xmin, ymin, xmax, ymax) in enumerate(discrete_ranges):
        cx, cy = range_centers[idx]
        count = counts.get(idx, 0)
        color = colors[idx % len(colors)]
        anchor_label = f"anchor {idx} (n={count})"
        if count > 0:
            ax.scatter(cx, cy, color=color, s=40 + 12 * count, alpha=0.6, label=anchor_label)
        else:
            ax.scatter(cx, cy, marker="x", s=55, color="red", linewidths=1.1, label=anchor_label)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Anchor usage ({total} entries)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.set_aspect("equal", "box")
    plt.tight_layout()
    plt.savefig(args.plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved visualization to {args.plot_path}")

    if args.illustration_path:

        def rotate_xy(xy):
            return np.array([xy[1], -xy[0]])

        def rotate_rect(xmin, ymin, xmax, ymax):
            corners = np.array(
                [
                    [xmin, ymin],
                    [xmax, ymin],
                    [xmax, ymax],
                    [xmin, ymax],
                ]
            )
            rotated = np.array([rotate_xy(pt) for pt in corners])
            min_corner = rotated.min(axis=0)
            max_corner = rotated.max(axis=0)
            width_height = max_corner - min_corner
            return min_corner, width_height

        rotated_centers = np.array([rotate_xy(pt) for pt in range_centers])
        robot_rot = rotate_xy(np.asarray(args.robot_coords))
        basket_rot = rotate_xy(np.asarray(args.basket_coords))
        rect_origin, rect_size = rotate_rect(*args.workspace_rect)

        fig2, ax_overview = plt.subplots(figsize=(8, 6.5))

        rect_patch = patches.Rectangle(
            rect_origin,
            rect_size[0],
            rect_size[1],
            linewidth=1.2,
            edgecolor="g",
            facecolor="g",
            alpha=0.25,
            label="Workspace",
        )
        ax_overview.add_patch(rect_patch)
        ax_overview.plot(robot_rot[0], robot_rot[1], "rs", markersize=8, label="Robot")
        ax_overview.plot(basket_rot[0], basket_rot[1], "bo", markersize=8, label="Basket")

        for idx, (cx, cy) in enumerate(rotated_centers):
            count = counts.get(idx, 0)
            color = colors[idx % len(colors)]
            marker = "o" if count > 0 else "x"
            size = 70 if count > 0 else 55
            anchor_label = f"anchor {idx} (n={count})"
            ax_overview.scatter(
                cx,
                cy,
                marker=marker,
                s=size,
                color=color if count > 0 else "red",
                alpha=0.7 if count > 0 else 1.0,
                label=anchor_label,
            )

        ax_overview.set_xlabel("Original Y-Coordinate")
        ax_overview.set_ylabel("Original X-Coordinate")
        ax_overview.set_title("Workspace overview (rotated)", pad=12)
        ax_overview.grid(True)
        ax_overview.set_aspect("equal", adjustable="box")
        ax_overview.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.2,
        )
        plt.tight_layout(rect=[0.0, 0.0, 0.8, 0.95])
        plt.savefig(args.illustration_path, dpi=200, bbox_inches="tight")
        plt.close(fig2)
        print(f"Illustration saved to {args.illustration_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Visualize BDDL init distributions, sampling verification, and anchor mappings."
    )
    subparsers = parser.add_subparsers(dest="command")

    dist_parser = subparsers.add_parser(
        "distribution",
        help="Plot sampled init distribution of a single-object BDDL task.",
    )
    dist_parser.add_argument("--bddl-file", required=True, help="Path to the BDDL file.")
    dist_parser.add_argument("--samples", type=int, default=100, help="Number of environment resets to sample.")
    dist_parser.add_argument("--camera-height", type=int, default=128, help="Render height.")
    dist_parser.add_argument("--camera-width", type=int, default=128, help="Render width.")
    dist_parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory. Default: ./tmp/new_scene_overview/<bddl_file_stem>",
    )
    dist_parser.add_argument("--seed", type=int, default=None, help="Optional seed for reproducible sampling.")
    dist_parser.add_argument("--no-illustration", action="store_true", help="Disable the extra illustration output.")
    dist_parser.add_argument("--illustration-path", type=str, default=None, help="Custom path for illustration image.")
    dist_parser.add_argument("--include-robot", action="store_true", help="Show robot base in the illustration.")
    dist_parser.add_argument(
        "--target-region-key",
        type=str,
        default=None,
        help="Target region to highlight. Use 'goal' to infer from the goal predicate.",
    )

    verify_parser = subparsers.add_parser(
        "verify",
        help="Validate that a BDDL sampler lands on discrete points.",
    )
    verify_parser.add_argument("--bddl-file", type=str, required=True)
    verify_parser.add_argument("--object-name", type=str, default=None)
    verify_parser.add_argument("--region-key", type=str, default=None)
    verify_parser.add_argument("--samples", type=int, default=50)
    verify_parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Extra slack (meters) when matching sampled positions to target patches.",
    )
    verify_parser.add_argument("--plot-path", type=str, default="init_points_scatter.png")
    verify_parser.add_argument("--illustration-path", type=str, default=None)
    verify_parser.add_argument(
        "--workspace-rect",
        type=float,
        nargs=4,
        default=[-0.4, -0.4, 0.1, 0.1],
        metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"),
    )
    verify_parser.add_argument(
        "--robot-coords",
        type=float,
        nargs=2,
        default=[-0.6, 0.0],
        metavar=("X", "Y"),
    )
    verify_parser.add_argument(
        "--basket-coords",
        type=float,
        nargs=2,
        default=[-0.01, 0.30],
        metavar=("X", "Y"),
    )
    verify_parser.add_argument("--zoom-margin", type=float, default=0.01)
    verify_parser.add_argument("--seed", type=int, default=None)
    verify_parser.add_argument(
        "--region-sampling-strategy",
        type=str,
        default="random",
        choices=["random", "round_robin", "cycle", "ordered"],
    )
    verify_parser.add_argument("--region-sampling-quota", type=int, default=1)

    anchor_parser = subparsers.add_parser(
        "anchors",
        help="Visualize which discrete points each anchor index refers to.",
    )
    anchor_parser.add_argument("--bddl-file", type=str, required=True)
    anchor_parser.add_argument("--anchor-json", type=str, default=None)
    anchor_parser.add_argument("--anchor-hdf5", type=str, default=None)
    anchor_parser.add_argument("--anchors", type=int, nargs="+", default=None)
    anchor_parser.add_argument("--plot-path", type=str, default="anchor_points_from_list.png")
    anchor_parser.add_argument("--illustration-path", type=str, default=None)
    anchor_parser.add_argument(
        "--workspace-rect",
        type=float,
        nargs=4,
        default=[-0.4, -0.4, 0.1, 0.1],
        metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"),
    )
    anchor_parser.add_argument(
        "--robot-coords",
        type=float,
        nargs=2,
        default=[-0.6, 0.0],
        metavar=("X", "Y"),
    )
    anchor_parser.add_argument(
        "--basket-coords",
        type=float,
        nargs=2,
        default=[-0.01, 0.30],
        metavar=("X", "Y"),
    )

    return parser


def parse_args(argv=None):
    parser = build_parser()
    known_commands = {"distribution", "verify", "anchors"}
    argv = list(argv or [])
    if not argv or argv[0] not in known_commands:
        argv.insert(0, "distribution")
    return parser.parse_args(argv)


def main():
    import sys

    args = parse_args(sys.argv[1:])
    if args.command == "verify":
        run_verify(args)
    elif args.command == "anchors":
        run_anchors(args)
    else:
        run_distribution(args)


if __name__ == "__main__":
    main()

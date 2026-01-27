#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from typing import Optional

import numpy as np

import libero.libero.envs.bddl_utils as BDDLUtils


def load_summary(path: Path):
    with path.open("r") as f:
        return json.load(f)


def format_rate(value):
    try:
        return f"{float(value):.3f}"
    except Exception:
        return "nan"


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
    if not cleaned:
        raise ValueError("No ranges defined for the requested region.")
    return cleaned


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
        f"Could not infer region for {object_name} from initial_state definitions."
    )


def parse_anchor_id(anchor_id):
    try:
        return int(anchor_id)
    except Exception:
        return None


def compute_success_rate(stats):
    if stats is None:
        return None
    if "success_rate" in stats and stats["success_rate"] is not None:
        return float(stats["success_rate"])
    success = stats.get("success")
    rollouts = stats.get("rollouts")
    if success is None or rollouts in (None, 0):
        return None
    return float(success) / float(rollouts)


def build_anchor_stats(anchors):
    stats_by_id = {}
    skipped = []
    for anchor_id, stats in anchors.items():
        parsed = parse_anchor_id(anchor_id)
        if parsed is None:
            skipped.append(anchor_id)
            continue
        stats_by_id[int(parsed)] = stats
    return stats_by_id, skipped


def plot_anchor_success(
    anchors,
    bddl_path: Path,
    plot_path: Path,
    show_ranges: bool,
    annotate: bool,
    illustration_path: Optional[Path],
    workspace_rect,
    robot_coords,
    basket_coords,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

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
        [(np.mean([r[0], r[2]]), np.mean([r[1], r[3]])) for r in discrete_ranges]
    )

    stats_by_id, skipped = build_anchor_stats(anchors)
    if skipped:
        preview = ", ".join(map(str, skipped[:5]))
        print(f"[warning] skipped non-integer anchor ids: {preview}")

    rates = []
    stats_list = []
    for anchor_id in range(len(discrete_ranges)):
        stats = stats_by_id.get(anchor_id, {})
        stats_list.append(stats)
        rate = compute_success_rate(stats)
        rates.append(rate if rate is not None else np.nan)

    x_centers = np.unique(range_centers[:, 0])
    y_centers = np.unique(range_centers[:, 1])
    x_centers.sort()
    y_centers.sort()

    if len(x_centers) * len(y_centers) != len(range_centers):
        raise ValueError(
            "Anchor centers do not form a full grid; cannot build 2D matrix plot."
        )

    x_spacing = np.min(np.diff(x_centers)) if len(x_centers) > 1 else 0.1
    y_spacing = np.min(np.diff(y_centers)) if len(y_centers) > 1 else 0.1
    x_edges = np.concatenate(([x_centers[0] - x_spacing / 2], x_centers + x_spacing / 2))
    y_edges = np.concatenate(([y_centers[0] - y_spacing / 2], y_centers + y_spacing / 2))

    rate_grid = np.full((len(y_centers), len(x_centers)), np.nan)
    label_grid = [[None for _ in range(len(x_centers))] for _ in range(len(y_centers))]
    for idx, (cx, cy) in enumerate(range_centers):
        col = int(np.where(np.isclose(x_centers, cx))[0][0])
        row = int(np.where(np.isclose(y_centers, cy))[0][0])
        rate_grid[row, col] = rates[idx]
        stats = stats_list[idx]
        success = stats.get("success")
        rollouts = stats.get("rollouts")
        label_grid[row][col] = f"{idx}\n{success}/{rollouts}"

    cmap = plt.cm.YlOrRd
    fig, ax = plt.subplots(figsize=(7.2, 6))
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        rate_grid,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        shading="auto",
    )

    if annotate:
        for row in range(len(y_centers)):
            for col in range(len(x_centers)):
                label = label_grid[row][col]
                if label is None:
                    continue
                ax.text(
                    x_centers[col],
                    y_centers[row],
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                    fontweight="bold" if rate_grid[row, col] >= 0.8 else "normal",
                )

    if show_ranges:
        for xmin, ymin, xmax, ymax in discrete_ranges:
            rect = patches.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                linewidth=0.8,
                edgecolor="black",
                facecolor="none",
                alpha=0.3,
            )
            ax.add_patch(rect)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Anchor Success Rates")
    ax.set_aspect("equal", "box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Success Rate")

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved visualization to {plot_path}")

    if illustration_path is None:
        return

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
    robot_rot = rotate_xy(np.asarray(robot_coords))
    basket_rot = rotate_xy(np.asarray(basket_coords))
    rect_origin, rect_size = rotate_rect(*workspace_rect)

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
        ax_overview.scatter(
            cx,
            cy,
            marker="o",
            s=55,
            color="white",
            edgecolor="black",
            linewidths=0.6,
        )
        ax_overview.text(
            cx,
            cy,
            str(idx),
            ha="center",
            va="center",
            fontsize=8,
            color="black",
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

    illustration_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0.0, 0.0, 0.8, 0.95])
    plt.savefig(illustration_path, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"Illustration saved to {illustration_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Print per-anchor rollout stats from rollout_summary.json"
    )
    parser.add_argument(
        "summary_path",
        type=Path,
        help="Path to rollout_summary.json",
    )
    parser.add_argument(
        "--bddl-file",
        type=Path,
        default=None,
        help="Optional BDDL file used to visualize anchor locations.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help="Where to save the anchor success plot (PNG).",
    )
    parser.add_argument(
        "--show-ranges",
        action="store_true",
        help="Overlay anchor range rectangles on the plot.",
    )
    parser.add_argument(
        "--no-annotate",
        action="store_true",
        help="Disable per-anchor text annotations.",
    )
    parser.add_argument(
        "--illustration-path",
        type=Path,
        default=None,
        help="Optional overview (rotated) plot with workspace and robot/basket markers.",
    )
    parser.add_argument(
        "--no-illustration",
        action="store_true",
        help="Disable the workspace overview plot when a BDDL file is provided.",
    )
    parser.add_argument(
        "--workspace-rect",
        type=float,
        nargs=4,
        default=[-0.4, -0.4, 0.1, 0.1],
        metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"),
        help="Original coordinate rectangle defining the workspace (before rotation).",
    )
    parser.add_argument(
        "--robot-coords",
        type=float,
        nargs=2,
        default=[-0.6, 0.0],
        metavar=("X", "Y"),
        help="Robot base coordinates in the original frame.",
    )
    parser.add_argument(
        "--basket-coords",
        type=float,
        nargs=2,
        default=[-0.01, 0.30],
        metavar=("X", "Y"),
        help="Basket coordinates in the original frame.",
    )
    args = parser.parse_args()

    summary_path = args.summary_path.expanduser().resolve()
    if not summary_path.exists():
        raise FileNotFoundError(f"summary not found: {summary_path}")

    summary = load_summary(summary_path)
    anchors = summary.get("anchors", {})

    if not anchors:
        print("No anchor stats found.")
        return

    def anchor_sort_key(aid):
        try:
            return int(aid)
        except Exception:
            return aid

    total = len(anchors)
    print(f"Total anchors: {total}")
    for anchor_id in sorted(anchors.keys(), key=anchor_sort_key):
        stats = anchors[anchor_id]
        rollouts = stats.get("rollouts")
        success = stats.get("success")
        success_rate = format_rate(stats.get("success_rate"))
        print(
            f"anchor {anchor_id}: rollouts={rollouts}, "
            f"success={success}, success_rate={success_rate}"
        )

    if args.bddl_file is None:
        return

    bddl_path = args.bddl_file.expanduser().resolve()
    if not bddl_path.exists():
        raise FileNotFoundError(f"BDDL file not found: {bddl_path}")

    plot_path = (
        args.plot_path.expanduser().resolve()
        if args.plot_path is not None
        else summary_path.parent / "anchor_success_plot.png"
    )
    if args.no_illustration:
        illustration_path = None
    elif args.illustration_path is not None:
        illustration_path = args.illustration_path.expanduser().resolve()
    else:
        illustration_path = summary_path.parent / "anchor_overview.png"
    plot_anchor_success(
        anchors=anchors,
        bddl_path=bddl_path,
        plot_path=plot_path,
        show_ranges=args.show_ranges,
        annotate=not args.no_annotate,
        illustration_path=illustration_path,
        workspace_rect=args.workspace_rect,
        robot_coords=args.robot_coords,
        basket_coords=args.basket_coords,
    )


if __name__ == "__main__":
    main()

import argparse
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs import bddl_utils as BDDLUtils


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


def match_range(xy, ranges, tol):
    for idx, (xmin, ymin, xmax, ymax) in enumerate(ranges):
        if (
            xmin - tol <= xy[0] <= xmax + tol
            and ymin - tol <= xy[1] <= ymax + tol
        ):
            return idx
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Validate that a BDDL file samples an object from discrete points."
    )
    parser.add_argument("--bddl-file", type=str, required=True)
    parser.add_argument("--object-name", type=str, default=None)
    parser.add_argument("--region-key", type=str, default=None)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Extra slack (in meters) when matching sampled positions to target patches.",
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default="init_points_scatter.png",
        help="Where to save the scatter plot for sampled vs defined points.",
    )
    parser.add_argument(
        "--illustration-path",
        type=str,
        default=None,
        help="If set, also draw overview + zoomed workspace illustration.",
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
    parser.add_argument(
        "--zoom-margin",
        type=float,
        default=0.01,
        help="Extra padding (meters) added around the zoomed-in region in the illustration.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed to reproduce the sampler's output.",
    )
    parser.add_argument(
        "--region-sampling-strategy",
        type=str,
        default="random",
        choices=["random", "round_robin", "cycle", "ordered"],
        help="Sampling strategy used by region samplers. 'ordered' cycles through ranges without RNG.",
    )
    parser.add_argument(
        "--region-sampling-quota",
        type=int,
        default=1,
        help="Number of times each range should appear per sampling cycle (>=1).",
    )
    args = parser.parse_args()

    parsed = BDDLUtils.robosuite_parse_problem(args.bddl_file)

    object_name = args.object_name
    if object_name is None:
        if parsed["obj_of_interest"]:
            object_name = parsed["obj_of_interest"][0]
        else:
            raise ValueError("Please provide --object-name (obj_of_interest is empty).")

    region_key = args.region_key or infer_region_key(
        parsed["initial_state"], object_name
    )
    if region_key not in parsed["regions"]:
        available = ", ".join(parsed["regions"].keys())
        raise ValueError(
            f"Region '{region_key}' not present in BDDL. Options: {available}"
        )

    discrete_ranges = sanitize_ranges(parsed["regions"][region_key]["ranges"])
    range_centers = np.array(
        [(np.mean([r[0], r[2]]), np.mean([r[1], r[3]])) for r in discrete_ranges]
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
        print(
            f"  point#{idx + 1}: center=({cx:.4f}, {cy:.4f}), "
            f"hits={hits[idx]}"
        )

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
        ax_overview.plot(
            robot_rot[0], robot_rot[1], "rs", markersize=8, label="Robot"
        )
        ax_overview.text(
            robot_rot[0], robot_rot[1] + 0.05, "Robot", ha="center"
        )
        ax_overview.plot(
            basket_rot[0], basket_rot[1], "bo", markersize=8, label="Basket"
        )
        ax_overview.text(
            basket_rot[0], basket_rot[1] + 0.05, "Basket", ha="center"
        )

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
        ax_zoom.set_xlim(
            zoom_origin[0] - margin, zoom_origin[0] + zoom_size[0] + margin
        )
        ax_zoom.set_ylim(
            zoom_origin[1] - margin, zoom_origin[1] + zoom_size[1] + margin
        )

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


if __name__ == "__main__":
    main()

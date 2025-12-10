import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import libero.libero.envs.bddl_utils as BDDLUtils


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


def load_anchor_indices(json_path=None, hdf5_path=None, inline=None):
    anchors = []

    def extend_from_value(val):
        if val is None:
            return
        if isinstance(val, (list, tuple)):
            anchors.extend(val)
        elif isinstance(val, (np.ndarray,)):
            anchors.extend(val.tolist())
        else:
            raise ValueError(f"Unsupported anchor container type: {type(val)}")

    if json_path:
        with open(json_path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for key in ("anchor_idx", "anchor_id", "anchors"):
                if key in data:
                    extend_from_value(data[key])
                    break
            else:
                raise ValueError(
                    f"JSON file {json_path} does not contain anchor indices."
                )
        elif isinstance(data, list):
            extend_from_value(data)
        else:
            raise ValueError(f"Unsupported JSON structure in {json_path}")

    if hdf5_path:
        try:
            import h5py
        except ImportError as exc:
            raise ImportError(
                "h5py is required to read anchor indices from an HDF5 file."
            ) from exc

        with h5py.File(hdf5_path, "r") as f:
            def try_read(attrs):
                for key in ("anchor_idx", "anchor_id"):
                    if key in attrs:
                        extend_from_value(attrs[key])
                        return True
                return False

            if "data" in f:
                try_read(f["data"].attrs)
                for grp in f["data"].values():
                    if isinstance(grp, h5py.Group):
                        try_read(grp.attrs)
            else:
                try_read(f.attrs)

    if inline:
        extend_from_value(inline)

    anchors = [int(a) for a in anchors]
    return anchors


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize which discrete points each anchor index refers to, "
            "using anchor indices from JSON/HDF5 or inline values."
        )
    )
    parser.add_argument("--bddl-file", type=str, required=True)
    parser.add_argument(
        "--anchor-json",
        type=str,
        default=None,
        help="Path to JSON containing anchor indices (list or {'anchor_idx': [...]}).",
    )
    parser.add_argument(
        "--anchor-hdf5",
        type=str,
        default=None,
        help="Optional HDF5 path; collects any 'anchor_idx' or 'anchor_id' attributes.",
    )
    parser.add_argument(
        "--anchors",
        type=int,
        nargs="+",
        default=None,
        help="Inline anchor indices if you prefer not to use a file.",
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default="anchor_points_from_list.png",
        help="Where to save the visualization.",
    )
    parser.add_argument(
        "--illustration-path",
        type=str,
        default=None,
        help="Optional overview (rotated) plot with workspace and robot/basket markers.",
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

    if not any([args.anchor_json, args.anchor_hdf5, args.anchors]):
        raise ValueError("Provide at least one anchor source (JSON, HDF5, or --anchors).")

    bddl_path = Path(args.bddl_file)
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
        [(np.mean([r[0], r[2]]), np.mean([r[1], r[3]])) for r in discrete_ranges]
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

    # Pre-compute nearest-neighbor spacing for annotation.
    nearest_dists = []
    for i in range(len(range_centers)):
        others = np.delete(range_centers, i, axis=0)
        if len(others) == 0:
            nearest_dists.append(np.nan)
            continue
        dists = np.linalg.norm(others - range_centers[i], axis=1)
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
        spacing = nearest_dists[idx]
        if count > 0:
            ax.scatter(
                cx,
                cy,
                color=color,
                s=40 + 12 * count,
                alpha=0.6,
                label=anchor_label,
            )
        else:
            ax.scatter(
                cx,
                cy,
                marker="x",
                s=55,
                color="red",
                linewidths=1.1,
                label=anchor_label,
            )

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
        ax_overview.plot(
            robot_rot[0], robot_rot[1], "rs", markersize=8, label="Robot"
        )
        ax_overview.plot(
            basket_rot[0], basket_rot[1], "bo", markersize=8, label="Basket"
        )

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


if __name__ == "__main__":
    main()

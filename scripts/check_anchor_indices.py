import argparse
import json
from pathlib import Path
from collections import defaultdict

import h5py
import numpy as np

import init_path  # noqa: F401
import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import OffScreenRenderEnv


def sanitize_ranges(raw_ranges):
    cleaned = []
    for entry in raw_ranges:
        if len(entry) != 4:
            continue
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


def match_range(xy, ranges):
    for idx, (xmin, ymin, xmax, ymax) in enumerate(ranges):
        if xmin <= xy[0] <= xmax and ymin <= xy[1] <= ymax:
            return idx
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Inspect per-demo anchor_idx by recomputing from init_state positions."
    )
    parser.add_argument("--demo-file", required=True, help="Path to *_demo.hdf5")
    parser.add_argument(
        "--bddl-file",
        default=None,
        help="Optional BDDL path (default: read from HDF5 attr bddl_file_name)",
    )
    parser.add_argument(
        "--max-demos",
        type=int,
        default=0,
        help="Limit number of demos to check (0 = all)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-demo match info for the first few samples.",
    )
    args = parser.parse_args()

    demo_path = Path(args.demo_file).expanduser().resolve()
    if not demo_path.exists():
        raise FileNotFoundError(f"HDF5 not found: {demo_path}")

    with h5py.File(demo_path, "r") as f:
        data = f["data"]
        demo_keys = sorted([k for k in data.keys() if k.startswith("demo_")], key=lambda x: int(x.split("_")[1]))
        if args.max_demos > 0:
            demo_keys = demo_keys[: args.max_demos]
        if not demo_keys:
            raise ValueError("No demo_* groups found in HDF5")
        bddl_file = args.bddl_file or data.attrs.get("bddl_file_name", None)
        if bddl_file is None:
            raise ValueError("No bddl_file_name found; please pass --bddl-file explicitly")

        # Parse BDDL for anchor ranges and target object name
        parsed = BDDLUtils.robosuite_parse_problem(str(Path(bddl_file).expanduser().resolve()))
        target_object = parsed["obj_of_interest"][0]
        region_key = [st[2] for st in parsed["initial_state"] if st[1] == target_object][0]
        ranges = sanitize_ranges(parsed["regions"][region_key]["ranges"])

        def get_obj_key(name):
            candidates = [
                f"{name.replace('_main', '')}_pos",
                f"{name.split('_')[0]}_pos",
            ]
            return candidates

        env = OffScreenRenderEnv(
            bddl_file_name=bddl_file,
            camera_heights=128,
            camera_widths=128,
        )
        env.seed(0)

        counts = defaultdict(int)
        anchor_counts = defaultdict(int)
        mismatches = []

        for idx, demo_key in enumerate(demo_keys):
            grp = data[demo_key]
            anchor_idx = grp.attrs.get("anchor_idx", None)
            anchor_counts[anchor_idx] += 1

            init_state = grp.attrs.get("init_state", None)
            if init_state is None:
                init_state = grp["states"][0]

            obs = env.set_init_state(init_state)

            obj_key = None
            for cand in get_obj_key(target_object):
                if cand in obs:
                    obj_key = cand
                    break
            if obj_key is None:
                raise KeyError(f"Object position key for {target_object} not found in obs keys {list(obs.keys())[:10]}")

            xy = np.array(obs[obj_key][:2])
            matched = match_range(xy, ranges)

            counts[(matched, anchor_idx)] += 1
            if args.verbose and idx < 10:
                print(f"{demo_key}: matched={matched} anchor_idx={anchor_idx} xy={xy.tolist()}")
            if matched != anchor_idx:
                mismatches.append((demo_key, matched, anchor_idx, xy))

        env.close()

    print("Summary (matched -> anchor_idx):")
    for key, cnt in sorted(counts.items(), key=lambda x: (str(x[0]), -x[1])):
        print(f"  {key}: {cnt}")

    print("Anchor counts from file:")
    for a, cnt in sorted(anchor_counts.items(), key=lambda x: (str(x[0]), -x[1])):
        print(f"  anchor {a}: {cnt}")

    if mismatches:
        print(f"[warning] {len(mismatches)} demos have mismatch between matched range and anchor_idx.")
        for entry in mismatches[:10]:
            demo, matched, anchor_idx, xy = entry
            print(f"  {demo}: matched={matched}, anchor_idx={anchor_idx}, xy={xy.tolist()}")
    else:
        print("All checked demos have anchor_idx equal to matched range.")


if __name__ == "__main__":
    main()

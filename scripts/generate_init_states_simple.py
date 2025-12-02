import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

import init_path  # noqa: F401
import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv


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
        if xmin - tol <= xy[0] <= xmax + tol and ymin - tol <= xy[1] <= ymax + tol:
            return idx
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-anchor init_states for a single BDDL task"
    )
    parser.add_argument("--bddl-file", required=True, help="Path to the task's BDDL file")
    parser.add_argument("--per-anchor", type=int, default=5, help="Number of init states per anchor point")
    parser.add_argument("--tol", type=float, default=0.01, help="Tolerance for matching anchor ranges")
    parser.add_argument("--seed", type=int, default=1234, help="Base seed (different from demo collection seed)")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path (.pruned_init). Default: <init_states>/<bddl_folder>/<bddl_name>.pruned_init",
    )
    args = parser.parse_args()

    bddl_path = Path(args.bddl_file).expanduser().resolve()
    if not bddl_path.exists():
        raise FileNotFoundError(f"BDDL file not found: {bddl_path}")

    out_path = (
        Path(args.out).expanduser().resolve()
        if args.out
        else Path(get_libero_path("init_states"))
        / bddl_path.parent.name
        / f"{bddl_path.stem}.pruned_init"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    parsed = BDDLUtils.robosuite_parse_problem(str(bddl_path))
    target_object = parsed["obj_of_interest"][0]
    region_key = [st[2] for st in parsed["initial_state"] if st[1] == target_object][0]
    anchor_ranges = sanitize_ranges(parsed["regions"][region_key]["ranges"])

    env_args = {
        "bddl_file_name": str(bddl_path),
        "camera_heights": 128,
        "camera_widths": 128,
        "region_sampling_strategy": "round_robin",
        "region_sampling_quota": args.per_anchor,
    }

    env = OffScreenRenderEnv(**env_args)
    env.seed(args.seed)  # set RNG once; resets will advance the stream

    total_needed = args.per_anchor * len(anchor_ranges)
    anchor_counts = {i: 0 for i in range(len(anchor_ranges))}
    init_states = []
    anchor_indices = []

    while sum(anchor_counts.values()) < total_needed:
        obs = env.reset()
        target_pos = obs[f"{target_object.replace('_main', '')}_pos"].copy()
        anchor_idx = match_range(target_pos[:2], anchor_ranges, args.tol)
        if anchor_idx is None:
            continue
        if anchor_counts[anchor_idx] >= args.per_anchor:
            continue
        init_states.append(np.array(env.get_sim_state()))
        anchor_indices.append(anchor_idx)
        anchor_counts[anchor_idx] += 1
        print(
            f"[info] collected anchor {anchor_idx} | count {anchor_counts[anchor_idx]}/{args.per_anchor} "
            f"({sum(anchor_counts.values())}/{total_needed})"
        )

    env.close()

    init_states = torch.tensor(np.stack(init_states, axis=0))
    torch.save(init_states, out_path)

    anchors_meta = out_path.with_suffix(out_path.suffix + ".anchors.json")
    with open(anchors_meta, "w") as f:
        json.dump({"anchor_idx": anchor_indices}, f, indent=2)

    print(f"[info] saved {len(init_states)} init states to {out_path}")
    print(f"[info] anchor indices saved to {anchors_meta}")


if __name__ == "__main__":
    main()

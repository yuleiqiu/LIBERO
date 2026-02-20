#!/usr/bin/env python3
"""
Check one-step state replay error for raw demos (states + actions).

Purpose:
- Verify that replaying actions from raw demos reproduces next-state transitions.

Example:
  python standalone/inspect/check_state_replay.py \
      --demo-file path/to/demo.hdf5 \
      --max-demos 2 --max-steps 10 --report-each
"""

import argparse
from pathlib import Path

import h5py
import numpy as np

from standalone.rollout_env import resolve_bddl_path
from libero.libero.envs import OffScreenRenderEnv


def demo_sort_key(name: str):
    try:
        return int(name.split("_")[1])
    except Exception:
        return name


def parse_demo_ids(raw: str):
    if not raw:
        return None
    ids = []
    for token in [t.strip() for t in raw.split(",") if t.strip()]:
        if token.startswith("demo_"):
            ids.append(token)
        elif token.isdigit():
            ids.append(f"demo_{int(token)}")
        else:
            raise ValueError(f"invalid demo id: {token}")
    return ids


def main():
    parser = argparse.ArgumentParser(description="Measure one-step replay state error.")
    parser.add_argument("--demo-file", required=True, help="Path to raw demo.hdf5")
    parser.add_argument("--demo-ids", default="", help="Comma-separated demo ids (e.g., 0,1)")
    parser.add_argument("--max-demos", type=int, default=0, help="Max demos to check (0 = all)")
    parser.add_argument("--max-steps", type=int, default=5, help="Max steps per demo")
    parser.add_argument(
        "--report-action-stats",
        action="store_true",
        help="Print action min/max/mean/std for each demo",
    )
    parser.add_argument("--report-each", action="store_true", help="Print per-step errors")
    args = parser.parse_args()

    demo_path = Path(args.demo_file).expanduser().resolve()
    if not demo_path.exists():
        raise FileNotFoundError(f"demo not found: {demo_path}")

    with h5py.File(demo_path, "r") as f:
        data = f["data"]
        bddl_name = data.attrs.get("bddl_file_name", None)
        if bddl_name is None:
            raise ValueError("bddl_file_name not found in demo")
        bddl_path = resolve_bddl_path(bddl_name, demo_path)
        if bddl_path is None:
            raise FileNotFoundError(f"bddl file not found: {bddl_name}")

        demo_ids = parse_demo_ids(args.demo_ids)
        demo_keys = sorted(
            [k for k in data.keys() if k.startswith("demo_")], key=demo_sort_key
        )
        if demo_ids is not None:
            missing = [k for k in demo_ids if k not in demo_keys]
            if missing:
                raise ValueError(f"requested demos not found: {missing}")
            demo_keys = demo_ids
        if args.max_demos > 0:
            demo_keys = demo_keys[: args.max_demos]

        env = OffScreenRenderEnv(bddl_file_name=bddl_path, use_camera_obs=False)
        all_errors = []
        all_qpos_errors = []
        all_qvel_errors = []
        nq = env.env.sim.model.nq
        nv = env.env.sim.model.nv

        for demo_key in demo_keys:
            states = data[f"{demo_key}/states"][()]
            actions = data[f"{demo_key}/actions"][()]
            steps = min(len(actions) - 1, args.max_steps)
            if steps <= 0:
                print(f"{demo_key}: not enough steps to compare")
                continue

            if args.report_action_stats:
                action_mean = np.mean(actions, axis=0)
                action_std = np.std(actions, axis=0)
                action_min = np.min(actions, axis=0)
                action_max = np.max(actions, axis=0)
                print(
                    f"{demo_key}: action mean={action_mean} std={action_std} "
                    f"min={action_min} max={action_max}"
                )

            env.reset()
            env.set_init_state(states[0])

            demo_errors = []
            demo_qpos_errors = []
            demo_qvel_errors = []
            for t in range(steps):
                env.step(actions[t])
                state_next = env.env.sim.get_state().flatten()
                target = states[t + 1]
                err = float(np.linalg.norm(target - state_next))
                qpos_err = float(np.linalg.norm(target[:nq] - state_next[:nq]))
                qvel_err = float(
                    np.linalg.norm(target[nq : nq + nv] - state_next[nq : nq + nv])
                )
                demo_errors.append(err)
                demo_qpos_errors.append(qpos_err)
                demo_qvel_errors.append(qvel_err)
                if args.report_each:
                    print(
                        f"{demo_key} step {t:02d} error: {err:.6f} "
                        f"qpos: {qpos_err:.6f} qvel: {qvel_err:.6f}"
                    )

            mean_err = float(np.mean(demo_errors))
            max_err = float(np.max(demo_errors))
            mean_qpos = float(np.mean(demo_qpos_errors))
            max_qpos = float(np.max(demo_qpos_errors))
            mean_qvel = float(np.mean(demo_qvel_errors))
            max_qvel = float(np.max(demo_qvel_errors))
            all_errors.extend(demo_errors)
            all_qpos_errors.extend(demo_qpos_errors)
            all_qvel_errors.extend(demo_qvel_errors)
            print(
                f"{demo_key}: mean={mean_err:.6f} max={max_err:.6f} steps={steps} "
                f"qpos_mean={mean_qpos:.6f} qpos_max={max_qpos:.6f} "
                f"qvel_mean={mean_qvel:.6f} qvel_max={max_qvel:.6f}"
            )

        env.close()

    if all_errors:
        all_errors = np.array(all_errors, dtype=np.float64)
        all_qpos_errors = np.array(all_qpos_errors, dtype=np.float64)
        all_qvel_errors = np.array(all_qvel_errors, dtype=np.float64)
        print(
            f"overall: mean={all_errors.mean():.6f} "
            f"std={all_errors.std():.6f} "
            f"min={all_errors.min():.6f} "
            f"max={all_errors.max():.6f} "
            f"qpos_mean={all_qpos_errors.mean():.6f} "
            f"qpos_max={all_qpos_errors.max():.6f} "
            f"qvel_mean={all_qvel_errors.mean():.6f} "
            f"qvel_max={all_qvel_errors.max():.6f}"
        )


if __name__ == "__main__":
    main()

"""
Standalone runner for the scripted pick-and-place policy from
scripts/collect_demonstration_by_anchor.py.

Example:
python scripts/run_scripted_policy.py \
    --bddl-file templates/tasks/put_the_blue_bowl_on_the_plate.bddl \
    --init-state-file path/to/state_0.npz --episodes 10 --render
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import robosuite as suite
import torch

import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import TASK_MAPPING


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


class BasePolicy:
    def __init__(self, inject_noise=False, debug=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.trajectory = None
        self.debug = debug

    def generate_trajectory(self, target_pos, destination_pos):
        raise NotImplementedError

    def _log_step(self, curr_wp, next_wp, target_xyz, gripper, current_eef_pos):
        if not self.debug:
            return
        next_t = next_wp["t"] if next_wp is not None else None
        print(
            f"[debug] step={self.step_count} curr_t={curr_wp['t']} next_t={next_t} "
            f"target_xyz={target_xyz} grip={gripper} eef={current_eef_pos}"
        )

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        denom = next_waypoint["t"] - curr_waypoint["t"]
        if abs(denom) < 1e-6:
            t_frac = 1.0
        else:
            t_frac = (t - curr_waypoint["t"]) / denom
            t_frac = np.clip(t_frac, 0.0, 1.0)
        curr_xyz = curr_waypoint["xyz"]
        curr_grip = curr_waypoint["gripper"]
        next_xyz = next_waypoint["xyz"]
        next_grip = next_waypoint["gripper"]
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        gripper = curr_grip if t_frac < 1.0 else next_grip
        return xyz, gripper

    def __call__(self, target_pos, destination_pos, current_eef_pos):
        if self.step_count == 0:
            self.generate_trajectory(target_pos, destination_pos)

        while self.trajectory and self.trajectory[0]["t"] <= self.step_count:
            self.curr_waypoint = self.trajectory.pop(0)
            if self.debug:
                next_wp = self.trajectory[0] if self.trajectory else None
                print(
                    f"[debug] step={self.step_count} -> curr_wp t={self.curr_waypoint['t']} "
                    f"xyz={self.curr_waypoint['xyz']} grip={self.curr_waypoint['gripper']} "
                    f"next_wp={next_wp}"
                )

        if self.trajectory:
            next_waypoint = self.trajectory[0]
            target_xyz, gripper = self.interpolate(
                self.curr_waypoint, next_waypoint, self.step_count
            )
            if self.inject_noise and not np.isclose(
                target_xyz[2], next_waypoint["xyz"][2], atol=1e-3
            ):
                scale = 0.02
                target_xyz += np.random.uniform(-scale, scale, target_xyz.shape)
            xyz = target_xyz - current_eef_pos
            self._log_step(self.curr_waypoint, next_waypoint, target_xyz, gripper, current_eef_pos)
        else:
            target_xyz = self.curr_waypoint["xyz"]
            gripper = self.curr_waypoint["gripper"]
            xyz = target_xyz - current_eef_pos
            self._log_step(self.curr_waypoint, None, target_xyz, gripper, current_eef_pos)

        self.step_count += 1
        return xyz, gripper


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


class FixedPickAndPlacePolicy(BasePolicy):
    """
    Deterministic pick-and-place trajectory matching the original scripted behavior.
    """

    def __init__(self):
        super().__init__(inject_noise=False, debug=os.environ.get("DEBUG_TRAJ", "0") == "1")

    def generate_trajectory(self, target_pos, destination_pos):
        self.trajectory = [
            {"t": 0, "xyz": target_pos + np.array([0, 0, 0.3]), "gripper": -1},
            {"t": 200, "xyz": target_pos + np.array([0, 0, 0.04]), "gripper": -1},
            {"t": 300, "xyz": target_pos + np.array([0, 0, 0.04]), "gripper": 1},
            {"t": 380, "xyz": target_pos + np.array([0, 0, 0.2]), "gripper": 1},
            {"t": 470, "xyz": destination_pos + np.array([0, 0, 0.25]), "gripper": 1},
            {"t": 540, "xyz": destination_pos + np.array([0, 0, 0.1]), "gripper": 1},
            {"t": 570, "xyz": destination_pos + np.array([0, 0, 0.1]), "gripper": -1},
            {"t": 590, "xyz": destination_pos + np.array([0, 0, 0.3]), "gripper": -1},
            {"t": 600, "xyz": destination_pos + np.array([0, 0, 0.3]), "gripper": -1},
        ]


class RandomizedPickAndPlacePolicy(BasePolicy):
    """
    Same keyframes as the fixed policy, but injects random mid-waypoints and optional noise.
    """

    def __init__(self, inject_noise=False, transfer_offset_max=0.06, approach_offset_max=0.06):
        super().__init__(inject_noise, debug=os.environ.get("DEBUG_TRAJ", "0") == "1")
        self.transfer_offset_max = transfer_offset_max
        self.approach_offset_max = approach_offset_max
        self.transfer_height_range = (0.22, 0.36)
        self.approach_height_range = (0.18, 0.30)

    def _midpoints(self, start, goal, offset_max, height_range):
        mode = np.random.choice(["direct", "arc", "double_arc"])
        mids = []
        if mode == "arc":
            offset = random_lateral_offset(start, goal, offset_max)
            mid = (start + goal) / 2.0 + offset
            mid[2] = np.random.uniform(*height_range)
            mids.append(mid)
        elif mode == "double_arc":
            offset1 = random_lateral_offset(start, goal, offset_max)
            offset2 = random_lateral_offset(goal, start, offset_max)
            mid1 = start + 0.35 * (goal - start) + offset1
            mid2 = start + 0.7 * (goal - start) + offset2
            mid1[2] = np.random.uniform(*height_range)
            mid2[2] = np.random.uniform(*height_range)
            mids.extend([mid1, mid2])
        return mids

    def _transfer_midpoints(self, start, goal):
        return self._midpoints(start, goal, self.transfer_offset_max, self.transfer_height_range)

    def _approach_midpoints(self, start, goal):
        return self._midpoints(start, goal, self.approach_offset_max, self.approach_height_range)

    def generate_trajectory(self, target_pos, destination_pos):
        def maybe_jitter(z, scale=0.005):
            if not self.inject_noise:
                return z
            return z + np.random.uniform(-scale, scale)

        waypoints = []

        def add_abs(t, pos, grip):
            waypoints.append({"t": t, "xyz": pos, "gripper": grip})

        start_hover = target_pos + np.array([0, 0, maybe_jitter(0.3)])
        approach_goal = target_pos + np.array([0, 0, maybe_jitter(0.05)])
        add_abs(0, start_hover, -1)
        approach_mids = self._approach_midpoints(start_hover, approach_goal)
        if approach_mids:
            approach_times = np.linspace(20, 180, num=len(approach_mids) + 2)[1:-1]
            for at, ap in zip(approach_times, approach_mids):
                add_abs(int(at), ap, -1)
        add_abs(200, approach_goal, -1)
        add_abs(300, target_pos + np.array([0, 0, maybe_jitter(0.05)]), 1)
        add_abs(380, target_pos + np.array([0, 0, maybe_jitter(0.2)]), 1)

        transfer_start_t = 380
        transfer_end_t = 470
        transfer_start = target_pos + np.array([0, 0, maybe_jitter(0.2)])
        transfer_goal = destination_pos + np.array([0, 0, maybe_jitter(0.25)])
        mids = self._transfer_midpoints(transfer_start, transfer_goal)
        if mids:
            mid_times = np.linspace(transfer_start_t + 10, transfer_end_t - 10, num=len(mids) + 2)[1:-1]
            for mt, mp in zip(mid_times, mids):
                add_abs(int(mt), mp, 1)
        add_abs(transfer_end_t, transfer_goal, 1)

        add_abs(540, destination_pos + np.array([0, 0, maybe_jitter(0.1)]), 1)
        add_abs(570, destination_pos + np.array([0, 0, maybe_jitter(0.1)]), -1)
        add_abs(590, destination_pos + np.array([0, 0, maybe_jitter(0.3)]), -1)
        add_abs(600, destination_pos + np.array([0, 0, maybe_jitter(0.3)]), -1)

        waypoints.sort(key=lambda w: w["t"])
        last_t = -1
        for wp in waypoints:
            if wp["t"] <= last_t:
                wp["t"] = last_t + 1
            last_t = wp["t"]

        self.trajectory = waypoints


def load_init_states(path: str) -> Sequence[np.ndarray]:
    """
    Load mujoco states (flattened) from npz/npy or torch-saved tensors (.pt/.pruned_init).
    """
    p = Path(path)
    if p.suffix in {".pt", ".pth", ".pruned_init"}:
        tensor = torch.load(path, map_location="cpu")
        return list(tensor.detach().cpu().numpy())

    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        for key in ["states", "mujoco_states", "init_states"]:
            if key in data:
                return list(data[key])
        first_key = list(data.keys())[0]
        return list(data[first_key])
    return list(data)


def build_env(args):
    controller_config = suite.load_controller_config(default_controller=args.controller)
    config = {
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    assert os.path.exists(args.bddl_file)
    problem_info = BDDLUtils.get_problem_info(args.bddl_file)
    task_name = args.bddl_file.split("/")[-1].split(".")[0]

    parsed = BDDLUtils.robosuite_parse_problem(args.bddl_file)
    target_object = parsed["obj_of_interest"][0]
    region_key = [st[2] for st in parsed["initial_state"] if st[1] == target_object][0]
    anchor_ranges = sanitize_ranges(parsed["regions"][region_key]["ranges"])
    anchor_centers = [
        (np.mean([r[0], r[2]]), np.mean([r[1], r[3]])) for r in anchor_ranges
    ]

    problem_name = problem_info["problem_name"]
    domain_name = problem_info["domain_name"]
    language_instruction = problem_info["language_instruction"]
    if "TwoArm" in problem_name:
        config["env_configuration"] = args.config

    env = TASK_MAPPING[problem_name](
        bddl_file_name=args.bddl_file,
        **config,
        has_renderer=args.render,
        has_offscreen_renderer=not args.render,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=args.control_freq,
        region_sampling_strategy="round_robin",
        region_sampling_quota=1,
    )

    if args.seed is not None:
        env.seed(args.seed)

    env_info = json.dumps(config)
    return env, env_info, {
        "problem_name": problem_name,
        "domain_name": domain_name,
        "language_instruction": language_instruction,
        "anchor_ranges": anchor_ranges,
        "anchor_centers": anchor_centers,
        "task_name": task_name,
    }


def rollout_policy(
    env,
    policy,
    init_state: Optional[np.ndarray],
    max_steps: int,
    hold_steps: int,
    render: bool,
) -> Tuple[bool, int, str]:
    """
    Run one episode with the scripted policy and optional fixed initial mujoco state.
    """
    reset_ok = False
    obs = None
    while not reset_ok:
        try:
            obs = env.reset()
            reset_ok = True
        except Exception:
            continue

    if init_state is not None:
        obs = regenerate_obs_from_state(env, init_state)

    obj_of_interest = env.obj_of_interest.copy()
    target_object_name = obj_of_interest[0]
    destination_name = obj_of_interest[-1]

    target_pos = obs[f"{target_object_name.replace('_main', '')}_pos"].copy()
    print(target_object_name, target_pos)
    destination_pos = obs[f"{destination_name.replace('_main', '')}_pos"].copy()

    success = False
    task_completed = False
    task_completion_hold_count = -1
    fail_reason = None

    for _ in range(max_steps):
        current_eef_pos = obs["robot0_eef_pos"].copy()

        if not success:
            xyz_delta, gripper = policy(target_pos, destination_pos, current_eef_pos)
            action = np.zeros(7)
            action[:3] = xyz_delta * 3
            action[3:6] = 0
            action[6] = gripper
        else:
            action = np.zeros(7)
            action[2] = 0.1
            action[6] = -1

        obs, _, success, _ = env.step(action)
        if render:
            env.render()

        if not success and getattr(policy, "trajectory", []) == []:
            fail_reason = "trajectory_exhausted"
            break

        if task_completion_hold_count == 0:
            task_completed = True
            break

        if success:
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1
            else:
                task_completion_hold_count = hold_steps
        else:
            task_completion_hold_count = -1
    else:
        fail_reason = "timeout"

    if not task_completed and fail_reason is None:
        fail_reason = "unknown"

    return task_completed, policy.step_count, fail_reason


def pick_policy(args):
    if args.policy == "fixed":
        return FixedPickAndPlacePolicy()
    return RandomizedPickAndPlacePolicy(
        inject_noise=args.inject_noise,
        transfer_offset_max=args.transfer_offset_max,
        approach_offset_max=args.approach_offset_max,
    )


def regenerate_obs_from_state(env, mujoco_state):
    """
    Align sim state with observations; uses env helper when available.
    """
    if hasattr(env, "regenerate_obs_from_state"):
        return env.regenerate_obs_from_state(mujoco_state)
    env.sim.set_state_from_flattened(mujoco_state)
    env.sim.forward()
    if hasattr(env, "_post_process"):
        env._post_process()
    if hasattr(env, "_update_observables"):
        env._update_observables(force=True)
    return env._get_observations()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bddl-file", type=str, required=True)
    parser.add_argument("--robots", nargs="+", type=str, default=["Panda"])
    parser.add_argument("--config", type=str, default="single-arm-opposed")
    parser.add_argument("--camera", type=str, default="agentview")
    parser.add_argument("--controller", type=str, default="OSC_POSE")
    parser.add_argument("--policy", choices=["fixed", "randomized"], default="randomized")
    parser.add_argument("--inject-noise", action="store_true")
    parser.add_argument("--transfer-offset-max", type=float, default=0.06)
    parser.add_argument("--approach-offset-max", type=float, default=0.06)
    parser.add_argument("--max-steps", type=int, default=750)
    parser.add_argument("--hold-steps", type=int, default=40)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--init-state-file", type=str, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument(
        "--debug-trajectory",
        action="store_true",
        help="Print generated waypoint list and current/next waypoint during rollout.",
    )
    args = parser.parse_args()

    env, env_info, meta = build_env(args)
    init_states: Optional[Sequence[np.ndarray]] = None
    if args.init_state_file:
        init_states = load_init_states(args.init_state_file)
        if args.start_index < 0 or args.start_index >= len(init_states):
            raise ValueError(f"start-index {args.start_index} out of range for {len(init_states)} states.")
        init_states = init_states[args.start_index :]

    successes = 0
    total = args.episodes if init_states is None else min(args.episodes, len(init_states))

    for ep in range(total):
        init_state = None if init_states is None else init_states[ep]
        policy = pick_policy(args)
        if args.debug_trajectory:
            policy.debug = True
        done, steps, fail_reason = rollout_policy(
            env,
            policy,
            init_state,
            max_steps=args.max_steps,
            hold_steps=args.hold_steps,
            render=args.render,
        )
        status = "success" if done else f"fail({fail_reason})"
        print(f"[episode {ep + 1}/{total}] {status} in {steps} policy steps")
        if done:
            successes += 1

    success_rate = successes / max(1, total)
    print(f"Success rate: {successes}/{total} = {success_rate:.2%}")
    env.close()


if __name__ == "__main__":
    main()

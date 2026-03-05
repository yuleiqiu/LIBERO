import argparse
import os

import imageio.v2 as imageio

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def resolve_task_bddl_path(task_suite_name: str, task_id: int):
    benchmark_dict = benchmark.get_benchmark_dict()
    try:
        task_suite = benchmark_dict[task_suite_name]()
        task = task_suite.get_task(task_id)
    except KeyError as exc:
        raise ValueError(
            f"Could not find task suite '{task_suite_name}' or task with ID {task_id}."
        ) from exc

    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )
    return task, task_bddl_file


def make_env(bddl_file: str, camera_width: int, camera_height: int):
    return OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_widths=camera_width,
        camera_heights=camera_height,
    )


def zero_action(env):
    action_dim = getattr(getattr(env, "env", None), "action_dim", 7)
    return [0.0] * action_dim


def get_camera_frame(obs, camera_name: str):
    key = f"{camera_name}_image"
    if key not in obs:
        raise KeyError(f"Camera '{key}' not found in observation.")
    return obs[key][::-1].copy()


def capture_rollout_frames(env, camera_name: str, num_steps: int):
    obs = env.reset()
    frames = [get_camera_frame(obs, camera_name)]
    action = zero_action(env)
    for _ in range(num_steps):
        obs, _, _, _ = env.step(action)
        frames.append(get_camera_frame(obs, camera_name))
    return frames



def main():
    parser = argparse.ArgumentParser(description="Visualize regions from a BDDL file in its environment.")
    parser.add_argument("--task-suite-name", type=str, required=True, help="Name of the task suite.")
    parser.add_argument("--task-id", type=int, required=True, help="ID of the task to visualize.")
    parser.add_argument("--output-path", type=str, default="initialization_sampler.mp4", help="Path to save the visualization image.")
    parser.add_argument("--camera-name", type=str, default="agentview", help="Camera to render from.")
    parser.add_argument("--camera-width", type=int, default=512, help="Camera width for rendering.")
    parser.add_argument("--camera-height", type=int, default=512, help="Camera height for rendering.")
    parser.add_argument("--num-steps", type=int, default=10, help="Number of zero-action steps to render after reset.")
    parser.add_argument("--fps", type=int, default=24, help="FPS for the output video.")
    args = parser.parse_args()

    try:
        task, task_bddl_file = resolve_task_bddl_path(args.task_suite_name, args.task_id)
        print(f"Visualizing task: {task.name} from {task_bddl_file}")
    except ValueError as exc:
        print(exc)
        return

    env = make_env(
        task_bddl_file,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
    )
    try:
        frames = capture_rollout_frames(env, camera_name=args.camera_name, num_steps=args.num_steps)
        imageio.mimwrite(args.output_path, frames, fps=args.fps)
        print(f"Visualization saved to {args.output_path}")
    finally:
        env.close()

if __name__ == "__main__":
    main()

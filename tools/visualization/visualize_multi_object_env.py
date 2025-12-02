import argparse
import math
import numpy as np
import imageio

from libero.libero.envs import OffScreenRenderEnv


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a multi-object BDDL environment by stacking several reset frames."
    )
    parser.add_argument("--bddl-file", type=str, required=True, help="Path to the multi-object BDDL file.")
    parser.add_argument("--output-path", type=str, default="multi_object_env_grid.png", help="Path to save the composite image.")
    parser.add_argument("--camera-name", type=str, default="agentview", help="Camera to render from (e.g., agentview or sideview).")
    parser.add_argument("--num-resets", type=int, default=9, help="How many resets to visualize.")
    parser.add_argument("--settle-steps", type=int, default=10, help="Empty steps after reset to let objects settle.")
    parser.add_argument("--camera-width", type=int, default=256, help="Camera width for rendering.")
    parser.add_argument("--camera-height", type=int, default=256, help="Camera height for rendering.")
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
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    args = parser.parse_args()

    grid_cols = max(1, math.ceil(math.sqrt(args.num_resets)))
    grid_rows = max(1, math.ceil(args.num_resets / grid_cols))

    env = OffScreenRenderEnv(
        bddl_file_name=args.bddl_file,
        camera_widths=args.camera_width,
        camera_heights=args.camera_height,
        region_sampling_strategy=args.region_sampling_strategy,
        region_sampling_quota=args.region_sampling_quota,
    )
    if args.seed is not None:
        env.seed(args.seed)

    frames = []
    for idx in range(args.num_resets):
        print(f"[{idx + 1}/{args.num_resets}] resetting env...")
        obs = env.reset()
        for _ in range(args.settle_steps):
            obs, _, _, _ = env.step([0.0] * 7)

        key = f"{args.camera_name}_image"
        if key not in obs:
            raise KeyError(f"Camera '{key}' not found in observation.")

        # Images from OffScreenRenderEnv come flipped vertically; flip them back.
        frame = obs[key][::-1].copy()
        frames.append(frame)

    env.close()

    if not frames:
        print("No frames captured; exiting without saving.")
        return

    img_h, img_w = frames[0].shape[:2]
    grid = np.zeros((grid_rows * img_h, grid_cols * img_w, 3), dtype=np.uint8)

    for i, frame in enumerate(frames):
        row = i // grid_cols
        col = i % grid_cols
        y0, y1 = row * img_h, (row + 1) * img_h
        x0, x1 = col * img_w, (col + 1) * img_w
        grid[y0:y1, x0:x1] = frame

    imageio.imwrite(args.output_path, grid)
    print(f"Composite visualization saved to {args.output_path}")


if __name__ == "__main__":
    main()

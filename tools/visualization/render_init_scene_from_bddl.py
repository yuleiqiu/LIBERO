import argparse
import math
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

from libero.libero.envs import OffScreenRenderEnv


def make_grid(frames):
    if not frames:
        raise ValueError("No frames to compose.")

    num_frames = len(frames)
    cols = math.ceil(math.sqrt(num_frames))
    rows = math.ceil(num_frames / cols)
    height, width = frames[0].shape[:2]
    grid = np.zeros((rows * height, cols * width, 3), dtype=np.uint8)

    for idx, frame in enumerate(frames):
        row = idx // cols
        col = idx % cols
        y0 = row * height
        x0 = col * width
        grid[y0 : y0 + height, x0 : x0 + width] = frame

    return grid


def project_world_point(sim, camera_name, point, image_height, image_width):
    cam_id = sim.model.camera_name2id(camera_name)
    cam_pos = sim.data.cam_xpos[cam_id]
    cam_rot = sim.data.cam_xmat[cam_id].reshape(3, 3)
    fovy = np.deg2rad(sim.model.cam_fovy[cam_id])

    # MuJoCo camera looks along its local -Z axis.
    point_cam = cam_rot.T @ (np.asarray(point, dtype=float) - cam_pos)
    depth = -point_cam[2]
    if depth <= 1e-6:
        return None

    focal = 0.5 * image_height / np.tan(fovy / 2.0)
    pixel_x = focal * (point_cam[0] / depth) + image_width / 2.0
    pixel_y = image_height / 2.0 - focal * (point_cam[1] / depth)
    return int(round(pixel_x)), int(round(pixel_y))


def overlay_world_axes(frame, sim, camera_name, origin, axis_length=0.10):
    image_height, image_width = frame.shape[:2]
    origin_px = project_world_point(sim, camera_name, origin, image_height, image_width)
    if origin_px is None:
        return frame

    axis_specs = [
        ("x", np.array([axis_length, 0.0, 0.0]), (255, 64, 64)),
        ("y", np.array([0.0, axis_length, 0.0]), (64, 220, 64)),
    ]

    annotated = frame.copy()
    cv2.circle(annotated, origin_px, 4, (255, 255, 255), -1)
    cv2.putText(
        annotated,
        "O",
        (origin_px[0] + 6, origin_px[1] - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    for label, offset, color in axis_specs:
        endpoint_px = project_world_point(
            sim,
            camera_name,
            np.asarray(origin, dtype=float) + offset,
            image_height,
            image_width,
        )
        if endpoint_px is None:
            continue
        cv2.arrowedLine(
            annotated,
            origin_px,
            endpoint_px,
            color,
            2,
            cv2.LINE_AA,
            tipLength=0.18,
        )
        cv2.putText(
            annotated,
            label,
            (endpoint_px[0] + 4, endpoint_px[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return annotated


def overlay_object_position(frame, object_name, position):
    annotated = frame.copy()
    text = f"({position[0]:+.3f}, {position[1]:+.3f})"
    cv2.putText(
        annotated,
        text,
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        text,
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return annotated


def resolve_init_states_path(path_str):
    candidate = Path(path_str).expanduser().resolve()
    if candidate.is_dir():
        matches = sorted(
            p
            for p in candidate.iterdir()
            if p.is_file()
            and (
                p.suffix in {".pt", ".pth"}
                or p.name.endswith((".init", ".pruned_init"))
            )
        )
        if not matches:
            raise FileNotFoundError(
                f"No init_states file (.init/.pruned_init/.pt/.pth) found under {candidate}"
            )
        if len(matches) > 1:
            raise ValueError(
                f"Multiple init_states files found under {candidate}: {[p.name for p in matches]}"
            )
        return matches[0]
    return candidate


def load_init_states(path_str):
    init_path = resolve_init_states_path(path_str)
    if not init_path.exists():
        raise FileNotFoundError(f"Init states file not found: {init_path}")
    init_states = torch.load(str(init_path), weights_only=False)
    if torch.is_tensor(init_states):
        init_states = init_states.cpu().numpy()
    else:
        init_states = np.asarray(init_states)
    if init_states.ndim < 2:
        raise ValueError(f"Expected stacked init states in {init_path}, got shape {init_states.shape}")
    return init_path, init_states


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render multiple initialization images directly from a BDDL file."
    )
    parser.add_argument("--bddl-file", required=True, help="Path to the BDDL file.")
    parser.add_argument(
        "--num-images",
        type=int,
        default=12,
        help="Number of initialization images to render.",
    )
    parser.add_argument(
        "--camera-name",
        type=str,
        default="agentview",
        help="Camera name to render from.",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=256,
        help="Render height in pixels.",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=256,
        help="Render width in pixels.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory. Default: ./tmp/new_scene_overview/<bddl_file_stem>",
    )
    parser.add_argument(
        "--init-states-file",
        type=str,
        default=None,
        help="Optional path to an init states file or directory containing one. If set, render these states instead of random resets.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    settle_steps = 15
    bddl_path = Path(args.bddl_file).expanduser().resolve()
    if not bddl_path.exists():
        raise FileNotFoundError(f"BDDL file not found: {bddl_path}")
    if args.num_images <= 0:
        raise ValueError("--num-images must be positive.")
    init_states_path = None
    init_states = None
    if args.init_states_file:
        init_states_path, init_states = load_init_states(args.init_states_file)

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else Path("./tmp/new_scene_overview")
        / (
            f"{bddl_path.stem}_{init_states_path.stem}"
            if init_states_path is not None
            else bddl_path.stem
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=args.camera_height,
        camera_widths=args.camera_width,
    )
    action = np.zeros(env.env.action_dim, dtype=float)
    frames = []
    axes_origin = np.array(env.env.workspace_offset, dtype=float)
    axes_origin[2] += 0.002
    tracked_object = next(
        (
            name
            for name in env.env.obj_of_interest
            if name in env.env.objects_dict
        ),
        None,
    )

    total_images = args.num_images if init_states is None else min(args.num_images, len(init_states))
    if init_states is not None and args.num_images > len(init_states):
        print(
            f"[warning] requested {args.num_images} images but only {len(init_states)} init states available; rendering {total_images}"
        )

    for idx in range(total_images):
        obs = env.reset()
        if init_states is not None:
            obs = env.set_init_state(init_states[idx])
        for _ in range(settle_steps):
            obs, _, _, _ = env.step(action)
        frame = obs[f"{args.camera_name}_image"][::-1].copy()
        frame = overlay_world_axes(
            frame,
            env.env.sim,
            args.camera_name,
            axes_origin,
        )
        if tracked_object is not None:
            object_pos = env.env.sim.data.body_xpos[env.env.obj_body_id[tracked_object]]
            frame = overlay_object_position(frame, tracked_object, object_pos)
        frames.append(frame)
        imageio.imwrite(out_dir / f"{idx:02d}.png", frame)

    grid = make_grid(frames)
    imageio.imwrite(out_dir / "grid.png", grid)
    env.close()

    print(f"[info] saved {total_images} init images to {out_dir}")
    print(f"[info] saved grid image to {out_dir / 'grid.png'}")


if __name__ == "__main__":
    main()

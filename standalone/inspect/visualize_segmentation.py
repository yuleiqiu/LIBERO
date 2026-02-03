import argparse
from pathlib import Path

import numpy as np
import cv2

from libero.libero.envs import SegmentationRenderEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize a segmentation frame.")
    parser.add_argument(
        "--bddl",
        required=True,
        help="Path to a .bddl task file.",
    )
    parser.add_argument(
        "--camera",
        default="agentview",
        help="Camera name(s), comma-separated (used to build default image keys).",
    )
    parser.add_argument(
        "--image-key",
        default="",
        help="Explicit obs image key(s), comma-separated (overrides --camera-derived keys).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=128,
        help="Camera height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=128,
        help="Camera width.",
    )
    parser.add_argument(
        "--output",
        default="tmp/seg_vis.png",
        help=(
            "Output image path. If multiple cameras are provided, a suffix is added. "
            "If this is a directory, files are written inside it."
        ),
    )
    parser.add_argument(
        "--random-colors",
        action="store_true",
        help="Use random colors for segmentation visualization.",
    )
    parser.add_argument(
        "--interest-only",
        action="store_true",
        help="Visualize get_segmentation_of_interest() mask instead of raw instances.",
    )
    return parser.parse_args()


def _split_csv(value: str):
    return [item.strip() for item in value.split(",") if item.strip()]


def _derive_image_keys(cameras):
    image_keys = []
    for name in cameras:
        if name.endswith("_image"):
            image_keys.append(name)
        else:
            image_keys.append(f"{name}_image")
    return image_keys


def _segmentation_key(env_key: str) -> str:
    if env_key.endswith("_image"):
        return f"{env_key[: -len('_image')]}_segmentation_instance"
    return f"{env_key}_segmentation_instance"


def _output_for_camera(base_path: Path, camera_name: str, multiple: bool) -> Path:
    if not multiple:
        return base_path
    if base_path.suffix:
        return base_path.with_name(f"{base_path.stem}_{camera_name}{base_path.suffix}")
    return base_path / f"seg_vis_{camera_name}.png"


def main():
    args = parse_args()
    bddl_path = Path(args.bddl).expanduser().resolve()
    if not bddl_path.exists():
        raise FileNotFoundError(f"BDDL not found: {bddl_path}")

    cameras = _split_csv(args.camera)
    if not cameras:
        raise ValueError("camera name is empty")

    image_keys = _split_csv(args.image_key) if args.image_key.strip() else []
    if image_keys:
        if len(image_keys) != len(cameras):
            raise ValueError(
                "image keys count must match camera count when --image-key is provided"
            )
    else:
        image_keys = _derive_image_keys(cameras)

    base_output = Path(args.output).expanduser().resolve()
    if base_output.exists() and base_output.is_dir():
        output_dir = base_output
    elif base_output.suffix:
        output_dir = base_output.parent
    else:
        output_dir = base_output
    output_dir.mkdir(parents=True, exist_ok=True)

    env = SegmentationRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_segmentations="instance",
        use_camera_obs=True,
        camera_names=cameras,
        camera_heights=args.height,
        camera_widths=args.width,
    )

    try:
        obs = env.reset()
        available = sorted([k for k in obs.keys() if "segmentation" in k])
        multiple = len(cameras) > 1
        for cam_name, env_key in zip(cameras, image_keys):
            seg_key = _segmentation_key(env_key)
            if seg_key not in obs:
                raise KeyError(
                    f"segmentation key not found: {seg_key}. "
                    f"available segmentation keys: {available}"
                )
            seg = np.squeeze(obs[seg_key])
            if args.interest_only:
                mask = np.squeeze(env.get_segmentation_of_interest(seg))
                rgb = np.zeros(mask.shape + (3,), dtype=np.uint8)
                rgb[mask == 1] = (255, 0, 0)
            else:
                rgb = env.segmentation_to_rgb(seg, random_colors=args.random_colors)
            output_path = _output_for_camera(base_output, cam_name, multiple)
            if not output_path.suffix:
                output_path = output_path.with_suffix(".png")
            rgb = rgb[::-1]
            cv2.imwrite(str(output_path), rgb[..., ::-1])
            print(f"[info] saved segmentation to {output_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()

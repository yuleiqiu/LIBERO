import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel


def demo_sort_key(name: str):
    if name.startswith("demo_"):
        suffix = name.split("demo_", 1)[1]
        if suffix.isdigit():
            return int(suffix)
    return name


def resolve_hdf5_file(hdf5_path: str) -> Path:
    path = Path(hdf5_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"hdf5 path does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"--hdf5-path must be a file, got: {path}")
    if path.suffix not in (".hdf5", ".h5"):
        raise ValueError(f"--hdf5-path must be .hdf5/.h5, got: {path}")
    return path


def sample_demo_key(data_group: h5py.Group, demo_key: str | None) -> str:
    demo_keys = sorted(
        [k for k in data_group.keys() if k.startswith("demo_")], key=demo_sort_key
    )
    if not demo_keys:
        raise ValueError("No demo groups found under /data")

    if demo_key is not None:
        if demo_key not in demo_keys:
            raise KeyError(
                f"Requested demo key '{demo_key}' not found. Available examples: {demo_keys[:8]}"
            )
        return demo_key

    return demo_keys[0]


def to_uint8_image(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return frame
    return np.clip(frame, 0, 255).astype(np.uint8)


def get_patch_tokens_and_size_check(
    model, processor, image: np.ndarray, device: torch.device
):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model(**inputs)

    tokens = out.last_hidden_state  # [1, 1+R+N, d]
    num_register_tokens = getattr(model.config, "num_register_tokens", 0)
    patch = tokens[:, 1 + num_register_tokens :, :]  # [1, N, d]
    patch = torch.nn.functional.normalize(patch, dim=-1)

    height, width = inputs["pixel_values"].shape[-2:]
    patch_size = getattr(model.config, "patch_size", None)
    seq_len = tokens.shape[1]
    num_patches = seq_len - 1 - num_register_tokens
    if patch_size is not None and patch_size > 0:
        grid_h = height // patch_size
        grid_w = width // patch_size
    else:
        grid_h = int(num_patches**0.5)
        grid_w = int(num_patches**0.5)

    size_info = {
        "height": int(height),
        "width": int(width),
        "patch_size": patch_size,
        "num_register_tokens": int(num_register_tokens),
        "seq_len": int(seq_len),
        "num_patches": int(num_patches),
        "grid_h": int(grid_h),
        "grid_w": int(grid_w),
    }
    return patch, size_info


def print_size_check(size_info: dict):
    print(
        "H,W =",
        (size_info["height"], size_info["width"]),
        "P =",
        size_info["patch_size"],
        "R =",
        size_info["num_register_tokens"],
        "L =",
        size_info["seq_len"],
        "N =",
        size_info["num_patches"],
        "grid =",
        size_info["grid_h"],
        "x",
        size_info["grid_w"],
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quick check: read one LIBERO demo and run DINOv2 size/token check on one frame per camera."
    )
    parser.add_argument(
        "--hdf5-path",
        type=str,
        required=True,
        help="Direct path to one LIBERO .hdf5/.h5 dataset file.",
    )
    parser.add_argument(
        "--camera-keys",
        type=str,
        nargs="+",
        default=["agentview_rgb", "eye_in_hand_rgb"],
        choices=["agentview_rgb", "eye_in_hand_rgb"],
        help=(
            "Camera observation keys to use from /data/demo_x/obs/. "
            "Default uses both views; pass one key to use only one view."
        ),
    )
    parser.add_argument(
        "--demo-key",
        type=str,
        default=None,
        help="Optional fixed demo key (e.g., demo_0). If omitted, use the first demo.",
    )
    parser.add_argument(
        "--flip-vertical",
        action="store_true",
        help="Flip each frame vertically before feeding ViT.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="facebook/dinov2-with-registers-base",
        help="HuggingFace model id for ViT feature extraction.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device, e.g., cuda, cuda:0, cpu. Defaults to cuda if available.",
    )
    parser.add_argument(
        "--frame-select",
        type=str,
        default="last",
        choices=["first", "last", "index"],
        help="Which frame to check per camera.",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Frame index used when --frame-select index.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    hdf5_path = resolve_hdf5_file(args.hdf5_path)

    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f:
            raise KeyError(f"Cannot find top-level 'data' group in {hdf5_path}")
        data_group = f["data"]

        demo_key = sample_demo_key(data_group, args.demo_key)
        obs_group = data_group[demo_key]["obs"]
        frames_by_camera = {}
        for camera_key in list(dict.fromkeys(args.camera_keys)):
            if camera_key not in obs_group:
                available = list(obs_group.keys())
                raise KeyError(
                    f"Camera key '{camera_key}' not found in demo {demo_key}. "
                    f"Available obs keys: {available}"
                )
            frames_by_camera[camera_key] = np.asarray(obs_group[camera_key][()])

    print(f"[Dataset] hdf5: {hdf5_path}")
    print(f"[Dataset] demo: {demo_key}")
    print(f"[Dataset] cameras: {list(dict.fromkeys(args.camera_keys))}")
    print(f"[Model] loading {args.model_name} on {device} ...")

    processor = AutoImageProcessor.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).eval().to(device)

    for camera_key in list(dict.fromkeys(args.camera_keys)):
        frames = frames_by_camera[camera_key]
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(
                f"Expected image tensor [T, H, W, 3], got shape {frames.shape} from {camera_key}"
            )

        if args.flip_vertical:
            frames = frames[:, ::-1, :, :]

        num_frames = int(frames.shape[0])
        if num_frames == 0:
            raise ValueError(f"Selected demo has 0 frames for camera '{camera_key}'.")

        print(
            f"[Dataset/{camera_key}] frames: {num_frames}, frame_shape: {tuple(frames.shape[1:])}"
        )

        if args.frame_select == "first":
            frame_idx = 0
        elif args.frame_select == "last":
            frame_idx = num_frames - 1
        else:
            if args.frame_index < 0 or args.frame_index >= num_frames:
                raise ValueError(
                    f"--frame-index out of range: {args.frame_index}, valid range [0, {num_frames - 1}]"
                )
            frame_idx = args.frame_index

        frame = to_uint8_image(frames[frame_idx])
        patch, size_info = get_patch_tokens_and_size_check(
            model, processor, frame, device=device
        )
        patch = patch[0].cpu()

        print(f"[SizeCheck/{camera_key}] frame_index={frame_idx}")
        print_size_check(size_info)
        print(f"[Done/{camera_key}] patch shape: {tuple(patch.shape)}")


if __name__ == "__main__":
    main()

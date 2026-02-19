#!/usr/bin/env python3
"""Visualize patch correspondence heatmaps with cross-demo goals for LIBERO demos."""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import h5py
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


CameraFrame = Dict[str, np.ndarray]


def demo_sort_key(name: str):
    if name.startswith("demo_"):
        suffix = name.split("demo_", 1)[1]
        if suffix.isdigit():
            return int(suffix)
    return name


def _decode_if_bytes(value):
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return value


def infer_control_freq(data_group: h5py.Group, default_fps: int = 20) -> int:
    # Preferred source in processed LIBERO datasets.
    raw_env_args = data_group.attrs.get("env_args")
    if raw_env_args is not None:
        try:
            env_args_str = _decode_if_bytes(raw_env_args)
            env_args = env_args_str if isinstance(env_args_str, dict) else json.loads(env_args_str)
            env_kwargs = env_args.get("env_kwargs", {})
            if "control_freq" in env_kwargs:
                return int(env_kwargs["control_freq"])
        except Exception:
            pass

    # Fallback source in replay datasets.
    raw_env_info = data_group.attrs.get("env_info")
    if raw_env_info is not None:
        try:
            env_info_str = _decode_if_bytes(raw_env_info)
            env_info = env_info_str if isinstance(env_info_str, dict) else json.loads(env_info_str)
            if "control_freq" in env_info:
                return int(env_info["control_freq"])
        except Exception:
            pass

    return int(default_fps)


def load_current_and_goal_frames(
    hdf5_path: str,
    demo_index: int,
    cam1_key: str,
    cam2_key: str,
) -> Tuple[List[CameraFrame], Dict[str, np.ndarray], str, str, int]:
    path = Path(hdf5_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"hdf5 path does not exist: {path}")

    with h5py.File(path, "r") as f:
        if "data" not in f:
            raise KeyError(f"Cannot find top-level 'data' group in {path}")

        data_group = f["data"]
        demo_keys = sorted(
            [k for k in data_group.keys() if k.startswith("demo_")],
            key=demo_sort_key,
        )
        if not demo_keys:
            raise ValueError("No demo groups found under /data")

        if demo_index < 0 or demo_index >= len(demo_keys):
            raise IndexError(
                f"demo_index {demo_index} out of range [0, {len(demo_keys) - 1}]"
            )
        goal_demo_index = demo_index + 1
        if goal_demo_index >= len(demo_keys):
            raise IndexError(
                f"goal demo index {goal_demo_index} (demo_index+1) out of range "
                f"[0, {len(demo_keys) - 1}]"
            )

        current_demo_key = demo_keys[demo_index]
        goal_demo_key = demo_keys[goal_demo_index]
        obs_group = data_group[current_demo_key]["obs"]
        goal_obs_group = data_group[goal_demo_key]["obs"]

        if cam1_key not in obs_group:
            raise KeyError(f"{cam1_key} not found in /data/{current_demo_key}/obs")
        if cam2_key not in obs_group:
            raise KeyError(f"{cam2_key} not found in /data/{current_demo_key}/obs")
        if cam1_key not in goal_obs_group:
            raise KeyError(f"{cam1_key} not found in /data/{goal_demo_key}/obs")
        if cam2_key not in goal_obs_group:
            raise KeyError(f"{cam2_key} not found in /data/{goal_demo_key}/obs")

        cam1_frames = np.asarray(obs_group[cam1_key][()])
        cam2_frames = np.asarray(obs_group[cam2_key][()])

        if cam1_frames.ndim != 4 or cam1_frames.shape[-1] != 3:
            raise ValueError(
                f"Expected {cam1_key} shape [T,H,W,3], got {cam1_frames.shape}"
            )
        if cam2_frames.ndim != 4 or cam2_frames.shape[-1] != 3:
            raise ValueError(
                f"Expected {cam2_key} shape [T,H,W,3], got {cam2_frames.shape}"
            )

        num_frames = min(int(cam1_frames.shape[0]), int(cam2_frames.shape[0]))
        if num_frames <= 0:
            raise ValueError(f"Demo {current_demo_key} has no RGB frames")

        frames = [
            {
                "cam1": cam1_frames[t],
                "cam2": cam2_frames[t],
            }
            for t in range(num_frames)
        ]

        goal_cam1_frames = np.asarray(goal_obs_group[cam1_key][()])
        goal_cam2_frames = np.asarray(goal_obs_group[cam2_key][()])
        if goal_cam1_frames.shape[0] <= 0 or goal_cam2_frames.shape[0] <= 0:
            raise ValueError(f"Demo {goal_demo_key} has no RGB frames")
        goal_last = {
            "cam1": goal_cam1_frames[-1],
            "cam2": goal_cam2_frames[-1],
        }

        fps = infer_control_freq(data_group, default_fps=20)

    return frames, goal_last, current_demo_key, goal_demo_key, fps


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)


def _to_hw(value) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    if isinstance(value, int):
        return int(value), int(value)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    if isinstance(value, dict):
        if "height" in value and "width" in value:
            return int(value["height"]), int(value["width"])
        if "shortest_edge" in value:
            edge = int(value["shortest_edge"])
            return edge, edge
    return None


def encode_patches(
    images: Sequence[np.ndarray],
    model,
    processor,
    device: torch.device,
) -> Tuple[torch.Tensor, Tuple[int, int], int, torch.Tensor]:
    inputs = processor(images=list(images), return_tensors="pt")
    pixel_values_cpu = inputs["pixel_values"]
    pixel_values = pixel_values_cpu.to(device)

    with torch.inference_mode():
        outputs = model(pixel_values=pixel_values)

    tokens = outputs.last_hidden_state  # [B, 1+R+N, d]
    num_register_tokens = int(getattr(model.config, "num_register_tokens", 0))
    patches = tokens[:, 1 + num_register_tokens :, :]  # [B, N, d]
    patches = F.normalize(patches, dim=-1)

    input_hw = (int(pixel_values.shape[-2]), int(pixel_values.shape[-1]))
    return patches, input_hw, num_register_tokens, pixel_values_cpu


def infer_patch_grid(
    num_patches: int,
    input_hw: Tuple[int, int],
    model,
) -> Tuple[int, int]:
    input_h, input_w = input_hw

    patch_hw = _to_hw(getattr(model.config, "patch_size", None))
    if patch_hw is not None:
        ph, pw = patch_hw
        if ph > 0 and pw > 0 and input_h % ph == 0 and input_w % pw == 0:
            hp, wp = input_h // ph, input_w // pw
            if hp * wp == num_patches:
                return hp, wp

    image_hw = _to_hw(getattr(model.config, "image_size", None))
    if image_hw is not None and patch_hw is not None:
        ih, iw = image_hw
        ph, pw = patch_hw
        if ph > 0 and pw > 0 and ih % ph == 0 and iw % pw == 0:
            hp, wp = ih // ph, iw // pw
            if hp * wp == num_patches:
                return hp, wp

    side = int(math.sqrt(num_patches))
    if side * side != num_patches:
        raise ValueError(
            f"Cannot infer patch grid from N={num_patches}. "
            "Need model.config.patch_size/image_size or square N."
        )
    return side, side


def compute_attn(
    xt: torch.Tensor,
    xg: torch.Tensor,
    tau: float,
    agg: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")

    sim = (xt @ xg.transpose(0, 1)) / tau  # [N, N]
    if agg == "logsumexp":
        score = torch.logsumexp(sim, dim=1)  # [N]
    elif agg == "max":
        score = torch.max(sim, dim=1).values  # [N]
    else:
        raise ValueError(f"Unsupported agg: {agg}")

    attn = torch.softmax(score, dim=0)  # [N]
    return attn, score


def upsample_patch_map(vec: torch.Tensor, hp: int, wp: int, out_hw: Tuple[int, int]) -> np.ndarray:
    out_h, out_w = out_hw
    grid = vec.view(1, 1, hp, wp)
    up = F.interpolate(grid, size=(out_h, out_w), mode="bilinear", align_corners=False)
    return up[0, 0].detach().cpu().numpy()


def normalize_percentile(x: np.ndarray, low_q: float = 5.0, high_q: float = 95.0) -> np.ndarray:
    lo = float(np.percentile(x, low_q))
    hi = float(np.percentile(x, high_q))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def normalize_minmax(x: np.ndarray) -> np.ndarray:
    lo = float(x.min())
    hi = float(x.max())
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def pixel_values_to_rgb(pixel_values_chw: torch.Tensor, processor) -> np.ndarray:
    mean = getattr(processor, "image_mean", [0.5, 0.5, 0.5])
    std = getattr(processor, "image_std", [0.5, 0.5, 0.5])

    if len(mean) != int(pixel_values_chw.shape[0]) or len(std) != int(pixel_values_chw.shape[0]):
        raise ValueError(
            f"Unexpected channel stats: C={pixel_values_chw.shape[0]}, "
            f"len(mean)={len(mean)}, len(std)={len(std)}"
        )

    mean_t = torch.as_tensor(
        mean, dtype=pixel_values_chw.dtype, device=pixel_values_chw.device
    ).view(-1, 1, 1)
    std_t = torch.as_tensor(
        std, dtype=pixel_values_chw.dtype, device=pixel_values_chw.device
    ).view(-1, 1, 1)
    img = (pixel_values_chw * std_t + mean_t).clamp(0.0, 1.0)
    img = (img.permute(1, 2, 0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
    return img


def heatmap_to_rgb(heat01: np.ndarray) -> np.ndarray:
    heat_u8 = np.clip(255.0 * heat01, 0, 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    return heat_rgb


def make_overlay(img_rgb: np.ndarray, heat01: np.ndarray, alpha: float) -> np.ndarray:
    heat_rgb = heatmap_to_rgb(heat01).astype(np.float32)
    base = img_rgb.astype(np.float32)
    out = (1.0 - alpha) * base + alpha * heat_rgb
    return np.clip(out, 0, 255).astype(np.uint8)


def save_panel(
    cam1_overlay: np.ndarray,
    cam1_goal: np.ndarray,
    cam2_overlay: np.ndarray,
    cam2_goal: np.ndarray,
) -> np.ndarray:
    top = np.hstack([cam1_overlay, cam1_goal])
    bottom = np.hstack([cam2_overlay, cam2_goal])
    panel = np.vstack([top, bottom])
    return panel


def parse_args():
    parser = argparse.ArgumentParser(description="Patch correspondence heatmap visualization")
    parser.add_argument("--hdf5-path", type=str, required=True, help="Path to one LIBERO hdf5 file")
    parser.add_argument(
        "--demo_index",
        type=int,
        default=0,
        help="Current demo index in sorted list (default: 0); goal demo is demo_index + 1.",
    )
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for PNG/MP4")
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/dinov2-with-registers-base",
        help="HuggingFace model name",
    )
    parser.add_argument("--tau", type=float, default=0.1, help="Temperature for similarity")
    parser.add_argument(
        "--agg",
        type=str,
        default="logsumexp",
        choices=["logsumexp", "max"],
        help="Aggregation over goal patches",
    )
    parser.add_argument("--alpha", type=float, default=0.45, help="Heatmap overlay alpha")
    parser.add_argument(
        "--heat_source",
        type=str,
        default="score",
        choices=["score", "attn"],
        help="Which heatmap to overlay: score(percentile) or attn(min-max).",
    )
    parser.add_argument("--device", type=str, default=None, help="cuda/cpu")
    parser.add_argument(
        "--cam1_key",
        type=str,
        default="agentview_rgb",
        help="Camera-1 observation key",
    )
    parser.add_argument(
        "--cam2_key",
        type=str,
        default="eye_in_hand_rgb",
        help="Camera-2 observation key",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Optional FPS override for MP4. Defaults to dataset control frequency.",
    )
    parser.add_argument(
        "--save_png",
        action="store_true",
        help="If set, also save per-frame PNGs. Default is video-only.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError(f"alpha must be in [0,1], got {args.alpha}")

    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    frames, goal_last, current_demo_key, goal_demo_key, dataset_fps = load_current_and_goal_frames(
        hdf5_path=args.hdf5_path,
        demo_index=args.demo_index,
        cam1_key=args.cam1_key,
        cam2_key=args.cam2_key,
    )

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModel.from_pretrained(args.model_name).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(args.model_name)

    goal_cam1 = to_uint8_rgb(goal_last["cam1"])
    goal_cam2 = to_uint8_rgb(goal_last["cam2"])

    goal_patches_batch, input_hw, num_register_tokens, goal_pixel_values_batch = encode_patches(
        [goal_cam1, goal_cam2],
        model=model,
        processor=processor,
        device=device,
    )
    goal_cam1_patch = goal_patches_batch[0]
    goal_cam2_patch = goal_patches_batch[1]
    goal_cam1_vis = pixel_values_to_rgb(goal_pixel_values_batch[0], processor=processor)[::-1]
    goal_cam2_vis = pixel_values_to_rgb(goal_pixel_values_batch[1], processor=processor)[::-1]

    num_patches = int(goal_cam1_patch.shape[0])
    dim = int(goal_cam1_patch.shape[1])
    hp, wp = infer_patch_grid(num_patches, input_hw=input_hw, model=model)

    print("[Info] model_name:", args.model_name)
    print("[Info] current_demo_key:", current_demo_key)
    print("[Info] current_demo_index:", args.demo_index)
    print("[Info] goal_demo_key:", goal_demo_key)
    print("[Info] goal_demo_index:", args.demo_index + 1)
    print("[Info] num_frames:", len(frames))
    print("[Info] R:", num_register_tokens)
    print("[Info] d:", dim)
    print("[Info] N:", num_patches)
    print("[Info] patch_grid:", (hp, wp))
    print("[Info] image_size:", input_hw)
    print("[Info] tau:", args.tau)
    print("[Info] agg:", args.agg)
    print("[Info] heat_source:", args.heat_source)

    mp4_fps = int(args.fps) if args.fps is not None else int(dataset_fps)
    writer = imageio.get_writer(str(out_dir / "attn.mp4"), fps=mp4_fps)
    print("[Info] mp4_fps:", mp4_fps)

    for t in tqdm(range(len(frames)), desc="Rendering heatmaps"):
        cur_cam1 = to_uint8_rgb(frames[t]["cam1"])
        cur_cam2 = to_uint8_rgb(frames[t]["cam2"])

        cur_patches_batch, cur_input_hw, _, cur_pixel_values_batch = encode_patches(
            [cur_cam1, cur_cam2],
            model=model,
            processor=processor,
            device=device,
        )

        if cur_input_hw != input_hw:
            raise RuntimeError(
                f"Processor output size changed from {input_hw} to {cur_input_hw} at frame {t}"
            )

        xt_cam1 = cur_patches_batch[0]
        xt_cam2 = cur_patches_batch[1]

        attn1, score1 = compute_attn(xt_cam1, goal_cam1_patch, tau=args.tau, agg=args.agg)
        attn2, score2 = compute_attn(xt_cam2, goal_cam2_patch, tau=args.tau, agg=args.agg)

        heat1_score = normalize_percentile(upsample_patch_map(score1, hp, wp, out_hw=input_hw))
        heat2_score = normalize_percentile(upsample_patch_map(score2, hp, wp, out_hw=input_hw))

        heat1_attn = normalize_minmax(upsample_patch_map(attn1, hp, wp, out_hw=input_hw))
        heat2_attn = normalize_minmax(upsample_patch_map(attn2, hp, wp, out_hw=input_hw))

        if args.heat_source == "attn":
            heat1 = heat1_attn
            heat2 = heat2_attn
        else:
            heat1 = heat1_score
            heat2 = heat2_score

        # Always flip vertically for visualization consistency with LIBERO camera convention.
        cam1_vis = pixel_values_to_rgb(cur_pixel_values_batch[0], processor=processor)[::-1]
        cam2_vis = pixel_values_to_rgb(cur_pixel_values_batch[1], processor=processor)[::-1]
        heat1 = heat1[::-1]
        heat2 = heat2[::-1]
        cam1_overlay = make_overlay(cam1_vis, heat1, alpha=args.alpha)
        cam2_overlay = make_overlay(cam2_vis, heat2, alpha=args.alpha)

        panel = save_panel(
            cam1_overlay=cam1_overlay,
            cam1_goal=goal_cam1_vis,
            cam2_overlay=cam2_overlay,
            cam2_goal=goal_cam2_vis,
        )

        writer.append_data(panel)
        if args.save_png:
            imageio.imwrite(out_dir / f"frame_{t:04d}.png", panel)

    writer.close()

    print(f"[Done] Saved MP4 to: {out_dir / 'attn.mp4'}")
    if args.save_png:
        print(f"[Done] Saved {len(frames)} PNG frames to: {out_dir}")


if __name__ == "__main__":
    main()

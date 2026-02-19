#!/usr/bin/env python3
"""Compare patch uniqueness confidence heatmaps across two LIBERO hdf5 files.

Goal patches come from one demo in a "single" hdf5, while current frames
come from another demo in a "multi" hdf5.
"""

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

METRIC_ORDER = ("s_lse", "margin", "p_max", "neg_entropy")
METRIC_TITLES = {
    "s_lse": "s_i=logsumexp_j S_ij",
    "margin": "top1-top2",
    "p_max": "max_j p_ij",
    "neg_entropy": "-H(p_i)",
}


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


def load_demo_frames(
    hdf5_path: str,
    demo_index: int,
    cam1_key: str,
    cam2_key: str,
) -> Tuple[List[CameraFrame], str, int]:
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

        demo_key = demo_keys[demo_index]
        obs_group = data_group[demo_key]["obs"]

        if cam1_key not in obs_group:
            raise KeyError(f"{cam1_key} not found in /data/{demo_key}/obs")
        if cam2_key not in obs_group:
            raise KeyError(f"{cam2_key} not found in /data/{demo_key}/obs")

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
            raise ValueError(f"Demo {demo_key} has no RGB frames")

        frames = [
            {
                "cam1": cam1_frames[t],
                "cam2": cam2_frames[t],
            }
            for t in range(num_frames)
        ]

        fps = infer_control_freq(data_group, default_fps=20)

    return frames, demo_key, fps


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


def compute_row_confidences(xt: torch.Tensor, xg: torch.Tensor, tau: float) -> Dict[str, torch.Tensor]:
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")

    sim = (xt @ xg.transpose(0, 1)) / tau  # [N_cur, N_goal]
    if int(sim.shape[1]) < 2:
        raise ValueError(f"Need at least 2 goal patches for margin, got {sim.shape[1]}")

    s_lse = torch.logsumexp(sim, dim=1)
    top2 = torch.topk(sim, k=2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]

    p = torch.softmax(sim, dim=1)
    p_max = torch.max(p, dim=1).values
    entropy = -(p * torch.log(torch.clamp(p, min=1e-12))).sum(dim=1)
    neg_entropy = -entropy

    return {
        "s_lse": s_lse,
        "margin": margin,
        "p_max": p_max,
        "neg_entropy": neg_entropy,
    }


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


def vector_to_heat(
    vec: torch.Tensor,
    hp: int,
    wp: int,
    out_hw: Tuple[int, int],
    map_mode: str,
    low_q: float,
    high_q: float,
) -> np.ndarray:
    if map_mode == "softmax_i":
        vec = torch.softmax(vec, dim=0)
        return normalize_minmax(upsample_patch_map(vec, hp, wp, out_hw=out_hw))
    if map_mode == "percentile":
        return normalize_percentile(
            upsample_patch_map(vec, hp, wp, out_hw=out_hw),
            low_q=low_q,
            high_q=high_q,
        )
    raise ValueError(f"Unsupported map_mode: {map_mode}")


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


def add_title(img_rgb: np.ndarray, title: str) -> np.ndarray:
    out = img_rgb.copy()
    h, w = out.shape[:2]
    bar_h = max(24, int(round(0.10 * h)))
    cv2.rectangle(out, (0, 0), (w - 1, bar_h), (0, 0, 0), thickness=-1)
    font_scale = max(0.5, h / 420.0)
    cv2.putText(
        out,
        title,
        (8, int(bar_h * 0.72)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness=1,
        lineType=cv2.LINE_AA,
    )
    return out


def render_camera_row(
    cur_vis: np.ndarray,
    goal_vis: np.ndarray,
    heat_by_metric: Dict[str, np.ndarray],
    alpha: float,
) -> np.ndarray:
    tiles = [add_title(cur_vis, "current")]
    for metric_name in METRIC_ORDER:
        overlay = make_overlay(cur_vis, heat_by_metric[metric_name], alpha=alpha)
        tiles.append(add_title(overlay, METRIC_TITLES[metric_name]))
    tiles.append(add_title(goal_vis, "goal"))
    return np.hstack(tiles)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-hdf5 patch uniqueness confidence heatmap visualization"
    )
    parser.add_argument(
        "--single_hdf5_path",
        type=str,
        required=True,
        help="Path to single-task LIBERO hdf5 (goal source).",
    )
    parser.add_argument(
        "--single_demo_index",
        type=int,
        default=0,
        help="Goal demo index in single_hdf5_path (default: 0)",
    )
    parser.add_argument(
        "--single_goal_frame",
        type=int,
        default=-1,
        help="Goal frame index in single demo (supports negative indexing, default: -1).",
    )
    parser.add_argument(
        "--multi_hdf5_path",
        type=str,
        required=True,
        help="Path to multi-task LIBERO hdf5 (current sequence source).",
    )
    parser.add_argument(
        "--multi_demo_index",
        type=int,
        default=0,
        help="Current demo index in multi_hdf5_path (default: 0)",
    )
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for PNG/MP4")
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/dinov2-with-registers-base",
        help="HuggingFace model name",
    )
    parser.add_argument("--tau", type=float, default=0.1, help="Temperature in S = (x_t @ x_g^T) / tau")
    parser.add_argument("--alpha", type=float, default=0.45, help="Heatmap overlay alpha")
    parser.add_argument(
        "--map_mode",
        type=str,
        default="percentile",
        choices=["percentile", "softmax_i"],
        help="Map confidence vector to 2D heatmap via percentile clip or softmax over i.",
    )
    parser.add_argument(
        "--percentile_low",
        type=float,
        default=5.0,
        help="Lower percentile for map_mode=percentile",
    )
    parser.add_argument(
        "--percentile_high",
        type=float,
        default=95.0,
        help="Upper percentile for map_mode=percentile",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional cap on rendered current frames (for quick experiments).",
    )
    parser.add_argument(
        "--video_name",
        type=str,
        default="uniqueness_single2multi.mp4",
        help="Output video filename",
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
    if args.percentile_high <= args.percentile_low:
        raise ValueError(
            f"percentile_high must be > percentile_low, got "
            f"{args.percentile_high} <= {args.percentile_low}"
        )

    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    multi_frames, multi_demo_key, multi_dataset_fps = load_demo_frames(
        hdf5_path=args.multi_hdf5_path,
        demo_index=args.multi_demo_index,
        cam1_key=args.cam1_key,
        cam2_key=args.cam2_key,
    )
    single_frames, single_demo_key, _ = load_demo_frames(
        hdf5_path=args.single_hdf5_path,
        demo_index=args.single_demo_index,
        cam1_key=args.cam1_key,
        cam2_key=args.cam2_key,
    )

    single_goal_frame = int(args.single_goal_frame)
    if single_goal_frame < 0:
        single_goal_frame = len(single_frames) + single_goal_frame
    if single_goal_frame < 0 or single_goal_frame >= len(single_frames):
        raise IndexError(
            f"single_goal_frame {args.single_goal_frame} out of range for {len(single_frames)} frames"
        )

    num_frames = len(multi_frames)
    if args.max_frames is not None:
        if args.max_frames <= 0:
            raise ValueError(f"max_frames must be > 0, got {args.max_frames}")
        num_frames = min(num_frames, int(args.max_frames))

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModel.from_pretrained(args.model_name).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(args.model_name)

    goal_cam1 = to_uint8_rgb(single_frames[single_goal_frame]["cam1"])
    goal_cam2 = to_uint8_rgb(single_frames[single_goal_frame]["cam2"])

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
    print("[Info] single_hdf5_path:", str(Path(args.single_hdf5_path).expanduser().resolve()))
    print("[Info] single_demo_key(goal):", single_demo_key)
    print("[Info] single_demo_index(goal):", args.single_demo_index)
    print("[Info] single_goal_frame:", single_goal_frame)
    print("[Info] multi_hdf5_path:", str(Path(args.multi_hdf5_path).expanduser().resolve()))
    print("[Info] multi_demo_key(current):", multi_demo_key)
    print("[Info] multi_demo_index(current):", args.multi_demo_index)
    print("[Info] rendered_num_frames(current):", num_frames)
    print("[Info] R:", num_register_tokens)
    print("[Info] d:", dim)
    print("[Info] N:", num_patches)
    print("[Info] patch_grid:", (hp, wp))
    print("[Info] image_size:", input_hw)
    print("[Info] tau:", args.tau)
    print("[Info] map_mode:", args.map_mode)
    print("[Info] metrics:", ", ".join(METRIC_ORDER))

    mp4_fps = int(args.fps) if args.fps is not None else int(multi_dataset_fps)
    mp4_path = out_dir / args.video_name
    writer = imageio.get_writer(str(mp4_path), fps=mp4_fps)
    print("[Info] mp4_fps:", mp4_fps)

    for t in tqdm(range(num_frames), desc="Rendering cross-hdf5 uniqueness heatmaps"):
        cur_cam1 = to_uint8_rgb(multi_frames[t]["cam1"])
        cur_cam2 = to_uint8_rgb(multi_frames[t]["cam2"])

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

        confs_cam1 = compute_row_confidences(cur_patches_batch[0], goal_cam1_patch, tau=args.tau)
        confs_cam2 = compute_row_confidences(cur_patches_batch[1], goal_cam2_patch, tau=args.tau)

        heat_cam1 = {
            name: vector_to_heat(
                confs_cam1[name],
                hp=hp,
                wp=wp,
                out_hw=input_hw,
                map_mode=args.map_mode,
                low_q=args.percentile_low,
                high_q=args.percentile_high,
            )
            for name in METRIC_ORDER
        }
        heat_cam2 = {
            name: vector_to_heat(
                confs_cam2[name],
                hp=hp,
                wp=wp,
                out_hw=input_hw,
                map_mode=args.map_mode,
                low_q=args.percentile_low,
                high_q=args.percentile_high,
            )
            for name in METRIC_ORDER
        }

        # Always flip vertically for visualization consistency with LIBERO camera convention.
        cam1_vis = pixel_values_to_rgb(cur_pixel_values_batch[0], processor=processor)[::-1]
        cam2_vis = pixel_values_to_rgb(cur_pixel_values_batch[1], processor=processor)[::-1]
        for metric_name in METRIC_ORDER:
            heat_cam1[metric_name] = heat_cam1[metric_name][::-1]
            heat_cam2[metric_name] = heat_cam2[metric_name][::-1]

        panel = np.vstack(
            [
                render_camera_row(cam1_vis, goal_cam1_vis, heat_cam1, alpha=args.alpha),
                render_camera_row(cam2_vis, goal_cam2_vis, heat_cam2, alpha=args.alpha),
            ]
        )

        writer.append_data(panel)
        if args.save_png:
            imageio.imwrite(out_dir / f"frame_{t:04d}.png", panel)

    writer.close()

    print(f"[Done] Saved MP4 to: {mp4_path}")
    if args.save_png:
        print(f"[Done] Saved {num_frames} PNG frames to: {out_dir}")


if __name__ == "__main__":
    main()

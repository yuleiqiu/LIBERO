#!/usr/bin/env python3
"""
Compare first-step actions from a trained ACT policy across different init states.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import imageio
from PIL import Image, ImageDraw, ImageFont

from libero.libero.envs import OffScreenRenderEnv
from standalone.configs import DataConfig, apply_policy_config, get_policy_param
from standalone.utils.train_utils import TRAIN_CONFIG_NAME
from standalone.dataset_utils.hdf5_sequence_dataset import HDF5SequenceDataset
from standalone.models.policy.act_policy import ACTPolicy
from standalone.rollout_env import (
    ObsHistory,
    build_obs_key_mapping,
    camera_names_from_mapping,
    extract_env_obs,
    infer_camera_size,
    read_bddl_from_hdf5,
    read_init_states_from_hdf5,
    resolve_bddl_path,
)


def parse_indices(raw: str):
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("no init indices provided")
    indices = []
    for part in parts:
        if not part.isdigit():
            raise ValueError(f"invalid init index: {part}")
        indices.append(int(part))
    return indices


def load_cfg(cfg_path: Path):
    with open(cfg_path, "r") as f:
        raw = json.load(f)
    data_cfg = DataConfig()
    for key, value in (raw.get("data") or {}).items():
        setattr(data_cfg, key, value)
    policy_cfg = raw.get("policy") or {}
    cfg = SimpleNamespace(data=data_cfg, policy=policy_cfg)
    apply_policy_config(cfg)
    return cfg


def resolve_init_states_path(init_states_arg: str, demo_path: Path) -> Path:
    init_path = Path(init_states_arg).expanduser().resolve()
    if init_path.is_file():
        return init_path
    if init_path.is_dir():
        bddl_file_name = read_bddl_from_hdf5(str(demo_path))
        if bddl_file_name is None:
            raise ValueError("bddl_file_name not found in hdf5")
        bddl_path = resolve_bddl_path(bddl_file_name, demo_path)
        if bddl_path is None:
            raise FileNotFoundError(f"bddl file not found: {bddl_file_name}")
        init_states_path = (
            init_path / Path(bddl_path).parent.name / f"{Path(bddl_path).stem}.pruned_init"
        )
        return init_states_path
    raise FileNotFoundError(f"init states path not found: {init_states_arg}")


def load_init_states(init_states_arg: str, demo_path: Path) -> np.ndarray:
    init_states_path = resolve_init_states_path(init_states_arg, demo_path)
    if not init_states_path.exists():
        raise FileNotFoundError(f"init states file not found: {init_states_path}")
    init_states = torch.load(str(init_states_path))
    if torch.is_tensor(init_states):
        init_states = init_states.cpu().numpy()
    return np.asarray(init_states)


def to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3:
        raise ValueError(f"expected image HWC or CHW, got shape {image.shape}")
    if image.shape[-1] in (1, 3):
        img = image
    elif image.shape[0] in (1, 3):
        img = np.transpose(image, (1, 2, 0))
    else:
        raise ValueError(f"cannot infer channels from shape {image.shape}")
    if img.dtype != np.uint8:
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255.0).astype(np.uint8)
    return img


def _multiline_text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    if hasattr(draw, "multiline_textbbox"):
        left, top, right, bottom = draw.multiline_textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    if hasattr(draw, "multiline_textsize"):
        return draw.multiline_textsize(text, font=font)
    lines = text.splitlines() or [""]
    widths = []
    heights = []
    for line in lines:
        if hasattr(draw, "textbbox"):
            left, top, right, bottom = draw.textbbox((0, 0), line, font=font)
            widths.append(right - left)
            heights.append(bottom - top)
        else:
            w, h = draw.textsize(line, font=font)
            widths.append(w)
            heights.append(h)
    return max(widths) if widths else 0, sum(heights) if heights else 0


def overlay_text(frame: np.ndarray, text: str, font: ImageFont.ImageFont) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    text_w, text_h = _multiline_text_size(draw, text, font)
    x = max((image.width - text_w) // 2, 0)
    y = max((image.height - text_h) // 2, 0)
    pad = 4
    rect = [x - pad, y - pad, x + text_w + pad, y + text_h + pad]
    draw.rectangle(rect, fill=(0, 0, 0))
    draw.multiline_text((x, y), text, fill=(255, 255, 255), font=font, align="center")
    return np.asarray(image)


def main():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Compare ACT actions across init states.")
    parser.add_argument("--ckpt", required=True, help="Path to standalone checkpoint (.pt)")
    parser.add_argument("--demo-file", required=True, help="Path to processed *_demo.hdf5")
    parser.add_argument(
        "--config",
        default="",
        help="Path to train_config.json (defaults to ckpt directory)",
    )
    parser.add_argument(
        "--init-states",
        default="",
        help="Optional path to .pruned_init or its root directory (uses HDF5 init states if empty).",
    )
    parser.add_argument(
        "--init-idxs",
        default="0,1",
        help="Comma-separated init state indices to compare (e.g., 0,1)",
    )
    parser.add_argument("--device", default="cuda:0", help="Device to run on")
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Number of zero-action warmup steps before querying the policy",
    )
    parser.add_argument(
        "--video-out",
        default="",
        help="Optional output path for a comparison video (requires exactly 2 init indices).",
    )
    parser.add_argument("--video-steps", type=int, default=60, help="Number of steps to render.")
    parser.add_argument("--video-fps", type=int, default=30, help="Video FPS.")
    parser.add_argument(
        "--video-scale",
        type=int,
        default=1,
        help="Scale factor for video frames (integer).",
    )
    parser.add_argument(
        "--diff-precision",
        type=int,
        default=4,
        help="Decimal precision for action diff overlay.",
    )
    args = parser.parse_args()

    demo_path = Path(args.demo_file).expanduser().resolve()
    if not demo_path.exists():
        raise FileNotFoundError(f"HDF5 not found: {demo_path}")

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    cfg_path = (
        Path(args.config).expanduser().resolve()
        if args.config
        else ckpt_path.parent / TRAIN_CONFIG_NAME
    )
    if not cfg_path.exists():
        raise FileNotFoundError(f"config not found: {cfg_path}")
    cfg = load_cfg(cfg_path)

    obs_keys = [k.strip() for k in cfg.data.obs_keys.split(",") if k.strip()]
    image_keys = [k.strip() for k in cfg.data.image_keys.split(",") if k.strip()]
    all_keys = obs_keys + image_keys
    policy_name = getattr(cfg.policy, "name", "act").lower()

    dataset = HDF5SequenceDataset(
        hdf5_path=str(demo_path),
        obs_keys=all_keys,
        obs_horizon=cfg.data.obs_horizon,
        predict_horizon=cfg.data.predict_horizon,
    )
    sample = dataset[0]
    action_dim = sample["actions"].shape[-1]
    qpos_dim = sum(np.prod(sample["obs"][k].shape[1:]) for k in obs_keys)
    image_shapes = {k: sample["obs"][k].shape[1:] for k in image_keys}

    exec_horizon = get_policy_param(cfg, "exec_horizon")
    model = ACTPolicy(
        obs_keys=obs_keys,
        image_keys=image_keys,
        obs_horizon=cfg.data.obs_horizon,
        predict_horizon=cfg.data.predict_horizon,
        exec_horizon=exec_horizon,
        qpos_dim=qpos_dim,
        action_dim=action_dim,
        model_type=policy_name,
        act_config=get_policy_param(cfg, "act_config"),
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)

    device = args.device if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    bddl_file_name = read_bddl_from_hdf5(str(demo_path))
    if bddl_file_name is None:
        raise ValueError("bddl_file_name not found in hdf5")
    bddl_path = resolve_bddl_path(bddl_file_name, demo_path)
    if bddl_path is None:
        raise FileNotFoundError(f"bddl file not found: {bddl_file_name}")

    obs_key_mapping = build_obs_key_mapping(cfg, obs_keys, image_keys)
    camera_names = camera_names_from_mapping(image_keys, obs_key_mapping) if image_keys else []
    cam_hw = infer_camera_size(image_shapes) if image_keys else None

    env_args = {"bddl_file_name": bddl_path}
    if image_keys:
        if cam_hw is None:
            raise ValueError("image_keys provided but camera size could not be inferred")
        camera_h, camera_w = cam_hw
        env_args.update(
            {
                "use_camera_obs": True,
                "camera_names": camera_names,
                "camera_heights": camera_h,
                "camera_widths": camera_w,
            }
        )
    else:
        env_args["use_camera_obs"] = False

    init_states = read_init_states_from_hdf5(str(demo_path))
    if args.init_states:
        init_states = load_init_states(args.init_states, demo_path)
    init_idxs = parse_indices(args.init_idxs)
    for idx in init_idxs:
        if idx < 0 or idx >= init_states.shape[0]:
            raise ValueError(f"init_idx out of range: {idx} (0..{init_states.shape[0]-1})")

    if args.video_out and len(init_idxs) != 2:
        raise ValueError("--video-out requires exactly 2 init indices.")

    if args.video_out:
        envs = [OffScreenRenderEnv(**env_args) for _ in range(2)]
        histories = [
            ObsHistory(obs_keys + image_keys, cfg.data.obs_horizon) for _ in range(2)
        ]
        dummy_action = np.zeros((action_dim,), dtype=np.float32)
        for i, init_idx in enumerate(init_idxs):
            model.reset()
            histories[i].reset()
            envs[i].reset()
            env_obs = envs[i].set_init_state(init_states[init_idx])
            obs = extract_env_obs(env_obs, obs_keys, image_keys, obs_key_mapping)
            histories[i].add(obs)
            for _ in range(int(args.warmup_steps)):
                env_obs, _, _, _ = envs[i].step(dummy_action)
                obs = extract_env_obs(env_obs, obs_keys, image_keys, obs_key_mapping)
                histories[i].add(obs)

        video_path = Path(args.video_out).expanduser().resolve()
        video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(video_path), fps=int(args.video_fps))
        font = ImageFont.load_default()

        for _ in range(int(args.video_steps)):
            actions = []
            rows = []
            for i in range(2):
                obs_input = histories[i].stack()
                action = model.get_action(obs_input)
                if torch.is_tensor(action):
                    action = action.detach().cpu().numpy()
                action = np.asarray(action).reshape(-1)
                actions.append(action)

                env_obs, _, _, _ = envs[i].step(action)
                obs = extract_env_obs(env_obs, obs_keys, image_keys, obs_key_mapping)
                histories[i].add(obs)

                if len(image_keys) < 2:
                    raise ValueError("video rendering expects at least 2 image keys.")
                img_left = to_rgb_uint8(obs[image_keys[0]])[::-1]
                img_right = to_rgb_uint8(obs[image_keys[1]])[::-1]
                row = np.hstack([img_left, img_right])
                rows.append(row)

            diff = np.abs(actions[0] - actions[1])
            precision = int(args.diff_precision)
            diff_str = np.array2string(diff, precision=precision, separator=", ")
            l2 = float(np.linalg.norm(diff))
            max_diff = float(np.max(diff))
            text = f"abs diff (l2={l2:.{precision}f}, max={max_diff:.{precision}f})\n{diff_str}"

            frame = np.vstack(rows)
            if args.video_scale > 1:
                frame = np.repeat(frame, args.video_scale, axis=0)
                frame = np.repeat(frame, args.video_scale, axis=1)
            frame = overlay_text(frame, text, font)
            writer.append_data(frame)

        writer.close()
        for env in envs:
            env.close()
        print(f"[info] video saved to {video_path}")
        return

    env = OffScreenRenderEnv(**env_args)
    dummy_action = np.zeros((action_dim,), dtype=np.float32)

    actions = []
    for init_idx in init_idxs:
        model.reset()
        history = ObsHistory(obs_keys + image_keys, cfg.data.obs_horizon)

        env.reset()
        env_obs = env.set_init_state(init_states[init_idx])
        obs = extract_env_obs(env_obs, obs_keys, image_keys, obs_key_mapping)
        history.add(obs)

        for _ in range(int(args.warmup_steps)):
            env_obs, _, _, _ = env.step(dummy_action)
            obs = extract_env_obs(env_obs, obs_keys, image_keys, obs_key_mapping)
            history.add(obs)

        obs_input = history.stack()
        action = model.get_action(obs_input)
        if torch.is_tensor(action):
            action = action.detach().cpu().numpy()
        action = np.asarray(action).reshape(-1)
        actions.append(action)
        print(f"init_idx {init_idx}: action {np.array2string(action, precision=6)}")

    if len(actions) >= 2:
        diff = np.abs(actions[0] - actions[1])
        print(f"abs diff (first two): {np.array2string(diff, precision=6)}")

    env.close()


if __name__ == "__main__":
    main()

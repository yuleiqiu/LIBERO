#!/usr/bin/env python3
"""
Compare first-step actions from a trained ACT policy across different init states.
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

try:
    import yaml
except ImportError as exc:
    raise ImportError("pyyaml is required; install with `pip install pyyaml`.") from exc

from libero.libero.envs import OffScreenRenderEnv
from standalone.configs import DataConfig, apply_policy_config, get_policy_param
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
        raw = yaml.safe_load(f) or {}
    data_cfg = DataConfig()
    for key, value in (raw.get("data") or {}).items():
        setattr(data_cfg, key, value)
    policy_cfg = raw.get("policy") or {}
    cfg = SimpleNamespace(data=data_cfg, policy=policy_cfg)
    apply_policy_config(cfg)
    return cfg


def main():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Compare ACT actions across init states.")
    parser.add_argument("--ckpt", required=True, help="Path to standalone checkpoint (.pt)")
    parser.add_argument("--demo-file", required=True, help="Path to processed *_demo.hdf5")
    parser.add_argument(
        "--config",
        default=str(repo_root / "standalone/configs/train_act.yaml"),
        help="Training config used to build the model",
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
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"config not found: {cfg_path}")
    cfg = load_cfg(cfg_path)

    demo_path = Path(args.demo_file).expanduser().resolve()
    if not demo_path.exists():
        raise FileNotFoundError(f"HDF5 not found: {demo_path}")

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

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
    init_idxs = parse_indices(args.init_idxs)
    for idx in init_idxs:
        if idx < 0 or idx >= init_states.shape[0]:
            raise ValueError(f"init_idx out of range: {idx} (0..{init_states.shape[0]-1})")

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

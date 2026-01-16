#!/usr/bin/env python3
"""
Evaluate action prediction fit on a demo dataset without rollout.
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

from torch.utils.data import DataLoader

from standalone.configs import DataConfig, apply_policy_config, get_policy_param
from standalone.dataset_utils.hdf5_sequence_dataset import HDF5SequenceDataset
from standalone.models.policy.act_policy import ACTPolicy


def load_cfg(cfg_path: Path):
    with open(cfg_path, "r") as f:
        raw = yaml.safe_load(f) or {}
    data_cfg = DataConfig()
    for key, value in (raw.get("data") or {}).items():
        setattr(data_cfg, key, value)
    policy_cfg = raw.get("policy") or {}
    cfg = SimpleNamespace(data=data_cfg, policy=policy_cfg, batch_size=raw.get("batch_size", 32))
    apply_policy_config(cfg)
    return cfg


def masked_stats(diff, mask):
    if mask is None:
        valid = diff
        denom = diff.shape[0]
    else:
        valid = diff[mask]
        denom = max(mask.sum(), 1)
    mse = float((valid ** 2).sum() / denom)
    l1 = float(np.abs(valid).sum() / denom)
    return mse, l1


def main():
    parser = argparse.ArgumentParser(description="Check action fit on a demo dataset.")
    parser.add_argument("--ckpt", required=True, help="Path to standalone checkpoint (.pt)")
    parser.add_argument("--demo-file", required=True, help="Path to processed *_demo.hdf5")
    parser.add_argument(
        "--config",
        default="standalone/configs/train_cnnmlp_quickcheck.yaml",
        help="Training config used to build the model",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"config not found: {cfg_path}")
    cfg = load_cfg(cfg_path)
    if args.batch_size:
        cfg.batch_size = args.batch_size

    demo_path = Path(args.demo_file).expanduser().resolve()
    if not demo_path.exists():
        raise FileNotFoundError(f"HDF5 not found: {demo_path}")

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    obs_keys = [k.strip() for k in cfg.data.obs_keys.split(",") if k.strip()]
    image_keys = [k.strip() for k in cfg.data.image_keys.split(",") if k.strip()]
    all_keys = obs_keys + image_keys
    policy_name = getattr(cfg.policy, "name", "mlp").lower()

    dataset = HDF5SequenceDataset(
        hdf5_path=str(demo_path),
        obs_keys=all_keys,
        obs_horizon=cfg.data.obs_horizon,
        predict_horizon=cfg.data.predict_horizon,
        action_shift=getattr(cfg.data, "action_shift", 0),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    sample = dataset[0]
    action_dim = sample["actions"].shape[-1]
    qpos_dim = sum(np.prod(sample["obs"][k].shape[1:]) for k in obs_keys)

    exec_horizon = get_policy_param(cfg, "exec_horizon")
    if policy_name not in ("act", "cnnmlp"):
        raise ValueError(f"unsupported policy: {policy_name}")
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

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # Baseline: constant mean action from dataset
    all_actions = []
    for item in dataset:
        all_actions.append(item["actions"])
    mean_action = np.mean(np.concatenate(all_actions, axis=0), axis=0, keepdims=True)

    sum_mse = 0.0
    sum_l1 = 0.0
    sum_mse_base = 0.0
    sum_l1_base = 0.0
    count = 0
    per_dim_sq = None
    per_dim_abs = None
    per_dim_sq_base = None
    per_dim_abs_base = None

    with torch.no_grad():
        for batch in loader:
            for key in batch["obs"]:
                batch["obs"][key] = batch["obs"][key].to(device)
            actions = batch["actions"].to(device)
            action_mask = batch.get("action_mask")
            if action_mask is not None:
                action_mask = action_mask.to(device)

            pred = model.forward(batch["obs"])
            if pred.ndim == 2:
                pred = pred.unsqueeze(1)
            pred = pred[:, : actions.shape[1]]

            diff = pred - actions
            if action_mask is not None:
                mask = action_mask[:, : actions.shape[1]].unsqueeze(-1) > 0
                diff_np = diff[mask].detach().cpu().numpy()
                gt_np = actions[mask].detach().cpu().numpy()
                pred_np = pred[mask].detach().cpu().numpy()
            else:
                diff_np = diff.detach().cpu().numpy().reshape(-1, action_dim)
                gt_np = actions.detach().cpu().numpy().reshape(-1, action_dim)
                pred_np = pred.detach().cpu().numpy().reshape(-1, action_dim)

            mse = float(np.mean(diff_np ** 2))
            l1 = float(np.mean(np.abs(diff_np)))
            sum_mse += mse
            sum_l1 += l1

            base_diff = mean_action - gt_np
            mse_base = float(np.mean(base_diff ** 2))
            l1_base = float(np.mean(np.abs(base_diff)))
            sum_mse_base += mse_base
            sum_l1_base += l1_base
            count += 1

            if per_dim_sq is None:
                per_dim_sq = np.zeros(action_dim, dtype=np.float64)
                per_dim_abs = np.zeros(action_dim, dtype=np.float64)
                per_dim_sq_base = np.zeros(action_dim, dtype=np.float64)
                per_dim_abs_base = np.zeros(action_dim, dtype=np.float64)

            per_dim_sq += np.mean((pred_np - gt_np) ** 2, axis=0)
            per_dim_abs += np.mean(np.abs(pred_np - gt_np), axis=0)
            per_dim_sq_base += np.mean(base_diff ** 2, axis=0)
            per_dim_abs_base += np.mean(np.abs(base_diff), axis=0)

    avg_mse = sum_mse / max(count, 1)
    avg_l1 = sum_l1 / max(count, 1)
    avg_mse_base = sum_mse_base / max(count, 1)
    avg_l1_base = sum_l1_base / max(count, 1)
    per_dim_sq = per_dim_sq / max(count, 1)
    per_dim_abs = per_dim_abs / max(count, 1)
    per_dim_sq_base = per_dim_sq_base / max(count, 1)
    per_dim_abs_base = per_dim_abs_base / max(count, 1)

    print("[fit] avg mse:", avg_mse)
    print("[fit] avg l1 :", avg_l1)
    print("[base] avg mse:", avg_mse_base)
    print("[base] avg l1 :", avg_l1_base)
    print("[fit] per-dim mse:", np.array2string(per_dim_sq, precision=6))
    print("[fit] per-dim l1 :", np.array2string(per_dim_abs, precision=6))
    print("[base] per-dim mse:", np.array2string(per_dim_sq_base, precision=6))
    print("[base] per-dim l1 :", np.array2string(per_dim_abs_base, precision=6))


if __name__ == "__main__":
    main()

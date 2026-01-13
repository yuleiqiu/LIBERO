import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

try:
    import draccus
except ImportError as exc:
    raise ImportError("draccus is required; install with `pip install draccus`.") from exc

from standalone.configs import EvalConfig, apply_policy_config, get_policy_param
from standalone.dataset_utils.hdf5_sequence_dataset import (
    HDF5SequenceDataset,
    load_obs_stats,
)
from standalone.models.policy.act_policy import ACTPolicy


def build_splits(dataset_len, train_ratio, val_ratio, seed):
    assert train_ratio + val_ratio <= 1.0 + 1e-8
    train_size = int(dataset_len * train_ratio)
    val_size = int(dataset_len * val_ratio)
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_len, generator=g).tolist()
    train_idx = indices[:train_size]
    val_idx = indices[train_size : train_size + val_size]
    eval_idx = indices[train_size + val_size :]
    return train_idx, val_idx, eval_idx


def select_eval_indices(split_dict):
    for key in ("val", "eval", "train"):
        if key in split_dict and len(split_dict[key]) > 0:
            return key, split_dict[key]
    return "train", split_dict.get("train", [])


@draccus.wrap()
def main(cfg: EvalConfig):
    apply_policy_config(cfg)
    if not cfg.data.demo_file:
        raise ValueError("data.demo_file is required")
    if not cfg.ckpt:
        raise ValueError("ckpt is required")
    obs_keys = [k.strip() for k in cfg.data.obs_keys.split(",") if k.strip()]
    image_keys = [k.strip() for k in cfg.data.image_keys.split(",") if k.strip()]
    all_keys = obs_keys + image_keys
    policy_name = getattr(cfg.policy, "name", "mlp").lower()

    demo_path = Path(cfg.data.demo_file).expanduser().resolve()
    if not demo_path.exists():
        raise FileNotFoundError(f"HDF5 not found: {demo_path}")

    ckpt_path = Path(cfg.ckpt).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = cfg.device if torch.cuda.is_available() else "cpu"

    dataset = HDF5SequenceDataset(
        hdf5_path=str(demo_path),
        obs_keys=all_keys,
        obs_horizon=cfg.data.obs_horizon,
        predict_horizon=cfg.data.predict_horizon,
    )

    split_key = None
    if cfg.split_path:
        split_path = Path(cfg.split_path).expanduser().resolve()
        if split_path.exists():
            with open(split_path, "r") as f:
                split_dict = json.load(f)
            split_key, eval_idx = select_eval_indices(split_dict)
        else:
            raise FileNotFoundError(f"split path not found: {split_path}")
    else:
        train_idx, val_idx, eval_idx = build_splits(
            len(dataset), cfg.data.train_ratio, cfg.data.val_ratio, cfg.data.seed
        )
        split_key = "val" if len(val_idx) > 0 else "eval"
        if len(eval_idx) == 0:
            split_key = "train"
            eval_idx = train_idx
        else:
            eval_idx = val_idx if split_key == "val" else eval_idx

    ckpt = torch.load(ckpt_path, map_location="cpu")
    obs_stats = None
    if policy_name not in ("act", "cnnmlp"):
        if cfg.data.obs_stats_path:
            obs_stats = load_obs_stats(cfg.data.obs_stats_path)
        elif isinstance(ckpt, dict) and ckpt.get("obs_stats") is not None:
            obs_stats = ckpt["obs_stats"]
        if obs_stats is not None and image_keys:
            for key in image_keys:
                obs_stats.pop(key, None)
        if obs_stats is not None:
            dataset.set_obs_stats(obs_stats)

    eval_dataset = Subset(dataset, eval_idx)
    loader = DataLoader(
        eval_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    sample = dataset[eval_idx[0]]
    action_dim = sample["actions"].shape[-1]
    exec_horizon = get_policy_param(cfg, "exec_horizon")
    if policy_name not in ("act", "cnnmlp"):
        raise ValueError(f"unsupported policy: {policy_name}")
    qpos_dim = sum(np.prod(sample["obs"][k].shape[1:]) for k in obs_keys)
    for key in image_keys:
        if key not in sample["obs"]:
            raise KeyError(f"image key not found in obs: {key}")
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
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    losses = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"eval on {split_key} split"):
            for key in batch["obs"]:
                batch["obs"][key] = batch["obs"][key].to(device)
            batch["actions"] = batch["actions"].to(device)
            if "action_mask" in batch:
                batch["action_mask"] = batch["action_mask"].to(device)
            losses.append(model.compute_loss(batch).item())

    avg_loss = sum(losses) / max(len(losses), 1)
    print(f"[info] eval split={split_key} | loss={avg_loss:.6f}")


if __name__ == "__main__":
    main()

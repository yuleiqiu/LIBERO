import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, Subset
from tqdm import tqdm

try:
    import draccus
except ImportError as exc:
    raise ImportError("draccus is required; install with `pip install draccus`.") from exc

from standalone.configs import TrainConfig
from standalone.dataset_utils.hdf5_sequence_dataset import (
    HDF5SequenceDataset,
    compute_obs_stats,
    load_obs_stats,
    save_obs_stats,
)
from standalone.models.policy.mlp_policy import MLPPolicy


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


@draccus.wrap()
def main(cfg: TrainConfig):
    if not cfg.data.demo_file:
        raise ValueError("data.demo_file is required")
    set_seed(cfg.data.seed)

    demo_path = Path(cfg.data.demo_file).expanduser().resolve()
    if not demo_path.exists():
        raise FileNotFoundError(f"HDF5 not found: {demo_path}")

    obs_keys = [k.strip() for k in cfg.data.obs_keys.split(",") if k.strip()]
    image_keys = [k.strip() for k in cfg.data.image_keys.split(",") if k.strip()]
    all_keys = obs_keys + image_keys

    save_dir = Path(cfg.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    split_path = save_dir / "split_indices.json"

    device = cfg.device if torch.cuda.is_available() else "cpu"

    base_dataset = HDF5SequenceDataset(
        hdf5_path=str(demo_path),
        obs_keys=all_keys,
        obs_horizon=cfg.data.obs_horizon,
        predict_horizon=cfg.data.predict_horizon,
    )

    train_idx, val_idx, eval_idx = build_splits(
        len(base_dataset), cfg.data.train_ratio, cfg.data.val_ratio, cfg.data.seed
    )
    with open(split_path, "w") as f:
        json.dump({"train": train_idx, "val": val_idx, "eval": eval_idx}, f, indent=2)

    obs_stats = None
    if cfg.data.normalize_obs:
        stats_path = (
            Path(cfg.data.obs_stats_path).expanduser().resolve()
            if cfg.data.obs_stats_path
            else save_dir / "obs_stats.json"
        )
        if stats_path.exists():
            obs_stats = load_obs_stats(stats_path)
        else:
            obs_stats = compute_obs_stats(
                base_dataset, train_idx, ignore_keys=image_keys
            )
            save_obs_stats(stats_path, obs_stats)
        if obs_stats is not None and image_keys:
            for key in image_keys:
                obs_stats.pop(key, None)
        base_dataset.set_obs_stats(obs_stats)

    train_dataset = Subset(base_dataset, train_idx)
    val_dataset = Subset(base_dataset, val_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=RandomSampler(train_dataset),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = None
    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    sample = base_dataset[train_idx[0]]
    obs_dim = sum(np.prod(sample["obs"][k].shape) for k in obs_keys)
    image_shapes = {}
    for key in image_keys:
        if key not in sample["obs"]:
            raise KeyError(f"image key not found in obs: {key}")
        image_shapes[key] = sample["obs"][key].shape[1:]
        obs_dim += sample["obs"][key].shape[0] * cfg.model.image_embed_dim
    act_shape = sample["actions"].shape
    action_dim = act_shape[-1]
    model = MLPPolicy(
        input_dim=obs_dim,
        action_dim=action_dim,
        predict_horizon=cfg.data.predict_horizon,
        exec_horizon=cfg.model.exec_horizon,
        hidden_dims=cfg.model.hidden_dims,
        action_squash=cfg.model.action_squash,
        obs_keys=obs_keys,
        image_keys=image_keys,
        image_shapes=image_shapes,
        image_embed_dim=cfg.model.image_embed_dim,
        image_encoder_pretrained=cfg.model.image_encoder_pretrained,
        image_encoder_remove_layer_num=cfg.model.image_encoder_remove_layer_num,
        image_encoder_no_stride=cfg.model.image_encoder_no_stride,
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    printed_batch = False
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"train epoch {epoch}"):
            if not printed_batch:
                print(
                    "[debug] batch structure:",
                    {
                        k: (list(v.keys()) if isinstance(v, dict) else type(v))
                        for k, v in batch.items()
                    },
                )
                print(
                    "[debug] obs shapes:",
                    {
                        k: f"{tuple(v.shape)} # (batch, obs_horizon, obs_dim...)"
                        for k, v in batch["obs"].items()
                    },
                )
                print(
                    "[debug] actions shape:",
                    f"{tuple(batch['actions'].shape)} # (batch, predict_horizon, action_dim)",
                )
                if "action_mask" in batch:
                    print(
                        "[debug] action_mask shape:",
                        f"{tuple(batch['action_mask'].shape)} # (batch, predict_horizon)",
                    )
                printed_batch = True
            for key in batch["obs"]:
                batch["obs"][key] = batch["obs"][key].to(device)
            batch["actions"] = batch["actions"].to(device)
            if "action_mask" in batch:
                batch["action_mask"] = batch["action_mask"].to(device)
            loss = model.compute_loss(batch)

            optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            train_losses.append(loss.item())

        avg_train = sum(train_losses) / max(len(train_losses), 1)
        print(f"[info] epoch {epoch:03d} | train loss {avg_train:.6f}")

        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    for key in batch["obs"]:
                        batch["obs"][key] = batch["obs"][key].to(device)
                    batch["actions"] = batch["actions"].to(device)
                    if "action_mask" in batch:
                        batch["action_mask"] = batch["action_mask"].to(device)
                    val_losses.append(model.compute_loss(batch).item())
            avg_val = sum(val_losses) / max(len(val_losses), 1)
            print(f"[info] epoch {epoch:03d} | val loss {avg_val:.6f}")

        ckpt_path = save_dir / "model_last.pt"
        torch.save({"model": model.state_dict(), "obs_stats": obs_stats}, ckpt_path)

    print(f"[info] finished training. ckpt saved to {ckpt_path}")


if __name__ == "__main__":
    main()

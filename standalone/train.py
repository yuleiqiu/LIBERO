import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, Subset
from tqdm import tqdm

from standalone.dataset_utils.hdf5_sequence_dataset import (
    HDF5SequenceDataset,
    compute_obs_stats,
    load_obs_stats,
    save_obs_stats,
)
from standalone.models.mlp_policy import MLPPolicy


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone MLP training")
    parser.add_argument("--demo-file", required=True, help="Path to *_demo.hdf5")
    parser.add_argument(
        "--obs-keys",
        default="gripper_states,joint_states",
        help="Comma-separated obs keys to use",
    )
    parser.add_argument("--obs-horizon", type=int, default=1)
    parser.add_argument("--predict-horizon", type=int, default=1)
    parser.add_argument("--exec-horizon", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--normalize-obs", action="store_true")
    parser.add_argument("--obs-stats-path", type=str, default=None)
    parser.add_argument(
        "--save-dir", type=str, default="standalone/standalone_runs/run_001"
    )
    parser.add_argument("--grad-clip", type=float, default=None)
    return parser.parse_args()


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


def main():
    args = parse_args()
    set_seed(args.seed)

    demo_path = Path(args.demo_file).expanduser().resolve()
    if not demo_path.exists():
        raise FileNotFoundError(f"HDF5 not found: {demo_path}")

    obs_keys = [k.strip() for k in args.obs_keys.split(",") if k.strip()]

    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    split_path = save_dir / "split_indices.json"

    device = args.device if torch.cuda.is_available() else "cpu"

    base_dataset = HDF5SequenceDataset(
        hdf5_path=str(demo_path),
        obs_keys=obs_keys,
        obs_horizon=args.obs_horizon,
        predict_horizon=args.predict_horizon,
    )

    train_idx, val_idx, eval_idx = build_splits(
        len(base_dataset), args.train_ratio, args.val_ratio, args.seed
    )
    with open(split_path, "w") as f:
        json.dump({"train": train_idx, "val": val_idx, "eval": eval_idx}, f, indent=2)

    obs_stats = None
    if args.normalize_obs:
        stats_path = (
            Path(args.obs_stats_path).expanduser().resolve()
            if args.obs_stats_path
            else save_dir / "obs_stats.json"
        )
        if stats_path.exists():
            obs_stats = load_obs_stats(stats_path)
        else:
            obs_stats = compute_obs_stats(base_dataset, train_idx)
            save_obs_stats(stats_path, obs_stats)
        base_dataset.set_obs_stats(obs_stats)

    train_dataset = Subset(base_dataset, train_idx)
    val_dataset = Subset(base_dataset, val_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=RandomSampler(train_dataset),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = None
    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    sample = base_dataset[train_idx[0]]
    obs_dim = sum(np.prod(sample["obs"][k].shape) for k in obs_keys)
    act_shape = sample["actions"].shape
    action_dim = act_shape[-1]
    model = MLPPolicy(
        input_dim=obs_dim,
        action_dim=action_dim,
        predict_horizon=args.predict_horizon,
        exec_horizon=args.exec_horizon,
        obs_keys=obs_keys,
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    printed_batch = False
    for epoch in range(1, args.epochs + 1):
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
            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
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

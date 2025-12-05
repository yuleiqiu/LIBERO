import argparse
import json
import os
from pathlib import Path

import h5py
import torch
import yaml
import wandb
from easydict import EasyDict
from hydra import compose, initialize_config_dir
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, RandomSampler, Subset
from tqdm import tqdm

import init_path  # noqa: F401
from libero.libero import get_libero_path
from libero.lifelong.algos import get_algo_class
from libero.lifelong.datasets import SequenceVLDataset, get_dataset
from libero.lifelong.utils import control_seed, create_experiment_dir, get_task_embs, safe_device, torch_save_model


def load_cfg(overrides):
    """
    Load the default Hydra config with optional overrides (e.g., ["policy=bc_rnn_policy"]).
    """
    config_dir = to_absolute_path("libero/configs")
    with initialize_config_dir(config_dir=config_dir, job_name="single_task"):
        hydra_cfg = compose(config_name="config", overrides=overrides)
    cfg = EasyDict(yaml.safe_load(OmegaConf.to_yaml(hydra_cfg)))
    return cfg


def read_language_from_hdf5(hdf5_path: str) -> str:
    with h5py.File(hdf5_path, "r") as f:
        problem_info = json.loads(f["data"].attrs["problem_info"])
    return problem_info["language_instruction"]


def train_single_task(cfg: EasyDict,
                      dataset: SequenceVLDataset,
                      task_emb: torch.Tensor,
                      save_dir: str,
                      train_ratio: float,
                      val_ratio: float,
                      ckpt_mode: str,
                      ckpt_interval: int
                    ) -> None:
    """
    Minimal single-task training loop (no benchmark dependency).
    """
    algo_cls = get_algo_class(cfg.lifelong.algo)
    algo = safe_device(algo_cls(n_tasks=1, cfg=cfg), cfg.device)
    algo.start_task(0)

    # Split dataset into train / val / eval (eval is the leftover if ratios don't sum to 1)
    total_len = len(dataset)
    assert train_ratio + val_ratio <= 1 + 1e-8, "train_ratio + val_ratio must be <= 1"
    train_size = int(total_len * train_ratio)
    val_size = int(total_len * val_ratio)
    overflow = max(train_size + val_size - total_len, 0)
    if overflow > 0:
        if val_size >= overflow:
            val_size -= overflow
        else:
            train_size = max(train_size - (overflow - val_size), 0)
            val_size = 0

    g = torch.Generator().manual_seed(cfg.seed)
    indices = torch.randperm(total_len, generator=g).tolist()
    train_idx = indices[:train_size]
    val_idx = indices[train_size : train_size + val_size]
    eval_idx = indices[train_size + val_size :]

    split_path = os.path.join(save_dir, "split_indices.json")
    with open(split_path, "w") as f:
        json.dump({"train": train_idx, "val": val_idx, "eval": eval_idx}, f, indent=2)
    print(
        f"[info] dataset split saved to {split_path} "
        f"(train {len(train_idx)}, val {len(val_idx)}, eval {len(eval_idx)})"
    )
    if cfg.use_wandb:
        wandb.run.summary.update(
            {
                "split/train_size": len(train_idx),
                "split/val_size": len(val_idx),
                "split/eval_size": len(eval_idx),
            }
        )

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        sampler=RandomSampler(train_dataset),
        persistent_workers=True,
    )
    val_loader = None
    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.train.batch_size,
            num_workers=0,
            shuffle=False,
            persistent_workers=False,
        )

    best_loss = float("inf")
    model_best_path = os.path.join(save_dir, "model_best.pth")
    model_last_path = os.path.join(save_dir, "model_last.pth")

    n_epochs = cfg.train.n_epochs
    val_every = getattr(cfg.train, "val_every", 5)

    # Zero-shot evaluation on validation set (epoch 0)
    algo.policy.eval()
    val_losses = []
    if val_loader is not None:
        with torch.no_grad():
            for data in tqdm(val_loader, desc="val epoch 0", leave=True):
                loss = algo.eval_observe(data)
                val_losses.append(loss)
    val_avg = sum(val_losses) / max(len(val_losses), 1) if val_losses else float("inf")
    best_loss = val_avg
    if val_loader is not None:
        if ckpt_mode == "best":
            torch_save_model(algo.policy, model_best_path, cfg=cfg)
            print(f"[info] epoch 000 | val avg loss {val_avg:.4f} | saved checkpoint (best)")
        else:
            print(f"[info] epoch 000 | val avg loss {val_avg:.4f}")
    else:
        print("[info] epoch 000 | no val split; skip zero-shot val")
    last_interval_ckpt = None

    if cfg.use_wandb and val_loader is not None:
        wandb.log({"val_loss": val_avg, "epoch": 0})

    for epoch in range(1, n_epochs + 1):
        algo.policy.train()
        train_losses = []
        for data in tqdm(train_loader, desc=f"train epoch {epoch}", leave=True):
            loss = algo.observe(data)
            train_losses.append(loss)

        train_avg = sum(train_losses) / max(len(train_losses), 1)
        print(f"[info] epoch {epoch:03d} | train avg loss {train_avg:.4f}")
        if cfg.use_wandb:
            wandb.log({"train_loss": train_avg, "epoch": epoch})

        if epoch % val_every == 0 or epoch == n_epochs:
            algo.policy.eval()
            val_losses = []
            with torch.no_grad():
                for data in tqdm(val_loader, desc=f"val epoch {epoch}", leave=True):
                    loss = algo.eval_observe(data)
                    val_losses.append(loss)
            val_avg = sum(val_losses) / max(len(val_losses), 1)
            print(
                f"[info] epoch {epoch:03d} | val avg loss {val_avg:.4f} "
                f"| train avg loss {train_avg:.4f}"
            )
            if cfg.use_wandb:
                wandb.log({"val_loss": val_avg, "train_loss": train_avg, "epoch": epoch})

            if val_avg < best_loss:
                best_loss = val_avg
                if ckpt_mode == "best":
                    torch_save_model(algo.policy, model_best_path, cfg=cfg)
                    print(f"[info] saved best checkpoint to {model_best_path}")

            if ckpt_mode == "interval":
                if epoch % ckpt_interval == 0 or epoch == n_epochs:
                    ckpt_path = os.path.join(save_dir, f"model_epoch_{epoch:03d}.pth")
                    torch_save_model(algo.policy, ckpt_path, cfg=cfg)
                    last_interval_ckpt = ckpt_path
                    print(f"[info] saved interval checkpoint to {ckpt_path}")

    if ckpt_mode == "last":
        torch_save_model(algo.policy, model_last_path, cfg=cfg)
        print(f"[info] saved last checkpoint to {model_last_path}")

    final_ckpt = None
    if ckpt_mode == "best":
        final_ckpt = model_best_path
        ckpt_label = "best_ckpt"
    elif ckpt_mode == "last":
        final_ckpt = model_last_path
        ckpt_label = "last_ckpt"
    elif ckpt_mode == "interval":
        final_ckpt = last_interval_ckpt
        ckpt_label = "interval_ckpt"

    print(
        f"[info] finished training. best val loss={best_loss:.4f}, "
        f"{ckpt_label}={(final_ckpt or 'N/A')}"
    )
    if cfg.use_wandb:
        wandb.run.summary["best_val_loss"] = best_loss
        if final_ckpt:
            wandb.run.summary["ckpt_path"] = final_ckpt


def main():
    parser = argparse.ArgumentParser(description="Single-task training directly from HDF5")
    parser.add_argument("--demo-file", required=True, help="Path to processed *_demo.hdf5")
    parser.add_argument(
        "--config-override",
        nargs="*",
        default=[],
        help="Hydra-style overrides, e.g., policy=bc_rnn_policy lifelong=er train.n_epochs=20",
    )
    parser.add_argument(
        "--ckpt-mode",
        choices=["best", "last", "interval"],
        default="best",
        help="Checkpoint saving strategy",
    )
    parser.add_argument(
        "--ckpt-interval",
        type=int,
        default=10,
        help="Checkpoint save interval for 'interval' mode; if epochs < interval, save final",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="CUDA device id (e.g., 0 -> cuda:0)",
    )
    args = parser.parse_args()

    cfg = load_cfg(args.config_override)
    cfg.device = f"cuda:{args.device_id}"
    control_seed(cfg.seed)

    demo_path = Path(args.demo_file).expanduser().resolve()
    if not demo_path.exists():
        raise FileNotFoundError(f"HDF5 not found: {demo_path}")

    # Use demo parent as dataset folder so robomimic obs utils find assets if needed.
    cfg.folder = cfg.folder or str(demo_path.parent)
    cfg.bddl_folder = cfg.bddl_folder or get_libero_path("bddl_files")
    cfg.init_states_folder = cfg.init_states_folder or get_libero_path("init_states")

    # Build dataset and shape meta from the single HDF5.
    base_dataset, shape_meta = get_dataset(
        dataset_path=str(demo_path),
        obs_modality=cfg.data.obs.modality,
        initialize_obs_utils=True,
        seq_len=cfg.data.seq_len,
    )
    language_instruction = read_language_from_hdf5(str(demo_path))

    # 构造一个与语言编码维度匹配的零向量，关闭语言条件
    lang_dim = cfg.policy.language_encoder.network_kwargs.get("input_size", 1)
    task_emb = torch.zeros((lang_dim,), dtype=torch.float32)
    dataset = SequenceVLDataset(base_dataset, task_emb)

    # Prepare experiment directory and attach shape meta.
    cfg.shape_meta = shape_meta
    create_experiment_dir(cfg)
    os.makedirs(cfg.experiment_dir, exist_ok=True)
    if cfg.use_wandb:
        wandb.init(project=cfg.wandb_project, config=cfg)
        wandb.run.name = cfg.experiment_name
    print("\n")
    print(f"[info] experiment dir: {cfg.experiment_dir}")
    print(f"[info] training on HDF5: {demo_path}")
    print(f"[info] language: {language_instruction}")
    print(f"[info] algo: {cfg.lifelong.algo}, policy: {cfg.policy.policy_type}")
    
    # split ratios from config (data.train_dataset_ratio / data.val_dataset_ratio)
    train_ratio = cfg.data.train_dataset_ratio
    val_ratio = cfg.data.val_dataset_ratio

    train_single_task(
        cfg,
        dataset,
        task_emb,
        cfg.experiment_dir,
        train_ratio,
        val_ratio,
        args.ckpt_mode,
        args.ckpt_interval,
    )
    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

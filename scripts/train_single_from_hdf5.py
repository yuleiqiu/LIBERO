import argparse
import json
import os
from pathlib import Path

import h5py
import torch
import yaml
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


def read_language_from_hdf5(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        problem_info = json.loads(f["data"].attrs["problem_info"])
    return problem_info["language_instruction"]


def train_single_task(cfg, dataset, task_emb, save_dir, train_ratio, val_ratio):
    """
    Minimal single-task training loop (no benchmark dependency).
    """
    algo_cls = get_algo_class(cfg.lifelong.algo)
    algo = safe_device(algo_cls(n_tasks=1, cfg=cfg), cfg.device)
    algo.start_task(0)

    # Split dataset into train / val / eval (eval indices saved for external eval)
    total_len = len(dataset)
    train_size = int(total_len * train_ratio)
    val_size = int(total_len * val_ratio)
    eval_size = total_len - train_size - val_size
    if val_size <= 0:
        val_size = 1
        train_size = max(train_size - 1, 1)
        eval_size = total_len - train_size - val_size
    if eval_size <= 0:
        eval_size = 1
        train_size = max(train_size - 1, 1)
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

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        sampler=RandomSampler(train_dataset),
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=getattr(cfg.eval, "batch_size", cfg.train.batch_size),
        num_workers=getattr(cfg.eval, "num_workers", 0),
        shuffle=False,
        persistent_workers=False,
    )

    best_loss = float("inf")
    ckpt_path = os.path.join(save_dir, "task0_model.pth")

    n_epochs = cfg.train.n_epochs
    val_every = getattr(cfg.eval, "eval_every", 1)

    # Zero-shot evaluation on validation set (epoch 0)
    algo.policy.eval()
    val_losses = []
    with torch.no_grad():
        for data in tqdm(val_loader, desc="val epoch 0", leave=True):
            loss = algo.eval_observe(data)
            val_losses.append(loss)
    val_avg = sum(val_losses) / max(len(val_losses), 1)
    best_loss = val_avg
    torch_save_model(algo.policy, ckpt_path, cfg=cfg)
    print(f"[info] epoch 000 | val avg loss {val_avg:.4f} | saved checkpoint")

    for epoch in range(1, n_epochs + 1):
        algo.policy.train()
        train_losses = []
        for data in tqdm(train_loader, desc=f"train epoch {epoch}", leave=True):
            loss = algo.observe(data)
            train_losses.append(loss)

        train_avg = sum(train_losses) / max(len(train_losses), 1)
        print(f"[info] epoch {epoch:03d} | train avg loss {train_avg:.4f}")

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

            if val_avg < best_loss:
                best_loss = val_avg
                torch_save_model(algo.policy, ckpt_path, cfg=cfg)
                print(f"[info] saved best checkpoint to {ckpt_path}")

    print(f"[info] finished training. best val loss={best_loss:.4f}, ckpt={ckpt_path}")


def main():
    parser = argparse.ArgumentParser(description="Single-task training directly from HDF5")
    parser.add_argument("--demo-file", required=True, help="Path to processed *_demo.hdf5")
    parser.add_argument(
        "--config-override",
        nargs="*",
        default=[],
        help="Hydra-style overrides, e.g., policy=bc_rnn_policy lifelong=er train.n_epochs=20",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Val split ratio")
    args = parser.parse_args()

    cfg = load_cfg(args.config_override)
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
    print("\n")
    print(f"[info] experiment dir: {cfg.experiment_dir}")
    print(f"[info] training on HDF5: {demo_path}")
    print(f"[info] language: {language_instruction}")
    print(f"[info] algo: {cfg.lifelong.algo}, policy: {cfg.policy.policy_type}")
    
    # pp(cfg.data)
    # pp(cfg.policy)
    # pp(cfg.train)
    # pp(cfg.eval)
    # pp(cfg.lifelong)

    train_single_task(cfg, dataset, task_emb, cfg.experiment_dir, args.train_ratio, args.val_ratio)


if __name__ == "__main__":
    main()

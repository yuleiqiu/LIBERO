import json
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, Subset
from tqdm import tqdm

try:
    import draccus
except ImportError as exc:
    raise ImportError("draccus is required; install with `pip install draccus`.") from exc

from standalone.configs import TrainConfig, serialize_policy_config
from standalone.dataset_utils.hdf5_sequence_dataset import (
    HDF5SequenceDataset,
    compute_obs_stats,
    load_obs_stats,
    save_obs_stats,
)
from standalone.dataset_utils.normalizer_utils import (
    build_identity_normalizer,
    build_linear_normalizer,
    compute_linear_stats,
)
from standalone.models.policy.policy_factory import build_policy, get_policy_name
from standalone.utils.train_utils import (
    build_scheduler,
    build_optimizer,
    load_init_states_with_anchors,
    make_split_indices,
    prepare_train_config,
    sample_per_anchor,
)


def set_seed(seed: int) -> None:
    """Set random seeds for numpy and torch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@draccus.wrap()
def main(cfg: TrainConfig) -> None:
    cfg, save_dir, _ = prepare_train_config(cfg)
    set_seed(cfg.data.seed)

    wandb = None
    if cfg.logging.use_wandb:
        try:
            import wandb as wandb_lib
        except ImportError as exc:
            raise ImportError(
                "wandb is required when use_wandb=True; install with `pip install wandb`."
            ) from exc
        wandb = wandb_lib
        if is_dataclass(cfg):
            wandb_config = asdict(cfg)
        else:
            wandb_config = getattr(cfg, "__dict__", cfg)
        if isinstance(wandb_config, dict):
            wandb_config["policy"] = serialize_policy_config(cfg)
        wandb.init(
            project=cfg.logging.wandb_project,
            entity=cfg.logging.wandb_entity or None,
            config=wandb_config,
        )
        if cfg.logging.experiment_name:
            wandb.run.name = cfg.logging.experiment_name

    demo_path = Path(cfg.data.demo_file).expanduser().resolve()
    if not demo_path.exists():
        raise FileNotFoundError(f"HDF5 not found: {demo_path}")

    obs_keys = [k.strip() for k in cfg.data.obs_keys.split(",") if k.strip()]
    image_keys = [k.strip() for k in cfg.data.image_keys.split(",") if k.strip()]
    all_keys = obs_keys + image_keys
    policy_name = get_policy_name(cfg)
    if policy_name in ("act", "cnnmlp", "dp") and cfg.data.normalize_obs:
        print("[warn] ACT/CNNMLP/DP policy ignores obs normalization; disabling normalize_obs.")
        cfg.data.normalize_obs = False
    split_path = save_dir / "split_indices.json"

    device = cfg.training.device if torch.cuda.is_available() else "cpu"
    val_every = int(cfg.training.val_every)
    rollout_every = int(cfg.rollout.every)
    save_ckpt_every = int(getattr(cfg.training, "save_ckpt_every", 1))
    if val_every < 0:
        raise ValueError("val_every must be >= 0")
    if rollout_every < 0:
        raise ValueError("rollout_every must be >= 0")
    if save_ckpt_every <= 0:
        raise ValueError("training.save_ckpt_every must be >= 1")
    if cfg.rollout.steps <= 0:
        raise ValueError("rollout_steps must be >= 1")
    if cfg.rollout.warmup_steps < 0:
        raise ValueError("rollout_warmup_steps must be >= 0")
    if cfg.rollout.num_procs <= 0:
        raise ValueError("rollout_num_procs must be >= 1")
    if cfg.rollout.env_horizon <= 0:
        raise ValueError("rollout.env_horizon must be >= 1")
    save_topk = int(getattr(cfg.training, "save_topk", 0) or 0)
    if save_topk > 0:
        if rollout_every <= 0:
            raise ValueError("training.save_topk requires rollout.every > 0")
        if save_ckpt_every % rollout_every != 0:
            raise ValueError(
                "training.save_topk requires save_ckpt_every to be a multiple of rollout.every"
            )

    base_dataset = HDF5SequenceDataset(
        hdf5_path=str(demo_path),
        obs_keys=all_keys,
        obs_horizon=cfg.data.obs_horizon,
        predict_horizon=cfg.data.predict_horizon,
        action_shift=getattr(cfg.data, "action_shift", 0),
        image_keys=image_keys,
        image_norm=cfg.data.image_norm,
        image_transforms=cfg.data.image_transforms,
    )

    train_idx, val_idx = make_split_indices(
        len(base_dataset), cfg.data.train_ratio, cfg.data.val_ratio, cfg.data.seed
    )
    with open(split_path, "w") as f:
        json.dump({"train": train_idx, "val": val_idx}, f, indent=2)
    if wandb is not None:
        wandb.run.summary.update(
            {
                "split/train_size": len(train_idx),
                "split/val_size": len(val_idx),
            }
        )

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
        batch_size=cfg.training.batch_size,
        sampler=RandomSampler(train_dataset),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = None
    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
    sample = base_dataset[train_idx[0]]
    action_dim = sample["actions"].shape[-1]
    # print(f"[debug] action_dim: {action_dim}")
    proprio_dim = sum(np.prod(sample["obs"][k].shape[1:]) for k in obs_keys)
    # print(f"[debug] proprio_dim: {proprio_dim}")
    image_shapes = {}
    obs_shapes = {}
    for key, value in sample["obs"].items():
        obs_shapes[key] = value.shape
        if key in image_keys:
            image_shapes[key] = value.shape[1:]
    dp_normalizer = None
    ckpt_extra = {}
    if policy_name == "dp":
        normalizer_cfg = cfg.policy.dp.normalizer
        identity_normalizer = build_identity_normalizer(
            obs_shapes=obs_shapes,
            obs_keys=list(obs_shapes.keys()),
            action_dim=action_dim,
            last_n_dims=normalizer_cfg.last_n_dims,
            include_actions=True,
        )
        if not normalizer_cfg.enable:
            dp_normalizer = identity_normalizer
        else:
            lowdim_keys = [k for k in obs_keys if k not in image_keys]
            obs_keys_for_norm = lowdim_keys if normalizer_cfg.normalize_obs else []
            include_actions = bool(normalizer_cfg.normalize_actions)
            stats = {}
            if obs_keys_for_norm or include_actions:
                stats = compute_linear_stats(
                    base_dataset,
                    train_idx,
                    obs_keys_for_norm,
                    image_keys=image_keys,
                    last_n_dims=normalizer_cfg.last_n_dims,
                    include_actions=include_actions,
                )
            if stats:
                dp_normalizer = build_linear_normalizer(
                    stats,
                    mode=normalizer_cfg.mode,
                    output_min=normalizer_cfg.output_min,
                    output_max=normalizer_cfg.output_max,
                    range_eps=normalizer_cfg.range_eps,
                    fit_offset=normalizer_cfg.fit_offset,
                )
                for key in identity_normalizer.fields:
                    if key not in dp_normalizer.fields:
                        dp_normalizer[key] = identity_normalizer[key]
            else:
                dp_normalizer = identity_normalizer
        ckpt_extra["normalizer"] = dp_normalizer.state_dict()
    model = build_policy(
        cfg,
        obs_keys,
        image_keys,
        action_dim,
        proprio_dim=proprio_dim,
        obs_shapes=obs_shapes,
    )
    if policy_name == "dp":
        if dp_normalizer is None:
            raise ValueError("dp normalizer is required but was not initialized")
        model.set_normalizer(dp_normalizer)
    model.to(device)
    optimizer = build_optimizer(cfg, model, policy_name)
    total_steps = int(cfg.training.epochs) * max(len(train_loader), 1)
    scheduler = build_scheduler(cfg, optimizer, policy_name, total_steps)
    def _model_state_for_ckpt():
        state = model.state_dict()
        if policy_name == "dp":
            state = {
                key: value
                for key, value in state.items()
                if not key.startswith("diffusion_model.normalizer.")
            }
        return state

    rollout_state = None
    if rollout_every > 0:
        if cfg.rollout.per_anchor <= 0:
            raise ValueError("rollout_per_anchor must be >= 1")
        init_states, anchor_map, init_states_path, anchor_indices = load_init_states_with_anchors(
            cfg, demo_path
        )
        if not anchor_map:
            raise ValueError(f"no anchors found in {init_states_path}")
        anchor_summary = ", ".join(
            f"{anchor}:{len(anchor_map[anchor])}"
            for anchor in sorted(anchor_map.keys())
        )
        print(
            f"[info] loaded init states for rollout: {init_states.shape[0]} | anchors {anchor_summary}"
        )
        rollout_state = {
            "init_states": init_states,
            "anchor_map": anchor_map,
            "image_shapes": image_shapes,
            "anchor_indices": anchor_indices,
        }
        from standalone.rollout_env import run_env_rollouts

        rollout_runner = run_env_rollouts
    else:
        rollout_runner = None

    last_ckpt_path = save_dir / "model_last.pt"
    final_ckpt_path = None
    printed_batch = False
    topk_path = save_dir / "topk.json"
    topk_records = []
    save_topk = int(getattr(cfg.training, "save_topk", 0) or 0)
    if topk_path.exists():
        try:
            with open(topk_path, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                topk_records = data
        except Exception:
            topk_records = []
    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        train_losses = []
        train_stats = {}
        train_stat_count = 0
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
            if wandb is not None:
                loss, stats = model.compute_loss(batch, return_stats=True)
            else:
                loss = model.compute_loss(batch)
                stats = None

            optimizer.zero_grad()
            loss.backward()
            if cfg.training.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            train_losses.append(loss.item())
            if stats:
                for key, value in stats.items():
                    train_stats[key] = train_stats.get(key, 0.0) + float(value)
                train_stat_count += 1

        avg_train = sum(train_losses) / max(len(train_losses), 1)
        print(f"[info] epoch {epoch:03d} | train loss {avg_train:.6f}")
        avg_train_stats = (
            {k: v / train_stat_count for k, v in train_stats.items()}
            if train_stat_count
            else {}
        )
        avg_val = None
        do_val = val_loader is not None and val_every > 0 and epoch % val_every == 0
        if do_val:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    for key in batch["obs"]:
                        batch["obs"][key] = batch["obs"][key].to(device)
                    batch["actions"] = batch["actions"].to(device)
                    if "action_mask" in batch:
                        batch["action_mask"] = batch["action_mask"].to(device)
                    loss = model.compute_loss(batch)
                    val_losses.append(loss.item())
            avg_val = sum(val_losses) / max(len(val_losses), 1)
            print(f"[info] epoch {epoch:03d} | val loss {avg_val:.6f}")
        rollout_success = None
        should_save = epoch % save_ckpt_every == 0
        if rollout_runner is not None and epoch % rollout_every == 0:
            rng = np.random.default_rng(cfg.data.seed + epoch)
            rollout_indices = sample_per_anchor(
                rollout_state["anchor_map"], cfg.rollout.per_anchor, rng
            )
            video_dir = save_dir / "rollout_videos" / "val" / f"epoch_{epoch:03d}"
            rollout_cfg = SimpleNamespace(
                data=cfg.data,
                steps=int(cfg.rollout.steps),
                warmup_steps=int(cfg.rollout.warmup_steps),
                n_rollouts=len(rollout_indices),
                env_horizon=int(cfg.rollout.env_horizon),
                sample_index=0,
                save_videos=len(rollout_indices),
                video_camera="",
                video_fps=30,
                video_dir=str(video_dir),
                use_mp=bool(cfg.rollout.use_mp),
                num_procs=int(cfg.rollout.num_procs),
            )
            rollout_details = rollout_runner(
                rollout_cfg,
                model,
                obs_keys,
                image_keys,
                obs_stats,
                demo_path,
                action_dim,
                rollout_state["image_shapes"],
                init_states_override=rollout_state["init_states"],
                rollout_order_override=rollout_indices,
                anchor_ids=rollout_state["anchor_indices"],
            )
            if rollout_details is not None:
                rollout_success = rollout_details.get("success_rate")
            if wandb is not None and rollout_details is not None:
                anchor_counts = defaultdict(int)
                anchor_success = defaultdict(int)
                for result in rollout_details.get("episode_results", []):
                    anchor_id = result.get("anchor_id")
                    if anchor_id is None:
                        continue
                    anchor_counts[anchor_id] += 1
                    if result.get("success"):
                        anchor_success[anchor_id] += 1
                anchor_table = wandb.Table(columns=["anchor_id", "success"])
                for anchor_id in sorted(anchor_counts.keys()):
                    success_str = f"{anchor_success[anchor_id]}/{anchor_counts[anchor_id]}"
                    anchor_table.add_data(anchor_id, success_str)

                wandb.log(
                    {
                        "rollout/success_rate": rollout_details.get("success_rate"),
                        "rollout/anchor_success": anchor_table,
                    },
                    step=epoch,
                )

        if wandb is not None:
            log_data = {"epoch": epoch, "train/loss": avg_train}
            for key, value in avg_train_stats.items():
                log_data[f"train/{key}"] = value
            if avg_val is not None:
                log_data["val/loss"] = avg_val
            wandb.log(log_data, step=epoch)

        if should_save:
            torch.save(
                {"model": _model_state_for_ckpt(), "obs_stats": obs_stats, **ckpt_extra},
                last_ckpt_path,
            )
            final_ckpt_path = last_ckpt_path
            if (
                save_topk > 0
                and rollout_success is not None
                and rollout_every > 0
            ):
                score = float(rollout_success)
                topk_records = [
                    r for r in topk_records if Path(r.get("path", "")).exists()
                ]
                if len(topk_records) < save_topk:
                    ckpt_path = save_dir / f"model_topk_epoch_{epoch:03d}.pt"
                    torch.save(
                        {"model": _model_state_for_ckpt(), "obs_stats": obs_stats, **ckpt_extra},
                        ckpt_path,
                    )
                    topk_records.append(
                        {"epoch": epoch, "success_rate": score, "path": str(ckpt_path)}
                    )
                else:
                    worst_idx = min(
                        range(len(topk_records)),
                        key=lambda i: float(topk_records[i].get("success_rate", -1e9)),
                    )
                    worst_score = float(topk_records[worst_idx].get("success_rate", -1e9))
                    if score > worst_score:
                        worst_path = topk_records[worst_idx].get("path")
                        if worst_path and Path(worst_path).exists():
                            Path(worst_path).unlink()
                        ckpt_path = save_dir / f"model_topk_epoch_{epoch:03d}.pt"
                        torch.save(
                            {"model": _model_state_for_ckpt(), "obs_stats": obs_stats, **ckpt_extra},
                            ckpt_path,
                        )
                        topk_records[worst_idx] = {
                            "epoch": epoch,
                            "success_rate": score,
                            "path": str(ckpt_path),
                        }
                topk_records = sorted(
                    topk_records, key=lambda r: float(r.get("success_rate", -1e9)), reverse=True
                )
                with open(topk_path, "w") as f:
                    json.dump(topk_records, f, indent=2)

    if final_ckpt_path is not None:
        print(f"[info] finished training. ckpt saved to {final_ckpt_path}")
    else:
        print("[info] finished training. no checkpoint saved")
    if wandb is not None:
        if final_ckpt_path is not None:
            wandb.run.summary["ckpt_path"] = str(final_ckpt_path)
        wandb.finish()


if __name__ == "__main__":
    main()

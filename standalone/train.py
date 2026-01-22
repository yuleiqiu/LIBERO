import json
import sys
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, Subset
from tqdm import tqdm

try:
    import draccus
except ImportError as exc:
    raise ImportError("draccus is required; install with `pip install draccus`.") from exc

from standalone.configs import (
    TrainConfig,
    apply_policy_config,
    serialize_policy_config,
)
from standalone.dataset_utils.hdf5_sequence_dataset import (
    HDF5SequenceDataset,
    compute_obs_stats,
    load_obs_stats,
    save_obs_stats,
)
from standalone.models.policy.policy_factory import build_policy, get_policy_name
from standalone.utils.train_utils import (
    TRAIN_CONFIG_NAME,
    apply_config_dict,
    cfg_to_dict,
    load_config_json,
    load_init_states_with_anchors,
    make_splits,
    merge_config_with_overrides,
    resolve_run_dir,
    sample_per_anchor,
    write_run_metadata,
)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


RESUME_OVERRIDE_ALLOWLIST = [
    "training.device",
    "logging.use_wandb",
    "logging.wandb_project",
    "logging.wandb_entity",
    "logging.experiment_name",
]


def reject_draccus_config():
    config_arg = None
    for idx, arg in enumerate(sys.argv):
        if arg == "--config" and idx + 1 < len(sys.argv):
            config_arg = sys.argv[idx + 1]
            break
        if arg.startswith("--config="):
            config_arg = arg.split("=", 1)[1]
            break
    if config_arg:
        raise ValueError(
            "YAML configs are disabled. Use CLI overrides with dataclass defaults, "
            "or pass saved_config_path/resume to load train_config.json."
        )


def apply_resume_config(cfg):
    if not getattr(cfg, "resume", False) and not getattr(cfg, "saved_config_path", None):
        return cfg, None
    cfg_dict = cfg_to_dict(cfg)
    config_path = None
    if cfg.saved_config_path:
        config_path = Path(cfg.saved_config_path).expanduser().resolve()
    if config_path is None and getattr(cfg, "resume", False):
        config_path = (
            Path(cfg.paths.save_dir).expanduser().resolve() / TRAIN_CONFIG_NAME
        )
    if config_path is None:
        return cfg, None
    if config_path.suffix.lower() in (".yml", ".yaml"):
        raise ValueError(
            "YAML configs are disabled. Use train_config.json or CLI overrides."
        )
    if not config_path.exists():
        if config_path.name == TRAIN_CONFIG_NAME:
            legacy_path = config_path.with_name("config.json")
            if legacy_path.exists():
                config_path = legacy_path
            else:
                raise FileNotFoundError(f"config not found: {config_path}")
        else:
            raise FileNotFoundError(f"config not found: {config_path}")
    saved_cfg = load_config_json(config_path)
    if "saved_config_path" not in saved_cfg and "config_path" in saved_cfg:
        saved_cfg["saved_config_path"] = saved_cfg["config_path"]
    defaults_dict = cfg_to_dict(TrainConfig())
    merged_cfg = merge_config_with_overrides(
        saved_cfg, cfg_dict, RESUME_OVERRIDE_ALLOWLIST, defaults=defaults_dict
    )
    merged_cfg["saved_config_path"] = str(config_path)
    merged_cfg["resume"] = bool(getattr(cfg, "resume", False))
    apply_config_dict(cfg, merged_cfg)
    if getattr(cfg, "resume", False):
        cfg.paths.save_dir = str(config_path.parent)
    return cfg, config_path


@draccus.wrap()
def main(cfg: TrainConfig):
    reject_draccus_config()
    cfg, _ = apply_resume_config(cfg)
    apply_policy_config(cfg)
    if not cfg.data.demo_file:
        raise ValueError("data.demo_file is required")
    set_seed(cfg.data.seed)

    if cfg.resume:
        save_dir = Path(cfg.paths.save_dir).expanduser().resolve()
        if not save_dir.exists():
            raise FileNotFoundError(f"resume dir not found: {save_dir}")
        print(f"[info] resuming in {save_dir}")
    else:
        save_dir = resolve_run_dir(Path(cfg.paths.save_dir))
        save_dir.mkdir(parents=True, exist_ok=True)
        cfg.paths.save_dir = str(save_dir)
        print(f"[info] saving to {save_dir}")

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
    if policy_name in ("act", "cnnmlp") and cfg.data.normalize_obs:
        print("[warn] ACT/CNNMLP policy ignores obs normalization; disabling normalize_obs.")
        cfg.data.normalize_obs = False
    if not cfg.resume:
        cfg_dict = cfg_to_dict(cfg)
        if isinstance(cfg_dict, dict):
            cfg_dict["policy"] = serialize_policy_config(cfg)
        write_run_metadata(save_dir, cfg, cfg_dict=cfg_dict)

    split_path = save_dir / "split_indices.json"

    device = cfg.training.device if torch.cuda.is_available() else "cpu"
    val_every = int(cfg.training.val_every)
    rollout_every = int(cfg.rollout.every)
    ckpt_mode = str(getattr(cfg.training, "ckpt_mode", "last")).lower()
    if val_every < 0:
        raise ValueError("val_every must be >= 0")
    if rollout_every < 0:
        raise ValueError("rollout_every must be >= 0")
    if ckpt_mode not in ("last", "best", "all"):
        raise ValueError("training.ckpt_mode must be one of: last, best, all")
    if cfg.rollout.steps <= 0:
        raise ValueError("rollout_steps must be >= 1")
    if cfg.rollout.warmup_steps < 0:
        raise ValueError("rollout_warmup_steps must be >= 0")
    if cfg.rollout.num_procs <= 0:
        raise ValueError("rollout_num_procs must be >= 1")
    if cfg.rollout.env_horizon <= 0:
        raise ValueError("rollout.env_horizon must be >= 1")

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

    train_idx, val_idx = make_splits(
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
    qpos_dim = sum(np.prod(sample["obs"][k].shape[1:]) for k in obs_keys)
    # print(f"[debug] qpos_dim: {qpos_dim}")
    image_shapes = {}
    for key in image_keys:
        if key not in sample["obs"]:
            raise KeyError(f"image key not found in obs: {key}")
        image_shapes[key] = sample["obs"][key].shape[1:]
    model = build_policy(cfg, obs_keys, image_keys, action_dim, qpos_dim)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

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

    if ckpt_mode == "best":
        if rollout_every <= 0 or rollout_runner is None:
            raise ValueError("training.ckpt_mode=best requires rollout.every > 0")
        if cfg.training.epochs < rollout_every:
            raise ValueError(
                "training.ckpt_mode=best requires at least one rollout; "
                "increase epochs or reduce rollout.every"
            )

    best_rollout = None
    best_rollout_epoch = None
    last_ckpt_path = save_dir / "model_last.pt"
    best_ckpt_path = save_dir / "model_best.pt"
    final_ckpt_path = None
    printed_batch = False
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
            if ckpt_mode == "best" and rollout_success is not None:
                rollout_success = float(rollout_success)
                if best_rollout is None or rollout_success > best_rollout:
                    best_rollout = rollout_success
                    best_rollout_epoch = epoch
                    torch.save(
                        {"model": model.state_dict(), "obs_stats": obs_stats},
                        best_ckpt_path,
                    )
                    final_ckpt_path = best_ckpt_path
                    print(
                        f"[info] saved best ckpt (success_rate={best_rollout:.3f}) "
                        f"to {best_ckpt_path}"
                    )
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

        if ckpt_mode == "all":
            ckpt_path = save_dir / f"model_epoch_{epoch:03d}.pt"
            torch.save({"model": model.state_dict(), "obs_stats": obs_stats}, ckpt_path)
            final_ckpt_path = ckpt_path
        elif ckpt_mode == "last":
            torch.save({"model": model.state_dict(), "obs_stats": obs_stats}, last_ckpt_path)
            final_ckpt_path = last_ckpt_path

    if ckpt_mode == "best" and best_rollout is None:
        print(
            "[warning] no rollout success_rate observed; "
            "saving final model as model_best.pt"
        )
        torch.save({"model": model.state_dict(), "obs_stats": obs_stats}, best_ckpt_path)
        final_ckpt_path = best_ckpt_path

    if final_ckpt_path is not None:
        print(f"[info] finished training. ckpt saved to {final_ckpt_path}")
    else:
        print("[info] finished training. no checkpoint saved")
    if wandb is not None:
        if best_rollout is not None:
            wandb.run.summary["best_rollout_success"] = best_rollout
            if best_rollout_epoch is not None:
                wandb.run.summary["best_rollout_epoch"] = best_rollout_epoch
        if final_ckpt_path is not None:
            wandb.run.summary["ckpt_path"] = str(final_ckpt_path)
        wandb.finish()


if __name__ == "__main__":
    main()

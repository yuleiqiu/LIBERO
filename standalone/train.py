import json
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

try:
    import draccus
except ImportError as exc:
    raise ImportError("draccus is required; install with `pip install draccus`.") from exc

from standalone.configs import TrainConfig, serialize_policy_config
from standalone.dataset_utils.hdf5_sequence_dataset import HDF5SequenceDataset
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
    make_episode_split_keys,
    prepare_train_config,
    rollout_sanity_check,
)
from standalone.utils.rollout_utils import set_rollout_seed


def set_seed(seed: int) -> None:
    """Set random seeds for numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict) -> None:
    if not isinstance(state, dict):
        return
    py_state = state.get("python")
    if py_state is not None:
        random.setstate(py_state)
    np_state = state.get("numpy")
    if np_state is not None:
        np.random.set_state(np_state)
    torch_state = state.get("torch")
    if torch_state is not None:
        torch.set_rng_state(torch_state)
    if torch.cuda.is_available():
        cuda_states = state.get("torch_cuda")
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device=device)


def _parse_wandb_tags(raw_tags: object) -> list:
    """Parse wandb tags from a comma-separated string or iterable."""
    if raw_tags is None:
        return []
    if isinstance(raw_tags, str):
        return [tag.strip() for tag in raw_tags.split(",") if tag.strip()]
    if isinstance(raw_tags, (list, tuple)):
        tags = []
        for tag in raw_tags:
            tag_str = str(tag).strip()
            if tag_str:
                tags.append(tag_str)
        return tags
    tag_str = str(raw_tags).strip()
    return [tag_str] if tag_str else []


@draccus.wrap()
def main(cfg: TrainConfig) -> None:
    cfg, save_dir, _ = prepare_train_config(cfg)
    set_seed(cfg.data.seed)

    wandb = None
    wandb_run_id_path = save_dir / "wandb_run_id.txt"
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
        wandb_init_kwargs = dict(
            project=cfg.logging.wandb_project,
            entity=cfg.logging.wandb_entity or None,
            config=wandb_config,
        )
        if cfg.logging.wandb_group:
            wandb_init_kwargs["group"] = cfg.logging.wandb_group
        wandb_tags = _parse_wandb_tags(getattr(cfg.logging, "wandb_tags", None))
        if wandb_tags:
            wandb_init_kwargs["tags"] = wandb_tags
        if cfg.resume and wandb_run_id_path.exists():
            with open(wandb_run_id_path, "r") as f:
                run_id = f.read().strip()
            if run_id:
                wandb_init_kwargs["id"] = run_id
                wandb_init_kwargs["resume"] = "must"
                print(f"[info] resuming wandb run id={run_id}")
        wandb.init(**wandb_init_kwargs)
        if wandb.run is not None:
            with open(wandb_run_id_path, "w") as f:
                f.write(str(wandb.run.id) + "\n")
        if cfg.logging.experiment_name:
            wandb.run.name = cfg.logging.experiment_name

    demo_path = Path(cfg.data.demo_file).expanduser().resolve()
    if not demo_path.exists():
        raise FileNotFoundError(f"HDF5 not found: {demo_path}")

    obs_keys = [k.strip() for k in cfg.data.obs_keys.split(",") if k.strip()]
    image_keys = [k.strip() for k in cfg.data.image_keys.split(",") if k.strip()]
    # mask_keys is a list parallel to image_keys; empty string means no mask for that camera
    _mask_keys_raw = [k.strip() for k in (cfg.data.mask_keys or "").split(",")]
    while len(_mask_keys_raw) < len(image_keys):
        _mask_keys_raw.append("")
    mask_keys = _mask_keys_raw[: len(image_keys)]
    active_mask_keys = [k for k in mask_keys if k]
    all_keys = obs_keys + image_keys + active_mask_keys
    policy_name = get_policy_name(cfg)
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
    if rollout_every > 0:
        rollout_sanity_check(cfg, demo_path)
    save_topk = int(getattr(cfg.training, "save_topk", 0) or 0)
    if save_topk > 0:
        if rollout_every <= 0:
            raise ValueError("training.save_topk requires rollout.every > 0")
        if save_ckpt_every % rollout_every != 0:
            raise ValueError(
                "training.save_topk requires save_ckpt_every to be a multiple of rollout.every"
            )

    dp_action_horizon = None
    dp_action_start_offset = None
    if policy_name == "dp":
        dp_cfg = cfg.policy.dp
        dp_action_horizon = dp_cfg.action_horizon
        if dp_action_horizon is None:
            dp_action_horizon = getattr(dp_cfg.model, "horizon", None)
        dp_action_start_offset = dp_cfg.action_start_offset
        if dp_action_start_offset is None and dp_action_horizon is not None:
            dp_action_start_offset = -(cfg.data.obs_horizon - 1)
        if dp_action_horizon is not None and dp_action_horizon <= 0:
            raise ValueError("dp.action_horizon must be >= 1")
        if (dp_action_horizon is not None) or (dp_action_start_offset not in (None, 0)):
            if not getattr(dp_cfg.model, "do_mask_loss_for_padding", False):
                print(
                    "[info] enabling dp.model.do_mask_loss_for_padding for padded action segments."
                )
                dp_cfg.model.do_mask_loss_for_padding = True

    with h5py.File(str(demo_path), "r") as f:
        all_demo_keys = sorted(str(k) for k in f["data"].keys())
    if not all_demo_keys:
        raise ValueError(f"no demos found in {demo_path}")

    val_ratio = float(cfg.data.val_ratio)
    if val_ratio < 0 or val_ratio >= 1.0:
        raise ValueError("data.val_ratio must be in [0, 1)")
    train_ratio = 1.0 - val_ratio

    split_data = None
    if split_path.exists():
        with open(split_path, "r") as f:
            split_data = json.load(f)
        if split_data.get("split_unit") != "episode":
            raise ValueError(
                f"existing split file is not episode-based: {split_path}. "
                "Delete it or use a fresh run directory."
            )
        train_episodes = [str(k) for k in split_data.get("train_episodes", [])]
        val_episodes = [str(k) for k in split_data.get("val_episodes", [])]
        print(f"[info] loaded episode split from {split_path}")
    else:
        train_episodes, val_episodes = make_episode_split_keys(
            all_demo_keys, val_ratio, cfg.data.seed
        )

    train_set = set(train_episodes)
    val_set = set(val_episodes)
    if len(train_set) != len(train_episodes):
        raise ValueError("duplicate episodes found in train split")
    if len(val_set) != len(val_episodes):
        raise ValueError("duplicate episodes found in val split")
    overlap = train_set.intersection(val_set)
    if overlap:
        raise ValueError(f"episode leakage detected in split: {sorted(overlap)}")
    all_demo_set = set(all_demo_keys)
    split_union = train_set.union(val_set)
    missing = sorted(split_union - all_demo_set)
    if missing:
        raise ValueError(f"split contains episodes not in dataset: {missing[:5]}")
    uncovered = sorted(all_demo_set - split_union)
    if uncovered:
        raise ValueError(
            "split does not cover all episodes. "
            f"Found {len(uncovered)} uncovered episode(s)."
        )
    if len(train_episodes) == 0:
        raise ValueError("train split is empty")
    if val_ratio > 0 and len(val_episodes) == 0:
        raise ValueError("val_ratio > 0 but val split is empty")

    dataset_kwargs = dict(
        hdf5_path=str(demo_path),
        obs_keys=all_keys,
        obs_horizon=cfg.data.obs_horizon,
        predict_horizon=cfg.data.predict_horizon,
        action_horizon=dp_action_horizon,
        action_start_offset=dp_action_start_offset,
        image_keys=image_keys,
        image_norm=cfg.data.image_norm,
        image_transforms=cfg.data.image_transforms,
    )
    train_dataset = HDF5SequenceDataset(demos=train_episodes, **dataset_kwargs)
    val_dataset = (
        HDF5SequenceDataset(demos=val_episodes, **dataset_kwargs)
        if len(val_episodes) > 0
        else None
    )
    train_sample_count = len(train_dataset)
    val_sample_count = len(val_dataset) if val_dataset is not None else 0
    if train_sample_count == 0:
        raise ValueError("train split has no samples")

    split_data = {
        "split_unit": "episode",
        "seed": int(cfg.data.seed),
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "num_episodes_total": len(all_demo_keys),
        "num_episodes_train": len(train_episodes),
        "num_episodes_val": len(val_episodes),
        "num_samples_train": train_sample_count,
        "num_samples_val": val_sample_count,
        "train_episodes": train_episodes,
        "val_episodes": val_episodes,
    }
    with open(split_path, "w") as f:
        json.dump(split_data, f, indent=2)
    print(
        "[info] split | "
        f"train episodes {len(train_episodes)} ({train_sample_count} samples), "
        f"val episodes {len(val_episodes)} ({val_sample_count} samples)"
    )
    if wandb is not None:
        wandb.run.summary.update(
            {
                "split/train_episodes": len(train_episodes),
                "split/val_episodes": len(val_episodes),
                "split/train_samples": train_sample_count,
                "split/val_samples": val_sample_count,
            }
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        sampler=RandomSampler(train_dataset),
        num_workers=8,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
        )
    sample = train_dataset[0]
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
                    train_dataset,
                    range(len(train_dataset)),
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
        mask_keys=mask_keys if active_mask_keys else None,
    )
    if policy_name == "dp":
        if dp_normalizer is None:
            raise ValueError("dp normalizer is required but was not initialized")
        model.set_normalizer(dp_normalizer)
    model.to(device)
    optimizer = build_optimizer(cfg, model, policy_name)
    steps_per_epoch = len(train_loader)
    total_steps = int(cfg.training.epochs) * max(steps_per_epoch, 1)
    scheduler = build_scheduler(cfg, optimizer, policy_name, total_steps)

    last_ckpt_path = save_dir / "model_last.pt"
    start_epoch = 0
    if cfg.resume:
        if not last_ckpt_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {last_ckpt_path}")
        resume_ckpt = torch.load(last_ckpt_path, map_location="cpu", weights_only=False)
        resume_model_state = (
            resume_ckpt.get("model")
            if isinstance(resume_ckpt, dict) and "model" in resume_ckpt
            else resume_ckpt
        )
        model.load_state_dict(resume_model_state)

        loaded_optimizer = False
        loaded_scheduler = False
        loaded_normalizer = False
        if isinstance(resume_ckpt, dict):
            if policy_name == "dp" and dp_normalizer is not None:
                normalizer_state = resume_ckpt.get("normalizer")
                if normalizer_state is not None:
                    dp_normalizer.load_state_dict(normalizer_state)
                    model.set_normalizer(dp_normalizer)
                    loaded_normalizer = True
            opt_state = resume_ckpt.get("optimizer")
            if opt_state is not None:
                optimizer.load_state_dict(opt_state)
                _move_optimizer_state_to_device(optimizer, torch.device(device))
                loaded_optimizer = True
            sched_state = resume_ckpt.get("scheduler")
            if scheduler is not None and sched_state is not None:
                scheduler.load_state_dict(sched_state)
                loaded_scheduler = True
            start_epoch = int(resume_ckpt.get("epoch", 0) or 0)
            _restore_rng_state(resume_ckpt.get("rng_state"))
        print(
            "[info] resumed training state: "
            f"epoch={start_epoch}, optimizer={'yes' if loaded_optimizer else 'no'}, "
            f"scheduler={'yes' if loaded_scheduler else 'no'}, "
            f"normalizer={'yes' if loaded_normalizer else 'no'}"
        )

    if start_epoch >= int(cfg.training.epochs):
        print(
            f"[info] checkpoint epoch {start_epoch} >= training.epochs {cfg.training.epochs}; nothing to do."
        )
        if wandb is not None:
            wandb.run.summary["ckpt_path"] = str(last_ckpt_path)
            wandb.finish()
        return

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
        rollout_indices = []
        for anchor_id in sorted(anchor_map.keys()):
            anchor_state_indices = list(anchor_map[anchor_id])
            if len(anchor_state_indices) < cfg.rollout.per_anchor:
                raise ValueError(
                    f"anchor {anchor_id} has only {len(anchor_state_indices)} states; "
                    f"need {cfg.rollout.per_anchor}"
                )
            rollout_indices.extend(anchor_state_indices[: cfg.rollout.per_anchor])
        print(
            f"[info] fixed rollout indices: {len(rollout_indices)} "
            f"(per_anchor={cfg.rollout.per_anchor})"
        )
        rollout_state = {
            "init_states": init_states,
            "anchor_map": anchor_map,
            "image_shapes": image_shapes,
            "anchor_indices": anchor_indices,
            "rollout_indices": rollout_indices,
        }
        from standalone.rollout_env import run_env_rollouts

        rollout_runner = run_env_rollouts
    else:
        rollout_runner = None

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
    for epoch in range(start_epoch + 1, cfg.training.epochs + 1):
        model.train()
        train_losses = []
        train_stats = {}
        train_stat_count = 0
        for batch in train_loader:
            if not printed_batch:
                obs_horizon = next(iter(batch["obs"].values())).shape[1]
                predict_horizon = batch["actions"].shape[1]
                action_dim = batch["actions"].shape[2]
                obs_key_list = list(batch["obs"].keys())
                action_mask_shape = (
                    tuple(batch["action_mask"].shape) if "action_mask" in batch else None
                )
                print(
                    "[debug] batch:",
                    f"obs_keys={obs_key_list}, actions={tuple(batch['actions'].shape)}, "
                    f"action_mask={action_mask_shape}",
                )
                print(
                    "[debug] horizons:",
                    f"obs_horizon={obs_horizon}, predict_horizon={predict_horizon}, action_dim={action_dim}",
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
        global_step = epoch * steps_per_epoch
        avg_train_stats = (
            {k: v / train_stat_count for k, v in train_stats.items()}
            if train_stat_count
            else {}
        )
        avg_val = None
        force_first_epoch_val = (start_epoch == 0 and epoch == 1)
        do_val = (
            val_loader is not None
            and val_every > 0
            and (force_first_epoch_val or epoch % val_every == 0)
        )
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
        epoch_summary = [
            f"[info] epoch {epoch}/{cfg.training.epochs}",
            f"steps this epoch: {steps_per_epoch}",
            f"global step: {global_step}/{total_steps}",
            f"train loss {avg_train:.6f}",
        ]
        if avg_val is not None:
            epoch_summary.append(f"val loss {avg_val:.6f}")
        print(" | ".join(epoch_summary))
        rollout_success = None
        should_save = epoch % save_ckpt_every == 0
        if rollout_runner is not None and epoch % rollout_every == 0:
            set_rollout_seed(int(cfg.data.seed))
            rollout_indices = rollout_state["rollout_indices"]
            video_dir = save_dir / "rollout_videos" / "val" / f"epoch_{epoch:03d}"
            rollout_cfg = SimpleNamespace(
                data=cfg.data,
                bddl_file=cfg.rollout.bddl_file,
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
                wandb.log(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "rollout/success_rate": rollout_details.get("success_rate"),
                    },
                    step=global_step,
                )

        if wandb is not None:
            log_data = {
                "epoch": epoch,
                "global_step": global_step,
                "train/loss": avg_train,
            }
            for key, value in avg_train_stats.items():
                log_data[f"train/{key}"] = value
            if avg_val is not None:
                log_data["val/loss"] = avg_val
            wandb.log(log_data, step=global_step)

        if should_save:
            ckpt_payload = {
                "model": _model_state_for_ckpt(),
                "epoch": int(epoch),
                "optimizer": optimizer.state_dict(),
                "rng_state": _capture_rng_state(),
                **ckpt_extra,
            }
            if scheduler is not None:
                ckpt_payload["scheduler"] = scheduler.state_dict()
            torch.save(
                ckpt_payload,
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
                        {"model": _model_state_for_ckpt(), **ckpt_extra},
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
                            {"model": _model_state_for_ckpt(), **ckpt_extra},
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

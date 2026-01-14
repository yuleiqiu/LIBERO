import json
import os
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, Subset
from tqdm import tqdm

try:
    import draccus
except ImportError as exc:
    raise ImportError("draccus is required; install with `pip install draccus`.") from exc

from standalone.configs import TrainConfig, apply_policy_config, get_policy_param
from standalone.dataset_utils.hdf5_sequence_dataset import (
    HDF5SequenceDataset,
    compute_obs_stats,
    load_obs_stats,
    save_obs_stats,
)
from standalone.models.policy.act_policy import ACTPolicy


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


def read_bddl_from_hdf5(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        data = f["data"]
        return data.attrs.get("bddl_file_name", None)


def resolve_bddl_path(bddl_file_name, demo_path):
    if not bddl_file_name:
        return None
    candidate = Path(bddl_file_name).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)
    if candidate.exists():
        return str(candidate.resolve())
    repo_root = Path(__file__).resolve().parents[1]
    repo_candidate = (repo_root / "libero/libero/bddl_files" / candidate).resolve()
    if repo_candidate.exists():
        return str(repo_candidate)
    demo_candidate = (demo_path.parent / candidate).resolve()
    if demo_candidate.exists():
        return str(demo_candidate)
    return None


def resolve_init_states_dir(cfg):
    init_dir = getattr(cfg, "rollout_init_states_dir", None)
    if init_dir:
        return Path(init_dir).expanduser().resolve()
    from libero.libero import get_libero_path

    return Path(get_libero_path("init_states")).expanduser().resolve()


def load_init_states_with_anchors(cfg, demo_path):
    bddl_file_name = read_bddl_from_hdf5(str(demo_path))
    if bddl_file_name is None:
        raise ValueError("bddl_file_name not found in hdf5; cannot resolve init states")
    bddl_path = resolve_bddl_path(bddl_file_name, demo_path)
    if bddl_path is None:
        raise FileNotFoundError(f"bddl file not found: {bddl_file_name}")
    init_dir = resolve_init_states_dir(cfg)
    init_states_path = init_dir / Path(bddl_path).parent.name / f"{Path(bddl_path).stem}.pruned_init"
    if not init_states_path.exists():
        raise FileNotFoundError(f"init states file not found: {init_states_path}")
    init_states = torch.load(str(init_states_path))
    if torch.is_tensor(init_states):
        init_states = init_states.cpu().numpy()
    else:
        init_states = np.asarray(init_states)
    anchors_meta = init_states_path.with_suffix(init_states_path.suffix + ".anchors.json")
    if not anchors_meta.exists():
        raise FileNotFoundError(f"anchors meta not found: {anchors_meta}")
    with open(anchors_meta, "r") as f:
        anchor_indices = json.load(f).get("anchor_idx", None)
    if anchor_indices is None:
        raise ValueError(f"anchor_idx not found in {anchors_meta}")
    if len(anchor_indices) != init_states.shape[0]:
        raise ValueError(
            f"anchor_idx length mismatch: {len(anchor_indices)} vs {init_states.shape[0]}"
        )
    by_anchor = defaultdict(list)
    for idx, anchor_id in enumerate(anchor_indices):
        by_anchor[int(anchor_id)].append(idx)
    return init_states, by_anchor, init_states_path, anchor_indices


def sample_per_anchor(by_anchor, per_anchor, rng):
    selected = []
    for anchor_id in sorted(by_anchor.keys()):
        indices = by_anchor[anchor_id]
        if len(indices) < per_anchor:
            raise ValueError(
                f"anchor {anchor_id} has {len(indices)} states; need {per_anchor}"
            )
        picks = rng.choice(indices, size=per_anchor, replace=False)
        selected.extend(picks.tolist())
    rng.shuffle(selected)
    return selected


@draccus.wrap()
def main(cfg: TrainConfig):
    apply_policy_config(cfg)
    if not cfg.data.demo_file:
        raise ValueError("data.demo_file is required")
    set_seed(cfg.data.seed)

    wandb = None
    if cfg.use_wandb:
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
        policy_cfg = getattr(cfg, "policy", None)
        if policy_cfg is not None and isinstance(wandb_config, dict):
            if is_dataclass(policy_cfg):
                wandb_config["policy"] = asdict(policy_cfg)
            elif isinstance(policy_cfg, dict):
                wandb_config["policy"] = dict(policy_cfg)
            else:
                wandb_config["policy"] = getattr(policy_cfg, "__dict__", str(policy_cfg))
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity or None,
            config=wandb_config,
        )
        if cfg.experiment_name:
            wandb.run.name = cfg.experiment_name

    demo_path = Path(cfg.data.demo_file).expanduser().resolve()
    if not demo_path.exists():
        raise FileNotFoundError(f"HDF5 not found: {demo_path}")

    obs_keys = [k.strip() for k in cfg.data.obs_keys.split(",") if k.strip()]
    image_keys = [k.strip() for k in cfg.data.image_keys.split(",") if k.strip()]
    all_keys = obs_keys + image_keys
    policy_name = getattr(cfg.policy, "name", "mlp").lower()
    if policy_name in ("act", "cnnmlp") and cfg.data.normalize_obs:
        print("[warn] ACT/CNNMLP policy ignores obs normalization; disabling normalize_obs.")
        cfg.data.normalize_obs = False

    save_dir = Path(cfg.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    split_path = save_dir / "split_indices.json"

    device = cfg.device if torch.cuda.is_available() else "cpu"
    val_every = int(cfg.val_every)
    rollout_every = int(cfg.rollout_every)
    if val_every < 0:
        raise ValueError("val_every must be >= 0")
    if rollout_every < 0:
        raise ValueError("rollout_every must be >= 0")
    if cfg.rollout_steps <= 0:
        raise ValueError("rollout_steps must be >= 1")
    if cfg.rollout_warmup_steps < 0:
        raise ValueError("rollout_warmup_steps must be >= 0")
    if cfg.rollout_num_procs <= 0:
        raise ValueError("rollout_num_procs must be >= 1")

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
    if wandb is not None:
        wandb.run.summary.update(
            {
                "split/train_size": len(train_idx),
                "split/val_size": len(val_idx),
                "split/eval_size": len(eval_idx),
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
    action_dim = sample["actions"].shape[-1]
    # print(f"[debug] action_dim: {action_dim}")
    exec_horizon = get_policy_param(cfg, "exec_horizon")
    if policy_name not in ("act", "cnnmlp"):
        raise ValueError(f"unsupported policy: {policy_name}")
    qpos_dim = sum(np.prod(sample["obs"][k].shape[1:]) for k in obs_keys)
    # print(f"[debug] qpos_dim: {qpos_dim}")
    image_shapes = {}
    for key in image_keys:
        if key not in sample["obs"]:
            raise KeyError(f"image key not found in obs: {key}")
        image_shapes[key] = sample["obs"][key].shape[1:]
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
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    rollout_state = None
    if rollout_every > 0:
        if cfg.rollout_per_anchor <= 0:
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

    best_val = None
    printed_batch = False
    for epoch in range(1, cfg.epochs + 1):
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
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
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
        avg_val_stats = {}

        do_val = val_loader is not None and val_every > 0 and epoch % val_every == 0
        if do_val:
            model.eval()
            val_losses = []
            val_stats = {}
            val_stat_count = 0
            with torch.no_grad():
                for batch in val_loader:
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
                    val_losses.append(loss.item())
                    if stats:
                        for key, value in stats.items():
                            val_stats[key] = val_stats.get(key, 0.0) + float(value)
                        val_stat_count += 1
            avg_val = sum(val_losses) / max(len(val_losses), 1)
            print(f"[info] epoch {epoch:03d} | val loss {avg_val:.6f}")
            if best_val is None or avg_val < best_val:
                best_val = avg_val
            if val_stat_count:
                avg_val_stats = {k: v / val_stat_count for k, v in val_stats.items()}

        if rollout_runner is not None and epoch % rollout_every == 0:
            rng = np.random.default_rng(cfg.data.seed + epoch)
            rollout_indices = sample_per_anchor(
                rollout_state["anchor_map"], cfg.rollout_per_anchor, rng
            )
            rollout_cfg = SimpleNamespace(
                data=cfg.data,
                steps=int(cfg.rollout_steps),
                warmup_steps=int(cfg.rollout_warmup_steps),
                n_eval=len(rollout_indices),
                sample_index=0,
                save_videos=0,
                video_camera="",
                video_fps=30,
                video_dir="",
                use_mp=bool(cfg.rollout_use_mp),
                num_procs=int(cfg.rollout_num_procs),
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
            for key, value in avg_val_stats.items():
                log_data[f"val/{key}"] = value
            wandb.log(log_data, step=epoch)

        ckpt_path = save_dir / "model_last.pt"
        torch.save({"model": model.state_dict(), "obs_stats": obs_stats}, ckpt_path)

    print(f"[info] finished training. ckpt saved to {ckpt_path}")
    if wandb is not None:
        if best_val is not None:
            wandb.run.summary["best_val_loss"] = best_val
        wandb.run.summary["ckpt_path"] = str(ckpt_path)
        wandb.finish()


if __name__ == "__main__":
    main()

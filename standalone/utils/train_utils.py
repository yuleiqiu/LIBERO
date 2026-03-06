import copy
import math
import json
import re
import shlex
import sys
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch

from standalone.configs import TrainConfig, apply_policy_config, serialize_policy_config
from standalone.configs.data import DataConfig
from standalone.utils.bddl_path_utils import (
    canonicalize_bddl_file_name,
    read_bddl_from_hdf5 as _read_bddl_from_hdf5,
    resolve_bddl_path as _resolve_bddl_path,
)

TRAIN_CONFIG_NAME = "train_config.json"
_RUN_DIR_PATTERN = re.compile(r"^run_(\d+)$")

RESUME_OVERRIDE_ALLOWLIST = [
    "training.device",
    "logging.use_wandb",
    "logging.wandb_project",
    "logging.wandb_entity",
    "logging.experiment_name",
    "logging.wandb_group",
    "logging.wandb_tags",
]


# --- Run directory helpers ---
def resolve_run_dir(save_dir: Path) -> Path:
    """Resolve or create a numbered run directory under save_dir."""
    save_dir = save_dir.expanduser()
    if _RUN_DIR_PATTERN.match(save_dir.name):
        return save_dir.resolve()
    base_dir = save_dir
    base_dir.mkdir(parents=True, exist_ok=True)
    run_ids = []
    for entry in base_dir.iterdir():
        if not entry.is_dir():
            continue
        match = _RUN_DIR_PATTERN.match(entry.name)
        if match:
            run_ids.append(int(match.group(1)))
    next_id = max(run_ids, default=-1) + 1
    return (base_dir / f"run_{next_id:03d}").resolve()


# --- Config serialization & merge helpers ---
def serialize_config(cfg: Any) -> Dict[str, Any]:
    """Convert a config object or dataclass to a plain dict."""
    if is_dataclass(cfg):
        return asdict(cfg)
    if isinstance(cfg, dict):
        return dict(cfg)
    return getattr(cfg, "__dict__", {"value": str(cfg)})


def _apply_dict_to_obj(obj: Any, data: Mapping[str, Any]) -> None:
    """Recursively apply a dict to a dataclass-like object."""
    for key, value in data.items():
        if not hasattr(obj, key):
            continue
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply_dict_to_obj(current, value)
        else:
            setattr(obj, key, value)


def apply_config_overrides(cfg: Any, data: Mapping[str, Any]) -> Any:
    """Apply a nested dict onto a config object and return it."""
    if isinstance(data, dict):
        _apply_dict_to_obj(cfg, data)
    return cfg


def load_config_json(path: Path) -> Dict[str, Any]:
    """Load a JSON config file."""
    with open(path, "r") as f:
        return json.load(f)


def _get_by_path(data: Mapping[str, Any], path: str) -> Any:
    """Get a nested dict value by dotted path; returns None if missing."""
    current = data
    for key in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _set_by_path(data: MutableMapping[str, Any], path: str, value: Any) -> None:
    """Set a nested dict value by dotted path."""
    current = data
    parts = path.split(".")
    for key in parts[:-1]:
        current = current.setdefault(key, {})
    current[parts[-1]] = value


def merge_config_overrides(
    saved: Dict[str, Any],
    overrides: Dict[str, Any],
    allowlist: Sequence[str],
    defaults: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Merge saved config with allowlisted overrides."""
    merged = copy.deepcopy(saved)
    for key in allowlist:
        override_value = _get_by_path(overrides, key)
        saved_value = _get_by_path(saved, key)
        default_value = _get_by_path(defaults, key) if defaults else None
        if defaults is not None and override_value == default_value:
            continue
        if override_value != saved_value:
            _set_by_path(merged, key, override_value)
    return merged


def reject_draccus_config() -> None:
    """Error if YAML config is used; only JSON/CLI allowed."""
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


def apply_resume_config(cfg: TrainConfig) -> Tuple[TrainConfig, Optional[Path]]:
    """Load and apply resume config if requested."""
    if not getattr(cfg, "resume", False) and not getattr(cfg, "saved_config_path", None):
        return cfg, None
    cfg_dict = serialize_config(cfg)
    config_path = None
    if cfg.saved_config_path:
        config_path = Path(cfg.saved_config_path).expanduser().resolve()
    if config_path is None and getattr(cfg, "resume", False):
        config_path = Path(cfg.paths.save_dir).expanduser().resolve() / TRAIN_CONFIG_NAME
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
    defaults_dict = serialize_config(TrainConfig())
    merged_cfg = merge_config_overrides(
        saved_cfg, cfg_dict, RESUME_OVERRIDE_ALLOWLIST, defaults=defaults_dict
    )
    merged_cfg["saved_config_path"] = str(config_path)
    merged_cfg["resume"] = bool(getattr(cfg, "resume", False))
    apply_config_overrides(cfg, merged_cfg)
    if getattr(cfg, "resume", False):
        cfg.paths.save_dir = str(config_path.parent)
    return cfg, config_path


def prepare_train_config(cfg: TrainConfig) -> Tuple[TrainConfig, Path, Optional[Path]]:
    """Apply resume logic, policy config, and resolve save_dir."""
    reject_draccus_config()
    cfg, config_path = apply_resume_config(cfg)
    apply_policy_config(cfg)
    if getattr(cfg.data, "sync_horizons_from_policy", False):
        policy_name = str(getattr(cfg.policy, "name", "")).lower()
        if policy_name == "dp":
            data_defaults = DataConfig()
            dp_model = getattr(cfg.policy, "dp", None)
            dp_model = getattr(dp_model, "model", None)
            if dp_model is not None:
                if (
                    cfg.data.obs_horizon == data_defaults.obs_horizon
                    and getattr(dp_model, "n_obs_steps", None) is not None
                ):
                    cfg.data.obs_horizon = int(dp_model.n_obs_steps)
                if (
                    cfg.data.predict_horizon == data_defaults.predict_horizon
                    and getattr(dp_model, "n_action_steps", None) is not None
                ):
                    cfg.data.predict_horizon = int(dp_model.n_action_steps)
    if not cfg.data.demo_file:
        raise ValueError("data.demo_file is required")

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

    if not cfg.resume:
        cfg_dict = serialize_config(cfg)
        if isinstance(cfg_dict, dict):
            cfg_dict["policy"] = serialize_policy_config(cfg)
        write_run_metadata(save_dir, cfg, cfg_dict=cfg_dict)

    return cfg, save_dir, config_path


# --- Run metadata helpers ---
def write_run_metadata(
    save_dir: Path, cfg: Any, cfg_dict: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Write config, command, and timestamp files to save_dir."""
    if cfg_dict is None:
        cfg_dict = serialize_config(cfg)
    with open(save_dir / TRAIN_CONFIG_NAME, "w") as f:
        json.dump(cfg_dict, f, indent=2)
    cmd = " ".join(shlex.quote(arg) for arg in sys.argv)
    with open(save_dir / "command.txt", "w") as f:
        f.write(cmd + "\n")
    with open(save_dir / "run_meta.json", "w") as f:
        json.dump({"started_at": datetime.utcnow().isoformat() + "Z"}, f, indent=2)
    return cfg_dict


# --- Dataset split helpers ---
def make_split_indices(
    dataset_len: int, train_ratio: float, val_ratio: float, seed: int
) -> Tuple[List[int], List[int]]:
    """Create train/val split indices with a fixed seed."""
    assert train_ratio + val_ratio <= 1.0 + 1e-8
    train_size = int(dataset_len * train_ratio)
    val_size = int(dataset_len * val_ratio)
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_len, generator=g).tolist()
    train_idx = indices[:train_size]
    val_idx = indices[train_size : train_size + val_size]
    return train_idx, val_idx


def make_episode_split_keys(
    demo_keys: Sequence[str], val_ratio: float, seed: int
) -> Tuple[List[str], List[str]]:
    """Create train/val episode splits with a fixed seed.

    Split semantics:
    - val_ratio is applied over episode count.
    - train_ratio is derived as (1 - val_ratio), with any flooring remainder in train.
    - When val_ratio > 0, val must contain at least one episode.
    """
    if val_ratio < 0 or val_ratio >= 1.0:
        raise ValueError("val_ratio must be in [0, 1)")

    episodes = sorted(str(key) for key in demo_keys)
    num_episodes = len(episodes)
    if num_episodes == 0:
        raise ValueError("no episodes found in dataset")
    if val_ratio > 0 and num_episodes < 2:
        raise ValueError("val_ratio > 0 requires at least 2 episodes")

    n_val = int(num_episodes * val_ratio)
    if val_ratio > 0:
        n_val = max(n_val, 1)
    n_train = num_episodes - n_val

    if n_val >= num_episodes:
        n_val = num_episodes - 1
        n_train = 1
    if n_train <= 0:
        raise ValueError("train split is empty; adjust ratios or dataset")
    if val_ratio > 0 and n_val <= 0:
        raise ValueError("val split is empty; adjust ratios or dataset")

    g = torch.Generator().manual_seed(seed)
    shuffled = [episodes[i] for i in torch.randperm(num_episodes, generator=g).tolist()]
    train_episodes = shuffled[:n_train]
    val_episodes = shuffled[n_train : n_train + n_val]

    overlap = set(train_episodes).intersection(val_episodes)
    if overlap:
        raise RuntimeError(f"episode leakage detected: {sorted(overlap)}")
    return train_episodes, val_episodes


# --- Rollout/init state helpers ---
def read_bddl_from_hdf5(hdf5_path: str) -> Optional[str]:
    """Read the BDDL file name from an HDF5 dataset."""
    return _read_bddl_from_hdf5(hdf5_path)


def resolve_bddl_path(bddl_file_name: Optional[str], demo_path: Path) -> Optional[str]:
    """Resolve a BDDL path from absolute, repo, or demo-relative locations."""
    return _resolve_bddl_path(bddl_file_name, demo_path)


def resolve_init_states_dir(cfg: Any) -> Path:
    """Resolve the init_states directory from cfg or libero defaults."""
    init_dir = getattr(getattr(cfg, "rollout", None), "init_states_dir", None)
    if init_dir is None:
        init_dir = getattr(cfg, "rollout_init_states_dir", None)
    if init_dir:
        return Path(init_dir).expanduser().resolve()
    from libero.libero import get_libero_path

    return Path(get_libero_path("init_states")).expanduser().resolve()


def resolve_rollout_bddl_path(cfg: Any, demo_path: Path) -> Tuple[Path, Optional[str], str]:
    """Resolve the rollout BDDL path from explicit override or HDF5 metadata."""
    rollout_cfg = getattr(cfg, "rollout", None)
    bddl_override = getattr(rollout_cfg, "bddl_file", None)
    if bddl_override:
        bddl_path = resolve_bddl_path(bddl_override, demo_path)
        if bddl_path is None:
            raise FileNotFoundError(f"rollout.bddl_file not found: {bddl_override}")
        return Path(bddl_path), canonicalize_bddl_file_name(bddl_override), "rollout.bddl_file"

    bddl_file_name = read_bddl_from_hdf5(str(demo_path))
    if bddl_file_name is None:
        raise ValueError("bddl_file_name not found in hdf5; cannot resolve rollout bddl")
    bddl_path = resolve_bddl_path(bddl_file_name, demo_path)
    if bddl_path is None:
        raise FileNotFoundError(f"bddl file not found: {bddl_file_name}")
    return Path(bddl_path), canonicalize_bddl_file_name(bddl_file_name), "data.attrs[bddl_file_name]"


def resolve_rollout_init_states_path(
    cfg: Any, demo_path: Path, bddl_path: Optional[Path] = None
) -> Tuple[Path, str]:
    """Resolve training-time rollout init-states path from override or BDDL-derived path."""
    rollout_cfg = getattr(cfg, "rollout", None)
    init_states_file = getattr(rollout_cfg, "init_states_file", None)
    if init_states_file:
        init_states_path = Path(init_states_file).expanduser().resolve()
        if not init_states_path.exists():
            raise FileNotFoundError(
                f"rollout.init_states_file not found: {init_states_path}"
            )
        return init_states_path, "rollout.init_states_file"

    if bddl_path is None:
        bddl_path, _, _ = resolve_rollout_bddl_path(cfg, demo_path)
    init_dir = resolve_init_states_dir(cfg)
    init_states_path = init_dir / bddl_path.parent.name / f"{bddl_path.stem}.pruned_init"
    return init_states_path, "rollout.init_states_dir"


def rollout_sanity_check(cfg: Any, demo_path: Path) -> Dict[str, Any]:
    """Resolve rollout BDDL/init paths early and warn on suspicious mismatches."""
    details: Dict[str, Any] = {"demo_path": str(demo_path)}
    try:
        bddl_path, bddl_ref, bddl_source = resolve_rollout_bddl_path(cfg, demo_path)
        init_states_path, init_source = resolve_rollout_init_states_path(
            cfg, demo_path, bddl_path=bddl_path
        )
    except Exception as exc:
        print(f"[warning] rollout sanity check failed: {exc}")
        details["error"] = str(exc)
        return details

    expected_init_path = (
        resolve_init_states_dir(cfg)
        / bddl_path.parent.name
        / f"{bddl_path.stem}.pruned_init"
    )
    anchors_meta = init_states_path.with_suffix(init_states_path.suffix + ".anchors.json")

    details.update(
        {
            "bddl_path": str(bddl_path),
            "bddl_ref": bddl_ref,
            "bddl_source": bddl_source,
            "init_states_path": str(init_states_path),
            "init_source": init_source,
            "expected_init_states_path": str(expected_init_path),
            "anchors_meta": str(anchors_meta),
        }
    )

    print("[info] rollout path sanity:")
    print(f"  bddl ({bddl_source}): {bddl_path}")
    print(f"  init ({init_source}): {init_states_path}")
    print(f"  anchors: {anchors_meta}")

    if not init_states_path.exists():
        print(f"[warning] rollout init states missing: {init_states_path}")
    if not anchors_meta.exists():
        print(f"[warning] rollout anchors meta missing: {anchors_meta}")
    if init_states_path != expected_init_path:
        print(
            "[warning] rollout init path does not match the BDDL-derived default: "
            f"{init_states_path} != {expected_init_path}"
        )
    if bddl_source != "data.attrs[bddl_file_name]":
        hdf5_bddl_ref = canonicalize_bddl_file_name(read_bddl_from_hdf5(str(demo_path)))
        if hdf5_bddl_ref and hdf5_bddl_ref != bddl_ref:
            print(
                "[warning] rollout BDDL override differs from HDF5 metadata: "
                f"{bddl_ref} != {hdf5_bddl_ref}"
            )
    return details


def load_init_states_with_anchors(
    cfg: Any, demo_path: Path
) -> Tuple[np.ndarray, Dict[int, List[int]], Path, List[int]]:
    """Load init states and anchor indices for rollout evaluation."""
    bddl_path, _, _ = resolve_rollout_bddl_path(cfg, demo_path)
    init_states_path, _ = resolve_rollout_init_states_path(cfg, demo_path, bddl_path=bddl_path)
    if not init_states_path.exists():
        raise FileNotFoundError(f"init states file not found: {init_states_path}")
    init_states = torch.load(str(init_states_path), weights_only=False)
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


def sample_per_anchor(
    by_anchor: Mapping[int, Sequence[int]],
    per_anchor: int,
    rng: np.random.Generator,
) -> List[int]:
    """Sample a fixed number of indices per anchor."""
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


# --- Optimizer helpers ---
def _resolve_opt_value(
    base_cfg: Any, override_cfg: Any, name: str, default: Any
) -> Any:
    """Select an optimizer field from override, then base, then default."""
    value = getattr(override_cfg, name, None) if override_cfg is not None else None
    if value is None and base_cfg is not None:
        value = getattr(base_cfg, name, None)
    if value is None:
        value = default
    return value


def _get_policy_optimizer_cfg(cfg: Any, policy_name: str) -> Any:
    """Return the optimizer config for a policy name."""
    if policy_name == "act":
        return getattr(cfg.policy.act, "optimizer", None)
    if policy_name == "dp":
        return getattr(cfg.policy.dp, "optimizer", None)
    return None


def _get_policy_scheduler_cfg(cfg: Any, policy_name: str) -> Any:
    """Return the scheduler config for a policy name."""
    if policy_name == "act":
        return getattr(cfg.policy.act, "scheduler", None)
    if policy_name == "dp":
        return getattr(cfg.policy.dp, "scheduler", None)
    return None


def build_optimizer(
    cfg: Any, model: torch.nn.Module, policy_name: str
) -> torch.optim.Optimizer:
    """Build an AdamW optimizer; ACT may split backbone params."""
    base_opt_cfg = getattr(cfg.training, "optimizer", None)
    policy_opt_cfg = _get_policy_optimizer_cfg(cfg, policy_name)

    base_lr = _resolve_opt_value(base_opt_cfg, policy_opt_cfg, "lr", None)
    if base_lr is None:
        base_lr = cfg.training.lr
    weight_decay = float(
        _resolve_opt_value(base_opt_cfg, policy_opt_cfg, "weight_decay", 1e-4)
    )
    betas = _resolve_opt_value(base_opt_cfg, policy_opt_cfg, "betas", [0.9, 0.999])
    eps = float(_resolve_opt_value(base_opt_cfg, policy_opt_cfg, "eps", 1e-8))

    if policy_name == "act":
        lr_backbone = float(getattr(cfg.policy.act, "lr_backbone", 0.0) or 0.0)
        backbones = getattr(getattr(model, "model", None), "backbones", None)
        backbone_params = []
        if lr_backbone > 0 and backbones is not None:
            backbone_params = [p for p in backbones.parameters() if p.requires_grad]
        if backbone_params:
            backbone_ids = {id(p) for p in backbone_params}
            other_params = [
                p
                for p in model.parameters()
                if p.requires_grad and id(p) not in backbone_ids
            ]
            return torch.optim.AdamW(
                [
                    {"params": other_params, "lr": base_lr, "weight_decay": weight_decay},
                    {
                        "params": backbone_params,
                        "lr": lr_backbone,
                        "weight_decay": weight_decay,
                    },
                ],
                betas=tuple(betas),
                eps=eps,
            )
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=base_lr,
        weight_decay=weight_decay,
        betas=tuple(betas),
        eps=eps,
    )


def build_scheduler(
    cfg: Any, optimizer: torch.optim.Optimizer, policy_name: str, total_steps: int
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Build a learning rate scheduler. Returns None when disabled."""
    base_cfg = getattr(cfg.training, "scheduler", None)
    policy_cfg = _get_policy_scheduler_cfg(cfg, policy_name)
    name = str(_resolve_opt_value(base_cfg, policy_cfg, "name", "none") or "none").lower()
    if name in ("none", "null", "disable", "disabled"):
        return None
    warmup_steps = int(_resolve_opt_value(base_cfg, policy_cfg, "warmup_steps", 0) or 0)
    min_lr = float(_resolve_opt_value(base_cfg, policy_cfg, "min_lr", 0.0) or 0.0)
    max_steps = _resolve_opt_value(base_cfg, policy_cfg, "num_training_steps", None)
    max_steps = int(max_steps) if max_steps is not None else int(total_steps)
    if max_steps <= 0:
        return None

    base_lrs = [group["lr"] for group in optimizer.param_groups]

    def _schedule(step: int) -> float:
        """Return the learning-rate scale factor for a given step."""
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(warmup_steps, 1))
        progress = (step - warmup_steps) / float(max(max_steps - warmup_steps, 1))
        progress = min(max(progress, 0.0), 1.0)
        if name == "linear":
            return 1.0 - progress
        if name == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        if name == "constant":
            return 1.0
        raise ValueError(f"unsupported scheduler name: {name}")

    if min_lr > 0.0:
        min_factors = [
            (min_lr / lr) if lr > 0 else 0.0 for lr in base_lrs
        ]

        def _make_lambda(min_factor: float):
            """Create a schedule function with a minimum lr clamp."""
            return lambda step: max(min_factor, _schedule(step))

        lr_lambdas = [_make_lambda(mf) for mf in min_factors]
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambdas)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_schedule)

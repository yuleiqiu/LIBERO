import copy
import json
import re
import shlex
import sys
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch

TRAIN_CONFIG_NAME = "train_config.json"
_RUN_DIR_PATTERN = re.compile(r"^run_(\d+)$")


def resolve_run_dir(save_dir: Path) -> Path:
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


def cfg_to_dict(cfg):
    if is_dataclass(cfg):
        return asdict(cfg)
    if isinstance(cfg, dict):
        return dict(cfg)
    return getattr(cfg, "__dict__", {"value": str(cfg)})


def _apply_dict_to_obj(obj, data):
    for key, value in data.items():
        if not hasattr(obj, key):
            continue
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply_dict_to_obj(current, value)
        else:
            setattr(obj, key, value)


def apply_config_dict(cfg, data):
    if isinstance(data, dict):
        _apply_dict_to_obj(cfg, data)
    return cfg


def load_config_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def _get_by_path(data, path):
    current = data
    for key in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _set_by_path(data, path, value):
    current = data
    parts = path.split(".")
    for key in parts[:-1]:
        current = current.setdefault(key, {})
    current[parts[-1]] = value


def merge_config_with_overrides(saved, overrides, allowlist, defaults=None):
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

def write_run_metadata(save_dir: Path, cfg, cfg_dict=None):
    if cfg_dict is None:
        cfg_dict = cfg_to_dict(cfg)
    with open(save_dir / TRAIN_CONFIG_NAME, "w") as f:
        json.dump(cfg_dict, f, indent=2)
    cmd = " ".join(shlex.quote(arg) for arg in sys.argv)
    with open(save_dir / "command.txt", "w") as f:
        f.write(cmd + "\n")
    with open(save_dir / "run_meta.json", "w") as f:
        json.dump({"started_at": datetime.utcnow().isoformat() + "Z"}, f, indent=2)
    return cfg_dict


def make_splits(dataset_len, train_ratio, val_ratio, seed):
    assert train_ratio + val_ratio <= 1.0 + 1e-8
    train_size = int(dataset_len * train_ratio)
    val_size = int(dataset_len * val_ratio)
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_len, generator=g).tolist()
    train_idx = indices[:train_size]
    val_idx = indices[train_size : train_size + val_size]
    return train_idx, val_idx


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
    repo_root = Path(__file__).resolve().parents[2]
    repo_candidate = (repo_root / "libero/libero/bddl_files" / candidate).resolve()
    if repo_candidate.exists():
        return str(repo_candidate)
    demo_candidate = (demo_path.parent / candidate).resolve()
    if demo_candidate.exists():
        return str(demo_candidate)
    return None


def resolve_init_states_dir(cfg):
    init_dir = getattr(getattr(cfg, "rollout", None), "init_states_dir", None)
    if init_dir is None:
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

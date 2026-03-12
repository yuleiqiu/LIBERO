import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Optional

from standalone.utils.rollout_utils import read_env_kwargs_from_hdf5
from standalone.utils.train_utils import (
    TRAIN_CONFIG_NAME,
    resolve_rollout_bddl_path,
    resolve_rollout_init_states_path,
)


ROLLOUT_SPEC_KEY = "rollout_spec"


def build_rollout_spec(cfg: Any, demo_path: Path) -> Dict[str, Any]:
    bddl_path, bddl_ref, _ = resolve_rollout_bddl_path(cfg, demo_path)
    init_states_path, _ = resolve_rollout_init_states_path(cfg, demo_path, bddl_path=bddl_path)
    env_kwargs = read_env_kwargs_from_hdf5(str(demo_path))
    return {
        "bddl_file": bddl_ref or str(bddl_path),
        "init_states": str(init_states_path),
        "env_kwargs": dict(env_kwargs),
    }


def load_rollout_spec(
    *, ckpt: Optional[Mapping[str, Any]] = None, run_config: Optional[Mapping[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    if isinstance(ckpt, Mapping):
        rollout_spec = ckpt.get(ROLLOUT_SPEC_KEY)
        if isinstance(rollout_spec, Mapping):
            return dict(rollout_spec)
    if isinstance(run_config, Mapping):
        rollout_spec = run_config.get(ROLLOUT_SPEC_KEY)
        if isinstance(rollout_spec, Mapping):
            return dict(rollout_spec)
    return None


def apply_rollout_spec_overrides(cfg: Any, rollout_spec: Optional[Mapping[str, Any]]) -> bool:
    if not isinstance(rollout_spec, Mapping):
        return False
    applied = False
    if not getattr(cfg, "bddl_file", None):
        bddl_file = rollout_spec.get("bddl_file")
        if bddl_file:
            cfg.bddl_file = str(bddl_file)
            applied = True
    if not getattr(cfg, "init_states", None):
        init_states = rollout_spec.get("init_states")
        if init_states:
            cfg.init_states = str(init_states)
            applied = True
    return applied


def get_rollout_env_kwargs(rollout_spec: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not isinstance(rollout_spec, Mapping):
        return {}
    env_kwargs = rollout_spec.get("env_kwargs")
    if not isinstance(env_kwargs, Mapping):
        return {}
    return {str(key): env_kwargs[key] for key in env_kwargs}


def write_rollout_spec_to_run_config(save_dir: Path, rollout_spec: Mapping[str, Any]) -> None:
    config_path = Path(save_dir) / TRAIN_CONFIG_NAME
    if not config_path.exists():
        return
    with open(config_path, "r") as f:
        cfg_dict = json.load(f)
    cfg_dict[ROLLOUT_SPEC_KEY] = dict(rollout_spec)
    with open(config_path, "w") as f:
        json.dump(cfg_dict, f, indent=2)

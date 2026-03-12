import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


MODEL_SPEC_KEY = "model_spec"
TRAIN_CONFIG_NAME = "train_config.json"


def _shape_to_list(shape: Sequence[int]) -> list[int]:
    return [int(dim) for dim in shape]


def _shape_map_to_lists(shape_map: Mapping[str, Sequence[int]]) -> Dict[str, list[int]]:
    return {
        str(key): _shape_to_list(shape)
        for key, shape in shape_map.items()
    }


def _shape_map_to_tuples(shape_map: Mapping[str, Sequence[int]]) -> Dict[str, Tuple[int, ...]]:
    return {
        str(key): tuple(int(dim) for dim in shape)
        for key, shape in shape_map.items()
    }


def build_model_spec(
    *,
    action_dim: int,
    proprio_dim: int,
    obs_shapes: Mapping[str, Sequence[int]],
    image_shapes: Mapping[str, Sequence[int]],
    obs_keys: Sequence[str],
    image_keys: Sequence[str],
    mask_keys: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    return {
        "action_dim": int(action_dim),
        "proprio_dim": int(proprio_dim),
        "obs_shapes": _shape_map_to_lists(obs_shapes),
        "image_shapes": _shape_map_to_lists(image_shapes),
        "obs_keys": [str(key) for key in obs_keys],
        "image_keys": [str(key) for key in image_keys],
        "mask_keys": [str(key) for key in (mask_keys or [])],
    }


def load_model_spec(
    *, ckpt: Optional[Mapping[str, Any]] = None, run_config: Optional[Mapping[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    if isinstance(ckpt, Mapping):
        model_spec = ckpt.get(MODEL_SPEC_KEY)
        if isinstance(model_spec, Mapping):
            return dict(model_spec)
    if isinstance(run_config, Mapping):
        model_spec = run_config.get(MODEL_SPEC_KEY)
        if isinstance(model_spec, Mapping):
            return dict(model_spec)
    return None


def unpack_model_spec(model_spec: Mapping[str, Any]) -> Tuple[int, Dict[str, Tuple[int, ...]], Dict[str, Tuple[int, ...]], int]:
    if not isinstance(model_spec, Mapping):
        raise TypeError(f"model_spec must be a mapping, got {type(model_spec)}")
    if "action_dim" not in model_spec:
        raise KeyError("model_spec missing action_dim")
    if "obs_shapes" not in model_spec:
        raise KeyError("model_spec missing obs_shapes")

    action_dim = int(model_spec["action_dim"])
    proprio_dim = int(model_spec.get("proprio_dim", 0))
    obs_shapes = _shape_map_to_tuples(model_spec.get("obs_shapes", {}))
    image_shapes = _shape_map_to_tuples(model_spec.get("image_shapes", {}))
    return action_dim, image_shapes, obs_shapes, proprio_dim


def write_model_spec_to_run_config(save_dir: Path, model_spec: Mapping[str, Any]) -> None:
    config_path = Path(save_dir) / TRAIN_CONFIG_NAME
    if not config_path.exists():
        return
    with open(config_path, "r") as f:
        cfg_dict = json.load(f)
    cfg_dict[MODEL_SPEC_KEY] = dict(model_spec)
    with open(config_path, "w") as f:
        json.dump(cfg_dict, f, indent=2)

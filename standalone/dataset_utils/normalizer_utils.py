import math
from typing import Dict, Iterable

import numpy as np
import torch

from standalone.models.algos.dp.utils.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)


def _flatten_last_dims(arr: np.ndarray, last_n_dims: int) -> np.ndarray:
    if last_n_dims < 0:
        raise ValueError("last_n_dims must be >= 0")
    if last_n_dims == 0:
        return arr.reshape(-1, 1)
    flat_dim = int(np.prod(arr.shape[-last_n_dims:]))
    return arr.reshape(-1, flat_dim)


def _init_stats(dim: int) -> Dict[str, np.ndarray]:
    return {
        "min": np.full((dim,), np.inf, dtype=np.float64),
        "max": np.full((dim,), -np.inf, dtype=np.float64),
        "sum": np.zeros((dim,), dtype=np.float64),
        "sumsq": np.zeros((dim,), dtype=np.float64),
        "count": np.array(0, dtype=np.int64),
    }


def _update_stats(stats: Dict[str, np.ndarray], data: np.ndarray) -> None:
    stats["min"] = np.minimum(stats["min"], data.min(axis=0))
    stats["max"] = np.maximum(stats["max"], data.max(axis=0))
    stats["sum"] += data.sum(axis=0)
    stats["sumsq"] += (data * data).sum(axis=0)
    stats["count"] += data.shape[0]


def compute_linear_stats(
    dataset,
    indices: Iterable[int],
    obs_keys,
    image_keys=None,
    action_key: str = "actions",
    last_n_dims: int = 1,
    include_actions: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    image_keys = set(image_keys or [])
    obs_keys = [k for k in obs_keys if k not in image_keys]
    stats = {}
    total = len(indices) if hasattr(indices, "__len__") else None
    checkpoints = []
    checkpoint_ptr = 0
    if total is not None and total > 0:
        checkpoints = [math.ceil(total * frac) for frac in (0.25, 0.5, 0.75, 1.0)]

    restore_transforms = None
    if hasattr(dataset, "set_image_transforms_enabled"):
        restore_transforms = dataset.image_transforms_enabled()
        dataset.set_image_transforms_enabled(False)

    try:
        for processed, idx in enumerate(indices, start=1):
            sample = dataset[idx]
            obs = sample["obs"]
            for key in obs_keys:
                if key not in obs:
                    continue
                data = np.asarray(obs[key], dtype=np.float32)
                flat = _flatten_last_dims(data, last_n_dims)
                if key not in stats:
                    stats[key] = _init_stats(flat.shape[1])
                _update_stats(stats[key], flat)

            if include_actions:
                actions = np.asarray(sample[action_key], dtype=np.float32)
                action_mask = sample.get("action_mask")
                if action_mask is not None:
                    action_mask = np.asarray(action_mask).astype(bool)
                    if action_mask.shape[0] == actions.shape[0]:
                        actions = actions[action_mask]
                    else:
                        raise ValueError(
                            f"action_mask shape {action_mask.shape} does not match actions {actions.shape}"
                        )
                if actions.size == 0:
                    continue
                flat = _flatten_last_dims(actions, last_n_dims)
                if "action" not in stats:
                    stats["action"] = _init_stats(flat.shape[1])
                _update_stats(stats["action"], flat)
            while checkpoint_ptr < len(checkpoints) and processed >= checkpoints[checkpoint_ptr]:
                pct = (checkpoint_ptr + 1) * 25
                print(
                    f"[info] compute normalizer stats: {pct}% ({processed}/{total})"
                )
                checkpoint_ptr += 1
    finally:
        if restore_transforms is not None:
            dataset.set_image_transforms_enabled(restore_transforms)

    output = {}
    for key, raw in stats.items():
        count = max(int(raw["count"]), 1)
        mean = raw["sum"] / count
        var = raw["sumsq"] / count - mean * mean
        var = np.maximum(var, 0.0)
        std = np.sqrt(var)
        output[key] = {
            "min": raw["min"].astype(np.float32),
            "max": raw["max"].astype(np.float32),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
        }
    return output


def build_linear_normalizer(
    stats: Dict[str, Dict[str, np.ndarray]],
    mode: str = "limits",
    output_min: float = -1.0,
    output_max: float = 1.0,
    range_eps: float = 1e-4,
    fit_offset: bool = True,
) -> LinearNormalizer:
    mode = str(mode).lower()
    normalizer = LinearNormalizer()

    for key, values in stats.items():
        input_min = torch.as_tensor(values["min"], dtype=torch.float32)
        input_max = torch.as_tensor(values["max"], dtype=torch.float32)
        input_mean = torch.as_tensor(values["mean"], dtype=torch.float32)
        input_std = torch.as_tensor(values["std"], dtype=torch.float32)

        if mode == "limits":
            if fit_offset:
                input_range = input_max - input_min
                ignore = input_range < range_eps
                safe_range = torch.where(
                    ignore,
                    torch.full_like(input_range, output_max - output_min),
                    input_range,
                )
                scale = (output_max - output_min) / safe_range
                offset = output_min - scale * input_min
                offset[ignore] = (output_max + output_min) / 2 - input_min[ignore]
            else:
                output_abs = min(abs(output_min), abs(output_max))
                input_abs = torch.maximum(input_min.abs(), input_max.abs())
                ignore = input_abs < range_eps
                input_abs[ignore] = output_abs
                scale = output_abs / input_abs
                offset = torch.zeros_like(input_mean)
        elif mode == "gaussian":
            ignore = input_std < range_eps
            scale = input_std.clone()
            scale[ignore] = 1.0
            scale = 1.0 / scale
            if fit_offset:
                offset = -input_mean * scale
            else:
                offset = torch.zeros_like(input_mean)
        else:
            raise ValueError(f"unsupported normalizer mode: {mode}")

        input_stats = {
            "min": input_min,
            "max": input_max,
            "mean": input_mean,
            "std": input_std,
        }
        field = SingleFieldLinearNormalizer.create_manual(
            scale=scale,
            offset=offset,
            input_stats_dict=input_stats,
        )
        normalizer[key] = field

    return normalizer


def build_identity_normalizer(
    obs_shapes: Dict[str, tuple],
    obs_keys,
    action_dim: int,
    last_n_dims: int = 1,
    include_actions: bool = True,
) -> LinearNormalizer:
    normalizer = LinearNormalizer()
    for key in obs_keys:
        shape = tuple(obs_shapes[key])
        dim = int(np.prod(shape[-last_n_dims:])) if last_n_dims > 0 else 1
        scale = torch.ones(dim, dtype=torch.float32)
        offset = torch.zeros(dim, dtype=torch.float32)
        stats = {
            "min": torch.full((dim,), -1.0, dtype=torch.float32),
            "max": torch.full((dim,), 1.0, dtype=torch.float32),
            "mean": torch.zeros(dim, dtype=torch.float32),
            "std": torch.ones(dim, dtype=torch.float32),
        }
        normalizer[key] = SingleFieldLinearNormalizer.create_manual(
            scale=scale,
            offset=offset,
            input_stats_dict=stats,
        )
    if include_actions:
        dim = int(action_dim)
        scale = torch.ones(dim, dtype=torch.float32)
        offset = torch.zeros(dim, dtype=torch.float32)
        stats = {
            "min": torch.full((dim,), -1.0, dtype=torch.float32),
            "max": torch.full((dim,), 1.0, dtype=torch.float32),
            "mean": torch.zeros(dim, dtype=torch.float32),
            "std": torch.ones(dim, dtype=torch.float32),
        }
        normalizer["action"] = SingleFieldLinearNormalizer.create_manual(
            scale=scale,
            offset=offset,
            input_stats_dict=stats,
        )
    return normalizer

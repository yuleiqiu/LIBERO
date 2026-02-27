#!/usr/bin/env python3
"""Inspect DP normalizer parameters and empirical pre/post normalization stats."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np
import torch

from standalone.dataset_utils.hdf5_sequence_dataset import HDF5SequenceDataset
from standalone.dataset_utils.normalizer_utils import (
    build_identity_normalizer,
    build_linear_normalizer,
    compute_linear_stats,
)
from standalone.models.algos.dp.utils.normalizer import LinearNormalizer
from standalone.utils.train_utils import TRAIN_CONFIG_NAME, make_episode_split_keys


class RunningStats:
    def __init__(self, dim: int):
        self.dim = int(dim)
        self.min = np.full((dim,), np.inf, dtype=np.float64)
        self.max = np.full((dim,), -np.inf, dtype=np.float64)
        self.sum = np.zeros((dim,), dtype=np.float64)
        self.sumsq = np.zeros((dim,), dtype=np.float64)
        self.count = 0

    def update(self, data: np.ndarray) -> None:
        if data.size == 0:
            return
        self.min = np.minimum(self.min, data.min(axis=0))
        self.max = np.maximum(self.max, data.max(axis=0))
        self.sum += data.sum(axis=0)
        self.sumsq += (data * data).sum(axis=0)
        self.count += int(data.shape[0])

    def finalize(self) -> Dict[str, np.ndarray]:
        if self.count == 0:
            nan = np.full((self.dim,), np.nan, dtype=np.float32)
            return {"min": nan, "max": nan, "mean": nan, "std": nan}
        mean = self.sum / self.count
        var = np.maximum(self.sumsq / self.count - mean * mean, 0.0)
        std = np.sqrt(var)
        return {
            "min": self.min.astype(np.float32),
            "max": self.max.astype(np.float32),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
        }


def _flatten_last_dims(arr: np.ndarray, last_n_dims: int) -> np.ndarray:
    if last_n_dims < 0:
        raise ValueError("last_n_dims must be >= 0")
    if last_n_dims == 0:
        return arr.reshape(-1, 1)
    flat_dim = int(np.prod(arr.shape[-last_n_dims:]))
    return arr.reshape(-1, flat_dim)


def _split_csv(text: str) -> List[str]:
    return [k.strip() for k in str(text or "").split(",") if k.strip()]


def _format_arr(arr: np.ndarray, max_dims: int) -> str:
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return f"{float(arr):.6f}"
    if arr.shape[0] <= max_dims:
        return np.array2string(arr, precision=6, floatmode="fixed")
    head = arr[:max_dims]
    return f"{np.array2string(head, precision=6, floatmode='fixed')} ... (dim={arr.shape[0]})"


def _parse_config(cfg_path: Path) -> dict:
    with open(cfg_path, "r") as f:
        raw = json.load(f)
    policy = raw.get("policy") or {}
    if str(policy.get("name", "")).lower() != "dp":
        raise ValueError("config policy.name is not 'dp'")
    if "dp" not in policy:
        raise ValueError("config missing policy.dp")
    return raw


def _load_train_episodes(split_path: Path) -> Optional[List[str]]:
    if not split_path.exists():
        return None
    with open(split_path, "r") as f:
        split = json.load(f)
    episodes = split.get("train_episodes")
    if isinstance(episodes, list) and episodes:
        return [str(x) for x in episodes]
    return None


def _list_all_demo_keys(demo_path: Path) -> List[str]:
    with h5py.File(str(demo_path), "r") as f:
        data_group = f.get("data")
        if data_group is None:
            raise KeyError("HDF5 missing 'data' group")
        keys = sorted(str(k) for k in data_group.keys())
    if not keys:
        raise ValueError(f"no demos found in {demo_path}")
    return keys


def _resolve_train_episodes(
    raw_cfg: dict,
    config_path: Path,
    demo_path: Path,
    split_source: str,
    val_ratio_override: Optional[float],
    split_seed_override: Optional[int],
) -> Tuple[Optional[List[str]], str]:
    split_source = str(split_source).lower()
    data_cfg = raw_cfg.get("data") or {}
    split_path = config_path.parent / "split_indices.json"

    if split_source not in {"auto", "file", "recompute", "all"}:
        raise ValueError(f"unsupported split_source: {split_source}")

    if split_source in {"auto", "file"}:
        file_episodes = _load_train_episodes(split_path)
        if file_episodes is not None:
            return file_episodes, f"split file: {split_path}"
        if split_source == "file":
            raise FileNotFoundError(f"split file missing or invalid: {split_path}")

    if split_source == "all":
        return None, "all demos (no split)"

    val_ratio = float(data_cfg.get("val_ratio", 0.0))
    seed = int(data_cfg.get("seed", 0))
    if val_ratio_override is not None:
        val_ratio = float(val_ratio_override)
    if split_seed_override is not None:
        seed = int(split_seed_override)

    all_demo_keys = _list_all_demo_keys(demo_path)
    train_episodes, _ = make_episode_split_keys(all_demo_keys, val_ratio=val_ratio, seed=seed)
    source = f"recomputed split (val_ratio={val_ratio}, seed={seed})"
    return train_episodes, source


def _resolve_dp_action_slice(raw_cfg: dict) -> Tuple[Optional[int], Optional[int]]:
    data_cfg = raw_cfg.get("data") or {}
    policy_dp = (raw_cfg.get("policy") or {}).get("dp") or {}
    model_cfg = policy_dp.get("model") or {}

    action_horizon = policy_dp.get("action_horizon")
    if action_horizon is None:
        action_horizon = model_cfg.get("horizon")

    action_start_offset = policy_dp.get("action_start_offset")
    obs_horizon = int(data_cfg.get("obs_horizon", 1))
    if action_start_offset is None and action_horizon is not None:
        action_start_offset = -(obs_horizon - 1)

    if action_horizon is not None:
        action_horizon = int(action_horizon)
    if action_start_offset is not None:
        action_start_offset = int(action_start_offset)
    return action_horizon, action_start_offset


def _build_dataset(raw_cfg: dict, demo_file_override: str, train_episodes: Optional[List[str]]):
    data_cfg = raw_cfg.get("data") or {}

    demo_file = str(demo_file_override or data_cfg.get("demo_file", "")).strip()
    if not demo_file:
        raise ValueError("demo file is empty; pass --demo-file or set data.demo_file in config")
    demo_path = Path(demo_file).expanduser().resolve()
    if not demo_path.exists():
        raise FileNotFoundError(f"demo file not found: {demo_path}")

    obs_keys = _split_csv(data_cfg.get("obs_keys", ""))
    image_keys = _split_csv(data_cfg.get("image_keys", ""))
    mask_keys_raw = _split_csv(data_cfg.get("mask_keys", ""))
    active_mask_keys = [k for k in mask_keys_raw if k]
    all_keys = obs_keys + image_keys + active_mask_keys

    action_horizon, action_start_offset = _resolve_dp_action_slice(raw_cfg)

    dataset = HDF5SequenceDataset(
        hdf5_path=str(demo_path),
        obs_keys=all_keys,
        obs_horizon=int(data_cfg.get("obs_horizon", 1)),
        predict_horizon=int(data_cfg.get("predict_horizon", 1)),
        action_horizon=action_horizon,
        action_start_offset=action_start_offset,
        image_keys=image_keys,
        image_norm=str(data_cfg.get("image_norm", "none")),
        demos=train_episodes,
    )
    return dataset, obs_keys, image_keys


def _build_normalizer_from_cfg(
    raw_cfg: dict,
    dataset: HDF5SequenceDataset,
    obs_keys: List[str],
    image_keys: List[str],
    indices: Iterable[int],
) -> LinearNormalizer:
    dp_cfg = (raw_cfg.get("policy") or {}).get("dp") or {}
    normalizer_cfg = dp_cfg.get("normalizer") or {}

    last_n_dims = int(normalizer_cfg.get("last_n_dims", 1))
    sample = dataset[0]
    obs_shapes = {k: v.shape for k, v in sample["obs"].items()}
    action_dim = int(sample["actions"].shape[-1])

    identity_normalizer = build_identity_normalizer(
        obs_shapes=obs_shapes,
        obs_keys=list(obs_shapes.keys()),
        action_dim=action_dim,
        last_n_dims=last_n_dims,
        include_actions=True,
    )

    if not bool(normalizer_cfg.get("enable", True)):
        return identity_normalizer

    lowdim_keys = [k for k in obs_keys if k not in set(image_keys)]
    obs_keys_for_norm = lowdim_keys if bool(normalizer_cfg.get("normalize_obs", True)) else []
    include_actions = bool(normalizer_cfg.get("normalize_actions", True))

    stats = {}
    if obs_keys_for_norm or include_actions:
        stats = compute_linear_stats(
            dataset,
            indices=indices,
            obs_keys=obs_keys_for_norm,
            image_keys=image_keys,
            last_n_dims=last_n_dims,
            include_actions=include_actions,
        )

    if not stats:
        return identity_normalizer

    normalizer = build_linear_normalizer(
        stats,
        mode=str(normalizer_cfg.get("mode", "limits")),
        output_min=float(normalizer_cfg.get("output_min", -1.0)),
        output_max=float(normalizer_cfg.get("output_max", 1.0)),
        range_eps=float(normalizer_cfg.get("range_eps", 1e-4)),
        fit_offset=bool(normalizer_cfg.get("fit_offset", True)),
    )
    for key in identity_normalizer.fields:
        if key not in normalizer.fields:
            normalizer[key] = identity_normalizer[key]
    return normalizer


def _load_normalizer_from_ckpt(ckpt_path: Path) -> LinearNormalizer:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "normalizer" not in ckpt:
        raise ValueError("checkpoint missing top-level 'normalizer' state")
    normalizer = LinearNormalizer()
    normalizer.load_state_dict(ckpt["normalizer"])
    return normalizer


def _print_param_summary(normalizer: LinearNormalizer, max_dims: int) -> None:
    print("\n=== Normalizer Parameters ===")
    for key in sorted(normalizer.fields.keys()):
        field = normalizer[key]
        params = field.params_dict
        scale = params["scale"].detach().cpu().numpy()
        offset = params["offset"].detach().cpu().numpy()
        print(f"[{key}] dim={scale.shape[0]}")
        print(f"  scale : {_format_arr(scale, max_dims)}")
        print(f"  offset: {_format_arr(offset, max_dims)}")
        for name in ("min", "max", "mean", "std"):
            stat_key = f"input_stats_{name}"
            if stat_key in params:
                value = params[stat_key].detach().cpu().numpy()
                print(f"  {stat_key}: {_format_arr(value, max_dims)}")


def _update_focus_counters(
    counters: List[Counter],
    data: np.ndarray,
    decimals: int,
) -> None:
    rounded = np.round(data, decimals=decimals)
    for dim in range(rounded.shape[1]):
        values, counts = np.unique(rounded[:, dim], return_counts=True)
        for value, count in zip(values.tolist(), counts.tolist()):
            counters[dim][float(value)] += int(count)


def _compute_empirical_stats(
    dataset: HDF5SequenceDataset,
    normalizer: LinearNormalizer,
    obs_eval_keys: List[str],
    include_action: bool,
    last_n_dims: int,
    sample_indices: Iterable[int],
    focus_keys: List[str],
    round_decimals: int,
    near_tol: float,
):
    raw_stats: Dict[str, RunningStats] = {}
    norm_stats: Dict[str, RunningStats] = {}
    norm_meta = {}
    focus_raw: Dict[str, List[Counter]] = {}
    focus_norm: Dict[str, List[Counter]] = {}

    def ensure_field(name: str, dim: int):
        if name not in raw_stats:
            raw_stats[name] = RunningStats(dim)
            norm_stats[name] = RunningStats(dim)
            norm_meta[name] = {
                "total": 0,
                "in_range": 0,
                "near_neg1": 0,
                "near_pos1": 0,
            }
            if name in focus_keys:
                focus_raw[name] = [Counter() for _ in range(dim)]
                focus_norm[name] = [Counter() for _ in range(dim)]

    for idx in sample_indices:
        sample = dataset[idx]

        for key in obs_eval_keys:
            if key not in sample["obs"] or key not in normalizer.fields:
                continue
            raw = np.asarray(sample["obs"][key], dtype=np.float32)
            raw_flat = _flatten_last_dims(raw, last_n_dims)
            if raw_flat.size == 0:
                continue
            norm = normalizer[key].normalize(raw).detach().cpu().numpy()
            norm_flat = _flatten_last_dims(norm, last_n_dims)

            ensure_field(key, raw_flat.shape[1])
            raw_stats[key].update(raw_flat)
            norm_stats[key].update(norm_flat)

            meta = norm_meta[key]
            meta["total"] += int(norm_flat.size)
            meta["in_range"] += int(np.count_nonzero((norm_flat >= -1.05) & (norm_flat <= 1.05)))
            meta["near_neg1"] += int(np.count_nonzero(np.abs(norm_flat + 1.0) <= near_tol))
            meta["near_pos1"] += int(np.count_nonzero(np.abs(norm_flat - 1.0) <= near_tol))

            if key in focus_raw:
                _update_focus_counters(focus_raw[key], raw_flat, round_decimals)
                _update_focus_counters(focus_norm[key], norm_flat, round_decimals)

        if include_action and "action" in normalizer.fields:
            actions = np.asarray(sample["actions"], dtype=np.float32)
            action_mask = sample.get("action_mask")
            if action_mask is not None:
                action_mask = np.asarray(action_mask).astype(bool)
                if action_mask.shape[0] == actions.shape[0]:
                    actions = actions[action_mask]
                else:
                    raise ValueError(
                        f"action_mask shape {action_mask.shape} does not match actions {actions.shape}"
                    )
            if actions.size > 0:
                raw_flat = _flatten_last_dims(actions, last_n_dims)
                norm = normalizer["action"].normalize(actions).detach().cpu().numpy()
                norm_flat = _flatten_last_dims(norm, last_n_dims)

                ensure_field("action", raw_flat.shape[1])
                raw_stats["action"].update(raw_flat)
                norm_stats["action"].update(norm_flat)

                meta = norm_meta["action"]
                meta["total"] += int(norm_flat.size)
                meta["in_range"] += int(np.count_nonzero((norm_flat >= -1.05) & (norm_flat <= 1.05)))
                meta["near_neg1"] += int(np.count_nonzero(np.abs(norm_flat + 1.0) <= near_tol))
                meta["near_pos1"] += int(np.count_nonzero(np.abs(norm_flat - 1.0) <= near_tol))

                if "action" in focus_raw:
                    _update_focus_counters(focus_raw["action"], raw_flat, round_decimals)
                    _update_focus_counters(focus_norm["action"], norm_flat, round_decimals)

    return raw_stats, norm_stats, norm_meta, focus_raw, focus_norm


def _print_empirical_summary(
    raw_stats: Dict[str, RunningStats],
    norm_stats: Dict[str, RunningStats],
    norm_meta: dict,
    max_dims: int,
) -> None:
    print("\n=== Empirical Stats (Pre vs Post Normalize) ===")
    for key in sorted(raw_stats.keys()):
        raw = raw_stats[key].finalize()
        norm = norm_stats[key].finalize()
        count = raw_stats[key].count
        meta = norm_meta.get(key, {})
        total = max(int(meta.get("total", 0)), 1)
        in_range_pct = 100.0 * float(meta.get("in_range", 0)) / total
        near_neg1_pct = 100.0 * float(meta.get("near_neg1", 0)) / total
        near_pos1_pct = 100.0 * float(meta.get("near_pos1", 0)) / total
        print(f"[{key}] samples={count}, dim={raw_stats[key].dim}")
        print(f"  raw  min : {_format_arr(raw['min'], max_dims)}")
        print(f"  raw  max : {_format_arr(raw['max'], max_dims)}")
        print(f"  raw mean : {_format_arr(raw['mean'], max_dims)}")
        print(f"  raw std  : {_format_arr(raw['std'], max_dims)}")
        print(f"  norm min : {_format_arr(norm['min'], max_dims)}")
        print(f"  norm max : {_format_arr(norm['max'], max_dims)}")
        print(f"  norm mean: {_format_arr(norm['mean'], max_dims)}")
        print(f"  norm std : {_format_arr(norm['std'], max_dims)}")
        print(
            "  norm coverage: "
            f"in[-1.05,1.05]={in_range_pct:.2f}% "
            f"near(-1)={near_neg1_pct:.2f}% near(+1)={near_pos1_pct:.2f}%"
        )


def _print_focus_counts(
    focus_raw: Dict[str, List[Counter]],
    focus_norm: Dict[str, List[Counter]],
    topk: int,
) -> None:
    if not focus_raw:
        return
    print("\n=== Focus Key Value Counts (rounded) ===")
    for key in sorted(focus_raw.keys()):
        print(f"[{key}]")
        raw_counters = focus_raw[key]
        norm_counters = focus_norm[key]
        for dim, (raw_counter, norm_counter) in enumerate(zip(raw_counters, norm_counters)):
            raw_top = raw_counter.most_common(topk)
            norm_top = norm_counter.most_common(topk)
            print(f"  dim {dim} raw top{topk}: {raw_top}")
            print(f"  dim {dim} norm top{topk}: {norm_top}")


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect DP normalizer stats.")
    parser.add_argument(
        "--config",
        default="",
        help="Path to train_config.json (defaults to ckpt dir/train_config.json when --ckpt is set).",
    )
    parser.add_argument(
        "--ckpt",
        default="",
        help="Optional checkpoint path; if provided, loads normalizer from checkpoint.",
    )
    parser.add_argument(
        "--demo-file",
        default="",
        help="Optional override for data.demo_file in config.",
    )
    parser.add_argument(
        "--split-source",
        default="auto",
        choices=["auto", "file", "recompute", "all"],
        help=(
            "How to choose train demos: "
            "auto=file if present else recompute; "
            "file=must use split_indices.json; "
            "recompute=ignore split file and regenerate from val_ratio+seed; "
            "all=use all demos."
        ),
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=None,
        help="Optional override for data.val_ratio when split-source is recompute/auto fallback.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Optional override for data.seed when split-source is recompute/auto fallback.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limit number of dataset samples used for empirical stats (0 means all).",
    )
    parser.add_argument(
        "--include-image-keys",
        action="store_true",
        help="Also evaluate image obs keys (default: low-dim obs only).",
    )
    parser.add_argument(
        "--focus-keys",
        default="gripper_states,action",
        help="Comma-separated keys to print rounded value counts for.",
    )
    parser.add_argument(
        "--round-decimals",
        type=int,
        default=4,
        help="Rounding decimals used by focus value counts.",
    )
    parser.add_argument(
        "--near-tol",
        type=float,
        default=1e-3,
        help="Tolerance for near -1 / near +1 ratio in normalized values.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=8,
        help="Top-k rounded values to print per focus dimension.",
    )
    parser.add_argument(
        "--max-print-dims",
        type=int,
        default=16,
        help="Max number of dims to print per vector field.",
    )
    parser.add_argument(
        "--no-empirical",
        action="store_true",
        help="Only print normalizer parameters; skip empirical pre/post stats.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    ckpt_path = Path(args.ckpt).expanduser().resolve() if args.ckpt else None
    config_path = Path(args.config).expanduser().resolve() if args.config else None
    if config_path is None:
        if ckpt_path is None:
            raise ValueError("pass --config, or pass --ckpt so config can default from ckpt dir")
        config_path = ckpt_path.parent / TRAIN_CONFIG_NAME
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")

    raw_cfg = _parse_config(config_path)
    data_cfg = raw_cfg.get("data") or {}
    demo_file = str(args.demo_file or data_cfg.get("demo_file", "")).strip()
    if not demo_file:
        raise ValueError("demo file is empty; pass --demo-file or set data.demo_file in config")
    demo_path = Path(demo_file).expanduser().resolve()
    if not demo_path.exists():
        raise FileNotFoundError(f"demo file not found: {demo_path}")

    train_episodes, split_info = _resolve_train_episodes(
        raw_cfg=raw_cfg,
        config_path=config_path,
        demo_path=demo_path,
        split_source=args.split_source,
        val_ratio_override=args.val_ratio,
        split_seed_override=args.split_seed,
    )

    dataset, obs_keys, image_keys = _build_dataset(raw_cfg, str(demo_path), train_episodes)
    if len(dataset) == 0:
        raise ValueError("dataset is empty")

    num_samples = len(dataset) if args.max_samples <= 0 else min(len(dataset), int(args.max_samples))
    sample_indices = range(num_samples)

    if ckpt_path is not None:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
        normalizer = _load_normalizer_from_ckpt(ckpt_path)
        source = f"checkpoint: {ckpt_path}"
    else:
        normalizer = _build_normalizer_from_cfg(raw_cfg, dataset, obs_keys, image_keys, sample_indices)
        source = "recomputed from config + dataset"

    dp_cfg = (raw_cfg.get("policy") or {}).get("dp") or {}
    normalizer_cfg = dp_cfg.get("normalizer") or {}
    last_n_dims = int(normalizer_cfg.get("last_n_dims", 1))

    print(f"normalizer source: {source}")
    print(f"config: {config_path}")
    print(f"demo file: {demo_path}")
    print(f"split source: {split_info}")
    print(f"dataset samples used: {num_samples}/{len(dataset)}")
    if train_episodes is not None:
        print(f"train episodes selected: {len(train_episodes)}")
    else:
        print("train episodes: all demos")
    print(f"last_n_dims: {last_n_dims}")

    _print_param_summary(normalizer, max_dims=int(args.max_print_dims))

    if args.no_empirical:
        return

    obs_eval_keys = list(obs_keys)
    if not args.include_image_keys:
        image_set = set(image_keys)
        obs_eval_keys = [k for k in obs_eval_keys if k not in image_set]

    focus_keys = _split_csv(args.focus_keys)
    include_action = "action" in normalizer.fields

    raw_stats, norm_stats, norm_meta, focus_raw, focus_norm = _compute_empirical_stats(
        dataset=dataset,
        normalizer=normalizer,
        obs_eval_keys=obs_eval_keys,
        include_action=include_action,
        last_n_dims=last_n_dims,
        sample_indices=sample_indices,
        focus_keys=focus_keys,
        round_decimals=int(args.round_decimals),
        near_tol=float(args.near_tol),
    )

    _print_empirical_summary(
        raw_stats=raw_stats,
        norm_stats=norm_stats,
        norm_meta=norm_meta,
        max_dims=int(args.max_print_dims),
    )
    _print_focus_counts(focus_raw=focus_raw, focus_norm=focus_norm, topk=int(args.topk))


if __name__ == "__main__":
    main()

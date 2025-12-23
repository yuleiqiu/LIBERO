import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class HDF5SequenceDataset(Dataset):
    def __init__(
        self,
        hdf5_path,
        obs_keys,
        obs_horizon=1,
        predict_horizon=1,
        action_key="actions",
        demos=None,
        obs_stats=None,
    ):
        self.hdf5_path = str(Path(hdf5_path).expanduser().resolve())
        self.obs_keys = list(obs_keys)
        self.obs_horizon = int(obs_horizon)
        self.predict_horizon = int(predict_horizon)
        self.action_key = action_key
        self._h5 = None
        self._indices = []
        self._obs_stats = obs_stats

        with h5py.File(self.hdf5_path, "r") as f:
            demo_keys = demos or list(f["data"].keys())
            for demo_key in sorted(demo_keys):
                demo_group = f["data"][demo_key]
                if "num_samples" in demo_group.attrs:
                    length = int(demo_group.attrs["num_samples"])
                else:
                    length = demo_group[action_key].shape[0]
                max_t = length
                for t in range(max(0, max_t)):
                    self._indices.append((demo_key, t))

    def __len__(self):
        return len(self._indices)

    def set_obs_stats(self, obs_stats):
        self._obs_stats = obs_stats

    def _get_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.hdf5_path, "r")
        return self._h5

    def __getitem__(self, idx):
        demo_key, t = self._indices[idx]
        f = self._get_h5()
        demo_group = f["data"][demo_key]

        obs = {}
        for key in self.obs_keys:
            start = max(0, t - self.obs_horizon + 1)
            arr = demo_group["obs"][key][start : t + 1]
            if arr.shape[0] < self.obs_horizon:
                pad = np.repeat(arr[[0]], self.obs_horizon - arr.shape[0], axis=0)
                arr = np.concatenate([pad, arr], axis=0)
            obs[key] = arr.astype(np.float32)

        actions = demo_group[self.action_key][t : t + self.predict_horizon].astype(
            np.float32
        )
        valid_len = actions.shape[0]
        action_mask = np.zeros((self.predict_horizon,), dtype=np.float32)
        action_mask[:valid_len] = 1.0
        if valid_len < self.predict_horizon:
            pad = np.zeros(
                (self.predict_horizon - valid_len, actions.shape[1]),
                dtype=np.float32,
            )
            actions = np.concatenate([actions, pad], axis=0)

        if self._obs_stats is not None:
            for key, value in obs.items():
                stats = self._obs_stats.get(key)
                if stats is None:
                    continue
                mean = stats["mean"]
                std = stats["std"]
                obs[key] = (value - mean) / std

        return {"obs": obs, "actions": actions, "action_mask": action_mask}


def save_obs_stats(path, stats):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        key: {"mean": value["mean"].tolist(), "std": value["std"].tolist()}
        for key, value in stats.items()
    }
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def load_obs_stats(path):
    path = Path(path).expanduser().resolve()
    with open(path, "r") as f:
        stats_json = json.load(f)
    return {
        key: {
            "mean": np.array(value["mean"], dtype=np.float32),
            "std": np.array(value["std"], dtype=np.float32),
        }
        for key, value in stats_json.items()
    }


def compute_obs_stats(dataset, indices, eps=1e-3):
    sums = {}
    sumsq = {}
    counts = {}

    for idx in tqdm(indices, desc="compute obs stats", leave=True):
        sample = dataset[idx]
        for key, value in sample["obs"].items():
            value = value.astype(np.float32, copy=False)
            sums.setdefault(key, 0.0)
            sumsq.setdefault(key, 0.0)
            counts.setdefault(key, 0)
            sums[key] = sums[key] + value.sum(axis=0)
            sumsq[key] = sumsq[key] + (value * value).sum(axis=0)
            counts[key] = counts[key] + value.shape[0]

    stats = {}
    for key in sums:
        mean = sums[key] / counts[key]
        var = sumsq[key] / counts[key] - mean * mean
        var = np.maximum(var, np.float32(0.0))
        std = np.sqrt(var) + np.float32(eps)
        stats[key] = {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}
    return stats

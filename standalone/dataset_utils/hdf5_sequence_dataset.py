from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from standalone.dataset_utils.image_normalization import normalize_images


class HDF5SequenceDataset(Dataset):
    def __init__(
        self,
        hdf5_path,
        obs_keys,
        obs_horizon=1,
        predict_horizon=1,
        action_horizon=None,
        action_start_offset=None,
        action_key="actions",
        demos=None,
        image_keys=None,
        image_norm="none",
        image_transforms=None,
    ):
        self.hdf5_path = str(Path(hdf5_path).expanduser().resolve())
        self.obs_keys = list(obs_keys)
        self.obs_horizon = int(obs_horizon)
        self.predict_horizon = int(predict_horizon)
        self.action_horizon = int(action_horizon) if action_horizon is not None else None
        self.action_start_offset = (
            int(action_start_offset) if action_start_offset is not None else None
        )
        self.action_key = action_key
        self._h5 = None
        self._indices = []
        self.image_keys = set(image_keys or [])
        self.image_norm = str(image_norm or "none").lower()
        self._image_transforms_cfg = image_transforms
        self._image_transform = None
        self._image_transforms_enabled = True
        self._init_image_transforms()

        with h5py.File(self.hdf5_path, "r") as f:
            demo_keys = demos or list(f["data"].keys())
            for demo_key in sorted(demo_keys):
                demo_group = f["data"][demo_key]
                if "num_samples" in demo_group.attrs:
                    length = int(demo_group.attrs["num_samples"])
                else:
                    length = demo_group[action_key].shape[0]
                for t in range(max(0, length)):
                    self._indices.append((demo_key, t))

    def __len__(self):
        return len(self._indices)

    def set_image_transforms_enabled(self, enabled: bool):
        self._image_transforms_enabled = bool(enabled)

    def image_transforms_enabled(self):
        return self._image_transforms_enabled

    def _get_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.hdf5_path, "r")
        return self._h5

    def _init_image_transforms(self):
        if not self._image_transforms_cfg or not self.image_keys:
            return
        if not getattr(self._image_transforms_cfg, "enable", False):
            return
        from standalone.dataset_utils.image_transforms import ImageTransforms

        self._image_transform = ImageTransforms(self._image_transforms_cfg)

    def _apply_image_transforms(self, arr):
        if self._image_transform is None or not self._image_transforms_enabled:
            return arr
        if arr.ndim < 3:
            raise ValueError(f"expected image dims >= 3, got shape {arr.shape}")
        channel_last = arr.shape[-1] in (1, 3)
        channel_first = arr.shape[-3] in (1, 3)
        if not (channel_last or channel_first):
            raise ValueError(f"cannot infer channel axis from shape {arr.shape}")
        if channel_last:
            arr = np.moveaxis(arr, -1, -3)
        x = torch.from_numpy(arr).to(dtype=torch.float32) / 255.0
        x = self._image_transform(x)
        if torch.is_tensor(x):
            x = x.clamp(0.0, 1.0).cpu().numpy()
        if channel_last:
            x = np.moveaxis(x, -3, -1)
        return x.astype(np.float32, copy=False)

    def _process_image(self, arr):
        arr = arr.astype(np.float32, copy=False)
        if self._image_transform is not None and self._image_transforms_enabled:
            arr = self._apply_image_transforms(arr)
            return normalize_images(arr, self.image_norm, input_scale="0_1")
        return normalize_images(arr, self.image_norm, input_scale="0_255")

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
            arr = arr.astype(np.float32, copy=False)
            if key in self.image_keys:
                arr = self._process_image(arr)
            obs[key] = arr

        action_start = t
        action_horizon = self.predict_horizon
        if self.action_horizon is not None:
            action_horizon = self.action_horizon
        if self.action_start_offset is not None:
            action_start = action_start + self.action_start_offset
        action_end = action_start + action_horizon

        base_action_dim = demo_group[self.action_key].shape[1]
        pad_front = max(0, -action_start)
        pad_front = min(pad_front, action_horizon)
        slice_start = max(0, action_start)
        slice_end = max(0, action_end)
        actions = demo_group[self.action_key][slice_start:slice_end].astype(np.float32)
        max_valid = max(0, action_horizon - pad_front)
        if actions.shape[0] > max_valid:
            actions = actions[:max_valid]
        valid_len = actions.shape[0]
        pad_back = action_horizon - (pad_front + valid_len)

        if pad_front > 0:
            front = np.zeros((pad_front, base_action_dim), dtype=np.float32)
            actions = np.concatenate([front, actions], axis=0)
        if pad_back > 0:
            back = np.zeros((pad_back, base_action_dim), dtype=np.float32)
            actions = np.concatenate([actions, back], axis=0)

        action_mask = np.zeros((action_horizon,), dtype=np.float32)
        if valid_len > 0:
            action_mask[pad_front : pad_front + valid_len] = 1.0

        return {"obs": obs, "actions": actions, "action_mask": action_mask}

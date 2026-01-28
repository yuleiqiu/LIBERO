from dataclasses import asdict, is_dataclass
import torch

from standalone.configs.policy import CNNMLPModelConfig
from standalone.models.algos.cnnmlp.core import CNNMLPModel
from standalone.models.encoders.image_map import ImageMapEncoder
from standalone.models.policy.base import ChunkPolicy


class CNNMLPPolicy(ChunkPolicy):
    def __init__(
        self,
        obs_keys,
        image_keys,
        obs_horizon,
        predict_horizon,
        exec_horizon,
        qpos_dim,
        action_dim,
        cnnmlp_config=None,
    ):
        super().__init__(predict_horizon=predict_horizon, exec_horizon=exec_horizon)
        self.obs_keys = list(obs_keys) if obs_keys is not None else []
        self.image_keys = list(image_keys) if image_keys is not None else []
        self.obs_horizon = int(obs_horizon)
        self.qpos_dim = int(qpos_dim)
        self.action_dim = int(action_dim)

        if not self.image_keys:
            raise ValueError("CNNMLPPolicy requires image_keys for camera inputs.")

        config = {}
        if cnnmlp_config:
            if is_dataclass(cnnmlp_config):
                config.update(asdict(cnnmlp_config))
            else:
                config.update(cnnmlp_config)
        model_cfg = config.pop("model", None) or {}
        if is_dataclass(model_cfg):
            model_cfg = asdict(model_cfg)
        model_config = CNNMLPModelConfig(**model_cfg)
        model_config.camera_names = list(self.image_keys)
        model_config.chunk_size = int(predict_horizon)
        model_config.qpos_dim = self.qpos_dim
        model_config.action_dim = self.action_dim

        train_backbone = float(config.get("lr_backbone", 0.0)) > 0
        image_encoder_kwargs = {
            "backbone": model_config.backbone,
            "position_embedding": model_config.position_embedding,
            "hidden_dim": int(model_config.hidden_dim),
            "dilation": bool(model_config.dilation),
            "train_backbone": train_backbone,
            "pretrained": bool(model_config.pretrained),
        }
        backbones = [
            ImageMapEncoder(**image_encoder_kwargs) for _ in self.image_keys
        ]
        self.model = CNNMLPModel(model_config, backbones=backbones)

    @staticmethod
    def _to_tensor(value):
        return value if torch.is_tensor(value) else torch.as_tensor(value)

    def _select_last_step(self, value):
        if value.ndim >= 3 and value.shape[1] == self.obs_horizon:
            return value[:, -1]
        if self.obs_horizon > 1 and value.ndim >= 2 and value.shape[0] == self.obs_horizon:
            return value[-1]
        return value

    def _build_qpos(self, obs):
        parts = []
        for key in self.obs_keys:
            if key not in obs:
                raise KeyError(f"missing lowdim key: {key}")
            x = self._to_tensor(obs[key])
            x = self._select_last_step(x)
            if x.ndim == 1:
                x = x.unsqueeze(0)
            parts.append(x.reshape(x.shape[0], -1))
        if not parts:
            raise ValueError("no obs_keys provided for CNNMLPPolicy input")
        qpos = torch.cat(parts, dim=-1)
        if qpos.shape[-1] != self.qpos_dim:
            raise ValueError(
                f"qpos dim mismatch: expected {self.qpos_dim}, got {qpos.shape[-1]}"
            )
        return qpos

    def _build_images(self, obs):
        images = []
        for key in self.image_keys:
            if key not in obs:
                raise KeyError(f"missing image key: {key}")
            x = self._to_tensor(obs[key])
            x = self._select_last_step(x)
            if x.ndim == 3:
                x = x.unsqueeze(0)
            if x.ndim != 4:
                raise ValueError(
                    f"expected image dims (B, H, W, C) or (B, C, H, W), got {x.shape}"
                )
            if x.shape[-1] in (1, 3):
                x = x.permute(0, 3, 1, 2)
            images.append(x)
        if not images:
            raise ValueError("no image_keys provided for CNNMLPPolicy input")
        images = torch.stack(images, dim=1).to(dtype=torch.float32)
        return images

    def forward(self, obs):
        if not isinstance(obs, dict):
            raise TypeError("CNNMLPPolicy expects a dict of observations")
        qpos = self._build_qpos(obs).to(dtype=torch.float32)
        image = self._build_images(obs)
        actions = self.model(qpos, image, env_state=None)
        if actions.ndim == 2:
            actions = actions.unsqueeze(1)
        if actions.shape[1] != self.predict_horizon:
            if actions.shape[1] == 1:
                actions = actions.repeat(1, self.predict_horizon, 1)
            else:
                actions = actions[:, : self.predict_horizon]
        return actions

import torch
import torch.nn as nn

from standalone.models.modules.image_encoders import ResnetEncoder
from standalone.models.policy.base import ChunkPolicy


class MLPPolicy(ChunkPolicy):
    def __init__(
        self,
        input_dim,
        action_dim,
        predict_horizon=1,
        exec_horizon=None,
        hidden_dims=(256, 256),
        action_squash=True,
        obs_keys=None,
        image_keys=None,
        image_shapes=None,
        image_embed_dim=128,
        image_encoder_pretrained=False,
        image_encoder_remove_layer_num=2,
        image_encoder_no_stride=False,
    ):
        super().__init__(predict_horizon=predict_horizon, exec_horizon=exec_horizon)
        self.obs_keys = list(obs_keys) if obs_keys is not None else []
        self.image_keys = list(image_keys) if image_keys is not None else []
        self.image_embed_dim = int(image_embed_dim)
        self.image_encoders = nn.ModuleDict()
        if self.image_keys:
            if image_shapes is None:
                raise ValueError("image_shapes is required when image_keys is set")
            for key in self.image_keys:
                if key not in image_shapes:
                    raise KeyError(f"missing image shape for key: {key}")
                input_shape = self._to_chw(image_shapes[key])
                self.image_encoders[key] = ResnetEncoder(
                    input_shape=input_shape,
                    output_size=self.image_embed_dim,
                    pretrained=image_encoder_pretrained,
                    remove_layer_num=image_encoder_remove_layer_num,
                    no_stride=image_encoder_no_stride,
                )
        layers = []
        prev = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev, dim))
            layers.append(nn.ReLU())
            prev = dim
        layers.append(nn.Linear(prev, action_dim * predict_horizon))
        self.net = nn.Sequential(*layers)
        self.action_squash = action_squash
        self.action_dim = int(action_dim)

    @staticmethod
    def _to_chw(shape):
        if len(shape) != 3:
            raise ValueError(f"expected 3D image shape, got {shape}")
        if shape[-1] in (1, 3):
            h, w, c = shape
            return (c, h, w)
        if shape[0] in (1, 3):
            return shape
        raise ValueError(f"cannot infer channel dim from shape: {shape}")

    def forward(self, obs):
        if isinstance(obs, dict):
            parts = []
            for key in self.obs_keys:
                if key not in obs:
                    raise KeyError(f"missing lowdim key: {key}")
                x = obs[key]
                if x.ndim == 2:
                    x = x.unsqueeze(1)
                parts.append(x.reshape(x.shape[0], -1))

            for key in self.image_keys:
                if key not in obs:
                    raise KeyError(f"missing image key: {key}")
                x = obs[key]
                if x.ndim == 4:
                    x = x.unsqueeze(1)
                if x.shape[-1] in (1, 3):
                    x = x.permute(0, 1, 4, 2, 3)
                x = x.float() / 255.0
                b, t, c, h, w = x.shape
                emb = self.image_encoders[key](x.reshape(b * t, c, h, w))
                emb = emb.view(b, t, -1)
                parts.append(emb.reshape(b, -1))

            if not parts:
                raise ValueError("no obs keys provided for policy input")
            x = torch.cat(parts, dim=-1)
        else:
            x = obs
        out = self.net(x)
        if self.action_squash:
            out = torch.tanh(out)
        return out.view(out.shape[0], self.predict_horizon, self.action_dim)

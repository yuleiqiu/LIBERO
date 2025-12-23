import torch
import torch.nn as nn

from standalone.models.base import ChunkPolicy


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
    ):
        super().__init__(predict_horizon=predict_horizon, exec_horizon=exec_horizon)
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
        self.obs_keys = list(obs_keys) if obs_keys is not None else None

    def forward(self, obs):
        if isinstance(obs, dict):
            parts = []
            keys = self.obs_keys or list(obs.keys())
            for key in keys:
                x = obs[key]
                parts.append(x.reshape(x.shape[0], -1))
            x = torch.cat(parts, dim=-1)
        else:
            x = obs
        out = self.net(x)
        if self.action_squash:
            out = torch.tanh(out)
        return out.view(out.shape[0], self.predict_horizon, self.action_dim)

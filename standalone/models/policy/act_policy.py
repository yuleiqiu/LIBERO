import torch

from standalone.act_standalone.main import get_args_parser
from standalone.act_standalone.models import build_ACT_model, build_CNNMLP_model
from standalone.models.policy.base import ChunkPolicy


class ACTPolicy(ChunkPolicy):
    def __init__(
        self,
        obs_keys,
        image_keys,
        obs_horizon,
        predict_horizon,
        exec_horizon,
        qpos_dim,
        action_dim,
        model_type="cnnmlp",
        act_config=None,
    ):
        super().__init__(predict_horizon=predict_horizon, exec_horizon=exec_horizon)
        self.obs_keys = list(obs_keys) if obs_keys is not None else []
        self.image_keys = list(image_keys) if image_keys is not None else []
        self.obs_horizon = int(obs_horizon)
        self.qpos_dim = int(qpos_dim)
        self.action_dim = int(action_dim)

        if not self.image_keys:
            raise ValueError("ACTPolicy requires image_keys for camera inputs.")

        config = {}
        if act_config:
            config.update(act_config)
        config.update(
            {
                "camera_names": list(self.image_keys),
                "num_queries": int(predict_horizon),
                "qpos_dim": self.qpos_dim,
                "action_dim": self.action_dim,
            }
        )
        args = get_args_parser(config)

        model_type = (model_type or "cnnmlp").lower()
        self._model_type = None
        if model_type in ("act", "detr_vae", "vae"):
            self.model = build_ACT_model(args)
            self._model_type = "act"
        elif model_type in ("cnnmlp",):
            self.model = build_CNNMLP_model(args)
            self._model_type = "cnnmlp"
        else:
            raise ValueError(f"Unknown ACT model_type: {model_type}")
        self.kl_weight = float(config.get("kl_weight", 1.0))

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
            raise ValueError("no obs_keys provided for ACTPolicy input")
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
                raise ValueError(f"expected image dims (B, H, W, C) or (B, C, H, W), got {x.shape}")
            if x.shape[-1] in (1, 3):
                x = x.permute(0, 3, 1, 2)
            images.append(x)
        if not images:
            raise ValueError("no image_keys provided for ACTPolicy input")
        return torch.stack(images, dim=1)

    def forward(self, obs):
        if not isinstance(obs, dict):
            raise TypeError("ACTPolicy expects a dict of observations")
        qpos = self._build_qpos(obs).to(dtype=torch.float32)
        image = self._build_images(obs).to(dtype=torch.float32)
        if self._model_type == "act":
            actions, _, _ = self.model(qpos, image, env_state=None)
        else:
            actions = self.model(qpos, image, env_state=None)
        if actions.ndim == 2:
            actions = actions.unsqueeze(1)
        if actions.shape[1] != self.predict_horizon:
            if actions.shape[1] == 1:
                actions = actions.repeat(1, self.predict_horizon, 1)
            else:
                actions = actions[:, : self.predict_horizon]
        return actions

    def compute_loss(self, batch, reduction="mean", return_stats=False):
        if self._model_type != "act":
            return super().compute_loss(
                batch, reduction=reduction, return_stats=return_stats
            )

        obs = batch["obs"]
        qpos = self._build_qpos(obs).to(dtype=torch.float32)
        image = self._build_images(obs).to(dtype=torch.float32)
        actions = batch["actions"]
        if actions.ndim == 2:
            actions = actions.unsqueeze(1)
        actions = actions[:, : self.predict_horizon]
        action_mask = batch.get("action_mask")
        if action_mask is None:
            is_pad = torch.zeros(
                actions.shape[0],
                actions.shape[1],
                dtype=torch.bool,
                device=actions.device,
            )
        else:
            is_pad = action_mask[:, : actions.shape[1]] <= 0

        a_hat, _, (mu, logvar) = self.model(
            qpos, image, env_state=None, actions=actions, is_pad=is_pad
        )
        total_kld = _kl_divergence(mu, logvar)[0][0]
        all_l1 = torch.abs(actions - a_hat)
        l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
        loss = l1 + total_kld * self.kl_weight

        if reduction not in ("mean", "sum"):
            raise NotImplementedError("Only mean/sum reductions are supported for ACT.")

        if not return_stats:
            return loss
        stats = {"action_l1": l1.detach(), "kl": total_kld.detach()}
        return loss, stats


def _kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    if batch_size == 0:
        raise ValueError("mu/logvar must have a non-zero batch size.")
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)
    return total_kld, dimension_wise_kld, mean_kld

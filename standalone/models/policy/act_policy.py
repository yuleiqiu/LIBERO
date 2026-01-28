from dataclasses import asdict, is_dataclass
from typing import Optional
import torch

from standalone.configs.policy import ACTModelConfig
from standalone.models.algos.act.models import ACTModel
from standalone.models.encoders.image_map import ImageMapEncoder
from standalone.models.policy.base import ChunkPolicy


class ACTTemporalEnsembler:
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        self.chunk_size = int(chunk_size)
        self.ensemble_weights = torch.exp(
            -temporal_ensemble_coeff * torch.arange(self.chunk_size)
        )
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self) -> None:
        self.ensembled_actions = None
        self.ensembled_actions_count = None

    def update(self, actions: torch.Tensor) -> torch.Tensor:
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(
            device=actions.device
        )
        if self.ensembled_actions is None:
            self.ensembled_actions = actions.clone()
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1),
                dtype=torch.long,
                device=self.ensembled_actions.device,
            )
        else:
            self.ensembled_actions *= self.ensemble_weights_cumsum[
                self.ensembled_actions_count - 1
            ]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[
                self.ensembled_actions_count
            ]
            self.ensembled_actions /= self.ensemble_weights_cumsum[
                self.ensembled_actions_count
            ]
            self.ensembled_actions_count = torch.clamp(
                self.ensembled_actions_count + 1, max=self.chunk_size
            )
            self.ensembled_actions = torch.cat(
                [self.ensembled_actions, actions[:, -1:]], dim=1
            )
            self.ensembled_actions_count = torch.cat(
                [
                    self.ensembled_actions_count,
                    torch.ones_like(self.ensembled_actions_count[-1:]),
                ]
            )
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action


class ACTPolicy(ChunkPolicy):
    def __init__(
        self,
        obs_keys,
        image_keys,
        obs_horizon,
        predict_horizon,
        exec_horizon,
        proprio_dim,
        action_dim,
        act_config=None,
    ):
        super().__init__(predict_horizon=predict_horizon, exec_horizon=exec_horizon)
        self.obs_keys = list(obs_keys) if obs_keys is not None else []
        self.image_keys = list(image_keys) if image_keys is not None else []
        self.obs_horizon = int(obs_horizon)
        self.proprio_dim = int(proprio_dim)
        self.action_dim = int(action_dim)

        if not self.image_keys:
            raise ValueError("ACTPolicy requires image_keys for camera inputs.")

        config = {}
        if act_config:
            if is_dataclass(act_config):
                config.update(asdict(act_config))
            else:
                config.update(act_config)
        model_cfg = config.pop("model", None) or {}
        if is_dataclass(model_cfg):
            model_cfg = asdict(model_cfg)
        if "proprio_dim" not in model_cfg and "qpos_dim" in model_cfg:
            model_cfg["proprio_dim"] = model_cfg.pop("qpos_dim")
        model_config = ACTModelConfig(**model_cfg)
        model_config.camera_names = list(self.image_keys)
        model_config.chunk_size = int(predict_horizon)
        model_config.proprio_dim = self.proprio_dim
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
        self.model = ACTModel(model_config, backbones=backbones)
        self.kl_weight = float(config.get("kl_weight", 10.0))
        self.temporal_ensemble_coeff = config.get("temporal_ensemble_coeff")
        self.temporal_ensembler: Optional[ACTTemporalEnsembler] = None
        if self.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(
                float(self.temporal_ensemble_coeff), self.predict_horizon
            )

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
        if qpos.shape[-1] != self.proprio_dim:
            raise ValueError(
                f"proprio dim mismatch: expected {self.proprio_dim}, got {qpos.shape[-1]}"
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
        images = torch.stack(images, dim=1).to(dtype=torch.float32)
        return images

    def reset(self):
        super().reset()
        if self.temporal_ensembler is not None:
            self.temporal_ensembler.reset()

    def get_action(self, obs, batched=False):
        if self.temporal_ensembler is None:
            return super().get_action(obs, batched=batched)
        device = next(self.parameters()).device
        obs = self._prepare_obs(obs, device, batched=batched)
        self.eval()
        with torch.no_grad():
            pred = self.forward(obs)
            if pred.ndim == 2:
                pred = pred.view(pred.shape[0], self.predict_horizon, -1)
        if pred.shape[0] != 1:
            raise ValueError("get_action expects a single observation (batch size 1)")
        action = self.temporal_ensembler.update(pred)
        return action[0]

    def forward(self, obs):
        if not isinstance(obs, dict):
            raise TypeError("ACTPolicy expects a dict of observations")
        qpos = self._build_qpos(obs).to(dtype=torch.float32)
        image = self._build_images(obs)
        actions, _, _ = self.model(qpos, image, env_state=None)
        if actions.ndim == 2:
            actions = actions.unsqueeze(1)
        if actions.shape[1] != self.predict_horizon:
            if actions.shape[1] == 1:
                actions = actions.repeat(1, self.predict_horizon, 1)
            else:
                actions = actions[:, : self.predict_horizon]
        return actions

    def compute_loss(self, batch, reduction="mean", return_stats=False):
        obs = batch["obs"]
        qpos = self._build_qpos(obs).to(dtype=torch.float32)
        image = self._build_images(obs)
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

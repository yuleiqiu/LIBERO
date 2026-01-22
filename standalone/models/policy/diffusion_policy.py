import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from standalone.models.algos.dp.core.diffusion_unet_core import DiffusionUnetCore
from standalone.models.algos.dp.model.common.normalizer import LinearNormalizer
from standalone.models.encoders.obs import build_obs_encoder
from standalone.models.policy.base import ChunkPolicy


class DiffusionPolicy(ChunkPolicy):
    def __init__(
        self,
        obs_keys,
        image_keys,
        obs_horizon,
        predict_horizon,
        exec_horizon,
        action_dim,
        obs_shapes,
        encoder_config=None,
        obs_encoder=None,
        dp_config=None,
        noise_scheduler=None,
    ):
        super().__init__(predict_horizon=predict_horizon, exec_horizon=exec_horizon)
        self.obs_keys = list(obs_keys) if obs_keys is not None else []
        self.image_keys = list(image_keys) if image_keys is not None else []
        self.obs_horizon = int(obs_horizon)
        self.action_dim = int(action_dim)
        self.obs_shapes = obs_shapes or {}

        shape_meta = self._build_shape_meta(self.obs_shapes)
        encoder_shapes = self._strip_time_dim(self.obs_shapes)
        if obs_encoder is None:
            obs_encoder = build_obs_encoder(
                obs_shapes=encoder_shapes,
                image_keys=self.image_keys,
                lowdim_keys=self.obs_keys,
                cfg=encoder_config,
            )

        cfg = dict(dp_config or {})
        cfg.pop("crop_shape", None)
        cfg.pop("obs_encoder_group_norm", None)
        cfg.pop("eval_fixed_crop", None)
        horizon = int(cfg.pop("horizon", self.obs_horizon + self.predict_horizon - 1))
        n_action_steps = int(cfg.pop("n_action_steps", self.predict_horizon))
        n_obs_steps = int(cfg.pop("n_obs_steps", self.obs_horizon))

        if noise_scheduler is None:
            scheduler_kwargs = dict(cfg.pop("noise_scheduler", {}))
            noise_scheduler = DDPMScheduler(**scheduler_kwargs)

        self.dp_policy = DiffusionUnetCore(
            shape_meta=shape_meta,
            noise_scheduler=noise_scheduler,
            horizon=horizon,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            obs_encoder=obs_encoder,
            **cfg,
        )

    def _build_shape_meta(self, obs_shapes):
        obs_meta = {}
        for key in self.obs_keys + self.image_keys:
            if key not in obs_shapes:
                raise KeyError(f"missing obs shape for key: {key}")
            shape = tuple(obs_shapes[key])
            if len(shape) >= 1 and shape[0] == self.obs_horizon:
                shape = shape[1:]
            obs_meta[key] = {
                "shape": list(shape),
                "type": "rgb" if key in self.image_keys else "low_dim",
            }
        return {"action": {"shape": (self.action_dim,)}, "obs": obs_meta}

    def _strip_time_dim(self, obs_shapes):
        cleaned = {}
        for key, shape in (obs_shapes or {}).items():
            shape = tuple(shape)
            if len(shape) >= 1 and shape[0] == self.obs_horizon:
                shape = shape[1:]
            cleaned[key] = shape
        return cleaned

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.dp_policy.set_normalizer(normalizer)

    def forward(self, obs):
        if not isinstance(obs, dict):
            raise TypeError("DiffusionPolicy expects a dict of observations")
        device = next(self.parameters()).device
        obs = self._prepare_obs(obs, device, batched=True)
        result = self.dp_policy.predict_action(obs)
        actions = result["action"]
        if actions.ndim == 2:
            actions = actions.unsqueeze(1)
        return actions

    def compute_loss(self, batch, reduction="mean", return_stats=False):
        if reduction not in ("mean", "sum"):
            raise NotImplementedError("Only mean/sum reductions are supported.")
        obs = batch["obs"]
        actions = batch["actions"]
        device = next(self.parameters()).device
        obs = self._prepare_obs(obs, device, batched=True)
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(actions)
        actions = actions.to(device=device, dtype=torch.float32)
        loss = self.dp_policy.compute_loss({"obs": obs, "action": actions})
        if reduction == "sum":
            loss = loss * actions.shape[0]
        if not return_stats:
            return loss
        stats = {"loss": loss.detach()}
        return loss, stats

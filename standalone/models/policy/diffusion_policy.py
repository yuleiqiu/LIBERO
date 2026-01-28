import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from standalone.models.algos.dp.core.diffusion_model import DiffusionModel
from standalone.models.algos.dp.utils.normalizer import LinearNormalizer
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
        horizon = int(cfg.pop("horizon", self.obs_horizon + self.predict_horizon - 1))
        n_action_steps = int(cfg.pop("n_action_steps", self.predict_horizon))
        n_obs_steps = int(cfg.pop("n_obs_steps", self.obs_horizon))

        if noise_scheduler is None:
            noise_scheduler_type = str(cfg.pop("noise_scheduler_type", "DDPM"))
            scheduler_cls = {"DDPM": DDPMScheduler, "DDIM": DDIMScheduler}.get(noise_scheduler_type)
            if scheduler_cls is None:
                raise ValueError(f"unsupported noise_scheduler_type: {noise_scheduler_type}")
            noise_scheduler = scheduler_cls(
                num_train_timesteps=int(cfg.pop("num_train_timesteps", 100)),
                beta_start=float(cfg.pop("beta_start", 0.0001)),
                beta_end=float(cfg.pop("beta_end", 0.02)),
                beta_schedule=str(cfg.pop("beta_schedule", "squaredcos_cap_v2")),
                prediction_type=str(cfg.pop("prediction_type", "epsilon")),
                clip_sample=bool(cfg.pop("clip_sample", True)),
                clip_sample_range=float(cfg.pop("clip_sample_range", 1.0)),
            )

        self.diffusion_model = DiffusionModel(
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
        self.diffusion_model.set_normalizer(normalizer)

    def forward(self, obs):
        if not isinstance(obs, dict):
            raise TypeError("DiffusionPolicy expects a dict of observations")
        device = next(self.parameters()).device
        obs = self._prepare_obs(obs, device, batched=True)
        result = self.diffusion_model.predict_action(obs)
        actions = result["action"]
        if actions.ndim == 2:
            actions = actions.unsqueeze(1)
        return actions

    def compute_loss(self, batch, reduction="mean", return_stats=False):
        if reduction not in ("mean", "sum"):
            raise NotImplementedError("Only mean/sum reductions are supported.")
        obs = batch["obs"]
        actions = batch["actions"]
        action_mask = batch.get("action_mask")
        device = next(self.parameters()).device
        obs = self._prepare_obs(obs, device, batched=True)
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(actions)
        actions = actions.to(device=device, dtype=torch.float32)
        loss = self.diffusion_model.compute_loss(
            {"obs": obs, "action": actions, "action_mask": action_mask}
        )
        if reduction == "sum":
            loss = loss * actions.shape[0]
        if not return_stats:
            return loss
        stats = {"loss": loss.detach()}
        return loss, stats

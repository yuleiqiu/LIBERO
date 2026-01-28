from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from ..utils.normalizer import LinearNormalizer
from ..model.diffusion.conditional_unet1d import ConditionalUnet1D
from ..utils.pytorch_util import dict_apply


class DiffusionModel(nn.Module):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            obs_encoder=None,
            num_inference_steps=None,
            diffusion_step_embed_dim=128,
            down_dims=(512,1024,2048),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            do_mask_loss_for_padding=False,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        if obs_encoder is None:
            raise ValueError("obs_encoder is required; robomimic dependency removed")

        # create diffusion model
        obs_feature_dim = getattr(obs_encoder, "output_dim", None)
        if obs_feature_dim is None and hasattr(obs_encoder, "output_shape"):
            obs_feature_dim = obs_encoder.output_shape()[0]
        if obs_feature_dim is None:
            raise ValueError("obs_encoder must define output_dim or output_shape()")
        input_dim = action_dim
        global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.kwargs = kwargs
        self.do_mask_loss_for_padding = bool(do_mask_loss_for_padding)

        downsampling_factor = 2 ** len(down_dims)
        if horizon % downsampling_factor != 0:
            raise ValueError(
                "horizon should be an integer multiple of the downsampling factor "
                f"(2 ** len(down_dims) = {downsampling_factor}). Got horizon={horizon}."
            )

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))
    
    # ========= inference  ============
    def conditional_sample(self, 
            shape,
            global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        device = global_cond.device if global_cond is not None else self.device
        dtype = global_cond.dtype if global_cond is not None else self.dtype
        trajectory = torch.randn(
            size=shape, 
            dtype=dtype,
            device=device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. predict model output
            model_output = model(trajectory, t, global_cond=global_cond)

            # 2. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps

        # handle different ways of passing observation
        # condition through global feature
        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        # reshape back to B, Do
        global_cond = nobs_features.reshape(B, -1)

        # run sampling
        nsample = self.conditional_sample(
            (B, T, Da),
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer = normalizer

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        trajectory = nactions
        # reshape B, T, ... to B*T
        this_nobs = dict_apply(nobs, 
            lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        # reshape back to B, Do
        global_cond = nobs_features.reshape(batch_size, -1)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        if self.do_mask_loss_for_padding:
            if "action_is_pad" in batch:
                in_episode_bound = ~batch["action_is_pad"].to(device=loss.device)
            elif "action_mask" in batch and batch["action_mask"] is not None:
                in_episode_bound = batch["action_mask"].to(device=loss.device) > 0
            else:
                raise ValueError(
                    "do_mask_loss_for_padding=True requires action_is_pad or action_mask in batch"
                )
            loss = loss * in_episode_bound.unsqueeze(-1)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss

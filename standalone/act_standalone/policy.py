"""Standalone ACT policy for inference.

Minimal usage:
    policy = ACTPolicy(config)
    policy.load_state_dict(ckpt)
    policy.eval()
    actions = policy(qpos, image)
"""
import os
import pickle

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from .main import build_ACT_model_and_optimizer


class ACTPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = build_ACT_model_and_optimizer(config)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        self.stats = _load_dataset_stats(config)
        self.kl_weight = config.get("kl_weight", 1.0)

    def _normalize_qpos(self, qpos):
        if self.stats is None:
            return qpos
        qpos_mean = self.stats["qpos_mean"].to(qpos.device)
        qpos_std = self.stats["qpos_std"].to(qpos.device)
        return (qpos - qpos_mean) / qpos_std

    def _denormalize_actions(self, actions):
        if self.stats is None:
            return actions
        action_mean = self.stats["action_mean"].to(actions.device)
        action_std = self.stats["action_std"].to(actions.device)
        return actions * action_std + action_mean

    def _normalize_actions(self, actions):
        if self.stats is None:
            return actions
        action_mean = self.stats["action_mean"].to(actions.device)
        action_std = self.stats["action_std"].to(actions.device)
        return (actions - action_mean) / action_std

    def _normalize_images(self, image):
        batch_size, num_cam, channels, height, width = image.shape
        image = image.reshape(batch_size * num_cam, channels, height, width)
        image = self.normalize(image)
        return image.reshape(batch_size, num_cam, channels, height, width)

    def forward(self, qpos, image, actions=None, is_pad=None):
        """
        qpos: Tensor [B, 14]
        image: Tensor [B, num_cam, C, H, W] with values in [0, 1]
        returns: actions [B, num_queries, 14]
        """
        camera_names = self.config.get("camera_names", [])
        if camera_names and image.shape[1] != len(camera_names):
            raise ValueError(
                f"Expected {len(camera_names)} cameras, got image shape {image.shape}."
            )

        qpos = self._normalize_qpos(qpos)
        image = self._normalize_images(image)
        env_state = None
        if actions is not None:
            actions = self._normalize_actions(actions)
            actions = actions[:, :self.model.num_queries]
            if is_pad is None:
                raise ValueError("is_pad must be provided when actions are given.")
            is_pad = is_pad[:, :self.model.num_queries]
            a_hat, is_pad_hat, (mu, logvar) = self.model(
                qpos, image, env_state, actions, is_pad
            )
            total_kld, _, _ = kl_divergence(mu, logvar)
            loss_dict = {}
            all_l1 = F.l1_loss(actions, a_hat, reduction="none")
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict["l1"] = l1
            loss_dict["kl"] = total_kld[0]
            loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"] * self.kl_weight
            loss_dict["is_pad_hat"] = is_pad_hat
            return loss_dict
        actions, _, _ = self.model(qpos, image, env_state)
        return self._denormalize_actions(actions)


def kl_divergence(mu, logvar):
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


def _load_dataset_stats(config):
    if "dataset_stats" in config and config["dataset_stats"] is not None:
        stats = config["dataset_stats"]
    else:
        stats_path = config.get("stats_path") or config.get("dataset_stats_path")
        if not stats_path:
            return None
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Dataset stats not found: {stats_path}")
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)

    return {
        "qpos_mean": torch.as_tensor(stats["qpos_mean"], dtype=torch.float32),
        "qpos_std": torch.as_tensor(stats["qpos_std"], dtype=torch.float32),
        "action_mean": torch.as_tensor(stats["action_mean"], dtype=torch.float32),
        "action_std": torch.as_tensor(stats["action_std"], dtype=torch.float32),
    }

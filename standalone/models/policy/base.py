import abc
from collections import deque

import torch
import torch.nn as nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def compute_loss(self, batch, reduction="mean", return_stats=False):
        raise NotImplementedError

    def _prepare_obs(self, obs, device, batched=False):
        if isinstance(obs, dict):
            out = {}
            for key, value in obs.items():
                tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
                if not batched:
                    tensor = tensor.unsqueeze(0)
                out[key] = tensor.to(device=device, dtype=torch.float32)
            return out
        tensor = obs if torch.is_tensor(obs) else torch.as_tensor(obs)
        if not batched:
            tensor = tensor.unsqueeze(0)
        return tensor.to(device=device, dtype=torch.float32)

    def get_action(self, obs, batched=False):
        device = next(self.parameters()).device
        obs = self._prepare_obs(obs, device, batched=batched)
        pred = self.forward(obs)
        # If model outputs a sequence (B, T, A), return the first action.
        if pred.ndim >= 3:
            return pred[:, 0]
        return pred


class ChunkPolicy(BasePolicy):
    def __init__(self, predict_horizon, exec_horizon=None):
        super().__init__()
        self.predict_horizon = int(predict_horizon)
        self.exec_horizon = (
            int(exec_horizon) if exec_horizon is not None else self.predict_horizon
        )
        if self.exec_horizon > self.predict_horizon:
            raise ValueError("exec_horizon cannot be larger than predict_horizon")
        self._action_queue = deque()

    def reset(self):
        self._action_queue.clear()

    def get_action(self, obs, batched=False):
        if len(self._action_queue) == 0:
            device = next(self.parameters()).device
            obs = self._prepare_obs(obs, device, batched=batched)
            self.eval()
            with torch.no_grad():
                pred = self.forward(obs)
                if pred.ndim == 2:
                    # （B, H*A） -> (B, H, A)
                    pred = pred.view(pred.shape[0], self.predict_horizon, -1)
            if pred.shape[0] != 1:
                raise ValueError("get_action expects a single observation (batch size 1)")
            actions = pred[0].detach()
            take = min(self.exec_horizon, actions.shape[0])
            self._action_queue.extend(actions[:take])
        return self._action_queue.popleft()

    def compute_loss(self, batch, reduction="mean", return_stats=False):
        pred = self.forward(batch["obs"])
        target = batch["actions"]

        if pred.ndim == 2:
            pred = pred.view(pred.shape[0], self.predict_horizon, -1)
        if target.ndim == 2:
            target = target.view(target.shape[0], self.predict_horizon, -1)

        loss = (pred - target).pow(2)
        mask = batch.get("action_mask")
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(-1)
            loss = loss * mask

        if reduction == "mean":
            if mask is not None:
                denom = mask.sum() * loss.shape[-1]
                loss = loss.sum() / denom.clamp_min(1.0)
            else:
                loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
        elif reduction == "none":
            pass
        else:
            raise NotImplementedError

        if not return_stats:
            return loss
        stats = {"action_mse": loss.detach() if loss.ndim == 0 else loss.mean()}
        return loss, stats

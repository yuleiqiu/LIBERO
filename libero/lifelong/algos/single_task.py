import copy

import torch
import torch.nn as nn

from libero.lifelong.algos.base import Sequential


class SingleTask(Sequential):
    """
    The sequential BC baseline.
    """

    def __init__(self, n_tasks, cfg):
        super().__init__(n_tasks, cfg)
        self.init_pi = copy.deepcopy(self.policy)

    def start_task(self, task):
        # re-initialize every new task
        self.policy = copy.deepcopy(self.init_pi)
        super().start_task(task)

    def observe(self, data, return_stats=False):
        """
        Single-task observe with optional stats logging.
        """
        data = self.map_tensor_to_device(data)
        self.optimizer.zero_grad()
        if return_stats:
            loss, stats = self.policy.compute_loss(data, return_stats=True)
        else:
            loss = self.policy.compute_loss(data)
            stats = None
        (self.loss_scale * loss).backward()
        if self.cfg.train.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.train.grad_clip)
        self.optimizer.step()

        if return_stats:
            stats = {k: (v.detach().item() if torch.is_tensor(v) else float(v)) for k, v in stats.items()}
            return loss.item(), stats
        return loss.item()

    def eval_observe(self, data, return_stats=False):
        data = self.map_tensor_to_device(data)
        with torch.no_grad():
            if return_stats:
                loss, stats = self.policy.compute_loss(data, return_stats=True)
            else:
                loss = self.policy.compute_loss(data)
                stats = None
        if return_stats:
            stats = {k: (v.detach().item() if torch.is_tensor(v) else float(v)) for k, v in stats.items()}
            return loss.item(), stats
        return loss.item()

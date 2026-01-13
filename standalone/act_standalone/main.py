"""Standalone ACT model builder utilities."""
from types import SimpleNamespace

import torch

from .models import build_ACT_model, build_CNNMLP_model


_DEFAULT_CONFIG = {
    "lr": 1e-4,
    "lr_backbone": 1e-5,
    "batch_size": 2,
    "weight_decay": 1e-4,
    "epochs": 300,
    "lr_drop": 200,
    "clip_max_norm": 0.1,
    "backbone": "resnet18",
    "dilation": False,
    "position_embedding": "sine",
    "camera_names": [],
    "enc_layers": 4,
    "dec_layers": 6,
    "dim_feedforward": 2048,
    "hidden_dim": 256,
    "dropout": 0.1,
    "nheads": 8,
    "num_queries": 400,
    "pre_norm": False,
    "masks": False,
    "qpos_dim": 14,
    "action_dim": 14,
}


def get_args_parser(config=None):
    """Create an args namespace from a dict config instead of argparse."""
    values = dict(_DEFAULT_CONFIG)
    if config:
        values.update(config)
    return SimpleNamespace(**values)


def build_ACT_model_and_optimizer(config):
    """Build the ACT model for inference. Optimizer is intentionally omitted."""
    args = get_args_parser(config)
    model = build_ACT_model(args)
    if torch.cuda.is_available():
        model.cuda()
    return model


def build_CNNMLP_model_and_optimizer(config):
    """Build the CNN-MLP model for inference. Optimizer is intentionally omitted."""
    args = get_args_parser(config)
    model = build_CNNMLP_model(args)
    if torch.cuda.is_available():
        model.cuda()
    return model

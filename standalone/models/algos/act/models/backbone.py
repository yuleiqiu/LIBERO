# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Backbone and positional embedding helpers for ACT/DETR-style models."""

import torch
import torch.distributed as dist
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d
from typing import Any, Dict, List, Tuple

from .position_encoding import build_position_encoding


def _is_main_process() -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


class BackboneBase(nn.Module):
    """
    Wrap a CNN backbone and expose the final feature map.
    Freezes backbone parameters when train_backbone is False.
    """

    def __init__(
        self,
        backbone: nn.Module,
        train_backbone: bool,
        num_channels: int,
    ) -> None:
        super().__init__()
        if not train_backbone:
            for param in backbone.parameters():
                param.requires_grad_(False)
        return_layers = {"layer4": "feature_map"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return a dict of feature maps keyed by layer name."""
        xs = self.body(tensor)
        return xs


class Backbone(BackboneBase):
    """ResNet backbone with FrozenBatchNorm2d and optional dilation."""

    def __init__(
        self,
        name: str,
        train_backbone: bool,
        dilation: bool,
    ) -> None:
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=_is_main_process(),
            norm_layer=FrozenBatchNorm2d,
        )  # pretrained # TODO do we want frozen batch_norm??
        num_channels = 512 if name in ("resnet18", "resnet34") else 2048
        super().__init__(backbone, train_backbone, num_channels)


class Joiner(nn.Sequential):
    """Bundle a backbone with its position embedding generator."""

    def __init__(self, backbone: nn.Module, position_embedding: nn.Module) -> None:
        super().__init__(backbone, position_embedding)

    def forward(self, tensor: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Return features and positional embeddings for each backbone output."""
        xs = self[0](tensor)
        out: List[torch.Tensor] = []
        pos: List[torch.Tensor] = []
        for _, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))

        return out, pos


def build_backbone(config: Any, train_backbone: bool = True) -> Joiner:
    """Build a backbone + positional embedding joiner from config."""
    position_embedding = build_position_encoding(
        config.hidden_dim, config.position_embedding
    )
    backbone = Backbone(config.backbone, train_backbone, config.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model

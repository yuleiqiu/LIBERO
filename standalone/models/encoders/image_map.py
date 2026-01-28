from typing import List, Tuple

import torch.nn as nn
import torchvision
from torch import Tensor
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from standalone.models.algos.act.models.position_encoding import build_position_encoding


class ImageMapEncoder(nn.Module):
    """Encoder that outputs a 2D feature map and positional embedding."""

    def __init__(
        self,
        backbone: str = "resnet18",
        position_embedding: str = "sine",
        hidden_dim: int = 256,
        dilation: bool = False,
        train_backbone: bool = True,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.position_embedding = build_position_encoding(
            hidden_dim, position_embedding
        )

        resnet_fn = getattr(torchvision.models, backbone, None)
        if resnet_fn is None:
            raise ValueError(f"unsupported backbone: {backbone}")
        weights = None
        if pretrained:
            enum_base = backbone
            if backbone.startswith("resnet"):
                enum_base = "ResNet" + backbone[len("resnet") :]
            else:
                enum_base = backbone[0].upper() + backbone[1:]
            enum_name = f"{enum_base}_Weights"
            weights_enum = getattr(torchvision.models, enum_name, None)
            if weights_enum is not None:
                weights = weights_enum.DEFAULT
        try:
            backbone_model = resnet_fn(
                replace_stride_with_dilation=[False, False, dilation],
                weights=weights,
                norm_layer=FrozenBatchNorm2d,
            )
        except TypeError:
            backbone_model = resnet_fn(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=pretrained,
                norm_layer=FrozenBatchNorm2d,
            )
        if not train_backbone:
            for param in backbone_model.parameters():
                param.requires_grad_(False)

        return_layers = {"layer4": "feature_map"}
        self.backbone = IntermediateLayerGetter(backbone_model, return_layers=return_layers)
        self.num_channels = 512 if backbone in ("resnet18", "resnet34") else 2048 # resnet18/34's layer4 output channels is 512, others are 2048

    def forward(self, images: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        xs = self.backbone(images)
        features = []
        pos = []
        for _, value in xs.items():
            features.append(value)
            pos.append(self.position_embedding(value).to(value.dtype))
        return features, pos

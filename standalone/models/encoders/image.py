import torch
import torch.nn as nn

try:
    import torchvision
except ImportError as exc:
    raise ImportError(
        "torchvision is required for ImageEncoder; install it first."
    ) from exc

from standalone.models.algos.dp.model.vision.crop_randomizer import CropRandomizer


class ImageEncoder(nn.Module):
    def __init__(
        self,
        input_shape,
        output_dim,
        backbone="resnet18",
        pretrained=False,
        remove_layer_num=2,
        no_stride=False,
        crop_randomizer=None,
    ):
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError(f"input_shape must be (C, H, W), got {input_shape}")
        if remove_layer_num < 0:
            raise ValueError("remove_layer_num must be non-negative")

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
            resnet = resnet_fn(weights=weights)
        except TypeError:
            resnet = resnet_fn(pretrained=pretrained)

        if input_shape[0] != 3:
            resnet.conv1 = nn.Conv2d(
                input_shape[0],
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

        if no_stride:
            resnet.conv1.stride = (1, 1)
            resnet.maxpool.stride = (1, 1)

        layers = list(resnet.children())
        if remove_layer_num >= len(layers):
            raise ValueError("remove_layer_num is too large for resnet")
        if remove_layer_num:
            layers = layers[:-remove_layer_num]
        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            out = self.backbone(dummy)
            if out.ndim == 4:
                out = self.pool(out)
            out_dim = out.reshape(1, -1).shape[1]

        self.proj = nn.Linear(out_dim, int(output_dim))
        self.output_dim = int(output_dim)

        self.randomizer = None
        if crop_randomizer:
            self.randomizer = CropRandomizer(**crop_randomizer)

    def forward(self, x):
        if self.randomizer is not None:
            x = self.randomizer.forward_in(x)
        h = self.backbone(x)
        if h.ndim == 4:
            h = self.pool(h)
        h = h.view(h.shape[0], -1)
        h = self.proj(h)
        if self.randomizer is not None:
            h = self.randomizer.forward_out(h)
        return h

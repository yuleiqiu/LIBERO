import torch
import torch.nn as nn

try:
    import torchvision
except ImportError as exc:
    raise ImportError(
        "torchvision is required for ResnetEncoder; install it first."
    ) from exc


class ResnetEncoder(nn.Module):
    def __init__(
        self,
        input_shape,
        output_size,
        pretrained=False,
        remove_layer_num=2,
        no_stride=False,
    ):
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError(f"input_shape must be (C, H, W), got {input_shape}")
        if remove_layer_num < 0:
            raise ValueError("remove_layer_num must be non-negative")

        if hasattr(torchvision.models, "ResNet18_Weights"):
            weights = (
                torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
            )
            resnet = torchvision.models.resnet18(weights=weights)
        else:
            resnet = torchvision.models.resnet18(pretrained=pretrained)

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
            raise ValueError("remove_layer_num is too large for resnet18")
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

        self.proj = nn.Linear(out_dim, int(output_size))

    def forward(self, x):
        h = self.backbone(x)
        if h.ndim == 4:
            h = self.pool(h)
        h = h.view(h.shape[0], -1)
        return self.proj(h)

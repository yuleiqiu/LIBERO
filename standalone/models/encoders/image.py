import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


def _replace_submodules(root_module: nn.Module, predicate, func) -> nn.Module:
    if predicate(root_module):
        return func(root_module)
    replace_list = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    for *parents, k in replace_list:
        parent_module = root_module
        if parents:
            parent_module = root_module.get_submodule(".".join(parents))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    assert not any(predicate(m) for _, m in root_module.named_modules(remove_duplicate=True))
    return root_module


class SpatialSoftmax(nn.Module):
    def __init__(self, input_shape, num_kp=None):
        super().__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape
        if num_kp is not None:
            self.nets = nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self._in_w),
            np.linspace(-1.0, 1.0, self._in_h),
        )
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features):
        if self.nets is not None:
            features = self.nets(features)
        features = features.reshape(-1, self._in_h * self._in_w)
        attention = torch.softmax(features, dim=-1)
        expected_xy = attention @ self.pos_grid
        return expected_xy.view(-1, self._out_c, 2)


class DPImageEncoder(nn.Module):
    def __init__(
        self,
        input_shape,
        output_dim=None,
        backbone="resnet18",
        pretrained=False,
        remove_layer_num=2,
        no_stride=False,
        crop_randomizer=None,
        use_group_norm=True,
        spatial_softmax_num_keypoints=32,
        mask_alpha=0.2,
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

        if use_group_norm:
            if pretrained:
                raise ValueError("use_group_norm requires pretrained=False")
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=max(1, x.num_features // 16), num_channels=x.num_features),
            )

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            out = self.backbone(dummy)
            feature_map_shape = out.shape[1:]

        self.pool = SpatialSoftmax(
            feature_map_shape, num_kp=int(spatial_softmax_num_keypoints)
        )
        base_dim = int(spatial_softmax_num_keypoints) * 2
        out_dim = int(output_dim) if output_dim is not None else base_dim
        self.proj = nn.Linear(base_dim, out_dim)
        self.relu = nn.ReLU()
        self.output_dim = out_dim

        self.randomizer = None
        if crop_randomizer:
            self.randomizer = CropRandomizer(**crop_randomizer)
        self.mask_alpha = mask_alpha

    def forward(self, x, mask=None):
        if self.randomizer is not None:
            x, crop_params = self.randomizer.forward_in(x, return_params=True)
            if mask is not None:
                mask = self.randomizer.forward_in(
                    mask, params=crop_params, return_params=False
                )
        h = self.backbone(x)
        if mask is not None:
            if torch.any((mask < 0) | (mask > 1)):
                raise ValueError("DPImageEncoder expects a binary mask with values in [0, 1]")
            mask_low = F.interpolate(mask.float(), size=h.shape[-2:], mode="nearest")
            h = h * (self.mask_alpha + (1.0 - self.mask_alpha) * mask_low)
        h = torch.flatten(self.pool(h), start_dim=1)
        h = self.relu(self.proj(h))
        if self.randomizer is not None:
            h = self.randomizer.forward_out(h)
        return h

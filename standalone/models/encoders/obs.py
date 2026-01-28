import dataclasses
import torch
import torch.nn as nn

from standalone.models.encoders.image import DPImageEncoder, ImageEncoder
from standalone.models.encoders.lowdim import LowdimEncoder


class ObsEncoder(nn.Module):
    def __init__(
        self,
        image_keys,
        lowdim_keys,
        image_encoder=None,
        lowdim_encoder=None,
        image_fusion="concat",
        output_dim=None,
    ):
        super().__init__()
        self.image_keys = list(image_keys or [])
        self.lowdim_keys = list(lowdim_keys or [])
        self.image_fusion = str(image_fusion)
        self.image_encoder = image_encoder
        self.lowdim_encoder = lowdim_encoder
        self.output_dim = None
        self.fusion_proj = None

        # TODO: For DP, support per-camera encoders with independent weights
        # to match the Diffusion Policy paper (currently encoders are shared).

        image_dim = 0
        if self.image_keys and self.image_encoder is None:
            raise ValueError("image_encoder is required when image_keys are provided")
        if self.image_encoder is not None:
            if isinstance(self.image_encoder, nn.ModuleList):
                if len(self.image_encoder) != len(self.image_keys):
                    raise ValueError(
                        "image_encoder ModuleList must match image_keys length "
                        f"({len(self.image_encoder)} vs {len(self.image_keys)})"
                    )
                dims = [int(getattr(enc, "output_dim", 0)) for enc in self.image_encoder]
                if not dims or any(d <= 0 for d in dims):
                    raise ValueError("all image encoders must define output_dim")
                if len(set(dims)) != 1:
                    raise ValueError("all image encoders must have the same output_dim")
                image_dim = dims[0]
            else:
                image_dim = int(getattr(self.image_encoder, "output_dim", 0))
                if image_dim <= 0:
                    raise ValueError("image_encoder must define output_dim")

        lowdim_dim = 0
        if self.lowdim_keys:
            if self.lowdim_encoder is None:
                raise ValueError("lowdim_encoder is required when lowdim_keys are provided")
            lowdim_dim = int(getattr(self.lowdim_encoder, "output_dim", 0))
            if lowdim_dim <= 0:
                raise ValueError("lowdim_encoder must define output_dim")

        if self.image_keys:
            if self.image_fusion == "mean":
                fused_image_dim = image_dim
            else:
                fused_image_dim = image_dim * len(self.image_keys)
        else:
            fused_image_dim = 0

        total_dim = fused_image_dim + lowdim_dim
        if total_dim <= 0:
            raise ValueError("ObsEncoder requires at least one image or lowdim key")

        if output_dim is not None and int(output_dim) != total_dim:
            self.fusion_proj = nn.Linear(total_dim, int(output_dim))
            self.output_dim = int(output_dim)
        else:
            self.output_dim = total_dim

    def _to_image_tensor(self, value):
        if value.ndim == 3:
            value = value.unsqueeze(0)
        if value.ndim != 4:
            raise ValueError(f"expected image dims (B, H, W, C) or (B, C, H, W), got {value.shape}")
        if value.shape[1] in (1, 3):
            return value
        if value.shape[-1] in (1, 3):
            return value.permute(0, 3, 1, 2)
        raise ValueError(f"cannot infer channel axis for image shape {value.shape}")

    def _to_lowdim_tensor(self, value):
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.ndim > 2:
            value = value.reshape(value.shape[0], -1)
        return value

    def forward(self, obs):
        features = []
        if self.image_keys:
            image_feats = []
            if isinstance(self.image_encoder, nn.ModuleList):
                for key, encoder in zip(self.image_keys, self.image_encoder, strict=True):
                    if key not in obs:
                        raise KeyError(f"missing image key: {key}")
                    x = self._to_image_tensor(obs[key])
                    image_feats.append(encoder(x))
            else:
                for key in self.image_keys:
                    if key not in obs:
                        raise KeyError(f"missing image key: {key}")
                    x = self._to_image_tensor(obs[key])
                    image_feats.append(self.image_encoder(x))
            if self.image_fusion == "mean":
                fused = torch.stack(image_feats, dim=0).mean(dim=0)
            else:
                fused = torch.cat(image_feats, dim=-1)
            features.append(fused)

        if self.lowdim_keys:
            parts = []
            for key in self.lowdim_keys:
                if key not in obs:
                    raise KeyError(f"missing lowdim key: {key}")
                x = self._to_lowdim_tensor(obs[key])
                parts.append(x)
            lowdim_in = torch.cat(parts, dim=-1)
            lowdim_feat = self.lowdim_encoder(lowdim_in)
            features.append(lowdim_feat)

        out = features[0] if len(features) == 1 else torch.cat(features, dim=-1)
        if self.fusion_proj is not None:
            out = self.fusion_proj(out)
        return out

    def output_shape(self):
        return (self.output_dim,)


def _flatten_shape(shape):
    if len(shape) == 0:
        return 1
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def build_obs_encoder(obs_shapes, image_keys, lowdim_keys, cfg=None):
    if dataclasses.is_dataclass(cfg):
        cfg = dataclasses.asdict(cfg)
    cfg = dict(cfg or {})
    image_cfg = dict(cfg.get("image", {}))
    lowdim_cfg = dict(cfg.get("lowdim", {}))
    fusion_cfg = dict(cfg.get("fusion", {}))

    image_keys = list(image_keys or [])
    lowdim_keys = list(lowdim_keys or [])

    image_encoder = None
    if image_keys:
        first_key = image_keys[0]
        if first_key not in obs_shapes:
            raise KeyError(f"missing obs shape for image key: {first_key}")
        shape = obs_shapes[first_key]
        for key in image_keys[1:]:
            if key not in obs_shapes:
                raise KeyError(f"missing obs shape for image key: {key}")
            if tuple(obs_shapes[key]) != tuple(shape):
                raise ValueError(
                    "image encoder expects identical shapes across image keys; "
                    f"{first_key}={shape}, {key}={obs_shapes[key]}"
                )
        if len(shape) >= 1 and shape[0] in (1, 3):
            input_shape = shape
        else:
            input_shape = (shape[-1], shape[-3], shape[-2])

        crop_cfg = dict(image_cfg.get("crop_randomizer") or {})
        encoder_input_shape = input_shape
        if crop_cfg.get("enable"):
            crop_cfg = dict(crop_cfg)
            crop_cfg.pop("enable", None)
            crop_cfg.setdefault("input_shape", input_shape)
            if crop_cfg.get("pos_enc"):
                encoder_input_shape = (
                    int(input_shape[0]) + 2,
                    int(input_shape[1]),
                    int(input_shape[2]),
                )
            image_cfg["crop_randomizer"] = crop_cfg
        else:
            image_cfg["crop_randomizer"] = None

        image_type = str(image_cfg.get("type", "resnet")).lower()
        if image_type == "resnet":
            image_encoder = ImageEncoder(
                input_shape=encoder_input_shape,
                output_dim=image_cfg.get("output_dim", 128),
                backbone=image_cfg.get("backbone", "resnet18"),
                pretrained=image_cfg.get("pretrained", False),
                remove_layer_num=image_cfg.get("remove_layer_num", 2),
                no_stride=image_cfg.get("no_stride", False),
                crop_randomizer=image_cfg.get("crop_randomizer"),
            )
        elif image_type == "dp_resnet":
            def _build_dp_encoder():
                return DPImageEncoder(
                    input_shape=encoder_input_shape,
                    output_dim=image_cfg.get("output_dim"),
                    backbone=image_cfg.get("backbone", "resnet18"),
                    pretrained=image_cfg.get("pretrained", False),
                    remove_layer_num=image_cfg.get("remove_layer_num", 2),
                    no_stride=image_cfg.get("no_stride", False),
                    crop_randomizer=image_cfg.get("crop_randomizer"),
                    use_group_norm=image_cfg.get("use_group_norm", True),
                    spatial_softmax_num_keypoints=image_cfg.get(
                        "spatial_softmax_num_keypoints", 32
                    ),
                )

            if image_cfg.get("use_separate_rgb_encoder_per_camera", False):
                image_encoder = nn.ModuleList([_build_dp_encoder() for _ in image_keys])
            else:
                image_encoder = _build_dp_encoder()
        else:
            raise ValueError(f"unsupported image encoder type: {image_type}")

    lowdim_encoder = None
    if lowdim_keys:
        input_dim = 0
        for key in lowdim_keys:
            if key not in obs_shapes:
                raise KeyError(f"missing obs shape for lowdim key: {key}")
            input_dim += _flatten_shape(obs_shapes[key])
        lowdim_type = str(lowdim_cfg.get("type", "mlp")).lower()
        if lowdim_type in ("identity", "none"):
            lowdim_encoder = nn.Identity()
            lowdim_encoder.output_dim = input_dim
        elif lowdim_type == "mlp":
            lowdim_encoder = LowdimEncoder(
                input_dim=input_dim,
                output_dim=lowdim_cfg.get("output_dim"),
                hidden_dims=lowdim_cfg.get("hidden_dims"),
            )
        else:
            raise ValueError(f"unsupported lowdim encoder type: {lowdim_type}")

    obs_encoder = ObsEncoder(
        image_keys=image_keys,
        lowdim_keys=lowdim_keys,
        image_encoder=image_encoder,
        lowdim_encoder=lowdim_encoder,
        image_fusion=fusion_cfg.get("image_fusion", "concat"),
        output_dim=fusion_cfg.get("output_dim"),
    )
    return obs_encoder

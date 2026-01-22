from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CropRandomizerConfig:
    enable: bool = False
    crop_height: int = 76
    crop_width: int = 76
    num_crops: int = 1
    pos_enc: bool = False


@dataclass
class ImageEncoderConfig:
    type: str = "resnet"
    backbone: str = "resnet18"
    output_dim: int = 128
    pretrained: bool = False
    remove_layer_num: int = 2
    no_stride: bool = False
    crop_randomizer: CropRandomizerConfig = field(default_factory=CropRandomizerConfig)


@dataclass
class LowdimEncoderConfig:
    type: str = "mlp"
    output_dim: Optional[int] = None
    hidden_dims: List[int] = field(default_factory=list)


@dataclass
class ObsEncoderFusionConfig:
    image_fusion: str = "concat"
    output_dim: Optional[int] = None


@dataclass
class ObsEncoderConfig:
    image: ImageEncoderConfig = field(default_factory=ImageEncoderConfig)
    lowdim: LowdimEncoderConfig = field(default_factory=LowdimEncoderConfig)
    fusion: ObsEncoderFusionConfig = field(default_factory=ObsEncoderFusionConfig)

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ImageTransformConfig:
    weight: float = 1.0
    type: str = "Identity"
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageTransformsConfig:
    enable: bool = False
    max_num_transforms: int = 3
    random_order: bool = False
    tfs: Dict[str, ImageTransformConfig] = field(
        default_factory=lambda: {
            "brightness": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"brightness": (0.8, 1.2)},
            ),
            "contrast": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"contrast": (0.8, 1.2)},
            ),
            "saturation": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"saturation": (0.5, 1.5)},
            ),
            "hue": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"hue": (-0.05, 0.05)},
            ),
            "sharpness": ImageTransformConfig(
                weight=1.0,
                type="SharpnessJitter",
                kwargs={"sharpness": (0.5, 1.5)},
            ),
            "random_crop": ImageTransformConfig(
                weight=0.0,
                type="RandomCrop",
                kwargs={"padding": 4, "pad_if_needed": True},
            ),
            "affine": ImageTransformConfig(
                weight=1.0,
                type="RandomAffine",
                kwargs={"degrees": (-5.0, 5.0), "translate": (0.05, 0.05)},
            ),
        }
    )


@dataclass
class DataConfig:
    demo_file: str = ""
    obs_keys: str = "gripper_states,joint_states"
    image_keys: str = "agentview_rgb,eye_in_hand_rgb"
    obs_horizon: int = 1
    predict_horizon: int = 1
    sync_horizons_from_policy: bool = False
    image_norm: str = "none"
    image_transforms: ImageTransformsConfig = field(default_factory=ImageTransformsConfig)
    obs_key_mapping: Optional[Dict[str, str]] = None
    train_ratio: float = 0.9
    val_ratio: float = 0.1
    seed: int = 10000

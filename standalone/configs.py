from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    demo_file: str = ""
    obs_keys: str = "gripper_states,joint_states"
    image_keys: str = "agentview_rgb,eye_in_hand_rgb"
    obs_horizon: int = 1
    predict_horizon: int = 1
    normalize_obs: bool = False
    obs_stats_path: Optional[str] = None
    train_ratio: float = 0.9
    val_ratio: float = 0.1
    seed: int = 10000


@dataclass
class ModelConfig:
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    action_squash: bool = True
    exec_horizon: Optional[int] = None
    image_embed_dim: int = 128
    image_encoder_pretrained: bool = False
    image_encoder_remove_layer_num: int = 2
    image_encoder_no_stride: bool = False


@dataclass
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    batch_size: int = 32
    epochs: int = 10
    lr: float = 1e-4
    device: str = "cuda:0"
    save_dir: str = "standalone/standalone_runs/run_001"
    grad_clip: Optional[float] = None


@dataclass
class EvalConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ckpt: str = ""
    split_path: Optional[str] = None
    batch_size: int = 32
    device: str = "cuda:0"


@dataclass
class RolloutConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ckpt: str = ""
    steps: int = 8
    sample_index: int = 0
    device: str = "cuda:0"

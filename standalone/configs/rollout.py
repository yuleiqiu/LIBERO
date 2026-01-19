from dataclasses import dataclass, field
from typing import Any, Optional

from standalone.configs.data import DataConfig
from standalone.configs.policy import PolicyConfig


@dataclass
class RolloutConfig:
    data: DataConfig = field(default_factory=DataConfig)
    policy: Any = field(default_factory=PolicyConfig)
    ckpt: str = ""
    use_ckpt_config: bool = True
    init_states: Optional[str] = None
    env_horizon: int = 2000
    steps: int = 8
    sample_index: int = 0
    n_rollouts: int = 1
    use_mp: bool = False
    num_procs: int = 1
    warmup_steps: int = 5
    save_videos: int = 0
    video_camera: str = ""
    video_fps: int = 30
    video_dir: str = ""
    device: str = "cuda:0"

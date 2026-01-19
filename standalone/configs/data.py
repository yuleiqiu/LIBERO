from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class DataConfig:
    demo_file: str = ""
    obs_keys: str = "gripper_states,joint_states"
    image_keys: str = "agentview_rgb,eye_in_hand_rgb"
    obs_horizon: int = 1
    predict_horizon: int = 1
    action_shift: int = 0  # Offset actions relative to obs; 1 means predict next-step actions.
    normalize_obs: bool = False
    obs_stats_path: Optional[str] = None
    obs_key_mapping: Optional[Dict[str, str]] = None
    train_ratio: float = 0.9
    val_ratio: float = 0.1
    seed: int = 10000

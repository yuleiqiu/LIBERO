from dataclasses import dataclass, field
from typing import Any, Optional

from standalone.configs.data import DataConfig
from standalone.configs.policy import PolicyConfig


@dataclass
class PathsConfig:
    save_dir: str = "standalone/standalone_runs/run_001"


@dataclass
class TrainLoopConfig:
    batch_size: int = 32
    epochs: int = 10
    val_every: int = 10
    lr: float = 1e-4
    grad_clip: Optional[float] = None
    device: str = "cuda:0"


@dataclass
class TrainRolloutConfig:
    every: int = 20
    init_states_dir: Optional[str] = field(
        default="./libero/libero/init_files",
        metadata={
            "help": (
                "Root directory for init states (e.g., ./libero/libero/init_files). "
                "Use the directory, not a single .init file."
            )
        },
    )
    env_horizon: int = 2000
    per_anchor: int = 2
    steps: int = 8
    warmup_steps: int = 5
    use_mp: bool = False
    num_procs: int = 1


@dataclass
class LoggingConfig:
    use_wandb: bool = False
    wandb_project: str = "libero-standalone"
    wandb_entity: Optional[str] = None
    experiment_name: Optional[str] = None


@dataclass
class TrainConfig:
    resume: bool = False
    saved_config_path: Optional[str] = None
    data: DataConfig = field(default_factory=DataConfig)
    policy: Any = field(default_factory=PolicyConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    training: TrainLoopConfig = field(default_factory=TrainLoopConfig)
    rollout: TrainRolloutConfig = field(default_factory=TrainRolloutConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

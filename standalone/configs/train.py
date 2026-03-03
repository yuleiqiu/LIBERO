from dataclasses import dataclass, field
from typing import Optional

from standalone.configs.data import DataConfig
from standalone.configs.policy import AdamWOptimizerConfig, PolicyConfig, SchedulerConfig


@dataclass
class PathsConfig:
    save_dir: str = "standalone/standalone_runs/run_001"


@dataclass
class TrainLoopConfig:
    batch_size: int = 32
    epochs: int = 10
    val_every: int = 10
    lr: float = 1e-4
    optimizer: AdamWOptimizerConfig = field(
        default_factory=lambda: AdamWOptimizerConfig(
            lr=1e-4,
            weight_decay=1e-6,
            betas=[0.9, 0.999],
            eps=1e-8,
        )
    )
    scheduler: SchedulerConfig = field(
        default_factory=lambda: SchedulerConfig(name="cosine", warmup_steps=500)
    )
    grad_clip: Optional[float] = None
    device: str = "cuda:0"
    save_ckpt_every: int = 1
    save_topk: int = 5


@dataclass
class TrainRolloutConfig:
    every: int = 20
    bddl_file: Optional[str] = field(
        default=None,
        metadata={"help": "Optional BDDL override for rollout env/init resolution."},
    )
    init_states_dir: Optional[str] = field(
        default="./libero/libero/init_files",
        metadata={
            "help": (
                "Root directory for init states (e.g., ./libero/libero/init_files). "
                "Use the directory, not a single .init file."
            )
        },
    )
    init_states_file: Optional[str] = field(
        default=None,
        metadata={"help": "Optional explicit .pruned_init file override for rollout."},
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
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    training: TrainLoopConfig = field(default_factory=TrainLoopConfig)
    rollout: TrainRolloutConfig = field(default_factory=TrainRolloutConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from standalone.configs.encoder import ObsEncoderConfig
from standalone.configs.normalizer import NormalizerConfig


@dataclass
class ACTModelConfig:
    """ACT model config.

    Attributes:
        backbone: ResNet backbone name.
        pretrained: Load torchvision pretrained weights for the backbone.
        dilation: Replace last strides with dilation in the backbone.
        position_embedding: Position embedding type (e.g., "sine").
        enc_layers: Transformer encoder layer count.
        dec_layers: Transformer decoder layer count.
        dim_feedforward: Transformer FFN dimension.
        hidden_dim: Transformer model dimension.
        dropout: Dropout rate in transformer blocks.
        nheads: Number of attention heads.
        pre_norm: Use pre-layernorm transformer blocks.
        action_dim: Action dimension.
        proprio_dim: Proprioception (low-dim) input dimension.
        chunk_size: Action chunk length / number of queries.
        camera_names: Ordered camera names for image inputs.
    """
    backbone: str = "resnet18"
    pretrained: bool = False
    dilation: bool = False
    position_embedding: str = "sine"
    enc_layers: int = 4
    dec_layers: int = 7
    dim_feedforward: int = 3200
    hidden_dim: int = 512
    dropout: float = 0.1
    nheads: int = 8
    pre_norm: bool = False
    action_dim: Optional[int] = None
    proprio_dim: Optional[int] = None
    chunk_size: int = 100
    camera_names: List[str] = field(default_factory=list)


@dataclass
class AdamWOptimizerConfig:
    """AdamW optimizer settings."""

    lr: Optional[float] = None
    weight_decay: Optional[float] = None
    betas: Optional[List[float]] = None
    eps: Optional[float] = None


@dataclass
class SchedulerConfig:
    """Learning rate scheduler settings."""
    name: str = "none"
    warmup_steps: int = 0
    num_training_steps: Optional[int] = None
    min_lr: float = 0.0


@dataclass
class ACTConfig:
    exec_horizon: Optional[int] = None
    lr_backbone: float = 1e-5
    kl_weight: float = 10.0
    temporal_ensemble_coeff: Optional[float] = None
    model: ACTModelConfig = field(default_factory=ACTModelConfig)
    optimizer: AdamWOptimizerConfig = field(default_factory=AdamWOptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    def act_config_dict(self):
        """Return ACT policy config; drop runtime/training-only keys."""
        data = asdict(self)
        data.pop("exec_horizon", None)
        data.pop("optimizer", None)
        return data


@dataclass
class CNNMLPModelConfig:
    """Model config for CNNMLP; pretrained/dilation control ResNet backbone weights/stride."""
    backbone: str = "resnet18"
    pretrained: bool = False
    hidden_dim: int = 256
    position_embedding: str = "sine"
    dilation: bool = False
    action_dim: Optional[int] = None
    qpos_dim: Optional[int] = None
    chunk_size: int = 400
    camera_names: List[str] = field(default_factory=list)


@dataclass
class CNNMLPConfig:
    exec_horizon: Optional[int] = None
    lr_backbone: float = 1e-5
    model: CNNMLPModelConfig = field(default_factory=CNNMLPModelConfig)
    optimizer: AdamWOptimizerConfig = field(default_factory=AdamWOptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    def cnnmlp_config_dict(self):
        data = asdict(self)
        data.pop("exec_horizon", None)
        data.pop("optimizer", None)
        return data

    def act_config_dict(self):
        return self.cnnmlp_config_dict()


@dataclass
class DiffusionModelConfig:
    horizon: Optional[int] = 16
    n_action_steps: Optional[int] = 8
    n_obs_steps: Optional[int] = 2
    num_inference_steps: Optional[int] = None
    diffusion_step_embed_dim: int = 128
    down_dims: List[int] = field(default_factory=lambda: [512, 1024, 2048])
    kernel_size: int = 5
    n_groups: int = 8
    cond_predict_scale: bool = True
    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0
    do_mask_loss_for_padding: bool = False

    def dp_config_dict(self):
        data = asdict(self)
        return {k: v for k, v in data.items() if v is not None}


@dataclass
class DiffusionConfig:
    exec_horizon: Optional[int] = None
    # DP-only data slicing controls (leave None to use defaults in train.py)
    action_horizon: Optional[int] = None
    action_start_offset: Optional[int] = None
    encoder: ObsEncoderConfig = field(default_factory=ObsEncoderConfig)
    model: DiffusionModelConfig = field(default_factory=DiffusionModelConfig)
    normalizer: NormalizerConfig = field(default_factory=NormalizerConfig)
    optimizer: AdamWOptimizerConfig = field(default_factory=AdamWOptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    def dp_config_dict(self):
        data = self.model.dp_config_dict()
        return data


@dataclass
class PolicyConfig:
    name: str = "cnnmlp"
    act: ACTConfig = field(default_factory=ACTConfig)
    cnnmlp: CNNMLPConfig = field(default_factory=CNNMLPConfig)
    dp: DiffusionConfig = field(default_factory=DiffusionConfig)


def apply_policy_config(cfg):
    policy_cfg = getattr(cfg, "policy", None)
    if not isinstance(policy_cfg, PolicyConfig):
        raise TypeError(f"policy must be PolicyConfig, got {type(policy_cfg)}")
    return policy_cfg


def resolve_policy_config(cfg):
    policy_cfg = getattr(cfg, "policy", None)
    if not isinstance(policy_cfg, PolicyConfig):
        raise TypeError(f"policy must be PolicyConfig, got {type(policy_cfg)}")
    policy_name = str(getattr(policy_cfg, "name", "")).lower()
    if policy_name == "act":
        return policy_cfg.act
    if policy_name == "cnnmlp":
        return policy_cfg.cnnmlp
    if policy_name == "dp":
        return policy_cfg.dp
    raise ValueError(f"unsupported policy: {policy_name}")


def get_policy_param(cfg, key, default=None):
    try:
        resolved = resolve_policy_config(cfg)
    except Exception:
        return default
    if key == "act_config":
        return resolved.act_config_dict()
    if key == "cnnmlp_config":
        return resolved.cnnmlp_config_dict()
    if key == "dp_config":
        return resolved.dp_config_dict()
    if key == "encoder_config":
        return resolved.encoder
    return getattr(resolved, key, default)


def serialize_policy_config(cfg):
    policy_cfg = getattr(cfg, "policy", None)
    if not isinstance(policy_cfg, PolicyConfig):
        raise TypeError(f"policy must be PolicyConfig, got {type(policy_cfg)}")
    policy_name = str(getattr(policy_cfg, "name", "")).lower()
    if policy_name == "act":
        return {"name": policy_cfg.name, "act": asdict(policy_cfg.act)}
    if policy_name == "cnnmlp":
        return {"name": policy_cfg.name, "cnnmlp": asdict(policy_cfg.cnnmlp)}
    if policy_name == "dp":
        return {"name": policy_cfg.name, "dp": asdict(policy_cfg.dp)}
    return {"name": policy_cfg.name}

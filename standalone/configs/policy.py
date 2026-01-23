from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from standalone.configs.encoder import ObsEncoderConfig
from standalone.configs.normalizer import NormalizerConfig


@dataclass
class ACTConfig:
    exec_horizon: Optional[int] = None
    lr_backbone: float = 1e-5
    kl_weight: float = 10.0
    backbone: str = "resnet18"
    dilation: bool = False
    position_embedding: str = "sine"
    enc_layers: int = 4
    dec_layers: int = 6
    dim_feedforward: int = 2048
    hidden_dim: int = 256
    dropout: float = 0.1
    nheads: int = 8
    pre_norm: bool = False
    masks: bool = False  # TODO: rename to return_interm_layers (backbone intermediate outputs).

    def act_config_dict(self):
        data = asdict(self)
        data.pop("exec_horizon", None)
        return data


@dataclass
class CNNMLPConfig:
    exec_horizon: Optional[int] = None
    backbone: str = "resnet18"
    lr_backbone: float = 1e-5
    hidden_dim: int = 256
    position_embedding: str = "sine"
    dilation: bool = False
    masks: bool = False

    def act_config_dict(self):
        data = asdict(self)
        data.pop("exec_horizon", None)
        return data


@dataclass
class DiffusionModelConfig:
    horizon: Optional[int] = None
    n_action_steps: Optional[int] = None
    n_obs_steps: Optional[int] = None
    num_inference_steps: Optional[int] = None
    diffusion_step_embed_dim: int = 256
    down_dims: List[int] = field(default_factory=lambda: [256, 512, 1024])
    kernel_size: int = 5
    n_groups: int = 8
    cond_predict_scale: bool = True
    noise_scheduler: Dict[str, Any] = field(default_factory=dict)

    def dp_config_dict(self):
        data = asdict(self)
        return {k: v for k, v in data.items() if v is not None}


@dataclass
class DiffusionConfig:
    exec_horizon: Optional[int] = None
    encoder: ObsEncoderConfig = field(default_factory=ObsEncoderConfig)
    model: DiffusionModelConfig = field(default_factory=DiffusionModelConfig)
    normalizer: NormalizerConfig = field(default_factory=NormalizerConfig)

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

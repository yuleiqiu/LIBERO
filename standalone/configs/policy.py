from dataclasses import asdict, dataclass, field
from typing import Optional


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
class PolicyConfig:
    name: str = "cnnmlp"
    act: ACTConfig = field(default_factory=ACTConfig)
    cnnmlp: CNNMLPConfig = field(default_factory=CNNMLPConfig)


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
    raise ValueError(f"unsupported policy: {policy_name}")


def get_policy_param(cfg, key, default=None):
    try:
        resolved = resolve_policy_config(cfg)
    except Exception:
        return default
    if key == "act_config":
        return resolved.act_config_dict()
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
    return {"name": policy_cfg.name}

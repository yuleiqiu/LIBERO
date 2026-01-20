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
    image_norm: str = "none"
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


def _coerce_act_config(value):
    if isinstance(value, ACTConfig):
        return value
    if value is None:
        return ACTConfig()
    if isinstance(value, dict):
        return ACTConfig(**value)
    raise TypeError(f"unsupported ACT config type: {type(value)}")


def _coerce_cnnmlp_config(value):
    if isinstance(value, CNNMLPConfig):
        return value
    if value is None:
        return CNNMLPConfig()
    if isinstance(value, dict):
        return CNNMLPConfig(**value)
    raise TypeError(f"unsupported CNNMLP config type: {type(value)}")


def _normalize_policy_config(cfg):
    policy_cfg = getattr(cfg, "policy", None)
    if policy_cfg is None:
        policy_cfg = PolicyConfig()
    elif isinstance(policy_cfg, PolicyConfig):
        policy_cfg.act = _coerce_act_config(policy_cfg.act)
        policy_cfg.cnnmlp = _coerce_cnnmlp_config(policy_cfg.cnnmlp)
    elif isinstance(policy_cfg, str):
        policy_cfg = PolicyConfig(name=policy_cfg)
    elif isinstance(policy_cfg, dict):
        policy_cfg = PolicyConfig(
            name=policy_cfg.get("name", "cnnmlp"),
            act=_coerce_act_config(policy_cfg.get("act")),
            cnnmlp=_coerce_cnnmlp_config(policy_cfg.get("cnnmlp")),
        )
    else:
        raise TypeError(f"unsupported policy config type: {type(policy_cfg)}")
    cfg.policy = policy_cfg
    return cfg.policy


def apply_policy_config(cfg):
    return _normalize_policy_config(cfg)


def resolve_policy_config(cfg):
    policy_cfg = _normalize_policy_config(cfg)
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


def normalize_policy_cli_args(argv):
    if not argv:
        return argv
    normalized = []
    skip_next = False
    for idx, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if arg == "--policy" and idx + 1 < len(argv):
            normalized.append(f"--policy.name={argv[idx + 1]}")
            skip_next = True
            continue
        if arg.startswith("--policy="):
            normalized.append(f"--policy.name={arg.split('=', 1)[1]}")
            continue
        normalized.append(arg)
    argv[:] = normalized
    return argv

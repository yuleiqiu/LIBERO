from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class PolicyConfig:
    name: str = "cnnmlp"
    config_path: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ACTConfig:
    exec_horizon: Optional[int] = None
    act_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "lr_backbone": 1e-5,
            "kl_weight": 10.0,
            "backbone": "resnet18",
            "dilation": False,
            "position_embedding": "sine",
            "image_norm": "none",
            "enc_layers": 4,
            "dec_layers": 6,
            "dim_feedforward": 2048,
            "hidden_dim": 256,
            "dropout": 0.1,
            "nheads": 8,
            "pre_norm": False,
            "masks": False,  # TODO: rename to return_interm_layers (backbone intermediate outputs).
        }
    )


@dataclass
class CNNMLPConfig:
    exec_horizon: Optional[int] = None
    act_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "backbone": "resnet18",
            "lr_backbone": 1e-5,
            "hidden_dim": 256,
            "position_embedding": "sine",
            "dilation": False,
            "masks": False,
        }
    )


def _normalize_policy_config(cfg):
    policy_cfg = getattr(cfg, "policy", None)
    if policy_cfg is None:
        cfg.policy = PolicyConfig()
    elif isinstance(policy_cfg, PolicyConfig):
        return policy_cfg
    elif isinstance(policy_cfg, str):
        cfg.policy = PolicyConfig(name=policy_cfg)
    elif isinstance(policy_cfg, dict):
        cfg.policy = PolicyConfig(**policy_cfg)
    else:
        raise TypeError(f"unsupported policy config type: {type(policy_cfg)}")
    return cfg.policy


def _get_policy_defaults(name):
    if name == "act":
        return asdict(ACTConfig())
    if name == "cnnmlp":
        return asdict(CNNMLPConfig())
    return {}


def apply_policy_config(cfg):
    policy_cfg = _normalize_policy_config(cfg)
    policy_path = getattr(policy_cfg, "config_path", None)
    if policy_path:
        print(
            "[warning] policy.config_path is ignored; use dataclass defaults and CLI overrides."
        )
    params = {}
    params.update(_get_policy_defaults(policy_cfg.name))
    if policy_cfg.params:
        params.update(policy_cfg.params)
    policy_cfg.params = params


def _apply_policy_overrides(cfg_obj, params):
    if not isinstance(params, dict):
        return cfg_obj
    if "exec_horizon" in params and params["exec_horizon"] is not None:
        cfg_obj.exec_horizon = params["exec_horizon"]
    if "act_config" in params and isinstance(params["act_config"], dict):
        cfg_obj.act_config.update(params["act_config"])
    return cfg_obj


def resolve_policy_config(cfg):
    policy_cfg = _normalize_policy_config(cfg)
    policy_name = str(getattr(policy_cfg, "name", "")).lower()
    params = getattr(policy_cfg, "params", None) or {}
    if policy_name == "act":
        return _apply_policy_overrides(ACTConfig(), params)
    if policy_name == "cnnmlp":
        return _apply_policy_overrides(CNNMLPConfig(), params)
    raise ValueError(f"unsupported policy: {policy_name}")


def get_policy_param(cfg, key, default=None):
    try:
        resolved = resolve_policy_config(cfg)
    except Exception:
        return default
    return getattr(resolved, key, default)

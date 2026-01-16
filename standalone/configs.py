from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:
    yaml = None


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
            "lr": 1e-4,
            "lr_backbone": 1e-5,
            "batch_size": 2,
            "weight_decay": 1e-4,
            "epochs": 300,
            "lr_drop": 200,
            # "clip_max_norm": 0.1,
            "kl_weight": 10.0,
            "backbone": "resnet18",
            "dilation": False,
            "position_embedding": "sine",
            "camera_names": [],
            "image_norm": "none",
            "enc_layers": 4,
            "dec_layers": 6,
            "dim_feedforward": 2048,
            "hidden_dim": 256,
            "dropout": 0.1,
            "nheads": 8,
            "num_queries": 400,
            "pre_norm": False,
            "masks": False,
            "qpos_dim": 14,
            "action_dim": 14,
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


@dataclass
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    policy: Any = field(default_factory=PolicyConfig)
    batch_size: int = 32
    epochs: int = 10
    val_every: int = 10
    rollout_every: int = 20
    rollout_init_states_dir: Optional[str] = None
    rollout_per_anchor: int = 2
    rollout_steps: int = 8
    rollout_warmup_steps: int = 5
    rollout_use_mp: bool = False
    rollout_num_procs: int = 1
    lr: float = 1e-4
    device: str = "cuda:0"
    save_dir: str = "standalone/standalone_runs/run_001"
    grad_clip: Optional[float] = None
    use_wandb: bool = False
    wandb_project: str = "libero-standalone"
    wandb_entity: Optional[str] = None
    experiment_name: Optional[str] = None


@dataclass
class EvalConfig:
    data: DataConfig = field(default_factory=DataConfig)
    policy: Any = field(default_factory=PolicyConfig)
    ckpt: str = ""
    split_path: Optional[str] = None
    batch_size: int = 32
    device: str = "cuda:0"


@dataclass
class RolloutConfig:
    data: DataConfig = field(default_factory=DataConfig)
    policy: Any = field(default_factory=PolicyConfig)
    ckpt: str = ""
    init_states: Optional[str] = None
    steps: int = 8
    sample_index: int = 0
    n_eval: int = 1
    use_mp: bool = False
    num_procs: int = 1
    warmup_steps: int = 5
    save_videos: int = 0
    video_camera: str = ""
    video_fps: int = 30
    video_dir: str = ""
    device: str = "cuda:0"


# Load a policy YAML file into a dict and validate basic shape.
def _load_policy_config(path: str):
    if yaml is None:
        raise ImportError("pyyaml is required; install with `pip install pyyaml`.")
    cfg_path = Path(path).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"policy config not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"policy config must be a dict, got {type(data)}")
    return data


# Merge new params without overwriting existing ones.
def _merge_policy_params(dst_params, src_dict):
    for key, value in src_dict.items():
        if key not in dst_params:
            dst_params[key] = value


# Normalize cfg.policy into a PolicyConfig (supports str/dict/None).
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


# Map policy name to built-in default params.
def _get_policy_defaults(name):
    if name == "act":
        return asdict(ACTConfig())
    if name == "cnnmlp":
        return asdict(CNNMLPConfig())
    return {}


# Accept either {name, params} or flat dict and return (name, params).
def _extract_policy_data(data):
    name = data.get("name")
    if "params" in data and isinstance(data["params"], dict):
        params = dict(data["params"])
    else:
        params = {k: v for k, v in data.items() if k != "name"}
    return name, params


# Merge defaults, file-based config, and inline overrides into policy.params.
def apply_policy_config(cfg):
    policy_cfg = _normalize_policy_config(cfg)
    params = {}
    params.update(_get_policy_defaults(policy_cfg.name))
    policy_path = getattr(policy_cfg, "config_path", None)
    if policy_path:
        data = _load_policy_config(policy_path)
        if isinstance(data, dict):
            cfg_name, cfg_params = _extract_policy_data(data)
            default_cfg = PolicyConfig()
            if cfg_name and policy_cfg.name == default_cfg.name:
                policy_cfg.name = cfg_name
                params.update(_get_policy_defaults(policy_cfg.name))
            params.update(cfg_params)
    if policy_cfg.params:
        params.update(policy_cfg.params)
    policy_cfg.params = params


# Convenience accessor for policy params after merge.
def get_policy_param(cfg, key, default=None):
    policy_cfg = getattr(cfg, "policy", None)
    if policy_cfg is None or not isinstance(policy_cfg, PolicyConfig):
        return default
    params = getattr(policy_cfg, "params", None) or {}
    return params.get(key, default)

from standalone.configs import resolve_policy_config
from standalone.models.policy.act_policy import ACTPolicy
from standalone.models.policy.diffusion_policy import DiffusionPolicy


def get_policy_name(cfg):
    policy_cfg = getattr(cfg, "policy", None)
    name = getattr(policy_cfg, "name", None)
    if not name:
        return "mlp"
    return str(name).lower()


def build_policy(cfg, obs_keys, image_keys, action_dim, qpos_dim=None, obs_shapes=None):
    policy_name = get_policy_name(cfg)
    resolved = resolve_policy_config(cfg)
    if policy_name in ("act", "cnnmlp"):
        if qpos_dim is None:
            raise ValueError("qpos_dim is required for ACT/CNNMLP policy")
        return ACTPolicy(
            obs_keys=obs_keys,
            image_keys=image_keys,
            obs_horizon=cfg.data.obs_horizon,
            predict_horizon=cfg.data.predict_horizon,
            exec_horizon=resolved.exec_horizon,
            qpos_dim=qpos_dim,
            action_dim=action_dim,
            model_type=policy_name,
            act_config=resolved.act_config_dict(),
        )
    if policy_name == "dp":
        if obs_shapes is None:
            raise ValueError("obs_shapes is required for DP policy")
        return DiffusionPolicy(
            obs_keys=obs_keys,
            image_keys=image_keys,
            obs_horizon=cfg.data.obs_horizon,
            predict_horizon=cfg.data.predict_horizon,
            exec_horizon=resolved.exec_horizon,
            action_dim=action_dim,
            obs_shapes=obs_shapes,
            encoder_config=resolved.encoder,
            dp_config=resolved.dp_config_dict(),
        )
    raise ValueError(f"unsupported policy: {policy_name}")

from standalone.configs import resolve_policy_config
from standalone.models.policy.act_policy import ACTPolicy


def get_policy_name(cfg):
    policy_cfg = getattr(cfg, "policy", None)
    name = getattr(policy_cfg, "name", None)
    if not name:
        return "mlp"
    return str(name).lower()


def build_policy(cfg, obs_keys, image_keys, action_dim, qpos_dim):
    policy_name = get_policy_name(cfg)
    if policy_name not in ("act", "cnnmlp"):
        raise ValueError(f"unsupported policy: {policy_name}")
    resolved = resolve_policy_config(cfg)
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

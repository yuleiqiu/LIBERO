from pathlib import Path

import numpy as np
import torch

try:
    import draccus
except ImportError as exc:
    raise ImportError("draccus is required; install with `pip install draccus`.") from exc

from standalone.configs import RolloutConfig, apply_policy_config, get_policy_param
from standalone.dataset_utils.hdf5_sequence_dataset import (
    HDF5SequenceDataset,
    load_obs_stats,
)
from standalone.models.policy.act_policy import ACTPolicy


@draccus.wrap()
def main(cfg: RolloutConfig):
    apply_policy_config(cfg)
    if not cfg.data.demo_file:
        raise ValueError("data.demo_file is required")
    if not cfg.ckpt:
        raise ValueError("ckpt is required")
    obs_keys = [k.strip() for k in cfg.data.obs_keys.split(",") if k.strip()]
    image_keys = [k.strip() for k in cfg.data.image_keys.split(",") if k.strip()]
    all_keys = obs_keys + image_keys
    policy_name = getattr(cfg.policy, "name", "mlp").lower()

    demo_path = Path(cfg.data.demo_file).expanduser().resolve()
    if not demo_path.exists():
        raise FileNotFoundError(f"HDF5 not found: {demo_path}")

    ckpt_path = Path(cfg.ckpt).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = cfg.device if torch.cuda.is_available() else "cpu"

    dataset = HDF5SequenceDataset(
        hdf5_path=str(demo_path),
        obs_keys=all_keys,
        obs_horizon=cfg.data.obs_horizon,
        predict_horizon=cfg.data.predict_horizon,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    obs_stats = None
    if policy_name not in ("act", "cnnmlp"):
        if cfg.data.obs_stats_path:
            obs_stats = load_obs_stats(cfg.data.obs_stats_path)
        elif isinstance(ckpt, dict) and ckpt.get("obs_stats") is not None:
            obs_stats = ckpt["obs_stats"]
        if obs_stats is not None and image_keys:
            for key in image_keys:
                obs_stats.pop(key, None)
        if obs_stats is not None:
            dataset.set_obs_stats(obs_stats)

    sample = dataset[cfg.sample_index]
    action_dim = sample["actions"].shape[-1]
    print(f"[debug] action_dim: {action_dim}")
    exec_horizon = get_policy_param(cfg, "exec_horizon")
    if policy_name not in ("act", "cnnmlp"):
        raise ValueError(f"unsupported policy: {policy_name}")
    qpos_dim = sum(np.prod(sample["obs"][k].shape[1:]) for k in obs_keys)
    for key in image_keys:
        if key not in sample["obs"]:
            raise KeyError(f"image key not found in obs: {key}")
    model = ACTPolicy(
        obs_keys=obs_keys,
        image_keys=image_keys,
        obs_horizon=cfg.data.obs_horizon,
        predict_horizon=cfg.data.predict_horizon,
        exec_horizon=exec_horizon,
        qpos_dim=qpos_dim,
        action_dim=action_dim,
        model_type=policy_name,
        act_config=get_policy_param(cfg, "act_config"),
    )
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.reset()

    obs = sample["obs"]
    for step in range(cfg.steps):
        if len(model._action_queue) == 0:
            print(f"[debug] refill queue at step {step}")
        action = model.get_action(obs)
        print(f"[rollout] step {step} | action shape {tuple(action.shape)}")


if __name__ == "__main__":
    main()

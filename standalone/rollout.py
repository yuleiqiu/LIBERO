from pathlib import Path

import numpy as np
import torch

try:
    import draccus
except ImportError as exc:
    raise ImportError("draccus is required; install with `pip install draccus`.") from exc

from standalone.configs import RolloutConfig, apply_policy_config
from standalone.dataset_utils.hdf5_sequence_dataset import (
    HDF5SequenceDataset,
    load_obs_stats,
)
from standalone.dataset_utils.normalizer_utils import build_identity_normalizer
from standalone.models.algos.dp.utils.normalizer import LinearNormalizer
from standalone.models.policy.policy_factory import build_policy, get_policy_name


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
    policy_name = get_policy_name(cfg)

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
    if policy_name not in ("act", "cnnmlp", "dp"):
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
    if policy_name not in ("act", "cnnmlp", "dp"):
        raise ValueError(f"unsupported policy: {policy_name}")
    qpos_dim = sum(np.prod(sample["obs"][k].shape[1:]) for k in obs_keys)
    for key in image_keys:
        if key not in sample["obs"]:
            raise KeyError(f"image key not found in obs: {key}")
    obs_shapes = {key: value.shape for key, value in sample["obs"].items()}
    model = build_policy(
        cfg,
        obs_keys,
        image_keys,
        action_dim,
        qpos_dim=qpos_dim,
        obs_shapes=obs_shapes,
    )
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    if policy_name == "dp":
        normalizer_state = ckpt.get("normalizer") if isinstance(ckpt, dict) else None
        if normalizer_state is None:
            print("[warning] dp normalizer missing in checkpoint; using identity.")
            dp_normalizer = build_identity_normalizer(
                obs_shapes=obs_shapes,
                obs_keys=list(obs_shapes.keys()),
                action_dim=action_dim,
                last_n_dims=cfg.policy.dp.normalizer.last_n_dims,
                include_actions=True,
            )
        else:
            dp_normalizer = LinearNormalizer()
            dp_normalizer.load_state_dict(normalizer_state)
        model.set_normalizer(dp_normalizer)
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

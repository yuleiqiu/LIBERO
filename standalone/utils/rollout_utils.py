from collections import defaultdict, deque
from dataclasses import is_dataclass
from pathlib import Path
import json
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch

try:
    import yaml
except ImportError:
    yaml = None

from standalone.configs import DataConfig, PolicyConfig, RolloutConfig
from standalone.utils.bddl_path_utils import (
    canonicalize_bddl_file_name,
    read_bddl_from_hdf5 as _read_bddl_from_hdf5,
    resolve_bddl_path as _resolve_bddl_path,
)
from standalone.dataset_utils.image_normalization import normalize_images

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OBS_KEY_MAPPING = {
    # TODO: This mapping is aligned with LIBERO/robosuite defaults (see libero/configs/data/default.yaml).
    "agentview_rgb": "agentview_image",
    "eye_in_hand_rgb": "robot0_eye_in_hand_image",
    "gripper_states": "robot0_gripper_qpos",
    "joint_states": "robot0_joint_pos",
    "ee_pos": "robot0_eef_pos",
}

ROLLOUT_ENV_KWARGS_ALLOWLIST = (
    "robots",
    "controller",
    "controller_configs",
    "gripper_types",
    "initialization_noise",
    "control_freq",
    "ignore_done",
    "reward_shaping",
    "camera_depths",
    "camera_segmentations",
    "renderer",
    "renderer_config",
)


def set_rollout_seed(seed: int) -> None:
    """Seed Python/NumPy/PyTorch RNGs for reproducible rollout-time policy sampling."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _decode_if_bytes(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, (bytes, bytearray, np.bytes_)):
        return value.decode("utf-8")
    return value


def _ensure_video_camera(
    video_writer: Optional[Any],
    video_camera: Optional[Union[str, Sequence[str]]],
    obs_sample: Mapping[str, Any],
) -> Optional[Any]:
    """Check that camera keys exist in obs; disable video if missing."""
    if not video_writer:
        return None
    missing: List[str] = []
    if isinstance(video_camera, (list, tuple)):
        for name in video_camera:
            if name not in obs_sample:
                missing.append(name)
    elif video_camera not in obs_sample:
        missing.append(str(video_camera))
    if missing:
        print(f"[warning] video_camera {missing} not in env obs; disabling video")
        return None
    return video_writer


def _is_missing_value(value: Any, default: Any) -> bool:
    """Return True if a value is unset (None/empty string/default)."""
    if value is None or value == "":
        return True
    return value == default


def _merge_dataclass_fields(target: Any, source: Mapping[str, Any], defaults: Any) -> None:
    """Recursively fill target fields from source when target is still default."""
    if not isinstance(source, dict):
        return
    for key, value in source.items():
        if not hasattr(target, key):
            continue
        current = getattr(target, key)
        default = getattr(defaults, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge_dataclass_fields(current, value, default)
            continue
        if _is_missing_value(current, default):
            setattr(target, key, value)


def apply_ckpt_config(cfg: Any, cfg_dict: Mapping[str, Any]) -> bool:
    """Fill cfg with checkpoint config fields; returns whether applied."""
    if not getattr(cfg, "use_ckpt_config", True):
        return False
    if not isinstance(cfg_dict, dict):
        return False
    if isinstance(cfg_dict.get("data"), dict):
        _merge_dataclass_fields(cfg.data, cfg_dict["data"], DataConfig())
    if isinstance(cfg_dict.get("policy"), dict):
        if not isinstance(cfg.policy, PolicyConfig):
            raise TypeError(f"policy must be PolicyConfig, got {type(cfg.policy)}")
        _merge_dataclass_fields(cfg.policy, cfg_dict["policy"], PolicyConfig())
    ckpt_rollout = cfg_dict.get("rollout") if isinstance(cfg_dict.get("rollout"), dict) else {}
    ckpt_env_horizon = ckpt_rollout.get("env_horizon")
    if ckpt_env_horizon is None:
        ckpt_env_horizon = cfg_dict.get("rollout_env_horizon")
    if ckpt_env_horizon is not None and cfg.env_horizon == RolloutConfig().env_horizon:
        cfg.env_horizon = int(ckpt_env_horizon)
    return True


def read_bddl_from_hdf5(hdf5_path: str) -> Optional[str]:
    """Read the BDDL file name from HDF5."""
    return _read_bddl_from_hdf5(hdf5_path)


def resolve_bddl_path(bddl_file_name: Optional[str], demo_path: Optional[Path]) -> Optional[str]:
    """Resolve a BDDL path from absolute, repo-relative, or demo-relative locations."""
    return _resolve_bddl_path(bddl_file_name, demo_path)


def resolve_rollout_bddl_path(cfg: Any, demo_path: Optional[Path]) -> Tuple[Path, Optional[str], str]:
    """Resolve rollout BDDL from explicit override or HDF5 metadata."""
    bddl_override = getattr(cfg, "bddl_file", None)
    if bddl_override:
        bddl_path = resolve_bddl_path(bddl_override, demo_path)
        if bddl_path is None:
            raise FileNotFoundError(f"bddl_file override not found: {bddl_override}")
        return Path(bddl_path), canonicalize_bddl_file_name(bddl_override), "cfg.bddl_file"

    if demo_path is None:
        raise ValueError("demo_path is required when bddl_file is not provided")
    bddl_file_name = read_bddl_from_hdf5(str(demo_path))
    if bddl_file_name is None:
        raise ValueError("bddl_file_name not found in hdf5; cannot create env")
    bddl_path = resolve_bddl_path(bddl_file_name, demo_path)
    if bddl_path is None:
        raise FileNotFoundError(f"bddl file not found: {bddl_file_name}")
    return Path(bddl_path), canonicalize_bddl_file_name(bddl_file_name), "data.attrs[bddl_file_name]"


def read_env_kwargs_from_hdf5(hdf5_path: str) -> Dict[str, Any]:
    """Read rollout-relevant env kwargs (e.g. controller configs) from HDF5 attrs."""
    with h5py.File(hdf5_path, "r") as f:
        data = f["data"]
        raw = data.attrs.get("env_args", None)
        if raw is None:
            raw = data.attrs.get("env_info", None)
    if raw is None:
        return {}

    raw = _decode_if_bytes(raw)
    payload: Dict[str, Any]
    if isinstance(raw, Mapping):
        payload = dict(raw)
    elif isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except Exception:
            print("[warning] failed to parse data attrs env_args/env_info; using default env kwargs")
            return {}
        if not isinstance(parsed, dict):
            return {}
        payload = parsed
    else:
        print(
            f"[warning] unsupported env_args/env_info type: {type(raw)}; using default env kwargs"
        )
        return {}

    env_kwargs = payload.get("env_kwargs", payload)
    if not isinstance(env_kwargs, dict):
        return {}
    return {
        key: env_kwargs[key]
        for key in ROLLOUT_ENV_KWARGS_ALLOWLIST
        if key in env_kwargs
    }


def read_init_states_from_hdf5(hdf5_path: str) -> np.ndarray:
    """Read one init_state per demo from HDF5 and stack into an array."""

    def demo_sort_key(name: str) -> Union[int, str]:
        """Sort key for demo_xx names."""
        try:
            return int(name.split("_")[1])
        except Exception:
            return name

    init_states: List[np.ndarray] = []
    with h5py.File(hdf5_path, "r") as f:
        data = f["data"]
        demo_keys = sorted([k for k in data.keys() if k.startswith("demo_")], key=demo_sort_key)
        if not demo_keys:
            raise ValueError(f"No demo_xx groups found under 'data' in {hdf5_path}")

        for demo_key in demo_keys:
            demo_grp = data[demo_key]
            init_state = demo_grp.attrs.get("init_state", None)
            if init_state is None and "states" in demo_grp and len(demo_grp["states"]) > 0:
                init_state = demo_grp["states"][0]
            if init_state is None:
                print(f"[warning] {demo_key} missing init_state; skipping")
                continue
            init_states.append(np.array(init_state))

    if not init_states:
        raise ValueError(f"No init states could be read from {hdf5_path}")

    return np.stack(init_states, axis=0)


def infer_rollout_io_specs(
    hdf5_path: str,
    obs_keys: Sequence[str],
    image_keys: Sequence[str],
    obs_horizon: int,
    extra_obs_keys: Optional[Sequence[str]] = None,
    action_key: str = "actions",
) -> Tuple[int, Dict[str, Tuple[int, ...]], Dict[str, Tuple[int, ...]], int]:
    """Infer rollout model IO specs from HDF5 without building dataset indices."""

    def demo_sort_key(name: str) -> Union[int, str]:
        try:
            return int(name.split("_")[1])
        except Exception:
            return name

    obs_horizon = int(obs_horizon)
    if obs_horizon <= 0:
        raise ValueError("obs_horizon must be >= 1")

    required = list(obs_keys) + list(image_keys) + list(extra_obs_keys or [])
    with h5py.File(hdf5_path, "r") as f:
        data = f["data"]
        demo_keys = [k for k in data.keys() if k.startswith("demo_")]
        if not demo_keys:
            demo_keys = list(data.keys())
        demo_keys = sorted(demo_keys, key=demo_sort_key)
        if not demo_keys:
            raise ValueError(f"No demo groups found under 'data' in {hdf5_path}")
        demo_group = data[demo_keys[0]]

        if action_key not in demo_group:
            raise KeyError(f"action key not found in demo group: {action_key}")
        actions = demo_group[action_key]
        if actions.ndim != 2:
            raise ValueError(f"expected action shape [T, A], got {actions.shape}")
        action_dim = int(actions.shape[-1])

        if "obs" not in demo_group:
            raise KeyError("obs group not found in demo group")
        obs_group = demo_group["obs"]
        obs_shapes: Dict[str, Tuple[int, ...]] = {}
        image_shapes: Dict[str, Tuple[int, ...]] = {}
        for key in required:
            if key not in obs_group:
                raise KeyError(f"obs key not found in hdf5 demo: {key}")
            ds = obs_group[key]
            if ds.ndim < 2:
                raise ValueError(f"expected obs shape [T, ...], got {ds.shape} for key {key}")
            shape = (obs_horizon,) + tuple(int(v) for v in ds.shape[1:])
            obs_shapes[key] = shape
            if key in image_keys:
                image_shapes[key] = shape[1:]

    proprio_dim = int(sum(np.prod(obs_shapes[k][1:]) for k in obs_keys))
    return action_dim, image_shapes, obs_shapes, proprio_dim


def load_init_states(cfg: Any, demo_path: Optional[Path]) -> np.ndarray:
    """Load init states from cfg.init_states or fall back to HDF5."""
    init_states_path = getattr(cfg, "init_states", None)
    if init_states_path:
        init_states_path = Path(init_states_path).expanduser().resolve()
        if not init_states_path.exists():
            raise FileNotFoundError(f"init states file not found: {init_states_path}")
        init_states = torch.load(str(init_states_path), weights_only=False)
        if torch.is_tensor(init_states):
            init_states = init_states.cpu().numpy()
        else:
            init_states = np.asarray(init_states)
        print(f"[info] loaded {init_states.shape[0]} init states from {init_states_path}")
        return init_states
    if demo_path is None:
        raise ValueError("demo_path is required when init_states is not provided")
    return read_init_states_from_hdf5(str(demo_path))


def load_anchor_indices(cfg: Any) -> Optional[List[int]]:
    """Load anchor indices associated with init_states."""
    init_states_path = getattr(cfg, "init_states", None)
    if not init_states_path:
        return None
    init_states_path = Path(init_states_path).expanduser().resolve()
    anchors_meta = init_states_path.with_suffix(init_states_path.suffix + ".anchors.json")
    if not anchors_meta.exists():
        print(f"[warning] anchors meta not found: {anchors_meta}")
        return None
    with open(anchors_meta, "r") as f:
        anchor_indices = json.load(f).get("anchor_idx", None)
    if anchor_indices is None:
        print(f"[warning] anchor_idx not found in {anchors_meta}")
    return anchor_indices


def load_default_obs_key_mapping() -> Dict[str, str]:
    """Load obs_key_mapping from default yaml; empty if missing."""
    if yaml is None:
        return {}
    config_path = REPO_ROOT / "libero/configs/data/default.yaml"
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    mapping = cfg.get("obs_key_mapping", {})
    return mapping or {}


def build_obs_key_mapping(
    cfg: Any, obs_keys: Sequence[str], image_keys: Sequence[str]
) -> Dict[str, str]:
    """Merge default + config mappings into the final obs_key mapping."""
    mapping: Dict[str, str] = {}
    mapping.update(DEFAULT_OBS_KEY_MAPPING)
    mapping.update(load_default_obs_key_mapping())
    if cfg.data.obs_key_mapping:
        mapping.update(cfg.data.obs_key_mapping)
    return {key: mapping.get(key, key) for key in obs_keys + image_keys}


def parse_mask_keys(mask_keys_raw: str, image_keys: Sequence[str]) -> List[str]:
    """Parse cfg.data.mask_keys into a list aligned with image_keys."""
    mask_keys = [k.strip() for k in str(mask_keys_raw or "").split(",")]
    while len(mask_keys) < len(image_keys):
        mask_keys.append("")
    return mask_keys[: len(image_keys)]


def active_mask_keys(mask_keys: Optional[Sequence[str]]) -> List[str]:
    """Return active non-empty mask keys."""
    return [key for key in (mask_keys or []) if key]


def image_mask_items(
    image_keys: Sequence[str], mask_keys: Optional[Sequence[str]]
) -> List[Tuple[str, str]]:
    """Return ordered `(image_key, mask_key)` pairs for active masks."""
    return [
        (img_key, mask_key)
        for img_key, mask_key in zip(image_keys, mask_keys or [])
        if mask_key
    ]


def infer_camera_size(image_shapes: Mapping[str, Sequence[int]]) -> Optional[Tuple[int, int]]:
    """Infer camera resolution (H, W) from image shapes."""
    if not image_shapes:
        return None
    heights = set()
    widths = set()
    for shape in image_shapes.values():
        if len(shape) != 3:
            raise ValueError(f"expected 3D image shape, got {shape}")
        if shape[-1] in (1, 3):
            h, w = shape[0], shape[1]
        elif shape[0] in (1, 3):
            h, w = shape[1], shape[2]
        else:
            raise ValueError(f"cannot infer channel dim from shape: {shape}")
        heights.add(int(h))
        widths.add(int(w))
    if len(heights) != 1 or len(widths) != 1:
        raise ValueError(f"mismatched camera sizes: heights={heights}, widths={widths}")
    return heights.pop(), widths.pop()


def camera_names_from_mapping(
    image_keys: Sequence[str], obs_key_mapping: Mapping[str, str]
) -> List[str]:
    """Map image keys to environment camera names."""
    names: List[str] = []
    for key in image_keys:
        env_key = obs_key_mapping.get(key, key)
        if env_key.endswith("_image"):
            names.append(env_key[: -len("_image")])
        else:
            names.append(env_key)
    return names


def segmentation_key(env_key: str) -> str:
    """Map an env image observation key to its instance-segmentation key."""
    if env_key.endswith("_image"):
        return f"{env_key[: -len('_image')]}_segmentation_instance"
    return f"{env_key}_segmentation_instance"


def build_mask_obs_batch(
    env_obs_list: Sequence[Mapping[str, Any]],
    env: Any,
    image_keys: Sequence[str],
    mask_keys: Sequence[str],
    obs_key_mapping: Mapping[str, str],
) -> List[Dict[str, np.ndarray]]:
    """Generate binary object-of-interest masks for each env observation."""
    mask_pairs = image_mask_items(image_keys, mask_keys)
    out: List[Dict[str, np.ndarray]] = [dict() for _ in env_obs_list]
    if not mask_pairs:
        return out
    if not hasattr(env, "get_segmentation_of_interest"):
        raise ValueError("active mask_keys require an env with get_segmentation_of_interest()")

    batched = len(env_obs_list) > 1
    for image_key, mask_key in mask_pairs:
        env_key = obs_key_mapping.get(image_key, image_key)
        seg_key = segmentation_key(env_key)
        seg_imgs: List[np.ndarray] = []
        for obs in env_obs_list:
            if seg_key not in obs:
                raise KeyError(
                    f"env obs missing segmentation key {seg_key} for image key {image_key}; "
                    f"available keys: {list(obs.keys())}"
                )
            seg_imgs.append(np.squeeze(np.asarray(obs[seg_key])))
        if batched:
            seg_masks = env.get_segmentation_of_interest(seg_imgs)
        else:
            seg_masks = [env.get_segmentation_of_interest(seg_imgs[0])]
        for idx, seg_mask in enumerate(seg_masks):
            out[idx][mask_key] = (np.squeeze(np.asarray(seg_mask)) == 1).astype(np.float32)
    return out


def extract_env_obs(
    env_obs: Mapping[str, Any],
    obs_keys: Sequence[str],
    image_keys: Sequence[str],
    obs_key_mapping: Mapping[str, str],
    extra_obs: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Select required keys from env obs and remap names."""
    out: Dict[str, Any] = {}
    for key in obs_keys + image_keys:
        env_key = obs_key_mapping.get(key, key)
        if env_key not in env_obs:
            raise KeyError(
                f"env obs missing key {env_key} (for {key}); available keys: {list(env_obs.keys())}"
            )
        out[key] = env_obs[env_key]
    if extra_obs:
        out.update(extra_obs)
    return out


def split_env_obs(env_obs: Any, env_num: int) -> List[Any]:
    """Split environment observations into env_num entries."""
    if env_num == 1:
        return [env_obs]
    if isinstance(env_obs, np.ndarray):
        obs_list = list(env_obs)
    elif isinstance(env_obs, (list, tuple)):
        obs_list = list(env_obs)
    else:
        obs_list = [env_obs]
    if len(obs_list) != env_num:
        raise ValueError(f"expected {env_num} env observations, got {len(obs_list)}")
    return obs_list


def stack_obs_batch(
    obs_list: Sequence[Mapping[str, Any]],
    obs_keys: Sequence[str],
    image_keys: Sequence[str],
    extra_keys: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    """Stack multiple obs into a batch (first dimension)."""
    if not obs_list:
        raise ValueError("cannot stack empty observation list")
    batch: Dict[str, np.ndarray] = {}
    for key in list(obs_keys) + list(image_keys) + list(extra_keys or []):
        batch[key] = np.stack([obs[key] for obs in obs_list], axis=0)
    return batch


def build_per_env_temporal_ensemblers(model: Any, env_num: int) -> Optional[List[Any]]:
    """Create one temporal ensembler per env when the policy uses ACT-style temporal ensembling."""
    temporal_ensembler = getattr(model, "temporal_ensembler", None)
    temporal_ensemble_coeff = getattr(model, "temporal_ensemble_coeff", None)
    if temporal_ensembler is None or temporal_ensemble_coeff is None:
        return None
    ensembler_cls = temporal_ensembler.__class__
    return [
        ensembler_cls(float(temporal_ensemble_coeff), int(model.predict_horizon))
        for _ in range(int(env_num))
    ]


def seed_rollout_env(env: Any, seed: int, env_num: int) -> None:
    """Seed a single env or all envs in a vector env with the configured rollout seed."""
    if env_num == 1:
        env.seed(seed)
    else:
        env.seed([int(seed)] * env_num)


def build_histories(
    cfg: Any,
    obs_keys: Sequence[str],
    image_keys: Sequence[str],
    env_num: int,
    extra_keys: Optional[Sequence[str]] = None,
):
    """Create one observation history buffer per active env slot."""
    return [
        ObsHistory(
            list(obs_keys) + list(image_keys) + list(extra_keys or []),
            cfg.data.obs_horizon,
            image_keys=image_keys,
            image_norm=cfg.data.image_norm,
        )
        for _ in range(env_num)
    ]


def reset_rollout_runtime(model: Any, histories: Sequence["ObsHistory"], temporal_ensemblers) -> None:
    """Reset policy runtime state and per-env rollout buffers at episode/batch start."""
    model.reset()
    for history in histories:
        history.reset()
    if temporal_ensemblers is not None:
        for ensembler in temporal_ensemblers:
            ensembler.reset()


def set_init_state_batch(env: Any, init_states_batch: np.ndarray, env_num: int):
    """Apply init states to the env and return one observation dict per env slot."""
    init_arg = init_states_batch[0] if env_num == 1 else init_states_batch
    env_obs = env.set_init_state(init_arg)
    return split_env_obs(env_obs, env_num)


def step_env_batch(env: Any, actions: np.ndarray, env_num: int):
    """Step a single env or vector env and normalize outputs to batched lists/arrays."""
    step_arg = actions[0] if env_num == 1 else actions
    env_obs, _, done, _ = env.step(step_arg)
    env_obs_list = split_env_obs(env_obs, env_num)
    if env_num == 1:
        done_array = np.asarray([bool(done)])
    else:
        done_array = np.asarray(done)
    return env_obs_list, done_array


def pending_env_indices(remaining: int, dones, action_queues, temporal_ensemblers):
    """Return env indices whose action queues need refilling at the current step."""
    if temporal_ensemblers is None:
        return [
            i
            for i in range(remaining)
            if not dones[i] and len(action_queues[i]) == 0
        ]
    return [i for i in range(remaining) if not dones[i]]


def pop_actions(action_queues, dones, remaining: int, env_num: int, action_dim: int) -> np.ndarray:
    """Pop one executable action per active env and pack them into a batched action array."""
    actions = np.zeros((env_num, action_dim), dtype=np.float32)
    for i in range(remaining):
        if dones[i]:
            continue
        if action_queues[i]:
            act = action_queues[i].popleft()
            if torch.is_tensor(act):
                act = act.detach().cpu().numpy()
            actions[i] = np.asarray(act).reshape(-1)
    return actions


def refill_action_queues(
    model: Any,
    obs_batch: Mapping[str, Any],
    pending: Sequence[int],
    action_queues: Sequence[deque],
    temporal_ensemblers: Optional[Sequence[Any]] = None,
) -> None:
    """Predict action chunks for pending envs and store executable actions in per-env queues."""
    if not pending:
        return

    model.eval()
    with torch.no_grad():
        pred = model.forward(obs_batch)
    if torch.is_tensor(pred):
        pred = pred.detach().cpu()
    if pred.ndim == 2:
        pred = pred.view(pred.shape[0], model.predict_horizon, -1)

    if temporal_ensemblers is None:
        for batch_idx, env_idx in enumerate(pending):
            actions_seq = pred[batch_idx]
            take = min(model.exec_horizon, actions_seq.shape[0])
            for step_action in actions_seq[:take]:
                action_queues[env_idx].append(step_action)
        return

    for batch_idx, env_idx in enumerate(pending):
        action = temporal_ensemblers[env_idx].update(pred[batch_idx : batch_idx + 1])[0]
        action_queues[env_idx].append(action)


class ObsHistory:
    """Rolling buffer of recent observations for obs_horizon stacking."""

    def __init__(
        self,
        keys: Sequence[str],
        horizon: int,
        image_keys: Optional[Sequence[str]] = None,
        image_norm: str = "none",
    ) -> None:
        """Create a fixed-length observation history buffer."""
        self.horizon = int(horizon)
        if self.horizon <= 0:
            raise ValueError("obs_horizon must be >= 1")
        self._buffers = {key: deque(maxlen=self.horizon) for key in keys}
        self._image_keys = set(image_keys or [])
        self._image_norm = str(image_norm or "none").lower()

    def reset(self) -> None:
        """Clear stored history."""
        for buf in self._buffers.values():
            buf.clear()

    def add(self, obs: Mapping[str, Any]) -> None:
        """Append one observation frame to history."""
        for key in self._buffers:
            if key not in obs:
                raise KeyError(f"missing key in obs history: {key}")
            self._buffers[key].append(np.asarray(obs[key]))

    def stack(self) -> Dict[str, np.ndarray]:
        """Stack history into fixed-length arrays and normalize images."""
        out: Dict[str, np.ndarray] = {}
        for key, buf in self._buffers.items():
            if not buf:
                raise ValueError(f"no observations collected for key {key}")
            arr = np.stack(list(buf), axis=0)
            if arr.shape[0] < self.horizon:
                pad = np.repeat(arr[[0]], self.horizon - arr.shape[0], axis=0)
                arr = np.concatenate([pad, arr], axis=0)
            out[key] = arr.astype(np.float32, copy=False)
        for key in self._image_keys:
            if key in out:
                out[key] = normalize_images(out[key], self._image_norm, input_scale="0_255")
        return out


def select_video_camera(
    cfg: Any, image_keys: Sequence[str], obs_key_mapping: Mapping[str, str]
) -> Optional[Union[str, List[str]]]:
    """Choose which camera to record; default to the first image key."""
    if getattr(cfg, "video_camera", ""):
        raw = str(cfg.video_camera)
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if not parts:
            return None
        if len(parts) == 1:
            return obs_key_mapping.get(parts[0], parts[0])
        return [obs_key_mapping.get(p, p) for p in parts]
    if not image_keys:
        return None
    first_key = image_keys[0]
    return obs_key_mapping.get(first_key, first_key)


def build_rollout_summary(
    n_rollouts: int, successes: int, episode_results: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Compute rollout success stats and per-anchor summary."""
    sr = successes / max(n_rollouts, 1)
    summary: Dict[str, Any] = {
        "total": {
            "success": int(successes),
            "rollouts": int(n_rollouts),
            "success_rate": float(sr),
        },
        "episode_results": [dict(result) for result in episode_results],
    }
    anchor_counts: Dict[int, int] = defaultdict(int)
    anchor_success: Dict[int, int] = defaultdict(int)
    for result in episode_results:
        anchor_id = result.get("anchor_id")
        if anchor_id is None:
            continue
        anchor_counts[int(anchor_id)] += 1
        if result.get("success"):
            anchor_success[int(anchor_id)] += 1
    if anchor_counts:
        anchors: Dict[str, Dict[str, float]] = {}
        for anchor_id in sorted(anchor_counts.keys()):
            count = anchor_counts[anchor_id]
            succ = anchor_success.get(anchor_id, 0)
            anchors[str(anchor_id)] = {
                "success": int(succ),
                "rollouts": int(count),
                "success_rate": float(succ / max(count, 1)),
            }
        summary["anchors"] = anchors
    return summary


def write_rollout_summary(
    video_dir: Optional[Union[str, Path]], summary: Mapping[str, Any]
) -> Optional[Path]:
    """Write rollout_summary.json and return its path."""
    if video_dir is None:
        return None
    out_dir = Path(video_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rollout_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    return out_path


def _derive_eval_video_dir(cfg: Any, run_config: Optional[Mapping[str, Any]]) -> Optional[Path]:
    """Infer eval video directory from run config or ckpt path."""
    if isinstance(run_config, dict):
        rollout_cfg = run_config.get("rollout")
        if isinstance(rollout_cfg, dict):
            train_video_dir = rollout_cfg.get("video_dir")
            if train_video_dir:
                try:
                    base_dir = Path(train_video_dir).expanduser().resolve().parent
                    return base_dir / "eval"
                except Exception:
                    pass
        paths_cfg = run_config.get("paths")
        if isinstance(paths_cfg, dict):
            save_dir = paths_cfg.get("save_dir")
            if save_dir:
                try:
                    base_dir = Path(save_dir).expanduser().resolve()
                    return base_dir / "rollout_videos" / "eval"
                except Exception:
                    pass
    ckpt_path = getattr(cfg, "ckpt", "")
    if ckpt_path:
        ckpt_dir = Path(ckpt_path).expanduser().resolve().parent
        return ckpt_dir / "rollout_videos" / "eval"
    return None


def resolve_video_dir(cfg: Any) -> Path:
    """Pick video output directory (cfg.video_dir takes precedence)."""
    if getattr(cfg, "video_dir", ""):
        return Path(cfg.video_dir).expanduser().resolve()
    ckpt_dir = Path(cfg.ckpt).expanduser().resolve().parent
    return ckpt_dir / "rollout_videos" / "eval"

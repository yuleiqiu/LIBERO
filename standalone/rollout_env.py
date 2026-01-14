from collections import defaultdict, deque
from pathlib import Path

import h5py
import numpy as np
import torch
import json
from tqdm import tqdm

try:
    import draccus
except ImportError as exc:
    raise ImportError("draccus is required; install with `pip install draccus`.") from exc

try:
    import yaml
except ImportError:
    yaml = None

from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
from libero.libero.utils.video_utils import VideoWriter

from standalone.configs import RolloutConfig, apply_policy_config, get_policy_param
from standalone.dataset_utils.hdf5_sequence_dataset import (
    HDF5SequenceDataset,
    load_obs_stats,
)
from standalone.models.policy.act_policy import ACTPolicy

DEFAULT_OBS_KEY_MAPPING = {
    "agentview_rgb": "agentview_image",
    "eye_in_hand_rgb": "robot0_eye_in_hand_image",
    "gripper_states": "robot0_gripper_qpos",
    "joint_states": "robot0_joint_pos",
}


def read_bddl_from_hdf5(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        data = f["data"]
        return data.attrs.get("bddl_file_name", None)


def read_init_states_from_hdf5(hdf5_path):
    """
    Collect one init_state per demo entry in the given HDF5.
    Prefers the per-demo attr 'init_state'; falls back to the first 'states' entry.
    """

    def demo_sort_key(name):
        try:
            return int(name.split("_")[1])
        except Exception:
            return name

    init_states = []
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

    init_states = np.stack(init_states, axis=0)
    return init_states


def load_init_states(cfg, demo_path):
    init_states_path = getattr(cfg, "init_states", None)
    if init_states_path:
        init_states_path = Path(init_states_path).expanduser().resolve()
        if not init_states_path.exists():
            raise FileNotFoundError(f"init states file not found: {init_states_path}")
        init_states = torch.load(str(init_states_path))
        if torch.is_tensor(init_states):
            init_states = init_states.cpu().numpy()
        else:
            init_states = np.asarray(init_states)
        print(f"[info] loaded {init_states.shape[0]} init states from {init_states_path}")
        return init_states
    return read_init_states_from_hdf5(str(demo_path))


def load_anchor_indices(cfg):
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


def load_default_obs_key_mapping():
    if yaml is None:
        return {}
    config_path = Path(__file__).resolve().parents[1] / "libero/configs/data/default.yaml"
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    mapping = cfg.get("obs_key_mapping", {})
    return mapping or {}


def build_obs_key_mapping(cfg, obs_keys, image_keys):
    mapping = {}
    mapping.update(DEFAULT_OBS_KEY_MAPPING)
    mapping.update(load_default_obs_key_mapping())
    if cfg.data.obs_key_mapping:
        mapping.update(cfg.data.obs_key_mapping)
    return {key: mapping.get(key, key) for key in obs_keys + image_keys}


def resolve_bddl_path(bddl_file_name, demo_path):
    if not bddl_file_name:
        return None
    candidate = Path(bddl_file_name).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)
    if candidate.exists():
        return str(candidate.resolve())
    repo_root = Path(__file__).resolve().parents[1]
    repo_candidate = (repo_root / "libero/libero/bddl_files" / candidate).resolve()
    if repo_candidate.exists():
        return str(repo_candidate)
    demo_candidate = (demo_path.parent / candidate).resolve()
    if demo_candidate.exists():
        return str(demo_candidate)
    return None


def infer_camera_size(image_shapes):
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


def camera_names_from_mapping(image_keys, obs_key_mapping):
    names = []
    for key in image_keys:
        env_key = obs_key_mapping.get(key, key)
        if env_key.endswith("_image"):
            names.append(env_key[: -len("_image")])
        else:
            names.append(env_key)
    return names


def extract_env_obs(env_obs, obs_keys, image_keys, obs_key_mapping):
    out = {}
    for key in obs_keys + image_keys:
        env_key = obs_key_mapping.get(key, key)
        if env_key not in env_obs:
            raise KeyError(
                f"env obs missing key {env_key} (for {key}); available keys: {list(env_obs.keys())}"
            )
        out[key] = env_obs[env_key]
    # print(f"[debug] extracted obs keys: {list(out.keys())}")
    # print(f"[debug] obs shapes: " + ", ".join(f"{k}: {v.shape}" for k, v in out.items()))
    return out


def split_env_obs(env_obs, env_num):
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


def stack_obs_batch(obs_list, obs_keys, image_keys):
    if not obs_list:
        raise ValueError("cannot stack empty observation list")
    batch = {}
    for key in obs_keys + image_keys:
        batch[key] = np.stack([obs[key] for obs in obs_list], axis=0)
    return batch


# def normalize_obs(obs, obs_stats):
#     if obs_stats is None:
#         return obs
#     out = {}
#     for key, value in obs.items():
#         stats = obs_stats.get(key)
#         if stats is None:
#             out[key] = value
#         else:
#             mean = stats["mean"]
#             std = stats["std"]
#             out[key] = ((value - mean) / std).astype(np.float32)
#     return out


class ObsHistory:
    def __init__(self, keys, horizon):
        self.horizon = int(horizon)
        if self.horizon <= 0:
            raise ValueError("obs_horizon must be >= 1")
        self._buffers = {key: deque(maxlen=self.horizon) for key in keys}

    def reset(self):
        for buf in self._buffers.values():
            buf.clear()

    def add(self, obs):
        for key in self._buffers:
            if key not in obs:
                raise KeyError(f"missing key in obs history: {key}")
            self._buffers[key].append(np.asarray(obs[key]))

    def stack(self):
        out = {}
        for key, buf in self._buffers.items():
            if not buf:
                raise ValueError(f"no observations collected for key {key}")
            arr = np.stack(list(buf), axis=0)
            if arr.shape[0] < self.horizon:
                pad = np.repeat(arr[[0]], self.horizon - arr.shape[0], axis=0)
                arr = np.concatenate([pad, arr], axis=0)
            out[key] = arr.astype(np.float32, copy=False)
        return out


def select_video_camera(cfg, image_keys, obs_key_mapping):
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


def resolve_video_dir(cfg):
    if getattr(cfg, "video_dir", ""):
        return Path(cfg.video_dir).expanduser().resolve()
    ckpt_dir = Path(cfg.ckpt).expanduser().resolve().parent
    return ckpt_dir / "rollout_videos"


def run_env_rollouts(
    cfg,
    model,
    obs_keys,
    image_keys,
    obs_stats,
    demo_path,
    action_dim,
    image_shapes,
    init_states_override=None,
    rollout_order_override=None,
    anchor_ids=None,
):
    bddl_file_name = read_bddl_from_hdf5(str(demo_path))
    if bddl_file_name is None:
        raise ValueError("bddl_file_name not found in hdf5; cannot create env")
    bddl_path = resolve_bddl_path(bddl_file_name, demo_path)
    if bddl_path is None:
        raise FileNotFoundError(f"bddl file not found: {bddl_file_name}")

    if init_states_override is None:
        init_states = load_init_states(cfg, demo_path)
    else:
        init_states = np.asarray(init_states_override)

    obs_key_mapping = build_obs_key_mapping(cfg, obs_keys, image_keys)
    camera_names = camera_names_from_mapping(image_keys, obs_key_mapping) if image_keys else []
    cam_hw = infer_camera_size(image_shapes) if image_keys else None

    env_args = {"bddl_file_name": bddl_path}
    if image_keys:
        if cam_hw is None:
            raise ValueError("image_keys provided but camera size could not be inferred")
        camera_h, camera_w = cam_hw
        env_args.update(
            {
                "use_camera_obs": True,
                "camera_names": camera_names,
                "camera_heights": camera_h,
                "camera_widths": camera_w,
            }
        )
    else:
        env_args["use_camera_obs"] = False

    save_videos = int(getattr(cfg, "save_videos", 0))
    video_writer = None
    video_camera = None
    video_dir = None
    if save_videos > 0:
        video_camera = select_video_camera(cfg, image_keys, obs_key_mapping)
        if not video_camera:
            print("[warning] save_videos requested but no image_keys; skipping video")
        else:
            video_dir = resolve_video_dir(cfg)
            video_writer = VideoWriter(
                video_path=str(video_dir),
                save_video=True,
                fps=int(getattr(cfg, "video_fps", 30)),
                single_video=False,
            )

    total_states = init_states.shape[0]
    if total_states == 0:
        raise ValueError("no init states found in hdf5")
    if anchor_ids is not None and len(anchor_ids) != total_states:
        raise ValueError(
            f"anchor_ids length mismatch: {len(anchor_ids)} vs {total_states}"
        )

    if rollout_order_override is not None:
        rollout_order = list(rollout_order_override)
        if not rollout_order:
            raise ValueError("rollout_order_override is empty")
        for idx in rollout_order:
            if idx < 0 or idx >= total_states:
                raise ValueError(
                    f"rollout_order_override index out of range: {idx} (0..{total_states - 1})"
                )
        n_eval = len(rollout_order)
    else:
        n_eval = int(cfg.n_eval)
        if n_eval <= 0:
            raise ValueError("n_eval must be >= 1")
        if n_eval > total_states:
            print(f"[warning] n_eval={n_eval} > init_states={total_states}; clipping")
            n_eval = total_states
        start_idx = int(cfg.sample_index)
        if start_idx < 0 or start_idx >= total_states:
            raise ValueError(
                f"sample_index out of range: {start_idx} (0..{total_states - 1})"
            )
        rollout_order = [(start_idx + i) % total_states for i in range(n_eval)]

    use_mp = bool(getattr(cfg, "use_mp", False))
    num_procs = int(getattr(cfg, "num_procs", 1))
    if num_procs <= 0:
        raise ValueError("num_procs must be >= 1")
    env_num = min(num_procs, n_eval) if use_mp else 1
    eval_loop_num = (n_eval + env_num - 1) // env_num

    max_steps = int(cfg.steps)
    if max_steps <= 0:
        raise ValueError("steps must be >= 1")

    episode_results = []
    if env_num == 1:
        env = OffScreenRenderEnv(**env_args)
        env.seed(cfg.data.seed)
        history = ObsHistory(obs_keys + image_keys, cfg.data.obs_horizon)
        successes = 0
        pbar = tqdm(total=n_eval, desc="rollout", leave=True)

        for ep_idx, init_idx in enumerate(rollout_order):
            model.reset()
            history.reset()
            env.reset()
            env_obs = env.set_init_state(init_states[init_idx])
            if video_writer:
                missing = []
                if isinstance(video_camera, (list, tuple)):
                    for name in video_camera:
                        if name not in env_obs:
                            missing.append(name)
                elif video_camera not in env_obs:
                    missing.append(video_camera)
                if missing:
                    print(
                        f"[warning] video_camera {missing} not in env obs; disabling video"
                    )
                    video_writer = None
            if video_writer and ep_idx < save_videos:
                video_writer.append_obs(env_obs, done=False, idx=ep_idx, camera_name=video_camera)
            obs = extract_env_obs(env_obs, obs_keys, image_keys, obs_key_mapping)
            history.add(obs)

            dummy = np.zeros((action_dim,), dtype=np.float32)
            for _ in range(int(cfg.warmup_steps)):
                env_obs, _, _, _ = env.step(dummy)
                if video_writer and ep_idx < save_videos:
                    video_writer.append_obs(
                        env_obs, done=False, idx=ep_idx, camera_name=video_camera
                    )
                obs = extract_env_obs(env_obs, obs_keys, image_keys, obs_key_mapping)
                history.add(obs)

            done = False
            steps_taken = 0
            while steps_taken < max_steps:
                steps_taken += 1
                obs_input = history.stack()
                action = model.get_action(obs_input)
                if torch.is_tensor(action):
                    action_np = action.detach().cpu().numpy()
                else:
                    action_np = np.asarray(action)
                action_np = action_np.reshape(-1)

                env_obs, _, done, _ = env.step(action_np)
                if video_writer and ep_idx < save_videos:
                    video_writer.append_obs(
                        env_obs, done=bool(done), idx=ep_idx, camera_name=video_camera
                    )
                obs = extract_env_obs(env_obs, obs_keys, image_keys, obs_key_mapping)
                history.add(obs)

                if done:
                    successes += 1
                    break

            print(
                f"[rollout] episode {ep_idx} | init_state {init_idx} | steps {steps_taken} | success {done}"
            )
            pbar.update(1)
            pbar.set_postfix(
                sr=f"{successes / max(ep_idx + 1, 1):.3f}",
                step=steps_taken,
            )
            result = {
                "rollout_idx": ep_idx,
                "init_idx": init_idx,
                "success": bool(done),
                "steps": steps_taken,
            }
            if anchor_ids is not None:
                result["anchor_id"] = int(anchor_ids[init_idx])
            episode_results.append(result)

        env.close()
        pbar.close()
        if video_writer:
            video_writer.save()
        sr = successes / max(n_eval, 1)
        print("[info] rollout summary:")
        print(f"  rollouts: {n_eval}")
        print(f"  success: {successes}/{n_eval} ({sr:.3f})")
        return {
            "n_eval": n_eval,
            "successes": successes,
            "success_rate": sr,
            "rollout_order": rollout_order,
            "episode_results": episode_results,
            "video_dir": str(video_dir) if video_dir is not None else None,
        }

    env = SubprocVectorEnv([lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)])
    env.seed(cfg.data.seed)
    histories = [ObsHistory(obs_keys + image_keys, cfg.data.obs_horizon) for _ in range(env_num)]

    max_record_videos = min(save_videos, n_eval)
    record_active = [False] * env_num
    video_ids = [None] * env_num

    successes = 0
    episodes_done = 0
    pbar = tqdm(total=n_eval, desc="rollout", leave=True)
    for loop_idx in range(eval_loop_num):
        if episodes_done >= n_eval:
            break
        batch_start = episodes_done
        model.reset()
        for history in histories:
            history.reset()

        remaining = min(env_num, n_eval - episodes_done)
        indices = rollout_order[episodes_done : episodes_done + remaining]
        if len(indices) < env_num:
            indices = indices + [indices[-1]] * (env_num - len(indices))
        init_states_batch = init_states[indices]

        env.reset()
        env_obs = env.set_init_state(init_states_batch)
        env_obs_list = split_env_obs(env_obs, env_num)
        if video_writer:
            missing = []
            if isinstance(video_camera, (list, tuple)):
                for name in video_camera:
                    if name not in env_obs_list[0]:
                        missing.append(name)
            elif video_camera not in env_obs_list[0]:
                missing.append(video_camera)
            if missing:
                print(
                    f"[warning] video_camera {missing} not in env obs; disabling video"
                )
                video_writer = None
        if video_writer:
            for i in range(env_num):
                record_active[i] = False
                video_ids[i] = None
            rec_slots = max_record_videos - episodes_done
            rec = max(0, min(rec_slots, remaining))
            for i in range(rec):
                record_active[i] = True
                video_ids[i] = episodes_done + i
                video_writer.append_obs(
                    env_obs_list[i], done=False, idx=video_ids[i], camera_name=video_camera
                )

        for i in range(env_num):
            obs = extract_env_obs(env_obs_list[i], obs_keys, image_keys, obs_key_mapping)
            histories[i].add(obs)

        dummy = np.zeros((env_num, action_dim), dtype=np.float32)
        for _ in range(int(cfg.warmup_steps)):
            env_obs, _, _, _ = env.step(dummy)
            env_obs_list = split_env_obs(env_obs, env_num)
            if video_writer:
                for i in range(remaining):
                    if record_active[i] and video_ids[i] is not None:
                        video_writer.append_obs(
                            env_obs_list[i],
                            done=False,
                            idx=video_ids[i],
                            camera_name=video_camera,
                        )
            for i in range(env_num):
                obs = extract_env_obs(env_obs_list[i], obs_keys, image_keys, obs_key_mapping)
                histories[i].add(obs)

        action_queues = [deque() for _ in range(env_num)]
        dones = [False] * env_num
        steps_by_env = [0] * env_num
        for k in range(remaining, env_num):
            dones[k] = True

        steps_taken = 0
        device = next(model.parameters()).device
        while steps_taken < max_steps:
            steps_taken += 1
            prev_done = sum(1 for d in dones[:remaining] if d)
            pending = [
                i
                for i in range(remaining)
                if not dones[i] and len(action_queues[i]) == 0
            ]
            if pending:
                obs_list = [histories[i].stack() for i in pending]
                obs_batch = stack_obs_batch(obs_list, obs_keys, image_keys)
                obs_batch = model._prepare_obs(obs_batch, device, batched=True)
                model.eval()
                with torch.no_grad():
                    pred = model.forward(obs_batch)
                if torch.is_tensor(pred):
                    pred = pred.detach().cpu()
                if pred.ndim == 2:
                    pred = pred.view(pred.shape[0], model.predict_horizon, -1)
                for idx, env_idx in enumerate(pending):
                    actions_seq = pred[idx]
                    take = min(model.exec_horizon, actions_seq.shape[0])
                    for step_action in actions_seq[:take]:
                        action_queues[env_idx].append(step_action)

            actions = np.zeros((env_num, action_dim), dtype=np.float32)
            for i in range(remaining):
                if dones[i]:
                    continue
                if action_queues[i]:
                    act = action_queues[i].popleft()
                    if torch.is_tensor(act):
                        act = act.cpu().numpy()
                    actions[i] = np.asarray(act).reshape(-1)

            env_obs, _, done, _ = env.step(actions)
            done_array = np.asarray(done)
            for i in range(remaining):
                if not dones[i]:
                    steps_by_env[i] += 1
                if bool(done_array[i]):
                    dones[i] = True
            curr_done = sum(1 for d in dones[:remaining] if d)
            newly_done = curr_done - prev_done
            if newly_done > 0:
                pbar.update(newly_done)
            current_successes = successes + sum(1 for d in dones[:remaining] if d)
            total_done = episodes_done + curr_done
            pbar.set_postfix(
                sr=f"{current_successes / max(total_done, 1):.3f}",
                active=remaining - curr_done,
                step=steps_taken,
            )

            env_obs_list = split_env_obs(env_obs, env_num)
            if video_writer:
                for i in range(remaining):
                    if record_active[i] and video_ids[i] is not None:
                        video_writer.append_obs(
                            env_obs_list[i],
                            done=bool(done_array[i]),
                            idx=video_ids[i],
                            camera_name=video_camera,
                        )
                        if bool(done_array[i]):
                            record_active[i] = False

            for i in range(env_num):
                obs = extract_env_obs(env_obs_list[i], obs_keys, image_keys, obs_key_mapping)
                histories[i].add(obs)

            if all(dones[:remaining]) and (not video_writer or not any(record_active[:remaining])):
                break

        successes += sum(1 for d in dones[:remaining] if d)
        episodes_done += remaining

        print(
            f"[rollout] batch {loop_idx} | episodes {episodes_done}/{n_eval} | steps {steps_taken}"
        )
        for i in range(remaining):
            print(
                f"[rollout] episode {batch_start + i} | init_state {indices[i]} | "
                f"steps {steps_by_env[i]} | success {dones[i]}"
            )
            result = {
                "rollout_idx": batch_start + i,
                "init_idx": indices[i],
                "success": bool(dones[i]),
                "steps": steps_by_env[i],
            }
            if anchor_ids is not None:
                result["anchor_id"] = int(anchor_ids[indices[i]])
            episode_results.append(result)

    env.close()
    pbar.close()
    if video_writer:
        video_writer.save()
    sr = successes / max(n_eval, 1)
    print("[info] rollout summary:")
    print(f"  rollouts: {n_eval}")
    print(f"  envs: {env_num} (use_mp={use_mp})")
    print(f"  success: {successes}/{n_eval} ({sr:.3f})")
    return {
        "n_eval": n_eval,
        "successes": successes,
        "success_rate": sr,
        "rollout_order": rollout_order,
        "episode_results": episode_results,
        "video_dir": str(video_dir) if video_dir is not None else None,
    }


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

    sample = dataset[0]
    action_dim = sample["actions"].shape[-1]
    print(f"[debug] action_dim: {action_dim}")

    image_shapes = {}
    for key in image_keys:
        if key not in sample["obs"]:
            raise KeyError(f"image key not found in obs: {key}")
        image_shapes[key] = sample["obs"][key].shape[1:]

    exec_horizon = get_policy_param(cfg, "exec_horizon")
    if policy_name not in ("act", "cnnmlp"):
        raise ValueError(f"unsupported policy: {policy_name}")
    qpos_dim = sum(np.prod(sample["obs"][k].shape[1:]) for k in obs_keys)
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

    run_env_rollouts(
        cfg,
        model,
        obs_keys,
        image_keys,
        obs_stats,
        demo_path,
        action_dim,
        image_shapes,
    )


if __name__ == "__main__":
    main()

from collections import deque
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

try:
    import draccus
except ImportError as exc:
    raise ImportError("draccus is required; install with `pip install draccus`.") from exc

import cv2

from libero.libero.envs import SegmentationRenderEnv, SubprocVectorEnv
from libero.libero.utils.video_utils import VideoWriter

from standalone.configs import apply_policy_config
from standalone.configs.rollout import SegRolloutConfig
from standalone.rollout_env import (
    ObsHistory,
    _derive_eval_video_dir,
    apply_ckpt_config,
    build_obs_key_mapping,
    build_rollout_summary,
    camera_names_from_mapping,
    extract_env_obs,
    infer_camera_size,
    load_anchor_indices,
    load_init_states,
    read_bddl_from_hdf5,
    resolve_bddl_path,
    resolve_video_dir,
    select_video_camera,
    split_env_obs,
    stack_obs_batch,
    write_rollout_summary,
)
from standalone.utils.train_utils import TRAIN_CONFIG_NAME, load_config_json
from standalone.dataset_utils.hdf5_sequence_dataset import (
    HDF5SequenceDataset,
    load_obs_stats,
)
from standalone.dataset_utils.normalizer_utils import build_identity_normalizer
from standalone.models.algos.dp.utils.normalizer import LinearNormalizer
from standalone.models.policy.policy_factory import build_policy, get_policy_name


def _segmentation_key(env_key: str) -> str:
    if env_key.endswith("_image"):
        return f"{env_key[: -len('_image')]}_segmentation_instance"
    return f"{env_key}_segmentation_instance"


def _ensure_video_camera(
    video_writer: Optional[Any],
    video_camera: Optional[Union[Sequence[str], str]],
    obs_sample: Mapping[str, Any],
) -> Optional[Any]:
    if not video_writer:
        return None
    missing = []
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


def _infer_layout(img: np.ndarray) -> str:
    if img.ndim == 2:
        return "hw"
    if img.ndim != 3:
        raise ValueError(f"unsupported image shape: {img.shape}")
    if img.shape[-1] in (1, 3, 4):
        return "hwc"
    if img.shape[0] in (1, 3, 4):
        return "chw"
    raise ValueError(f"cannot infer image layout from shape: {img.shape}")


def _to_hwc(img: np.ndarray, layout: str) -> Tuple[np.ndarray, str]:
    if layout == "hwc":
        return img, "hwc"
    if layout == "chw":
        return np.transpose(img, (1, 2, 0)), "chw"
    if layout == "hw":
        return img[..., None], "hw"
    raise ValueError(f"unknown layout: {layout}")


def _from_hwc(img: np.ndarray, layout: str, orig_layout: str) -> np.ndarray:
    if orig_layout == "hwc":
        return img
    if orig_layout == "chw":
        return np.transpose(img, (2, 0, 1))
    if orig_layout == "hw":
        return img[..., 0]
    raise ValueError(f"unknown layout: {orig_layout}")


def _apply_mask_image(
    image: np.ndarray,
    mask: np.ndarray,
    mode: str,
    fill_value: float,
    blur_ksize: int,
    blur_sigma: float,
) -> np.ndarray:
    layout = _infer_layout(image)
    img_hwc, orig_layout = _to_hwc(image, layout)

    img_dtype = img_hwc.dtype
    img = img_hwc.astype(np.float32)
    mask = mask.astype(np.float32)
    if mask.ndim == 2:
        mask = mask[..., None]

    if mode == "hard":
        out = img * mask + (1.0 - mask) * float(fill_value)
    elif mode == "soft":
        k = int(blur_ksize)
        if k % 2 == 0:
            k += 1
        blurred = cv2.GaussianBlur(img, (k, k), float(blur_sigma))
        out = img * mask + (1.0 - mask) * blurred
    else:
        raise ValueError(f"unknown seg_mode: {mode}")

    if np.issubdtype(img_dtype, np.integer):
        out = np.clip(out, 0, 255)
        out = out.astype(img_dtype)
    else:
        out = out.astype(img_dtype)

    return _from_hwc(out, layout, orig_layout)


def _to_uint8_image(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    arr = img.astype(np.float32)
    if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0:
        arr = arr * 255.0
    arr = np.clip(arr, 0, 255)
    return arr.astype(np.uint8)


def _ensure_rgb(img: np.ndarray) -> np.ndarray:
    layout = _infer_layout(img)
    img_hwc, _ = _to_hwc(img, layout)
    if img_hwc.shape[-1] == 1:
        img_hwc = np.repeat(img_hwc, 3, axis=-1)
    return _to_uint8_image(img_hwc)


def _build_mask_grid(
    env_obs: Mapping[str, Any],
    env: Any,
    camera_names: Sequence[str],
    mode: str,
    fill_value: float,
    blur_ksize: int,
    blur_sigma: float,
) -> Optional[np.ndarray]:
    rows = []
    missing = []
    for cam in camera_names:
        if cam not in env_obs:
            missing.append(cam)
            continue
        seg_key = _segmentation_key(cam)
        if seg_key not in env_obs:
            missing.append(seg_key)
            continue
        seg_img = np.squeeze(env_obs[seg_key])
        if isinstance(env, SubprocVectorEnv):
            seg_mask_list = env.get_segmentation_of_interest([seg_img])
            seg_mask = seg_mask_list[0]
        else:
            seg_mask = env.get_segmentation_of_interest(seg_img)
        mask = (np.squeeze(seg_mask) == 1).astype(np.float32)
        masked = _apply_mask_image(
            env_obs[cam],
            mask,
            mode=mode,
            fill_value=fill_value,
            blur_ksize=blur_ksize,
            blur_sigma=blur_sigma,
        )
        orig_rgb = _ensure_rgb(env_obs[cam])
        mask_rgb = _ensure_rgb((mask * 255.0).astype(np.uint8))
        masked_rgb = _ensure_rgb(masked)

        orig_rgb = orig_rgb[::-1]
        mask_rgb = mask_rgb[::-1]
        masked_rgb = masked_rgb[::-1]
        row = np.concatenate([orig_rgb, mask_rgb, masked_rgb], axis=1)
        rows.append(row)

    if missing:
        print(f"[warning] missing video keys: {missing}; disabling video grid")
        return None
    if not rows:
        return None
    if len(rows) == 1:
        return rows[0]
    return np.concatenate(rows, axis=0)


def _select_video_cameras(
    video_camera: Optional[Union[Sequence[str], str]],
    image_keys: Sequence[str],
    obs_key_mapping: Mapping[str, str],
    explicit: bool = False,
) -> Sequence[str]:
    cameras = []
    if explicit:
        if isinstance(video_camera, (list, tuple)):
            cameras = [str(name) for name in video_camera]
        elif video_camera:
            cameras = [str(video_camera)]
    if cameras:
        return cameras
    env_keys = [obs_key_mapping.get(k, k) for k in image_keys]
    preferred = []
    for name in ("agentview_image", "robot0_eye_in_hand_image"):
        if name in env_keys:
            preferred.append(name)
    if preferred:
        return preferred
    return env_keys[:2]


def _mask_env_obs_single(
    env_obs: Mapping[str, Any],
    env: Any,
    image_keys: Sequence[str],
    obs_key_mapping: Mapping[str, str],
    mode: str,
    fill_value: float,
    blur_ksize: int,
    blur_sigma: float,
) -> Dict[str, Any]:
    masked = dict(env_obs)
    for key in image_keys:
        env_key = obs_key_mapping.get(key, key)
        if env_key not in env_obs:
            continue
        seg_key = _segmentation_key(env_key)
        if seg_key not in env_obs:
            continue
        seg_img = np.squeeze(env_obs[seg_key])
        seg_mask = env.get_segmentation_of_interest(seg_img)
        mask = (np.squeeze(seg_mask) == 1).astype(np.float32)
        masked[env_key] = _apply_mask_image(
            env_obs[env_key],
            mask,
            mode=mode,
            fill_value=fill_value,
            blur_ksize=blur_ksize,
            blur_sigma=blur_sigma,
        )
    return masked


def _mask_env_obs_batch(
    env_obs_list: Sequence[Mapping[str, Any]],
    env: Any,
    image_keys: Sequence[str],
    obs_key_mapping: Mapping[str, str],
    mode: str,
    fill_value: float,
    blur_ksize: int,
    blur_sigma: float,
) -> Sequence[Dict[str, Any]]:
    masked_list = [dict(obs) for obs in env_obs_list]
    for key in image_keys:
        env_key = obs_key_mapping.get(key, key)
        seg_key = _segmentation_key(env_key)
        seg_imgs = []
        valid = True
        for obs in env_obs_list:
            if env_key not in obs or seg_key not in obs:
                valid = False
                break
            seg_imgs.append(np.squeeze(obs[seg_key]))
        if not valid:
            continue
        seg_masks = env.get_segmentation_of_interest(seg_imgs)
        for i, seg_mask in enumerate(seg_masks):
            mask = (np.squeeze(seg_mask) == 1).astype(np.float32)
            masked_list[i][env_key] = _apply_mask_image(
                env_obs_list[i][env_key],
                mask,
                mode=mode,
                fill_value=fill_value,
                blur_ksize=blur_ksize,
                blur_sigma=blur_sigma,
            )
    return masked_list


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

    env_args = {"bddl_file_name": bddl_path, "camera_segmentations": "instance"}
    env_horizon = getattr(cfg, "env_horizon", None)
    if env_horizon is not None:
        env_horizon = int(env_horizon)
        env_args["horizon"] = env_horizon
        min_needed = int(getattr(cfg, "steps", 0)) + int(getattr(cfg, "warmup_steps", 0))
        if env_horizon < min_needed:
            env_horizon = min_needed + 1
            env_args["horizon"] = env_horizon
            print(
                "[info] env_horizon < steps+warmup_steps; bumping horizon to steps+warmup_steps+1"
            )
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

    seg_mode = str(getattr(cfg, "seg_mode", "hard")).lower()
    seg_fill_value = float(getattr(cfg, "seg_fill_value", 0.0))
    seg_blur_ksize = int(getattr(cfg, "seg_blur_ksize", 11))
    seg_blur_sigma = float(getattr(cfg, "seg_blur_sigma", 5.0))
    video_show_masks = bool(getattr(cfg, "video_show_masks", False))

    video_dir = resolve_video_dir(cfg)
    save_videos = int(getattr(cfg, "save_videos", 0))
    video_writer = None
    video_camera = None
    video_cameras = []
    video_camera_explicit = bool(str(getattr(cfg, "video_camera", "")).strip())
    if save_videos > 0:
        video_camera = select_video_camera(cfg, image_keys, obs_key_mapping)
        if not video_camera:
            print("[warning] save_videos requested but no image_keys; skipping video")
        else:
            if video_show_masks:
                video_cameras = _select_video_cameras(
                    video_camera,
                    image_keys,
                    obs_key_mapping,
                    explicit=video_camera_explicit,
                )
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
        n_rollouts = len(rollout_order)
    else:
        n_rollouts = int(cfg.n_rollouts)
        if n_rollouts <= 0:
            raise ValueError("n_rollouts must be >= 1")
        if n_rollouts > total_states:
            print(
                f"[warning] n_rollouts={n_rollouts} > init_states={total_states}; clipping"
            )
            n_rollouts = total_states
        start_idx = int(cfg.sample_index)
        if start_idx < 0 or start_idx >= total_states:
            raise ValueError(
                f"sample_index out of range: {start_idx} (0..{total_states - 1})"
            )
        rollout_order = [(start_idx + i) % total_states for i in range(n_rollouts)]

    use_mp = bool(getattr(cfg, "use_mp", False))
    num_procs = int(getattr(cfg, "num_procs", 1))
    if num_procs <= 0:
        raise ValueError("num_procs must be >= 1")
    env_num = min(num_procs, n_rollouts) if use_mp else 1
    rollout_loop_num = (n_rollouts + env_num - 1) // env_num

    max_steps = int(cfg.steps)
    if max_steps <= 0:
        raise ValueError("steps must be >= 1")

    episode_results = []
    if env_num == 1:
        env = SegmentationRenderEnv(**env_args)
        env.seed(cfg.data.seed)
        history = ObsHistory(
            obs_keys + image_keys,
            cfg.data.obs_horizon,
            image_keys=image_keys,
            image_norm=cfg.data.image_norm,
        )
        successes = 0
        pbar = tqdm(total=n_rollouts, desc="rollout", leave=True)

        for ep_idx, init_idx in enumerate(rollout_order):
            model.reset()
            history.reset()
            env.reset()
            env_obs = env.set_init_state(init_states[init_idx])
            video_writer = _ensure_video_camera(video_writer, video_camera, env_obs)

            masked_obs = _mask_env_obs_single(
                env_obs,
                env,
                image_keys,
                obs_key_mapping,
                mode=seg_mode,
                fill_value=seg_fill_value,
                blur_ksize=seg_blur_ksize,
                blur_sigma=seg_blur_sigma,
            )
            obs = extract_env_obs(masked_obs, obs_keys, image_keys, obs_key_mapping)
            history.add(obs)

            dummy = np.zeros((action_dim,), dtype=np.float32)
            for _ in range(int(cfg.warmup_steps)):
                env_obs, _, _, _ = env.step(dummy)
                masked_obs = _mask_env_obs_single(
                    env_obs,
                    env,
                    image_keys,
                    obs_key_mapping,
                    mode=seg_mode,
                    fill_value=seg_fill_value,
                    blur_ksize=seg_blur_ksize,
                    blur_sigma=seg_blur_sigma,
                )
                obs = extract_env_obs(masked_obs, obs_keys, image_keys, obs_key_mapping)
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
                    if video_show_masks:
                        frame = _build_mask_grid(
                            env_obs,
                            env,
                            video_cameras,
                            mode=seg_mode,
                            fill_value=seg_fill_value,
                            blur_ksize=seg_blur_ksize,
                            blur_sigma=seg_blur_sigma,
                        )
                        if frame is not None:
                            video_writer.append_image(frame, idx=ep_idx)
                    else:
                        video_writer.append_obs(
                            env_obs, done=bool(done), idx=ep_idx, camera_name=video_camera
                        )
                masked_obs = _mask_env_obs_single(
                    env_obs,
                    env,
                    image_keys,
                    obs_key_mapping,
                    mode=seg_mode,
                    fill_value=seg_fill_value,
                    blur_ksize=seg_blur_ksize,
                    blur_sigma=seg_blur_sigma,
                )
                obs = extract_env_obs(masked_obs, obs_keys, image_keys, obs_key_mapping)
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
        sr = successes / max(n_rollouts, 1)
        print("[info] rollout summary:")
        print(f"  rollouts: {n_rollouts}")
        print(f"  success: {successes}/{n_rollouts} ({sr:.3f})")
        summary = build_rollout_summary(n_rollouts, successes, episode_results)
        summary_path = write_rollout_summary(video_dir, summary)
        return {
            "n_rollouts": n_rollouts,
            "successes": successes,
            "success_rate": sr,
            "rollout_order": rollout_order,
            "episode_results": episode_results,
            "video_dir": str(video_dir) if video_dir is not None else None,
            "summary_path": str(summary_path) if summary_path is not None else None,
        }

    env = SubprocVectorEnv([lambda: SegmentationRenderEnv(**env_args) for _ in range(env_num)])
    env.seed(cfg.data.seed)
    histories = [
        ObsHistory(
            obs_keys + image_keys,
            cfg.data.obs_horizon,
            image_keys=image_keys,
            image_norm=cfg.data.image_norm,
        )
        for _ in range(env_num)
    ]

    max_record_videos = min(save_videos, n_rollouts)
    record_active = [False] * env_num
    video_ids = [None] * env_num

    successes = 0
    episodes_done = 0
    pbar = tqdm(total=n_rollouts, desc="rollout", leave=True)
    for loop_idx in range(rollout_loop_num):
        if episodes_done >= n_rollouts:
            break
        batch_start = episodes_done
        model.reset()
        for history in histories:
            history.reset()

        remaining = min(env_num, n_rollouts - episodes_done)
        indices = rollout_order[episodes_done : episodes_done + remaining]
        if len(indices) < env_num:
            indices = indices + [indices[-1]] * (env_num - len(indices))
        init_states_batch = init_states[indices]

        env.reset()
        env_obs = env.set_init_state(init_states_batch)
        env_obs_list = split_env_obs(env_obs, env_num)
        video_writer = _ensure_video_camera(video_writer, video_camera, env_obs_list[0])
        if video_writer:
            for i in range(env_num):
                record_active[i] = False
                video_ids[i] = None
            rec_slots = max_record_videos - episodes_done
            rec = max(0, min(rec_slots, remaining))
            for i in range(rec):
                record_active[i] = True
                video_ids[i] = episodes_done + i

        masked_list = _mask_env_obs_batch(
            env_obs_list,
            env,
            image_keys,
            obs_key_mapping,
            mode=seg_mode,
            fill_value=seg_fill_value,
            blur_ksize=seg_blur_ksize,
            blur_sigma=seg_blur_sigma,
        )
        for i in range(env_num):
            obs = extract_env_obs(masked_list[i], obs_keys, image_keys, obs_key_mapping)
            histories[i].add(obs)

        dummy = np.zeros((env_num, action_dim), dtype=np.float32)
        for _ in range(int(cfg.warmup_steps)):
            env_obs, _, _, _ = env.step(dummy)
            env_obs_list = split_env_obs(env_obs, env_num)
            masked_list = _mask_env_obs_batch(
                env_obs_list,
                env,
                image_keys,
                obs_key_mapping,
                mode=seg_mode,
                fill_value=seg_fill_value,
                blur_ksize=seg_blur_ksize,
                blur_sigma=seg_blur_sigma,
            )
            for i in range(env_num):
                obs = extract_env_obs(masked_list[i], obs_keys, image_keys, obs_key_mapping)
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
                        if video_show_masks:
                            frame = _build_mask_grid(
                                env_obs_list[i],
                                env,
                                video_cameras,
                                mode=seg_mode,
                                fill_value=seg_fill_value,
                                blur_ksize=seg_blur_ksize,
                                blur_sigma=seg_blur_sigma,
                            )
                            if frame is not None:
                                video_writer.append_image(frame, idx=video_ids[i])
                        else:
                            video_writer.append_obs(
                                env_obs_list[i],
                                done=bool(done_array[i]),
                                idx=video_ids[i],
                                camera_name=video_camera,
                            )
                        if bool(done_array[i]):
                            record_active[i] = False

            masked_list = _mask_env_obs_batch(
                env_obs_list,
                env,
                image_keys,
                obs_key_mapping,
                mode=seg_mode,
                fill_value=seg_fill_value,
                blur_ksize=seg_blur_ksize,
                blur_sigma=seg_blur_sigma,
            )
            for i in range(env_num):
                obs = extract_env_obs(masked_list[i], obs_keys, image_keys, obs_key_mapping)
                histories[i].add(obs)

            if all(dones[:remaining]) and (
                not video_writer or not any(record_active[:remaining])
            ):
                break

        incomplete = remaining - sum(1 for d in dones[:remaining] if d)
        if incomplete > 0:
            pbar.update(incomplete)

        successes += sum(1 for d in dones[:remaining] if d)
        episodes_done += remaining

        print(
            f"[rollout] batch {loop_idx} | episodes {episodes_done}/{n_rollouts} | steps {steps_taken}"
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
    sr = successes / max(n_rollouts, 1)
    print("[info] rollout summary:")
    print(f"  rollouts: {n_rollouts}")
    print(f"  envs: {env_num} (use_mp={use_mp})")
    print(f"  success: {successes}/{n_rollouts} ({sr:.3f})")
    summary = build_rollout_summary(n_rollouts, successes, episode_results)
    summary_path = write_rollout_summary(video_dir, summary)
    return {
        "n_rollouts": n_rollouts,
        "successes": successes,
        "success_rate": sr,
        "rollout_order": rollout_order,
        "episode_results": episode_results,
        "video_dir": str(video_dir) if video_dir is not None else None,
        "summary_path": str(summary_path) if summary_path is not None else None,
    }


@draccus.wrap()
def main(cfg: SegRolloutConfig):
    if not cfg.ckpt:
        raise ValueError("ckpt is required")
    ckpt_path = Path(cfg.ckpt).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    run_config = None
    config_path = ckpt_path.parent / TRAIN_CONFIG_NAME
    if config_path.exists():
        run_config = load_config_json(config_path)
    elif isinstance(ckpt, dict) and isinstance(ckpt.get("config"), dict):
        run_config = ckpt["config"]
    if run_config is not None and apply_ckpt_config(cfg, run_config):
        print("[info] using config from checkpoint")
    if not getattr(cfg, "video_dir", ""):
        derived_dir = _derive_eval_video_dir(cfg, run_config)
        if derived_dir is not None:
            cfg.video_dir = str(derived_dir)

    apply_policy_config(cfg)
    if not cfg.data.demo_file:
        raise ValueError("data.demo_file is required")
    obs_keys = [k.strip() for k in cfg.data.obs_keys.split(",") if k.strip()]
    image_keys = [k.strip() for k in cfg.data.image_keys.split(",") if k.strip()]
    all_keys = obs_keys + image_keys
    policy_name = get_policy_name(cfg)

    demo_path = Path(cfg.data.demo_file).expanduser().resolve()
    if not demo_path.exists():
        raise FileNotFoundError(f"HDF5 not found: {demo_path}")

    device = cfg.device if torch.cuda.is_available() else "cpu"

    dataset = HDF5SequenceDataset(
        hdf5_path=str(demo_path),
        obs_keys=all_keys,
        obs_horizon=cfg.data.obs_horizon,
        predict_horizon=cfg.data.predict_horizon,
        action_shift=getattr(cfg.data, "action_shift", 0),
        image_keys=image_keys,
        image_norm=cfg.data.image_norm,
        image_transforms=None,
    )

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

    sample = dataset[0]
    action_dim = sample["actions"].shape[-1]
    print(f"[debug] action_dim: {action_dim}")

    image_shapes = {}
    for key in image_keys:
        if key not in sample["obs"]:
            raise KeyError(f"image key not found in obs: {key}")
        image_shapes[key] = sample["obs"][key].shape[1:]

    if policy_name not in ("act", "cnnmlp", "dp"):
        raise ValueError(f"unsupported policy: {policy_name}")
    proprio_dim = sum(np.prod(sample["obs"][k].shape[1:]) for k in obs_keys)
    obs_shapes = {key: value.shape for key, value in sample["obs"].items()}
    model = build_policy(
        cfg,
        obs_keys,
        image_keys,
        action_dim,
        proprio_dim=proprio_dim,
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

    anchor_ids = load_anchor_indices(cfg)
    run_env_rollouts(
        cfg,
        model,
        obs_keys,
        image_keys,
        obs_stats,
        demo_path,
        action_dim,
        image_shapes,
        anchor_ids=anchor_ids,
    )


if __name__ == "__main__":
    main()

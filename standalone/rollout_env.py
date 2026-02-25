from collections import deque
from pathlib import Path

import numpy as np
import torch

try:
    import draccus
except ImportError as exc:
    raise ImportError("draccus is required; install with `pip install draccus`.") from exc

from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
from libero.libero.utils.video_utils import VideoWriter

from standalone.configs import RolloutConfig, apply_policy_config
from standalone.utils.rollout_utils import (
    ObsHistory,
    _derive_eval_video_dir,
    _ensure_video_camera,
    apply_ckpt_config,
    build_obs_key_mapping,
    build_rollout_summary,
    camera_names_from_mapping,
    extract_env_obs,
    infer_camera_size,
    infer_rollout_io_specs,
    load_anchor_indices,
    read_env_kwargs_from_hdf5,
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
from standalone.dataset_utils.normalizer_utils import build_identity_normalizer
from standalone.models.algos.dp.utils.normalizer import LinearNormalizer
from standalone.models.policy.policy_factory import build_policy, get_policy_name


def run_env_rollouts(
    cfg,
    model,
    obs_keys,
    image_keys,
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
    dataset_env_kwargs = read_env_kwargs_from_hdf5(str(demo_path))
    if dataset_env_kwargs:
        env_args.update(dataset_env_kwargs)
        print(
            "[info] rollout env kwargs from dataset:",
            ", ".join(sorted(dataset_env_kwargs.keys())),
        )
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

    video_dir = resolve_video_dir(cfg)
    save_videos = int(getattr(cfg, "save_videos", 0))
    video_writer = None
    video_camera = None
    if save_videos > 0:
        video_camera = select_video_camera(cfg, image_keys, obs_key_mapping)
        if not video_camera:
            print("[warning] save_videos requested but no image_keys; skipping video")
        else:
            video_writer = VideoWriter(
                video_path=str(video_dir),
                save_video=True,
                fps=int(getattr(cfg, "video_fps", 30)),
                single_video=False,
                stream_write=True,
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
        env = OffScreenRenderEnv(**env_args)
        env.seed(cfg.data.seed)
        history = ObsHistory(
            obs_keys + image_keys,
            cfg.data.obs_horizon,
            image_keys=image_keys,
            image_norm=cfg.data.image_norm,
        )
        successes = 0

        for ep_idx, init_idx in enumerate(rollout_order):
            model.reset()
            history.reset()
            env.reset()
            env_obs = env.set_init_state(init_states[init_idx])
            video_writer = _ensure_video_camera(video_writer, video_camera, env_obs)
            obs = extract_env_obs(env_obs, obs_keys, image_keys, obs_key_mapping)
            history.add(obs)

            dummy = np.zeros((action_dim,), dtype=np.float32)
            for _ in range(int(cfg.warmup_steps)):
                env_obs, _, _, _ = env.step(dummy)
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

                # TODO: consider requiring success to hold for N consecutive steps
                if done:
                    successes += 1
                    break

            print(
                f"[rollout] episode {ep_idx} | init_state {init_idx} | steps {steps_taken} | success {done}"
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

    env = SubprocVectorEnv([lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)])
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

        for i in range(env_num):
            obs = extract_env_obs(env_obs_list[i], obs_keys, image_keys, obs_key_mapping)
            histories[i].add(obs)

        dummy = np.zeros((env_num, action_dim), dtype=np.float32)
        for _ in range(int(cfg.warmup_steps)):
            env_obs, _, _, _ = env.step(dummy)
            env_obs_list = split_env_obs(env_obs, env_num)
            for i in range(env_num):
                obs = extract_env_obs(env_obs_list[i], obs_keys, image_keys, obs_key_mapping)
                histories[i].add(obs)

        action_queues = [deque() for _ in range(env_num)]
        dones = [False] * env_num
        steps_by_env = [0] * env_num
        for k in range(remaining, env_num):
            dones[k] = True

        steps_taken = 0
        while steps_taken < max_steps:
            steps_taken += 1
            pending = [
                i
                for i in range(remaining)
                if not dones[i] and len(action_queues[i]) == 0
            ]
            if pending:
                obs_list = [histories[i].stack() for i in pending]
                obs_batch = stack_obs_batch(obs_list, obs_keys, image_keys)
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
                # TODO: consider requiring success to hold for N consecutive steps
                if bool(done_array[i]):
                    dones[i] = True

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
def main(cfg: RolloutConfig):
    if not cfg.ckpt:
        raise ValueError("ckpt is required")
    ckpt_path = Path(cfg.ckpt).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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
    policy_name = get_policy_name(cfg)

    demo_path = Path(cfg.data.demo_file).expanduser().resolve()
    if not demo_path.exists():
        raise FileNotFoundError(f"HDF5 not found: {demo_path}")

    device = cfg.device if torch.cuda.is_available() else "cpu"

    action_dim, image_shapes, obs_shapes, proprio_dim = infer_rollout_io_specs(
        hdf5_path=str(demo_path),
        obs_keys=obs_keys,
        image_keys=image_keys,
        obs_horizon=cfg.data.obs_horizon,
    )
    print(f"[debug] action_dim: {action_dim}")

    if policy_name not in ("act", "dp"):
        raise ValueError(f"unsupported policy: {policy_name}")
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
        demo_path,
        action_dim,
        image_shapes,
        anchor_ids=anchor_ids,
    )


if __name__ == "__main__":
    main()

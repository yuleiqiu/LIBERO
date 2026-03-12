from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch

try:
    import draccus
except ImportError as exc:
    raise ImportError("draccus is required; install with `pip install draccus`.") from exc

from libero.libero.envs import OffScreenRenderEnv, SegmentationRenderEnv, SubprocVectorEnv
from libero.libero.utils.video_utils import VideoWriter

from standalone.configs import RolloutConfig, apply_policy_config
from standalone.utils.rollout_utils import (
    _derive_eval_video_dir,
    _ensure_video_camera,
    active_mask_keys,
    apply_ckpt_config,
    build_histories,
    build_mask_obs_batch,
    build_per_env_temporal_ensemblers,
    build_obs_key_mapping,
    build_rollout_summary,
    camera_names_from_mapping,
    extract_env_obs,
    infer_camera_size,
    infer_rollout_io_specs,
    load_anchor_indices,
    pending_env_indices,
    parse_mask_keys,
    pop_actions,
    read_env_kwargs_from_hdf5,
    refill_action_queues,
    reset_rollout_runtime,
    seed_rollout_env,
    set_init_state_batch,
    step_env_batch,
    load_init_states,
    resolve_rollout_bddl_path,
    resolve_video_dir,
    select_video_camera,
    set_rollout_seed,
    stack_obs_batch,
    write_rollout_summary,
)
from standalone.utils.model_spec_utils import load_model_spec, unpack_model_spec
from standalone.utils.rollout_spec_utils import (
    apply_rollout_spec_overrides,
    get_rollout_env_kwargs,
    load_rollout_spec,
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
    mask_keys,
    demo_path: Optional[Path],
    action_dim,
    image_shapes,
    env_kwargs_override=None,
    init_states_override=None,
    rollout_order_override=None,
    anchor_ids=None,
):
    bddl_path, _, _ = resolve_rollout_bddl_path(cfg, demo_path)
    active_masks = active_mask_keys(mask_keys)

    if init_states_override is None:
        init_states = load_init_states(cfg, demo_path)
    else:
        init_states = np.asarray(init_states_override)

    obs_key_mapping = build_obs_key_mapping(cfg, obs_keys, image_keys)
    camera_names = camera_names_from_mapping(image_keys, obs_key_mapping) if image_keys else []
    cam_hw = infer_camera_size(image_shapes) if image_keys else None

    env_args = {"bddl_file_name": str(bddl_path)}
    if env_kwargs_override:
        env_args.update(dict(env_kwargs_override))
        print(
            "[info] rollout env kwargs from rollout_spec:",
            ", ".join(sorted(env_kwargs_override.keys())),
        )
    elif demo_path is not None:
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
        if active_masks:
            env_args["camera_segmentations"] = "instance"
    else:
        if active_masks:
            raise ValueError("mask_keys require image_keys during env rollout")
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

    env_cls = SegmentationRenderEnv if active_masks else OffScreenRenderEnv
    if env_num == 1:
        env = env_cls(**env_args)
    else:
        env = SubprocVectorEnv([lambda: env_cls(**env_args) for _ in range(env_num)])
    seed_rollout_env(env, int(cfg.data.seed), env_num)
    histories = build_histories(cfg, obs_keys, image_keys, env_num, extra_keys=active_masks)

    episode_results = []
    max_record_videos = min(save_videos, n_rollouts)
    record_active = [False] * env_num
    video_ids = [None] * env_num
    temporal_ensemblers = build_per_env_temporal_ensemblers(model, env_num)

    successes = 0
    episodes_done = 0
    for loop_idx in range(rollout_loop_num):
        if episodes_done >= n_rollouts:
            break
        batch_start = episodes_done
        reset_rollout_runtime(model, histories, temporal_ensemblers)

        remaining = min(env_num, n_rollouts - episodes_done)
        indices = rollout_order[episodes_done : episodes_done + remaining]
        if len(indices) < env_num:
            indices = indices + [indices[-1]] * (env_num - len(indices))
        init_states_batch = init_states[indices]

        env.reset()
        env_obs_list = set_init_state_batch(env, init_states_batch, env_num)
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

        mask_obs_list = build_mask_obs_batch(env_obs_list, env, image_keys, mask_keys, obs_key_mapping)
        for i in range(env_num):
            obs = extract_env_obs(
                env_obs_list[i],
                obs_keys,
                image_keys,
                obs_key_mapping,
                extra_obs=mask_obs_list[i],
            )
            histories[i].add(obs)

        dummy = np.zeros((env_num, action_dim), dtype=np.float32)
        for _ in range(int(cfg.warmup_steps)):
            env_obs_list, _ = step_env_batch(env, dummy, env_num)
            mask_obs_list = build_mask_obs_batch(
                env_obs_list, env, image_keys, mask_keys, obs_key_mapping
            )
            for i in range(env_num):
                obs = extract_env_obs(
                    env_obs_list[i],
                    obs_keys,
                    image_keys,
                    obs_key_mapping,
                    extra_obs=mask_obs_list[i],
                )
                histories[i].add(obs)

        action_queues = [deque() for _ in range(env_num)]
        dones = [False] * env_num
        steps_by_env = [0] * env_num
        for k in range(remaining, env_num):
            dones[k] = True

        steps_taken = 0
        while steps_taken < max_steps:
            steps_taken += 1
            pending = pending_env_indices(
                remaining, dones, action_queues, temporal_ensemblers
            )
            if pending:
                obs_list = [histories[i].stack() for i in pending]
                obs_batch = stack_obs_batch(obs_list, obs_keys, image_keys, extra_keys=active_masks)
                refill_action_queues(
                    model,
                    obs_batch,
                    pending,
                    action_queues,
                    temporal_ensemblers=temporal_ensemblers,
                )

            actions = pop_actions(action_queues, dones, remaining, env_num, action_dim)
            env_obs_list, done_array = step_env_batch(env, actions, env_num)
            for i in range(remaining):
                if not dones[i]:
                    steps_by_env[i] += 1
                # TODO: consider requiring success to hold for N consecutive steps
                if bool(done_array[i]):
                    dones[i] = True

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

            mask_obs_list = build_mask_obs_batch(
                env_obs_list, env, image_keys, mask_keys, obs_key_mapping
            )
            for i in range(env_num):
                obs = extract_env_obs(
                    env_obs_list[i],
                    obs_keys,
                    image_keys,
                    obs_key_mapping,
                    extra_obs=mask_obs_list[i],
                )
                histories[i].add(obs)

            if all(dones[:remaining]) and (not video_writer or not any(record_active[:remaining])):
                break

        successes += sum(1 for d in dones[:remaining] if d)
        episodes_done += remaining

        if env_num > 1:
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
    if env_num > 1:
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
    rollout_spec = load_rollout_spec(ckpt=ckpt, run_config=run_config)
    if apply_rollout_spec_overrides(cfg, rollout_spec):
        print("[info] using rollout_spec from checkpoint metadata")
    if not getattr(cfg, "video_dir", ""):
        derived_dir = _derive_eval_video_dir(cfg, run_config)
        if derived_dir is not None:
            cfg.video_dir = str(derived_dir)

    apply_policy_config(cfg)
    set_rollout_seed(int(cfg.data.seed))
    print(f"[info] rollout seed: {int(cfg.data.seed)}")
    obs_keys = [k.strip() for k in cfg.data.obs_keys.split(",") if k.strip()]
    image_keys = [k.strip() for k in cfg.data.image_keys.split(",") if k.strip()]
    mask_keys = parse_mask_keys(getattr(cfg.data, "mask_keys", ""), image_keys)
    active_masks = active_mask_keys(mask_keys)
    policy_name = get_policy_name(cfg)

    device = cfg.device if torch.cuda.is_available() else "cpu"
    explicit_env_override = bool(getattr(cfg, "bddl_file", None)) and bool(
        getattr(cfg, "init_states", None)
    )
    demo_path = None
    raw_demo_file = str(getattr(cfg.data, "demo_file", "") or "").strip()
    if raw_demo_file:
        candidate = Path(raw_demo_file).expanduser().resolve()
        if candidate.exists():
            demo_path = candidate
        elif not explicit_env_override:
            raise FileNotFoundError(f"HDF5 not found: {candidate}")
        else:
            print(
                "[warning] data.demo_file does not exist; ignoring it because "
                "bddl_file and init_states were provided explicitly"
            )
    model_spec = load_model_spec(ckpt=ckpt, run_config=run_config)
    if model_spec is not None:
        action_dim, image_shapes, obs_shapes, proprio_dim = unpack_model_spec(model_spec)
        print("[info] using model_spec from checkpoint metadata")
    else:
        if demo_path is None:
            raise ValueError(
                "data.demo_file is required when checkpoint metadata lacks model_spec"
            )
        action_dim, image_shapes, obs_shapes, proprio_dim = infer_rollout_io_specs(
            hdf5_path=str(demo_path),
            obs_keys=obs_keys,
            image_keys=image_keys,
            obs_horizon=cfg.data.obs_horizon,
            extra_obs_keys=active_masks,
        )
        print(f"[debug] action_dim: {action_dim}")
    if demo_path is None and not explicit_env_override:
        raise ValueError(
            "data.demo_file is required unless both bddl_file and init_states are provided"
        )

    if policy_name not in ("act", "dp"):
        raise ValueError(f"unsupported policy: {policy_name}")
    model = build_policy(
        cfg,
        obs_keys,
        image_keys,
        action_dim,
        proprio_dim=proprio_dim,
        obs_shapes=obs_shapes,
        mask_keys=mask_keys if active_masks else None,
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
    rollout_env_kwargs = {} if explicit_env_override else get_rollout_env_kwargs(rollout_spec)
    run_env_rollouts(
        cfg,
        model,
        obs_keys,
        image_keys,
        mask_keys,
        demo_path,
        action_dim,
        image_shapes,
        env_kwargs_override=rollout_env_kwargs,
        anchor_ids=anchor_ids,
    )


if __name__ == "__main__":
    main()

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from easydict import EasyDict
from hydra import compose, initialize_config_dir
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from tqdm import tqdm

import init_path  # noqa: F401
from libero.libero import get_libero_path
from libero.lifelong.algos import get_algo_class
from libero.lifelong.datasets import SequenceVLDataset, get_dataset
from libero.lifelong.metric import raw_obs_to_tensor_obs
from libero.lifelong.utils import control_seed, safe_device, torch_load_model
from libero.libero.envs import DummyVectorEnv, OffScreenRenderEnv, SubprocVectorEnv
from libero.libero.utils.video_utils import VideoWriter


def load_cfg(overrides):
    config_dir = to_absolute_path("libero/configs")
    with initialize_config_dir(config_dir=config_dir, job_name="single_task_rollout"):
        hydra_cfg = compose(config_name="config", overrides=overrides)
    cfg = EasyDict(yaml.safe_load(OmegaConf.to_yaml(hydra_cfg)))
    return cfg


def build_task_emb(cfg):
    lang_dim = cfg.policy.language_encoder.network_kwargs.get("input_size", 1)
    return torch.zeros((lang_dim,), dtype=torch.float32)


def easydict_to_plain(d):
    if isinstance(d, dict):
        return {k: easydict_to_plain(v) for k, v in d.items()}
    return d


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


def build_rollout_order(anchor_idx_list, total_len, target_n):
    if not anchor_idx_list:
        return list(range(min(target_n, total_len)))
    by_anchor = defaultdict(list)
    for idx, a in enumerate(anchor_idx_list):
        by_anchor[a].append(idx)
    order = []
    while len(order) < target_n and any(by_anchor.values()):
        for a in sorted(by_anchor.keys()):
            if by_anchor[a]:
                order.append(by_anchor[a].pop(0))
                if len(order) >= target_n:
                    break
    return order


def main():
    parser = argparse.ArgumentParser(description="Rollout checkpoint using init states stored in an HDF5 demo file")
    parser.add_argument("--checkpoint", required=True, help="Path to task0_model.pth")
    parser.add_argument("--demo-file", required=True, help="Path to processed *_demo.hdf5")
    parser.add_argument(
        "--config-override",
        nargs="*",
        default=[],
        help="Optional hydra-style overrides if checkpoint lacks cfg",
    )
    parser.add_argument("--device-id", type=int, default=0, help="CUDA device id for eval (e.g., 0 -> cuda:0)")
    parser.add_argument(
        "--n-eval",
        type=int,
        default=None,
        help="Override number of rollouts (default: cfg.eval.n_eval or number of init states)",
    )
    parser.add_argument(
        "--save-videos",
        type=int,
        default=0,
        help="If >0, save this many rollout videos (one mp4 per episode).",
    )
    args = parser.parse_args()

    demo_path = Path(args.demo_file).expanduser().resolve()
    if not demo_path.exists():
        raise FileNotFoundError(f"HDF5 not found: {demo_path}")

    device = f"cuda:{args.device_id}"
    map_location = device if torch.cuda.is_available() else "cpu"
    state_dict, ckpt_cfg, _ = torch_load_model(args.checkpoint, map_location=map_location)

    if ckpt_cfg is None:
        cfg = load_cfg(args.config_override)
    else:
        cfg = ckpt_cfg
        if args.config_override:
            base = OmegaConf.create(easydict_to_plain(cfg))
            override_conf = OmegaConf.from_dotlist(args.config_override)
            merged = OmegaConf.merge(base, override_conf)
            cfg = EasyDict(OmegaConf.to_container(merged, resolve=True))

    cfg.device = device
    control_seed(cfg.seed)

    cfg.folder = cfg.folder or str(demo_path.parent)
    cfg.bddl_folder = cfg.bddl_folder or get_libero_path("bddl_files")
    cfg.init_states_folder = cfg.init_states_folder or get_libero_path("init_states")

    # Initialize ObsUtils by building the dataset
    base_dataset, shape_meta = get_dataset(
        dataset_path=str(demo_path),
        obs_modality=cfg.data.obs.modality,
        initialize_obs_utils=True,
        seq_len=cfg.data.seq_len,
    )
    task_emb = build_task_emb(cfg)
    SequenceVLDataset(base_dataset, task_emb)

    algo_cls = get_algo_class(cfg.lifelong.algo)
    algo = safe_device(algo_cls(n_tasks=1, cfg=cfg), cfg.device)
    algo.policy.load_state_dict(state_dict)
    algo.policy.eval()

    init_states = read_init_states_from_hdf5(str(demo_path))
    anchor_indices = None  # ignore any anchor metadata; use sequential ordering
    if torch.is_tensor(init_states):
        init_states = init_states.cpu().numpy()

    bddl_file_name = read_bddl_from_hdf5(str(demo_path))
    if bddl_file_name is None:
        print("[warning] bddl_file_name not found in hdf5; abort rollout")
        return

    env_args = {
        "bddl_file_name": bddl_file_name,
        "camera_heights": cfg.data.img_h,
        "camera_widths": cfg.data.img_w,
    }
    max_steps = getattr(cfg.eval, "max_steps", 600)

    desired_n_eval = args.n_eval if args.n_eval is not None else getattr(cfg.eval, "n_eval", 20)
    max_available = init_states.shape[0]
    n_eval = min(desired_n_eval, max_available)

    rollout_order = build_rollout_order(anchor_indices, max_available, n_eval)
    if len(rollout_order) < n_eval:
        n_eval = len(rollout_order)
        print(f"[warning] not enough init states; using n_eval={n_eval}")

    use_mp = getattr(cfg.eval, "use_mp", False)
    cfg_num_procs = getattr(cfg.eval, "num_procs", 1)
    env_num = min(cfg_num_procs, n_eval) if use_mp else 1
    eval_loop_num = (n_eval + env_num - 1) // env_num

    max_record_videos = min(args.save_videos, n_eval)
    save_video = max_record_videos > 0
    video_writer = None
    record_active = [False] * env_num if save_video else []
    record_tail = [None] * env_num if save_video else []
    video_ids = [None] * env_num if save_video else []
    tail_after_done = 30
    if save_video:
        ckpt_dir = Path(args.checkpoint).expanduser().resolve().parent
        video_dir = ckpt_dir / "rollout_videos"
        os.makedirs(video_dir, exist_ok=True)
        video_writer = VideoWriter(video_path=str(video_dir), save_video=True, single_video=False)

    env = (
        DummyVectorEnv([lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)])
        if env_num == 1
        else SubprocVectorEnv([lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)])
    )
    env.seed(cfg.seed)

    anchor_success = defaultdict(int)
    anchor_trials = defaultdict(int)
    successes = 0
    episodes_done = 0
    batch_success_counts = []
    pbar = tqdm(total=n_eval, desc="rollout", leave=True)
    for loop_idx in range(eval_loop_num):
        if episodes_done >= n_eval:
            break
        algo.reset()
        indices = [
            rollout_order[(loop_idx * env_num + k) % len(rollout_order)]
            for k in range(min(env_num, n_eval - episodes_done))
        ]
        if len(indices) < env_num:
            indices = indices + [indices[-1]] * (env_num - len(indices))
        init_states_batch = init_states[indices]

        env.reset()
        obs = env.set_init_state(init_states_batch)
        dummy = np.zeros((env_num, 7))
        for _ in range(5):
            obs, _, _, _ = env.step(dummy)

        remaining = min(env_num, n_eval - episodes_done)
        prev_successes = successes
        dones = [False] * env_num
        for k in range(remaining, env_num):
            dones[k] = True
        steps = 0

        anchor_ids = None
        if anchor_indices:
            anchor_ids = [anchor_indices[idx] if idx < len(anchor_indices) else None for idx in indices]
            for k in range(remaining):
                if anchor_ids[k] is not None:
                    anchor_trials[anchor_ids[k]] += 1

        if save_video:
            record_active = [False] * env_num
            record_tail = [None] * env_num
            video_ids = [None] * env_num
            rec_slots = max_record_videos - episodes_done
            rec = max(0, min(rec_slots, remaining))
            for i in range(rec):
                record_active[i] = True
                record_tail[i] = None
                video_ids[i] = episodes_done + i

        single_env_done = False
        while steps < max_steps:
            steps += 1
            data = raw_obs_to_tensor_obs(obs, task_emb, cfg)
            actions = algo.policy.get_action(data)
            if torch.is_tensor(actions):
                actions_np = actions.detach().cpu().numpy()
            else:
                actions_np = np.asarray(actions)
            if env_num == 1 and actions_np.ndim == 1:
                actions_np = actions_np[None]
            obs, reward, done, info = env.step(actions)
            done_array = np.asarray(done)

            if env_num == 1 and bool(done_array.item()) and not single_env_done:
                successes += 1
                if anchor_ids and anchor_ids[0] is not None:
                    anchor_success[anchor_ids[0]] += 1
                single_env_done = True
            elif env_num > 1:
                for k in range(env_num):
                    dones[k] = dones[k] or done[k]
                if all(dones) and (not save_video or not any(record_active)):
                    break

            if video_writer:
                rec = 1 if env_num == 1 else min(env_num, remaining)
                for i in range(rec):
                    if not record_active[i] or video_ids[i] is None:
                        continue
                    done_flag = bool(done_array.item()) if env_num == 1 else bool(done[i])
                    gripper_open = False
                    if actions_np.ndim == 2 and i < actions_np.shape[0]:
                        gripper_open = actions_np[i, -1] < 0
                    elif actions_np.ndim == 1:
                        gripper_open = actions_np[-1] < 0
                    video_writer.append_obs(
                        obs if env_num == 1 else obs[i],
                        done=done_flag,
                        idx=video_ids[i],
                        camera_name="agentview_image",
                    )
                    if done_flag and gripper_open and record_tail[i] is None:
                        record_tail[i] = tail_after_done
                    if record_tail[i] is not None:
                        record_tail[i] -= 1
                        if record_tail[i] < 0:
                            record_active[i] = False

            if env_num == 1 and single_env_done:
                tail_active = save_video and record_active[0]
                if not tail_active:
                    break

        if env_num > 1:
            successes += sum(int(dones[k]) for k in range(remaining))
            if anchor_ids:
                for k in range(remaining):
                    if dones[k] and anchor_ids[k] is not None:
                        anchor_success[anchor_ids[k]] += 1

        batch_success = successes - prev_successes
        batch_success_counts.append(f"{batch_success}/{remaining}")
        episodes_done += remaining
        pbar.update(remaining)
        pbar.set_postfix(sr=successes / max(episodes_done, 1))

    env.close()
    pbar.close()
    if video_writer:
        video_writer.save()

    sr = successes / n_eval
    batch_str = ", ".join(batch_success_counts)
    print("[info] rollout summary:")
    print(f"  {n_eval} rollouts in {eval_loop_num} runs; {env_num} parallel environments per run")
    print(f"  success: {successes}/{n_eval} ({sr:.3f})")
    print(f"  success per run: {batch_str}")
    if anchor_indices:
        print("  per-anchor success:")
        for a in sorted(anchor_trials):
            trials = anchor_trials[a]
            suc = anchor_success.get(a, 0)
            rate = suc / max(trials, 1)
            print(f"    anchor {a}: {suc}/{trials} ({rate:.3f})")


if __name__ == "__main__":
    main()

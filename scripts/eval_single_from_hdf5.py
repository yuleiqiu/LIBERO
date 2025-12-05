import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import h5py
import torch
import numpy as np
import yaml
from tqdm import tqdm
from easydict import EasyDict
from hydra import compose, initialize_config_dir
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import init_path  # noqa: F401
from libero.libero import get_libero_path
from libero.lifelong.algos import get_algo_class
from libero.lifelong.datasets import SequenceVLDataset, get_dataset
from libero.lifelong.metric import raw_obs_to_tensor_obs
from libero.lifelong.utils import control_seed, safe_device, torch_load_model
from libero.libero.utils.video_utils import VideoWriter
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv, DummyVectorEnv


def load_cfg(overrides):
    config_dir = to_absolute_path("libero/configs")
    with initialize_config_dir(config_dir=config_dir, job_name="single_task_eval"):
        hydra_cfg = compose(config_name="config", overrides=overrides)
    cfg = EasyDict(yaml.safe_load(OmegaConf.to_yaml(hydra_cfg)))
    return cfg


def read_language_from_hdf5(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        problem_info = json.loads(f["data"].attrs["problem_info"])
    return problem_info["language_instruction"]


def build_task_emb(cfg):
    lang_dim = cfg.policy.language_encoder.network_kwargs.get("input_size", 1)
    return torch.zeros((lang_dim,), dtype=torch.float32)


def easydict_to_plain(d):
    """
    Recursively convert an EasyDict (or nested EasyDicts) to a plain dict.
    """
    if isinstance(d, dict):
        return {k: easydict_to_plain(v) for k, v in d.items()}
    return d


def main():
    parser = argparse.ArgumentParser(description="Eval single-task checkpoint on HDF5")
    parser.add_argument("--checkpoint", required=True, help="Path to task0_model.pth")
    parser.add_argument("--demo-file", required=True, help="Path to processed *_demo.hdf5")
    parser.add_argument(
        "--split-indices",
        type=str,
        default=None,
        help="Path to split_indices.json (if omitted, will look in checkpoint dir)",
    )
    parser.add_argument(
        "--init-states",
        type=str,
        default=None,
        help="Path to .pruned_init file for rollout (required if --rollout)",
    )
    parser.add_argument(
        "--config-override",
        nargs="*",
        default=[],
        help="Optional hydra-style overrides if checkpoint lacks cfg",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="CUDA device id for eval (e.g., 0 -> cuda:0)",
    )
    parser.add_argument(
        "--rollout",
        action="store_true",
        help="If set, also run environment rollouts to estimate success rate",
    )
    parser.add_argument(
        "--loss",
        action="store_true",
        help="If set, compute eval loss on provided split indices; otherwise skip loss.",
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

    # Ensure paths
    cfg.folder = cfg.folder or str(demo_path.parent)
    cfg.bddl_folder = cfg.bddl_folder or get_libero_path("bddl_files")
    cfg.init_states_folder = cfg.init_states_folder or get_libero_path("init_states")

    # Build dataset
    base_dataset, shape_meta = get_dataset(
        dataset_path=str(demo_path),
        obs_modality=cfg.data.obs.modality,
        initialize_obs_utils=True,
        seq_len=cfg.data.seq_len,
    )

    task_emb = build_task_emb(cfg)
    dataset = SequenceVLDataset(base_dataset, task_emb)

    algo_cls = get_algo_class(cfg.lifelong.algo)
    algo = safe_device(algo_cls(n_tasks=1, cfg=cfg), cfg.device)
    algo.policy.load_state_dict(state_dict)
    algo.policy.eval()

    # Optional loss evaluation
    if args.loss:
        split_path = args.split_indices
        if split_path is None:
            ckpt_dir = Path(args.checkpoint).expanduser().resolve().parent
            candidate = ckpt_dir / "split_indices.json"
            split_path = str(candidate) if candidate.exists() else None

        if split_path and os.path.exists(split_path):
            with open(split_path, "r") as f:
                split_idx = json.load(f)
            eval_idx = split_idx.get("eval") or split_idx.get("val", [])
            split_name = "eval" if "eval" in split_idx else "val"
            if len(eval_idx) == 0:
                print(f"[warning] {split_name} split empty, skip loss eval")
            else:
                print(f"[info] using {split_name} split for loss eval ({len(eval_idx)} samples)")
                dataset_loss = Subset(dataset, eval_idx)
                loader = DataLoader(
                    dataset_loss,
                    batch_size=getattr(cfg.eval, "batch_size", 64),
                    num_workers=getattr(cfg.eval, "num_workers", 4),
                    shuffle=False,
                    persistent_workers=False,
                )
                losses = []
                with torch.no_grad():
                    for data in tqdm(loader, desc="eval loss", leave=True):
                        loss = algo.eval_observe(data)
                        losses.append(loss)
                avg_loss = sum(losses) / max(len(losses), 1)
                print(f"[info] eval avg loss: {avg_loss:.4f}")
        else:
            print("[info] no split_indices provided/found; skip loss eval to avoid using train data")

    if not args.rollout:
        return

    if args.init_states is None:
        print("[warning] --rollout specified but no --init-states provided; skip rollout")
        return

    # Optional rollout success rate
    with h5py.File(str(demo_path), "r") as f:
        bddl_file_name = f["data"].attrs.get("bddl_file_name", None)
    if bddl_file_name is None:
        print("[warning] bddl_file_name not found in hdf5; skip rollout")
        return

    env_args = {
        "bddl_file_name": bddl_file_name,
        "camera_heights": cfg.data.img_h,
        "camera_widths": cfg.data.img_w,
    }
    max_steps = getattr(cfg.eval, "max_steps", 600)

    # load init states
    init_states_path = Path(args.init_states).expanduser().resolve()
    if not init_states_path.exists():
        print(f"[warning] init states file not found: {init_states_path}, skip rollout")
        return
    init_states = torch.load(str(init_states_path))
    # ensure numpy array for mujoco state setter
    if torch.is_tensor(init_states):
        init_states = init_states.cpu().numpy()
    anchors_meta = init_states_path.with_suffix(init_states_path.suffix + ".anchors.json")
    anchor_indices = None
    if anchors_meta.exists():
        try:
            with open(anchors_meta, "r") as f:
                anchor_indices = json.load(f).get("anchor_idx", None)
        except Exception as e:
            print(f"[warning] failed to read anchors meta {anchors_meta}: {e}")

    n_eval_cfg = getattr(cfg.eval, "n_eval", 20)

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

    max_available = init_states.shape[0]
    n_eval = min(n_eval_cfg, max_available)
    rollout_order = build_rollout_order(anchor_indices, max_available, n_eval)
    if len(rollout_order) < n_eval:
        n_eval = len(rollout_order)
        print(f"[warning] not enough init states; using n_eval={n_eval}")

    use_mp = getattr(cfg.eval, "use_mp", False)
    cfg_num_procs = getattr(cfg.eval, "num_procs", 1)
    env_num = min(cfg_num_procs, n_eval) if use_mp else 1

    eval_loop_num = (n_eval + env_num - 1) // env_num

    # Video writer: record only in the first batch, per-env mp4.
    record_envs = min(args.save_videos, env_num)
    save_video = record_envs > 0
    video_writer = None
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
        # pad to env_num for convenience, extras are marked done immediately
        if len(indices) < env_num:
            indices = indices + [indices[-1]] * (env_num - len(indices))
        init_states_batch = init_states[indices]

        env.reset()
        obs = env.set_init_state(init_states_batch)
        dummy = np.zeros((env_num, 7))
        for _ in range(5):
            obs, reward, done, info = env.step(dummy)

        remaining = min(env_num, n_eval - episodes_done)
        prev_successes = successes
        dones = [False] * env_num
        for k in range(remaining, env_num):
            dones[k] = True  # ignore padded envs
        steps = 0

        while steps < max_steps:
            steps += 1
            data = raw_obs_to_tensor_obs(obs, task_emb, cfg)
            actions = algo.policy.get_action(data)
            obs, reward, done, info = env.step(actions)

            if env_num == 1 and done:
                successes += 1
            elif env_num > 1:
                for k in range(env_num):
                    dones[k] = dones[k] or done[k]
                if all(dones):
                    break

            if video_writer and loop_idx == 0:
                if env_num == 1:
                    video_writer.append_obs(obs, done=done, idx=0)
                else:
                    rec = min(record_envs, remaining)
                    obs_rec = [obs[i] for i in range(rec)]
                    dones_rec = [dones[i] for i in range(rec)]
                    video_writer.append_vector_obs(obs_rec, dones_rec, camera_name="agentview_image")

            if env_num == 1 and done:
                break

        if env_num > 1:
            successes += sum(int(dones[k]) for k in range(remaining))

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
    print(
        "[info] rollout success: "
        f"batches {batch_str} | total {successes}/{n_eval} ({sr:.3f}) across {n_eval} episodes; "
        f"envs per batch: {env_num}"
    )


if __name__ == "__main__":
    main()

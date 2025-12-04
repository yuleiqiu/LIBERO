#!/usr/bin/env python3

"""Render and stack the initial states of every demo inside a dataset hdf5 file."""

# Standard library imports
import argparse
import json
import os
from pathlib import Path

# Third-party imports
import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.ioff()

# LIBERO-specific imports
from libero.libero.envs import OffScreenRenderEnv


def _decode_if_bytes(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _get_bddl_path(h5_group):
    """Fetch the bddl file path stored alongside the dataset metadata."""
    data_attrs = h5_group.attrs
    if "bddl_file_name" in data_attrs:
        return _decode_if_bytes(data_attrs["bddl_file_name"])

    env_args_raw = data_attrs.get("env_args")
    if env_args_raw is None:
        raise ValueError("Could not find bddl_file_name or env_args in dataset attrs.")

    env_args_str = _decode_if_bytes(env_args_raw)
    env_args = env_args_str if isinstance(env_args_raw, dict) else json.loads(env_args_str)
    for key in ("bddl_file", "bddl_file_name"):
        if key in env_args:
            return env_args[key]

    raise ValueError("bddl file path missing in env_args.")


def _load_init_state(h5_file, demo_key):
    """Return the stored init state for a given demo."""
    demo_grp = h5_file[f"data/{demo_key}"]
    if "init_state" in demo_grp.attrs:
        return np.array(demo_grp.attrs["init_state"])

    if "states" not in demo_grp:
        raise ValueError(f"Demo {demo_key} does not contain 'init_state' attr or 'states' dataset.")
    return np.array(demo_grp["states"][0])


def _tensor_to_display_np(tensor):
    """Convert CxHxW tensor to HxWxC numpy array (flipped for display)."""
    np_img = tensor.permute(1, 2, 0).numpy()
    return np_img[::-1]


def _resize_image(np_img, size):
    pil_img = Image.fromarray(np_img)
    resized = pil_img.resize((size, size), resample=Image.BILINEAR)
    return np.array(resized)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize each demo's initial state from a dataset hdf5 by stacking renders."
    )
    parser.add_argument("--dataset", required=True, help="Path to the demo hdf5 dataset.")
    parser.add_argument("--render-size", type=int, default=256, help="Square camera render size.")
    parser.add_argument("--output-size", type=int, default=640, help="Side length (pixels) of saved visualizations.")
    parser.add_argument("--stack-alpha", type=float, default=0.35, help="Transparency for stacked overlay (0-1).")
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save visualizations. Defaults to <dataset_dir>/<dataset_stem>_init_states.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with h5py.File(dataset_path, "r") as f:
        data_grp = f["data"]
        demo_keys = sorted([k for k in data_grp.keys() if k.startswith("demo_")])
        if not demo_keys:
            raise ValueError(f"No demo_* groups found under data/ in {dataset_path}")

        bddl_file = _get_bddl_path(data_grp)
        print(f"Found {len(demo_keys)} demos/init states in {dataset_path}")

        output_root = (
            Path(args.output_dir)
            if args.output_dir
            else dataset_path.parent / f"{dataset_path.stem}_init_states"
        )
        output_root.mkdir(parents=True, exist_ok=True)
        individual_dir = output_root / "individual"
        individual_dir.mkdir(exist_ok=True)

        env_args = {
            "bddl_file_name": bddl_file,
            "camera_heights": args.render_size,
            "camera_widths": args.render_size,
        }
        env = OffScreenRenderEnv(**env_args)

        images = []
        for idx, demo_key in enumerate(tqdm(demo_keys, desc="Rendering init states")):
            init_state = _load_init_state(f, demo_key)
            env.reset()
            env.set_init_state(init_state)
            for _ in range(5):
                obs, _, _, _ = env.step([0.0] * 7)
            images.append(torch.from_numpy(obs["agentview_image"]).permute(2, 0, 1))

        env.close()

    display_images = [_tensor_to_display_np(img) for img in images]
    resized_images = [_resize_image(img, args.output_size) for img in display_images]

    for idx, img in enumerate(resized_images):
        fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
        ax.imshow(img)
        ax.set_title(f"Demo {idx} init state")
        ax.axis("off")
        individual_path = individual_dir / f"demo_{idx:03d}.png"
        fig.savefig(individual_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    stack_alpha = min(max(args.stack_alpha, 0.05), 1.0)
    stack_fig, stack_ax = plt.subplots(figsize=(7, 7), dpi=200)
    for img in resized_images:
        stack_ax.imshow(img, alpha=stack_alpha)
    stack_ax.set_title(
        f"{dataset_path.stem}\nStacked demo init states (alpha={stack_alpha:.2f})"
    )
    stack_ax.axis("off")
    stacked_output = output_root / "stacked_overlay.png"
    stack_fig.savefig(stacked_output, bbox_inches="tight", pad_inches=0)
    plt.close(stack_fig)

    print(f"Saved individual images to {individual_dir}")
    print(f"Saved stacked overlay to {stacked_output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render one raw LIBERO demo to a side-by-side MP4 by replaying stored actions."""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional

import h5py
import imageio.v2 as imageio
import numpy as np

import init_path
import libero.libero.utils.utils as libero_utils
from libero.libero.envs import OffScreenRenderEnv
from standalone.utils.bddl_path_utils import canonicalize_bddl_file_name, resolve_bddl_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render one raw LIBERO demo from data/demo_x/actions to mp4."
    )
    parser.add_argument(
        "--input-demo-file",
        type=str,
        required=True,
        help="Raw HDF5 file containing data/demo_x/states and actions.",
    )
    parser.add_argument(
        "--demo-id",
        type=str,
        default=None,
        help="Demo to render, e.g. 1 or demo_1. Defaults to the first demo.",
    )
    parser.add_argument(
        "--bddl-file",
        type=str,
        default=None,
        help="Optional BDDL override. If unset, use data.attrs['bddl_file_name'].",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output mp4 path.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Output video fps.",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=128,
        help="Rendered camera height.",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=128,
        help="Rendered camera width.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap on replay steps.",
    )
    return parser.parse_args()


def ensure_file(path_str: str, desc: str) -> str:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{desc} not found: {path}")
    return str(path)


def sort_demo_keys(demo_keys: List[str]) -> List[str]:
    def sort_key(key: str):
        match = re.match(r"demo_(\d+)$", key)
        if match:
            return (0, int(match.group(1)))
        return (1, key)

    return sorted(demo_keys, key=sort_key)


def normalize_demo_key(data_group: h5py.Group, demo_id: Optional[str]) -> str:
    demo_keys = sort_demo_keys([key for key in data_group.keys() if key.startswith("demo_")])
    if not demo_keys:
        raise ValueError("No demo_* groups found under top-level 'data' group.")
    if demo_id is None:
        return demo_keys[0]

    demo_key = str(demo_id)
    if not demo_key.startswith("demo_"):
        demo_key = f"demo_{demo_key}"
    if demo_key not in data_group:
        raise ValueError(f"Requested demo {demo_key} not found. Available demos: {demo_keys[:10]}")
    return demo_key


def flip_image(img: np.ndarray) -> np.ndarray:
    return img[::-1]


def compose_frame(agentview_img: np.ndarray, eye_in_hand_img: np.ndarray) -> np.ndarray:
    return np.hstack((flip_image(agentview_img), flip_image(eye_in_hand_img)))


def resolve_bddl_file(
    data_group: h5py.Group, override_path: Optional[str], input_demo_file: str
) -> str:
    bddl_file = override_path or data_group.attrs.get("bddl_file_name", None)
    if bddl_file is None:
        raise ValueError("No bddl_file_name found; please pass --bddl-file explicitly")

    demo_path = Path(input_demo_file).expanduser().resolve()
    resolved_bddl = resolve_bddl_path(
        canonicalize_bddl_file_name(str(bddl_file)),
        demo_path,
    )
    if resolved_bddl is None:
        raise FileNotFoundError(f"BDDL file not found: {bddl_file}")
    return resolved_bddl


def maybe_load_model_xml(demo_group: h5py.Group) -> Optional[str]:
    model_xml = demo_group.attrs.get("model_file", None)
    if model_xml is None:
        return None
    return libero_utils.postprocess_model_xml(model_xml, {})


def main():
    args = parse_args()
    try:
        input_demo_file = ensure_file(args.input_demo_file, "Input demo file")
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_demo_file, "r") as h5_file:
        if "data" not in h5_file:
            raise KeyError(f"Input HDF5 missing top-level 'data' group: {input_demo_file}")

        data_group = h5_file["data"]
        bddl_file = resolve_bddl_file(data_group, args.bddl_file, input_demo_file)
        demo_key = normalize_demo_key(data_group, args.demo_id)
        demo_group = data_group[demo_key]

        if "states" not in demo_group:
            raise KeyError(f"{demo_key} missing dataset 'states'")
        if "actions" not in demo_group:
            raise KeyError(f"{demo_key} missing dataset 'actions'")

        states = np.asarray(demo_group["states"])
        actions = np.asarray(demo_group["actions"])
        model_xml = maybe_load_model_xml(demo_group)

    if len(states) == 0:
        raise ValueError(f"{demo_key} contains no states")

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=args.camera_height,
        camera_widths=args.camera_width,
    )

    done = False
    info = {}
    try:
        env.reset()
        if model_xml is not None:
            env.reset_from_xml_string(model_xml)
            env.sim.reset()

        obs = env.set_init_state(states[0])

        writer = imageio.get_writer(str(output_path), fps=int(args.fps))
        try:
            writer.append_data(
                compose_frame(obs["agentview_image"], obs["robot0_eye_in_hand_image"])
            )

            steps_to_run = len(actions)
            if args.max_steps is not None:
                steps_to_run = min(steps_to_run, int(args.max_steps))

            for action in actions[:steps_to_run]:
                obs, _, done, info = env.step(action)
                writer.append_data(
                    compose_frame(obs["agentview_image"], obs["robot0_eye_in_hand_image"])
                )
                if done:
                    break
        finally:
            writer.close()
    finally:
        env.close()

    print(f"[INFO] demo: {demo_key}")
    print(f"[INFO] actions: {len(actions)}")
    print(f"[INFO] output: {output_path}")
    print(f"[INFO] final success: {bool(info.get('success', done))}")


if __name__ == "__main__":
    main()

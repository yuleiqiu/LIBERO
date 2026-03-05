#!/usr/bin/env python3

import os
from typing import Iterable, List, Tuple

import h5py
import imageio.v2 as imageio
import numpy as np

from libero.libero import benchmark, get_libero_path


def sort_demo_keys(demo_keys: Iterable[str]) -> List[str]:
    return sorted(demo_keys, key=lambda key: int(key.split("_")[1]))


def list_demo_keys(data_group) -> List[str]:
    demo_keys = [key for key in data_group.keys() if key.startswith("demo_")]
    if not demo_keys:
        raise ValueError("No demo_* groups found under the HDF5 'data' group.")
    return sort_demo_keys(demo_keys)


def normalize_demo_key(data_group, demo_id: str = None) -> str:
    demo_keys = list_demo_keys(data_group)
    if demo_id is None:
        return demo_keys[0]

    target_key = demo_id if str(demo_id).startswith("demo_") else f"demo_{demo_id}"
    if target_key not in data_group:
        raise ValueError(f"Requested demo '{target_key}' not found. Available: {demo_keys}")
    return target_key


def resolve_hdf5_path(path_str: str) -> str:
    input_path = os.path.expanduser(path_str)
    if os.path.isdir(input_path):
        hdf5_files = sorted(
            file_name for file_name in os.listdir(input_path) if file_name.endswith(".hdf5")
        )
        if not hdf5_files:
            raise FileNotFoundError(f"No .hdf5 files found under {input_path}")
        if len(hdf5_files) > 1:
            raise ValueError(
                f"Multiple .hdf5 files found under {input_path}: {hdf5_files}. "
                "Please specify one explicitly."
            )
        return os.path.join(input_path, hdf5_files[0])
    return input_path


def build_task_context(benchmark_name: str, task_id: int) -> Tuple[object, object, str]:
    bench_instance = benchmark.get_benchmark(benchmark_name)()
    task = bench_instance.get_task(task_id)
    datasets_default_path = get_libero_path("datasets")
    demo_file = os.path.join(
        datasets_default_path,
        bench_instance.get_task_demonstration(task_id),
    )
    return bench_instance, task, demo_file


def fetch_demo_frames(h5_file: h5py.File, demo_key: str) -> Tuple[np.ndarray, np.ndarray]:
    agentview_images = h5_file[f"data/{demo_key}/obs/agentview_rgb"][()]
    eye_in_hand_images = h5_file[f"data/{demo_key}/obs/eye_in_hand_rgb"][()]
    return agentview_images, eye_in_hand_images


def flip_image(img: np.ndarray) -> np.ndarray:
    return img[::-1]


def compose_side_by_side_frame(agentview_img: np.ndarray, eye_in_hand_img: np.ndarray) -> np.ndarray:
    return np.hstack((flip_image(agentview_img), flip_image(eye_in_hand_img)))


def write_video(output_path: str, frames: Iterable[np.ndarray], fps: int = 60) -> None:
    writer = imageio.get_writer(output_path, fps=fps)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def render_demo_video(h5_file: h5py.File, demo_key: str, output_path: str, fps: int = 60) -> None:
    agentview_images, eye_in_hand_images = fetch_demo_frames(h5_file, demo_key)
    frames = (
        compose_side_by_side_frame(agentview_img, eye_in_hand_img)
        for agentview_img, eye_in_hand_img in zip(agentview_images, eye_in_hand_images)
    )
    write_video(output_path, frames, fps=fps)


def pad_image(img: np.ndarray, target_height: int = None, target_width: int = None) -> np.ndarray:
    height, width = img.shape[:2]
    target_height = target_height or height
    target_width = target_width or width
    pad_bottom = max(target_height - height, 0)
    pad_right = max(target_width - width, 0)
    if pad_bottom == 0 and pad_right == 0:
        return img
    return np.pad(
        img,
        pad_width=((0, pad_bottom), (0, pad_right), (0, 0)),
        mode="constant",
        constant_values=0,
    )


def hstack_with_padding(left_img: np.ndarray, right_img: np.ndarray) -> np.ndarray:
    target_height = max(left_img.shape[0], right_img.shape[0])
    left = pad_image(left_img, target_height=target_height)
    right = pad_image(right_img, target_height=target_height)
    return np.hstack((left, right))


def vstack_with_padding(top_img: np.ndarray, bottom_img: np.ndarray) -> np.ndarray:
    target_width = max(top_img.shape[1], bottom_img.shape[1])
    top = pad_image(top_img, target_width=target_width)
    bottom = pad_image(bottom_img, target_width=target_width)
    return np.vstack((top, bottom))

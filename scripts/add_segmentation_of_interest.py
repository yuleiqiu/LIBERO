import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np

import init_path
from libero.libero.envs import SegmentationRenderEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add segmentation_of_interest masks to a processed LIBERO dataset "
            "by replaying states and rendering instance segmentations."
        )
    )
    parser.add_argument("--input", required=True, help="Path to processed hdf5.")
    parser.add_argument("--output", required=True, help="Output hdf5 path.")
    parser.add_argument(
        "--cameras",
        default="",
        help=(
            "Comma-separated dataset camera prefixes to process "
            "(e.g. agentview,eye_in_hand). If empty, infer from obs keys."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it exists.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Overwrite segmentation datasets if already present.",
    )
    return parser.parse_args()


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _dataset_prefix_to_camera(prefix: str) -> str:
    if prefix == "eye_in_hand":
        return "robot0_eye_in_hand"
    return prefix


def _infer_hw(dset: h5py.Dataset) -> Tuple[int, int]:
    shape = dset.shape
    if len(shape) == 3:
        _, h, w = shape
        return int(h), int(w)
    if len(shape) != 4:
        raise ValueError(f"unexpected rgb shape: {shape}")
    # (T, H, W, C)
    if shape[-1] in (1, 3, 4):
        return int(shape[-3]), int(shape[-2])
    # (T, C, H, W)
    if shape[1] in (1, 3, 4):
        return int(shape[2]), int(shape[3])
    raise ValueError(f"cannot infer H/W from shape: {shape}")


def _refresh_segmentation_mapping(env: SegmentationRenderEnv) -> None:
    env.segmentation_id_mapping = {}
    env.instance_to_id = {}
    env.segmentation_robot_id = None

    instances = list(env.env.model.instances_to_ids.keys())
    for i, name in enumerate(instances):
        if name in ("Panda0", "OnTheGroundPanda0"):
            env.segmentation_robot_id = i
            break

    for i, name in enumerate(instances):
        if name not in ["Panda0", "RethinkMount0", "PandaGripper0"]:
            env.segmentation_id_mapping[i] = name

    env.instance_to_id = {
        v: k + 1 for k, v in env.segmentation_id_mapping.items()
    }


def _load_env_kwargs(data_grp: h5py.Group) -> Dict:
    env_kwargs: Dict = {}
    if "env_args" in data_grp.attrs:
        try:
            env_args = json.loads(data_grp.attrs["env_args"])
            env_kwargs = env_args.get("env_kwargs", {}) or {}
        except Exception:
            env_kwargs = {}
    elif "env_info" in data_grp.attrs:
        try:
            env_kwargs = json.loads(data_grp.attrs["env_info"]) or {}
        except Exception:
            env_kwargs = {}
    return dict(env_kwargs)


def _infer_camera_specs(
    obs_grp: h5py.Group, camera_prefixes: Sequence[str]
) -> List[Tuple[str, str, int, int]]:
    specs: List[Tuple[str, str, int, int]] = []
    for prefix in camera_prefixes:
        rgb_key = f"{prefix}_rgb"
        if rgb_key not in obs_grp:
            continue
        h, w = _infer_hw(obs_grp[rgb_key])
        cam_name = _dataset_prefix_to_camera(prefix)
        specs.append((prefix, cam_name, h, w))
    return specs


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"input file not found: {input_path}")
    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists: {output_path}")
        output_path.unlink()

    if output_path == input_path:
        raise ValueError("output path must be different from input path")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(input_path, output_path)

    with h5py.File(output_path, "r+") as f:
        if "data" not in f:
            raise KeyError("missing /data group in dataset")
        data_grp = f["data"]

        demos = sorted([k for k in data_grp.keys() if k.startswith("demo_")])
        if not demos:
            raise ValueError("no demo_* groups found")

        first_obs = data_grp[f"{demos[0]}/obs"]
        if args.cameras:
            prefixes = _split_csv(args.cameras)
        else:
            prefixes = [k[: -len("_rgb")] for k in first_obs.keys() if k.endswith("_rgb")]

        if not prefixes:
            raise ValueError("no rgb obs keys found to infer cameras")

        specs = _infer_camera_specs(first_obs, prefixes)
        if not specs:
            raise ValueError("failed to infer camera specs from obs")

        camera_names = [cam for _, cam, _, _ in specs]
        camera_heights = [h for _, _, h, _ in specs]
        camera_widths = [w for _, _, _, w in specs]

        bddl_file = data_grp.attrs.get("bddl_file_name", None)
        if not bddl_file:
            raise ValueError("bddl_file_name missing in data attrs")

        env_kwargs = _load_env_kwargs(data_grp)
        env_kwargs.update(
            {
                "bddl_file_name": bddl_file,
                "use_camera_obs": True,
                "has_renderer": False,
                "has_offscreen_renderer": True,
                "camera_names": camera_names,
                "camera_heights": camera_heights,
                "camera_widths": camera_widths,
                "camera_segmentations": "instance",
            }
        )

        env = SegmentationRenderEnv(**env_kwargs)
        try:
            for demo_key in demos:
                demo_grp = data_grp[demo_key]
                obs_grp = demo_grp.get("obs", None)
                if obs_grp is None:
                    print(f"[warning] {demo_key} missing obs group; skipping")
                    continue

                states = demo_grp["states"][()]
                model_xml = demo_grp.attrs.get("model_file", None)
                if model_xml is None:
                    print(f"[warning] {demo_key} missing model_file; skipping")
                    continue

                reset_success = False
                while not reset_success:
                    try:
                        env.reset()
                        reset_success = True
                    except Exception:
                        continue

                env.reset_from_xml_string(model_xml)
                env.sim.reset()
                _refresh_segmentation_mapping(env)

                masks_by_prefix: Dict[str, List[np.ndarray]] = {
                    prefix: [] for prefix, _, _, _ in specs
                }

                for state in states:
                    obs = env.regenerate_obs_from_state(state)
                    for prefix, cam_name, _, _ in specs:
                        seg_key = f"{cam_name}_segmentation_instance"
                        if seg_key not in obs:
                            raise KeyError(f"missing segmentation key: {seg_key}")
                        seg_img = np.squeeze(obs[seg_key])
                        seg_mask = env.get_segmentation_of_interest(seg_img)
                        seg_mask = np.squeeze(seg_mask).astype(np.int8)
                        masks_by_prefix[prefix].append(seg_mask)

                for prefix, masks in masks_by_prefix.items():
                    ds_name = f"{prefix}_segmentation_of_interest"
                    if ds_name in obs_grp:
                        if not args.overwrite_existing:
                            print(
                                f"[info] {demo_key}/{ds_name} exists; skipping "
                                "(use --overwrite-existing to replace)"
                            )
                            continue
                        del obs_grp[ds_name]
                    obs_grp.create_dataset(ds_name, data=np.stack(masks, axis=0))

                print(f"[info] added segmentation_of_interest for {demo_key}")
        finally:
            env.close()

        data_grp.attrs["segmentation_of_interest"] = True
        data_grp.attrs["segmentation_of_interest_source"] = "instance"

    print(f"[done] wrote dataset to {output_path}")


if __name__ == "__main__":
    main()

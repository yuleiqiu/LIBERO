#!/usr/bin/env python3
"""Inspect how train.py resolves demo, BDDL, and rollout init-state paths."""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import h5py

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from standalone.configs.train import TrainConfig
from standalone.utils.train_utils import resolve_init_states_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Check whether moving data.demo_file changes the rollout-related paths "
            "used by standalone/train.py."
        )
    )
    parser.add_argument(
        "--demo-file",
        required=True,
        help="Path to the HDF5 demo file currently used for training.",
    )
    parser.add_argument(
        "--as-if-demo-file",
        default="",
        help=(
            "Optional simulated new HDF5 path. The script will reuse the current "
            "HDF5 contents but resolve rollout paths as if the file lived here."
        ),
    )
    parser.add_argument(
        "--init-states-dir",
        default=None,
        help=(
            "Optional override for rollout.init_states_dir. Defaults to the same "
            "default as TrainConfig."
        ),
    )
    return parser.parse_args()


def read_bddl_from_hdf5(hdf5_path: Path) -> Optional[str]:
    with h5py.File(str(hdf5_path), "r") as f:
        data_group = f.get("data")
        if data_group is None:
            raise KeyError(f"HDF5 missing 'data' group: {hdf5_path}")
        return data_group.attrs.get("bddl_file_name", None)


def resolve_bddl_path_with_source(
    bddl_file_name: Optional[str], demo_path: Path
) -> Tuple[Optional[Path], str]:
    if not bddl_file_name:
        return None, "missing_attr"

    candidate = Path(bddl_file_name).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return candidate.resolve(), "absolute"

    if candidate.exists():
        return candidate.resolve(), "cwd_relative"

    repo_root = Path(__file__).resolve().parents[2]
    repo_candidate = (repo_root / "libero/libero/bddl_files" / candidate).resolve()
    if repo_candidate.exists():
        return repo_candidate, "repo_bddl_files"

    demo_candidate = (demo_path.parent / candidate).resolve()
    if demo_candidate.exists():
        return demo_candidate, "demo_relative"

    return None, "not_found"


def build_init_states_path(init_states_dir: Path, bddl_path: Optional[Path]) -> Optional[Path]:
    if bddl_path is None:
        return None
    return init_states_dir / bddl_path.parent.name / f"{bddl_path.stem}.pruned_init"


def print_report(label: str, demo_path: Path, init_states_dir: Path, bddl_name: Optional[str]) -> None:
    bddl_path, bddl_source = resolve_bddl_path_with_source(bddl_name, demo_path)
    init_states_path = build_init_states_path(init_states_dir, bddl_path)
    anchors_meta = None
    if init_states_path is not None:
        anchors_meta = init_states_path.with_suffix(init_states_path.suffix + ".anchors.json")

    print(f"[{label}]")
    print(f"demo_path={demo_path}")
    print(f"demo_exists={demo_path.exists()}")
    print(f"bddl_file_name={bddl_name}")
    print(f"bddl_resolution={bddl_source}")
    print(f"bddl_path={bddl_path if bddl_path is not None else 'None'}")
    print(f"init_states_dir={init_states_dir}")
    print(f"init_states_path={init_states_path if init_states_path is not None else 'None'}")
    print(
        "init_states_exists="
        f"{init_states_path.exists() if init_states_path is not None else False}"
    )
    print(f"anchors_meta={anchors_meta if anchors_meta is not None else 'None'}")
    print(
        "anchors_meta_exists="
        f"{anchors_meta.exists() if anchors_meta is not None else False}"
    )


def main():
    args = parse_args()
    demo_path = Path(args.demo_file).expanduser().resolve()
    if not demo_path.exists():
        raise FileNotFoundError(f"demo file not found: {demo_path}")

    cfg = TrainConfig()
    if args.init_states_dir is not None:
        cfg.rollout.init_states_dir = args.init_states_dir
    init_states_dir = resolve_init_states_dir(cfg)
    bddl_name = read_bddl_from_hdf5(demo_path)

    print_report("current", demo_path, init_states_dir, bddl_name)

    if args.as_if_demo_file:
        simulated_demo_path = Path(args.as_if_demo_file).expanduser().resolve()
        print()
        print_report("as_if_moved", simulated_demo_path, init_states_dir, bddl_name)


if __name__ == "__main__":
    main()

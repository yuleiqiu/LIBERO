#!/usr/bin/env python3
import argparse
import h5py
import numpy as np
import os
from typing import List, Tuple

from libero.libero import benchmark, get_libero_path

def preview_slice(dset: h5py.Dataset, max_elems: int = 8):
    """Return a tiny preview: first element or a small 1D slice when possible."""
    try:
        if dset.ndim == 0:
            arr = dset[()]
            return arr
        elif dset.ndim == 1:
            arr = dset[:min(len(dset), max_elems)]
            return arr
        else:
            # Take the first index on all but last dim; and a small slice on last
            idx = tuple(0 for _ in range(dset.ndim - 1))
            last_len = dset.shape[-1]
            arr = dset[idx + (slice(0, min(last_len, max_elems)),)]
            return arr
    except Exception as e:
        return f"[preview failed: {e}]"

def walk(group: h5py.Group, prefix: str = ""):
    # # Sort keys for stable output
    # for name, obj in sorted(group.items(), key=lambda kv: kv[0]):
    #     path = f"{prefix}{name}"
    #     if isinstance(obj, h5py.Group):
    #         print(f"{path}/ (Group)")
    #         walk(obj, prefix=f"{path}/")
    #     elif isinstance(obj, h5py.Dataset):
    #         try:
    #             print(f"{path}: shape={obj.shape}, dtype={obj.dtype}")
    #         except Exception as e:
    #             print(f"{path}: [could not read shape/dtype: {e}]")
    #         prev = preview_slice(obj)
    #         print(f"  demo-preview: {repr(prev)}")
    #     else:
    #         print(f"{path}: (Unknown HDF5 object type: {type(obj)})")
    for name, obj in sorted(group.items()):
        path = f"{prefix}{name}"
        if isinstance(obj, h5py.Group):
            print(f"{path}/ (Group)")
            walk(obj, prefix=f"{path}/")
        elif isinstance(obj, h5py.Dataset):
            print(f"{path}: shape={obj.shape}, dtype={obj.dtype}")



def list_demo_groups(root: h5py.Group) -> List[Tuple[str, h5py.Group]]:
    return [(name, obj) for name, obj in root.items() if isinstance(obj, h5py.Group)]

def main():
    p = argparse.ArgumentParser(
        description="Print the contents (shapes/dtypes + tiny preview) of a single 'demo' group."
    )
    p.add_argument('--task-id', type=int, default=0, help='Task ID to visualize')
    p.add_argument('--benchmark', type=str, default="libero_object", help='Benchmark name')
    # Root inside the HDF5 file (default '/')
    p.add_argument('--root', type=str, default='/data/', help="Root group path inside HDF5 file (default '/')")
    # Mutually exclusive options to select a demo either by name or by index
    gsel = p.add_mutually_exclusive_group()
    gsel.add_argument('--demo', type=str, help='Name of the demo group to print')
    gsel.add_argument('--demo-index', type=int, help='Index of demo group (0-based)')
    args = p.parse_args()

    datasets_default_path = get_libero_path("datasets")
    benchmark_instance = benchmark.get_benchmark_dict()[args.benchmark]()
    demo_file = os.path.join(datasets_default_path, benchmark_instance.get_task_demonstration(args.task_id))

    if not os.path.isfile(demo_file):
        raise SystemExit(f"Demonstration file not found: {demo_file}")

    with h5py.File(demo_file, "r") as f:
        print(f"Opened demo file: {demo_file}")

        # Normalize and resolve the root group. Accept forms like '/data', 'data/', 'data'.
        root_path_raw = args.root if args.root else '/'
        if root_path_raw in ('', '/'):  # file root
            root = f
            root_path = '/'
        else:
            # Strip leading slash for h5py indexing; internal groups don't start with '/'
            key = root_path_raw[1:] if root_path_raw.startswith('/') else root_path_raw
            key = key[:-1] if key.endswith('/') else key
            if key not in f:
                avail = ', '.join(sorted(list(f.keys())))
                raise SystemExit(f"--root {root_path_raw!r} not found. Top-level groups: {avail}")
            root = f[key]
            root_path = '/' + key
        if not isinstance(root, h5py.Group):
            raise SystemExit(f"--root {root_path_raw!r} exists but is not a Group")

        # Collect demo groups directly under the root (e.g., /data/demo_0, /data/demo_1, ...)
        demos = list_demo_groups(root)
        if not demos:
            raise SystemExit(f"No demo groups found under root {root_path!r} in file {demo_file}")

        # Select demo by name or index
        if args.demo is not None:
            matches = [obj for name, obj in demos if name == args.demo]
            if not matches:
                available = ", ".join(name for name, _ in demos)
                raise SystemExit(f"Demo {args.demo!r} not found under {root_path}. Available: {available}")
            demo_name, demo_group = args.demo, matches[0]
        else:
            idx = args.demo_index if args.demo_index is not None else 0
            if not (0 <= idx < len(demos)):
                raise SystemExit(f"--demo-index out of range [0, {len(demos)-1}]")
            demo_name, demo_group = demos[idx]

        print(f"== Demo selected: {demo_name} (under {root_path}) ==")
        # Walk only the selected demo group
        prefix_base = root_path if root_path != '/' else ''
        walk(demo_group, prefix=f"{prefix_base}/{demo_name}/")

if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import h5py


def parse_args():
    parser = argparse.ArgumentParser(description="Print obs keys and shapes from HDF5.")
    parser.add_argument("--demo-file", required=True, help="Path to *_demo.hdf5")
    parser.add_argument(
        "--demo-key",
        default=None,
        help="Optional demo key under data/ to inspect (default: first demo).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    demo_path = Path(args.demo_file).expanduser().resolve()
    if not demo_path.exists():
        raise FileNotFoundError(f"HDF5 not found: {demo_path}")

    with h5py.File(demo_path, "r") as f:
        data_group = f.get("data")
        if data_group is None:
            raise KeyError("HDF5 missing 'data' group.")
        demo_keys = sorted(data_group.keys())
        if not demo_keys:
            raise KeyError("No demos found under data/.")
        demo_key = args.demo_key or demo_keys[0]
        if demo_key not in data_group:
            raise KeyError(f"Demo key not found: {demo_key}")
        obs_group = data_group[demo_key].get("obs")
        if obs_group is None:
            raise KeyError(f"Demo {demo_key} missing obs group.")

        print(f"demo: {demo_key}")
        for key in obs_group.keys():
            dataset = obs_group[key]
            print(f"{key:25s} shape={dataset.shape} dtype={dataset.dtype}")


if __name__ == "__main__":
    main()

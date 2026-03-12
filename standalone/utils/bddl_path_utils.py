from pathlib import Path
from typing import Optional

import h5py


REPO_ROOT = Path(__file__).resolve().parents[2]
BDDL_ROOT = REPO_ROOT / "libero/libero/bddl_files"


def canonicalize_bddl_file_name(bddl_file_name: Optional[str]) -> Optional[str]:
    """Store BDDL refs relative to libero/libero/bddl_files when possible."""
    if bddl_file_name is None:
        return None

    raw = str(bddl_file_name).strip()
    if not raw:
        return raw

    normalized = raw.replace("\\", "/")
    marker = "bddl_files/"
    if marker in normalized:
        return normalized.split(marker, 1)[1]

    path = Path(raw).expanduser()
    try:
        resolved = path.resolve() if path.exists() or path.is_absolute() else None
    except Exception:
        resolved = None
    if resolved is not None:
        try:
            return str(resolved.relative_to(BDDL_ROOT)).replace("\\", "/")
        except ValueError:
            return str(resolved)
    return normalized


def read_bddl_from_hdf5(hdf5_path: str) -> Optional[str]:
    """Read the BDDL file name from an HDF5 dataset."""
    with h5py.File(hdf5_path, "r") as f:
        data = f["data"]
        return data.attrs.get("bddl_file_name", None)


def resolve_bddl_path(bddl_file_name: Optional[str], demo_path: Optional[Path]) -> Optional[str]:
    """Resolve a BDDL path from absolute, repo-relative, or demo-relative locations."""
    if not bddl_file_name:
        return None

    candidate = Path(str(bddl_file_name)).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return str(candidate.resolve())
    if candidate.exists():
        return str(candidate.resolve())

    canonical = canonicalize_bddl_file_name(str(bddl_file_name))
    if canonical:
        repo_candidate = (BDDL_ROOT / canonical).resolve()
        if repo_candidate.exists():
            return str(repo_candidate)

    if demo_path is not None:
        demo_candidate = (demo_path.parent / candidate).resolve()
        if demo_candidate.exists():
            return str(demo_candidate)
    return None

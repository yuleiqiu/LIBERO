# Standalone Structure

This folder hosts the training/rollout pipeline plus reusable model components.
The goal is to keep algorithm internals separate from policy wrappers so we can
swap or migrate algorithms with minimal friction.

## Core Layout

- `configs/`  
  Dataclass-based configs for data, policies, encoders, training, and rollout.

- `dataset_utils/`  
  HDF5 dataset, image transforms, and image normalization.
  Dataset-level ops should stay here (e.g., image augmentation).

- `models/encoders/`  
  Shared image / low-dim / obs encoders used across policies.

- `models/policy/`  
  Policy wrappers that inherit `ChunkPolicy` and adapt algorithms to the
  training/rollout interface (`forward`, `compute_loss`, `get_action`).

- `models/modules/`  
  Shared building blocks (legacy; will shrink as encoders/algos mature).

- `act_standalone/`, `dp_standalone/`  
  Imported or legacy algorithm code. These should gradually move under
  `models/algos/<name>/` and be wrapped by `models/policy/<name>_policy.py`.

- `train.py`, `rollout.py`, `rollout_env.py`  
  Training and evaluation entrypoints.

## Intended Direction

We aim to keep algorithm internals and policy wrappers separate:

- **Algorithm core**: `models/algos/<name>/...`
- **Policy wrapper**: `models/policy/<name>_policy.py`
- **Encoders**: `models/encoders/...` (shared across algorithms)

Two structural conventions:

- Inside `models/algos/<name>/`, prefer `core/` (algorithm entry), `model/` (networks),
  and `common/` (utils) to keep layouts consistent across algorithms.
- Reserve the term "policy" for framework wrappers only (`models/policy/`), not
  for algorithm internals.

Explicit rule:

- Put the algorithm entrypoint in `models/algos/<name>/core/` and wrap it with a
  `ChunkPolicy`/`BasePolicy` implementation under `models/policy/` to connect
  training, datasets, and rollout.

## Adding a New Algorithm

1) Put algorithm core under `models/algos/<name>/` (or temporary `*_standalone/`).  
2) Create a `ChunkPolicy` wrapper in `models/policy/`.  
3) Add/extend configs in `configs/` and update `policy_factory`.  
4) Keep image augmentation in `dataset_utils/`; keep algorithm normalization in
   the policy or algorithm core (not both).

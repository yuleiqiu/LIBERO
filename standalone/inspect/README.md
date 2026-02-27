# Inspect Tools

Small, standalone diagnostics for trained policies and datasets.

## compare_act_actions.py
Compare first-step actions across two init states. Uses env rendering and can save a
side-by-side video (two rows, two cameras per row).

```bash
python standalone/inspect/compare_act_actions.py \
  --ckpt standalone/standalone_runs/run_act_quickcheck/model_last.pt \
  --demo-file libero/datasets/processed/<task>_demo.hdf5 \
  --config standalone/standalone_runs/run_act_quickcheck/train_config.json \
  --init-idxs 0,5 --warmup-steps 0
```

Video mode:
```bash
python standalone/inspect/compare_act_actions.py \
  --ckpt standalone/standalone_runs/run_act_quickcheck/model_last.pt \
  --demo-file libero/datasets/processed/<task>_demo.hdf5 \
  --config standalone/standalone_runs/run_act_quickcheck/train_config.json \
  --init-idxs 0,1 --video-out tmp/compare.mp4 --video-steps 60
```

## print_demo_actions.py
Print raw action values (or just the gripper) from an HDF5 demo.

```bash
python standalone/inspect/print_demo_actions.py \
  libero/datasets/processed/<task>_demo.hdf5 --demo-ids 0 --gripper-only
```

## check_action_fit.py
Evaluate action prediction fit on a processed demo dataset without rollout.
Reports model error vs a mean-action baseline (overall + per-dim).

```bash
python standalone/inspect/check_action_fit.py \
  --ckpt standalone/standalone_runs/run_act_quickcheck/model_last.pt \
  --demo-file libero/datasets/processed/<task>_demo.hdf5 \
  --config standalone/standalone_runs/run_act_quickcheck/train_config.json
```

## inspect_dp_normalizer.py
Inspect DP normalizer parameters and verify empirical stats before/after normalization.
Supports loading normalizer from checkpoint or recomputing from config + dataset.

```bash
python standalone/inspect/inspect_dp_normalizer.py \
  --ckpt standalone/standalone_runs/train_dp_quickcheck/run_000/model_last.pt \
  --config standalone/standalone_runs/train_dp_quickcheck/run_000/train_config.json \
  --focus-keys gripper_states,action
```

Quick check on a subset:
```bash
python standalone/inspect/inspect_dp_normalizer.py \
  --ckpt standalone/standalone_runs/train_dp_quickcheck/run_000/model_last.pt \
  --max-samples 2000 \
  --focus-keys gripper_states,action
```

Recompute train split with explicit split params (same episode-level logic as training):
```bash
python standalone/inspect/inspect_dp_normalizer.py \
  --config standalone/standalone_runs/train_dp_quickcheck/run_000/train_config.json \
  --ckpt standalone/standalone_runs/train_dp_quickcheck/run_000/model_last.pt \
  --split-source recompute \
  --val-ratio 0.02 \
  --split-seed 10000 \
  --focus-keys gripper_states,action
```

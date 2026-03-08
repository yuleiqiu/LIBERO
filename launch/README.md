# Launch Layout

This repository separates reusable repo tooling from experiment launchers.

## Directory Roles

- `scripts/`: repository tools
- `launch/`: experiment launchers

Use `scripts/` for code that is part of the repo's functionality, such as:

- dataset creation
- dataset post-processing
- visualization
- debugging utilities
- migration or inspection tools

Use `launch/` for commands that orchestrate an experiment in a specific runtime
environment, such as:

- local training entrypoints
- local evaluation entrypoints
- Slurm submission wrappers
- sweep launchers

These files are not core algorithm implementations. They are operational entrypoints.

## Subdirectories

- `launch/local/`: launchers intended for direct execution on a workstation or server
- `launch/slurm/`: launchers intended for cluster submission

If more environments are needed later, add explicit folders instead of mixing styles in
one directory.

## Naming Rules

Prefer names that describe the action and the experiment type:

- `train_*.sh`
- `eval_*.sh`
- `sweep_*.sh`

Examples:

- `train_dp_mask_ablation.sh`
- `eval_dp_mask_ablation.sh`
- `sweep_dp_ta.sh`

Avoid vague names such as:

- `run.sh`
- `train_script.sh`
- `experiment.sh`

## Content Rules

Launchers may contain environment-specific settings, for example:

- GPU selection
- save directory layout
- wandb naming
- rollout frequency
- machine-specific paths

If logic becomes reusable across many launchers, move that logic back into Python code
or into `scripts/`, and keep the launcher thin.

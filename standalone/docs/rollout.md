# Rollout Design

This note explains the current rollout design in `standalone/rollout_env.py` and
`standalone/rollout_seg_env.py`, why it was refactored, and what state now lives
in rollout runtime instead of policy wrappers.

## Goals

The rollout refactor was driven by three requirements:

1. Single-env and vector-env rollout should follow the same execution logic.
2. ACT temporal ensembling should remain correct under multi-env rollout.
3. Policy wrappers should stay focused on model inference / loss, while rollout
   owns per-env runtime state.

The resulting design treats `env_num=1` as a special case of batched rollout,
instead of keeping a separate single-env online path.

## Main Idea

The policy is now used primarily as a chunk predictor:

- `model.forward(obs_batch)` produces action chunks.
- rollout owns the online execution state needed to consume those chunks.

In practice, rollout now owns:

- observation history (`ObsHistory`)
- per-env action queues
- per-env ACT temporal ensemblers
- done flags / step counters
- per-env video bookkeeping

This avoids the old split where:

- single-env rollout used `model.get_action(...)`
- vector rollout used external action queues

That split worked for basic chunk execution, but it became incorrect once ACT
temporal ensembling had to run independently per environment.

## Architecture

```text
                     +----------------------+
                     |   rollout_env.py     |
                     | rollout_seg_env.py   |
                     +----------+-----------+
                                |
                                v
                   +---------------------------+
                   | shared rollout runtime    |
                   | (rollout_utils.py)        |
                   |                           |
                   | - build_histories         |
                   | - seed_rollout_env        |
                   | - reset_rollout_runtime   |
                   | - set_init_state_batch    |
                   | - step_env_batch          |
                   | - pending_env_indices     |
                   | - refill_action_queues    |
                   | - pop_actions             |
                   +-------------+-------------+
                                 |
             +-------------------+-------------------+
             |                                       |
             v                                       v
   +---------------------+                +----------------------+
   | policy.forward(...) |                | policy.compute_loss  |
   | chunk inference     |                | training path        |
   +---------------------+                +----------------------+
             |
             v
   +------------------------------+
   | per-env rollout runtime      |
   |                              |
   | history[i]                   |
   | action_queue[i]              |
   | temporal_ensembler[i]        |
   | done[i], steps[i]            |
   +------------------------------+
             |
             v
   +------------------------------+
   | env.step(action or actions)  |
   +------------------------------+
```

## Current Flow

The normal rollout path is:

1. Build env args from BDDL, HDF5 env kwargs, camera config, and rollout config.
2. Create either:
   - `OffScreenRenderEnv` when `env_num == 1`
   - `SubprocVectorEnv` when `env_num > 1`
3. Seed the env backend through `seed_rollout_env(...)`.
4. Create one `ObsHistory` per env slot.
5. Create one temporal ensembler per env slot when the policy uses ACT temporal
   ensembling.
6. For each rollout batch:
   - reset model runtime and rollout runtime
   - reset env(s)
   - set init state(s)
   - collect initial observations
   - run warmup dummy actions
   - repeatedly:
     - decide which envs need more actions
     - build a batched observation input for those envs
     - call `refill_action_queues(...)`
     - pop one action per active env
     - call `step_env_batch(...)`
     - update histories, dones, step counters, and video state
7. Write `rollout_summary.json`.

## Why Single-Env Now Uses the Batch Path

Single-env rollout now uses the same refill logic as vector rollout:

- pending env selection
- action chunk refill
- action queue pop
- optional ACT temporal ensembling

The only difference is the env backend adapter:

- for `env_num == 1`, `set_init_state_batch(...)` unwraps the single init state
- for `env_num == 1`, `step_env_batch(...)` unwraps the single action before
  stepping and re-wraps outputs into batch-like containers

This removes logic drift between single-env and vector-env execution.

## ACT vs DP Semantics

The rollout runtime is shared, but the policies still differ semantically:

- `ACTPolicy`
  - inference is chunk-based
  - current wrapper effectively uses the latest observation frame
  - temporal ensembling is an inference-time runtime state
- `DiffusionPolicy`
  - inference is also chunk-based
  - uses stacked observation history
  - rollout queues remain external and per-env

The key point is that rollout no longer depends on `get_action(...)` as the main
execution interface. That makes multi-env execution and per-env temporal state
much easier to reason about.

## Summary Output

`rollout_summary.json` now includes:

- `total`
- `episode_results`
- optional per-anchor aggregates

`episode_results` are important for debugging single-vs-vector rollout behavior,
because aggregate success rate alone can hide per-init-state differences.

## Why These Helpers Live in `rollout_utils.py`

The following helpers are now shared by both standard and segmentation rollout:

- `seed_rollout_env`
- `build_histories`
- `reset_rollout_runtime`
- `set_init_state_batch`
- `step_env_batch`
- `pending_env_indices`
- `refill_action_queues`
- `pop_actions`

This keeps:

- `rollout_env.py` focused on plain env rollout
- `rollout_seg_env.py` focused on segmentation-specific masking / video logic
- `rollout_utils.py` focused on reusable rollout runtime behavior

## Remaining Boundaries

The current design still keeps a clean separation between:

- shared rollout runtime behavior
- env-specific observation processing

For example:

- plain rollout directly uses `extract_env_obs(...)`
- segmentation rollout first masks images, then uses `extract_env_obs(...)`

That boundary is intentional. The shared runtime should not need to know how an
env-specific rollout variant transforms observations before they enter history.

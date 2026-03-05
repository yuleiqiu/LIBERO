# Visualization Tools README

This document covers only the visualization scripts and helper modules that were modified or added in the recent refactor.

## 1. Change Summary

### 1.1 Kept and Refactored Entry Scripts

- `visualize_all_demos.py`
- `visualize_one_env.py`
- `plot_init_distribution_from_bddl.py`

### 1.2 Newly Added Internal Helper Modules

- `_demo_video_utils.py`
- `_bddl_vis_utils.py`

### 1.3 Removed (Functionality Merged)

- `visualize_one_demo.py` -> merged into `visualize_all_demos.py one`
- `visualize_two_task_demos_comparison.py` -> merged into `visualize_all_demos.py compare`
- `plot_anchors_from_list.py` -> merged into `plot_init_distribution_from_bddl.py anchors`
- `verify_discrete_init_points.py` -> merged into `plot_init_distribution_from_bddl.py verify`
- `visualize_multiple_init_states.py` (removed, no replacement entrypoint)
- `visualize_multi_object_env.py` (removed, no replacement entrypoint)
- `_env_visualization_utils.py` (removed)

---

## 2. `visualize_all_demos.py`

Unified demo visualization entrypoint with three modes:

- `all`: export all demos for one task
- `one`: export one demo from a dataset
- `compare`: compare demos from two tasks (2x2 composite video)

Script path:

- `tools/visualization/visualize_all_demos.py`

### 2.1 Default Behavior

If no subcommand is provided, it defaults to `all` (backward compatible with old usage).

### 2.2 Usage Examples

Export all demos for a task (implicit `all`):

```bash
python3 tools/visualization/visualize_all_demos.py \
  --benchmark libero_object \
  --task-id 0
```

Explicit `all`:

```bash
python3 tools/visualization/visualize_all_demos.py all \
  --benchmark libero_object \
  --task-id 0 \
  --fps 60
```

Export one demo:

```bash
python3 tools/visualization/visualize_all_demos.py one \
  --demo-file /path/to/demo.hdf5 \
  --demo-id 3 \
  --output-dir tmp/visualization \
  --fps 60
```

Cross-task comparison:

```bash
python3 tools/visualization/visualize_all_demos.py compare \
  --benchmark-a libero_object \
  --task-a-id 0 \
  --benchmark-b libero_goal \
  --task-b-id 1 \
  --fps 60
```

### 2.3 Key Arguments

`all` subcommand:

- `--benchmark`: benchmark name (default `libero_object`)
- `--task-id`: task ID
- `--hdf5-path`: optional manual hdf5 file/directory; overrides benchmark/task lookup
- `--output-dir`: output directory
- `--fps`: output frame rate

`one` subcommand:

- `--demo-file`: hdf5 file path (required)
- `--demo-id`: accepts `0` or `demo_0`; defaults to the first demo
- `--output-dir` / `--output-path`: output destination (`output-path` has higher priority)
- `--fps`: output frame rate

`compare` subcommand:

- `--benchmark-a` / `--task-a-id`
- `--benchmark-b` / `--task-b-id`
- `--fps`

---

## 3. `visualize_one_env.py`

This is the only remaining env visualization entrypoint. It:

- resolves BDDL from `task-suite-name + task-id`
- runs reset followed by zero-action rollout steps
- records a short video from a selected camera

Script path:

- `tools/visualization/visualize_one_env.py`

### 3.1 Usage Example

```bash
python3 tools/visualization/visualize_one_env.py \
  --task-suite-name libero_object \
  --task-id 0 \
  --camera-name agentview \
  --num-steps 10 \
  --fps 24 \
  --output-path initialization_sampler.mp4
```

### 3.2 Key Arguments

- `--task-suite-name`: task suite name (required)
- `--task-id`: task ID (required)
- `--camera-name`: camera key, default `agentview`
- `--camera-width` / `--camera-height`: render resolution, default `512x512`
- `--num-steps`: number of post-reset zero-action steps, default `10`
- `--fps`: output mp4 frame rate, default `24`
- `--output-path`: output video path

---

## 4. `plot_init_distribution_from_bddl.py`

Unified BDDL visualization entrypoint with three modes:

- `distribution`: sample and plot initialization distribution + optional illustration
- `verify`: validate sampled points against discrete patches
- `anchors`: visualize mapping from anchor indices to discrete points

Script path:

- `tools/visualization/plot_init_distribution_from_bddl.py`

### 4.1 Default Behavior

If no subcommand is given, it defaults to `distribution`.

### 4.2 Usage Examples

`distribution` (default):

```bash
python3 tools/visualization/plot_init_distribution_from_bddl.py \
  --bddl-file /path/to/task.bddl \
  --samples 100 \
  --out-dir tmp/new_scene_overview/task_xxx
```

Explicit `distribution`:

```bash
python3 tools/visualization/plot_init_distribution_from_bddl.py distribution \
  --bddl-file /path/to/task.bddl \
  --samples 200 \
  --target-region-key goal \
  --include-robot
```

`verify`:

```bash
python3 tools/visualization/plot_init_distribution_from_bddl.py verify \
  --bddl-file /path/to/task.bddl \
  --samples 80 \
  --tolerance 0.01 \
  --plot-path init_points_scatter.png
```

`anchors`:

```bash
python3 tools/visualization/plot_init_distribution_from_bddl.py anchors \
  --bddl-file /path/to/task.bddl \
  --anchor-json /path/to/anchors.json \
  --plot-path anchor_points.png
```

### 4.3 Outputs

`distribution` outputs by default:

- `<out-dir>/init_distribution.png`
- `<out-dir>/init_distribution_xy.npy`
- `<out-dir>/illustration.png` (unless `--no-illustration`)

`verify` outputs:

- scatter plot at `--plot-path`
- optional overview+zoom figure at `--illustration-path`

`anchors` outputs:

- anchor usage figure at `--plot-path`
- optional rotated overview figure at `--illustration-path`

---

## 5. Internal Modules

These modules are intended for reuse by scripts, not as standalone CLIs.

### 5.1 `_demo_video_utils.py`

Responsibilities:

- hdf5 demo key handling (sort/select)
- benchmark/task -> demo file resolution
- dual-view frame composition and video writing
- compare-mode padding / horizontal stack / vertical stack

Typical functions:

- `list_demo_keys`
- `normalize_demo_key`
- `build_task_context`
- `render_demo_video`
- `hstack_with_padding` / `vstack_with_padding`

### 5.2 `_bddl_vis_utils.py`

Responsibilities:

- normalize BDDL region ranges: `sanitize_ranges`
- infer region key from `initial_state`: `infer_region_key`

---

## 6. Old-to-New Migration Map

### 6.1 Demo Scripts

- Old: `visualize_one_demo.py`
  - New: `visualize_all_demos.py one`
- Old: `visualize_two_task_demos_comparison.py`
  - New: `visualize_all_demos.py compare`

### 6.2 BDDL Scripts

- Old: `verify_discrete_init_points.py`
  - New: `plot_init_distribution_from_bddl.py verify`
- Old: `plot_anchors_from_list.py`
  - New: `plot_init_distribution_from_bddl.py anchors`

### 6.3 Env Grid Scripts

- Old: `visualize_multiple_init_states.py` / `visualize_multi_object_env.py`
  - Removed in this refactor (no grid-env entrypoint kept)

---

## 7. Dependencies

At minimum, these scripts depend on:

- `h5py`
- `imageio`
- `numpy`
- `matplotlib` (for BDDL plotting workflows)
- LIBERO runtime dependencies (`robosuite` / MuJoCo stack)

If you hit `ModuleNotFoundError`, install missing packages in your LIBERO conda environment first.

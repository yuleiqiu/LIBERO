# Train Script (standalone/train.py)

This script trains ACT or CNNMLP policies from HDF5 demonstrations using
dataclass defaults plus CLI overrides. YAML configs are no longer supported.

## Quick start

CLI-only usage:

```bash
python standalone/train.py \
  --data.demo_file=./libero/datasets/libero_object_single/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5 \
  --policy.name=act \
  --paths.save_dir=standalone/standalone_runs/train_act_quickcheck
```

CNNMLP example:

```bash
python standalone/train.py \
  --data.demo_file=./libero/datasets/libero_object_single/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5 \
  --policy.name=cnnmlp \
  --policy.cnnmlp.model.hidden_dim=256 \
  --paths.save_dir=standalone/standalone_runs/train_cnnmlp_quickcheck
```

Diffusion Policy example (DP encoder with SpatialSoftmax + GroupNorm):

```bash
python standalone/train.py \
  --data.demo_file=./libero/datasets/libero_object_single/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5 \
  --policy.name=dp \
  --data.obs_horizon=2 \
  --data.predict_horizon=8 \
  --data.image_norm=scale_0_1 \
  --policy.dp.encoder.image.type=dp_resnet \
  --policy.dp.encoder.image.pretrained=false \
  --policy.dp.encoder.image.use_group_norm=true \
  --policy.dp.encoder.image.spatial_softmax_num_keypoints=32 \
  --policy.dp.model.horizon=16 \
  --policy.dp.model.n_obs_steps=2 \
  --policy.dp.model.n_action_steps=8 \
  --policy.dp.model.noise_scheduler_type=DDPM \
  --policy.dp.model.do_mask_loss_for_padding=true \
  --paths.save_dir=standalone/standalone_runs/train_dp_quickcheck
```

Resume or reproduce from a saved config:

```bash
python standalone/train.py --resume=true \
  --paths.save_dir=standalone/standalone_runs/train_act_quickcheck/run_000
```

```bash
python standalone/train.py --saved_config_path=standalone/standalone_runs/train_act_quickcheck/run_000/train_config.json
```

## Config structure

The main config is `TrainConfig` in `standalone/configs/train.py`, grouped into
`data`, `policy`, `paths`, `training`, `rollout`, and `logging`.

Train config fields (grouped):

- `data.demo_file`: path to the HDF5 demo file.
- `data.obs_keys` / `data.image_keys`: comma-separated keys from the dataset.
- `data.obs_horizon`, `data.predict_horizon`: temporal window sizes.
- `data.action_shift`: action offset relative to observations.
- `data.image_norm`: image normalization (`none`, `scale_0_1`, or `imagenet`).
- `data.image_transforms.*`: optional image augmentation settings.
- `data.normalize_obs`, `data.obs_stats_path`: observation normalization settings.
- `data.obs_key_mapping`: optional remap for dataset keys.
- `data.train_ratio`, `data.val_ratio`, `data.seed`: split ratios and RNG seed.
- `policy.name`: policy type (`act` or `cnnmlp`).
- `policy.act.model.*`: ACT model overrides (e.g., `policy.act.model.enc_layers`).
- `policy.act.kl_weight`, `policy.act.lr_backbone`: ACT wrapper settings.
- `policy.cnnmlp.model.*`: CNNMLP model overrides.
- `policy.cnnmlp.lr_backbone`: CNNMLP wrapper settings.
- `resume`, `saved_config_path`: resume from an existing run config (JSON).
- `paths.save_dir`: base directory for run outputs (see below).
- `training.batch_size`, `training.lr`, `training.epochs`: core optimization settings.
- `training.val_every`, `training.grad_clip`, `training.device`: validation cadence, grad clip, device.
- `training.save_ckpt_every`: save logic trigger period (in epochs).
- `training.save_topk`: number of top-K checkpoints to keep (by success_rate).
- `training.scheduler.*`: optional global LR scheduler defaults (overridden by policy schedulers).
- `policy.<name>.scheduler.*`: optional per-policy LR scheduler settings (see below).
- `rollout.every`: training-time rollout interval (0 disables rollouts).
- `rollout.init_states_dir`: init states root (falls back to LIBERO default if unset).
- `rollout.env_horizon`: environment horizon used during training rollouts.
- `rollout.per_anchor`: number of rollouts per anchor.
- `rollout.steps`, `rollout.warmup_steps`: rollout length and warmup steps.
- `rollout.use_mp`, `rollout.num_procs`: multiprocessing settings for rollouts.
- `logging.use_wandb`: enable/disable wandb.
- `logging.wandb_project`, `logging.wandb_entity`, `logging.experiment_name`: wandb metadata.

## Run directory behavior

`paths.save_dir` is treated as a base directory unless it already ends with
`run_###`.

Example:

- `paths.save_dir: standalone/standalone_runs/train_act_quickcheck`
- Creates `standalone/standalone_runs/train_act_quickcheck/run_000`
  (then `run_001`, `run_002`, ... on later runs)

Each run directory stores:

- `train_config.json`: full resolved config snapshot
- `command.txt`: the exact CLI used to launch the run
- `run_meta.json`: metadata like `started_at`
- `split_indices.json`: train/val split indices
- `model_last.pt`: latest checkpoint (always saved when save logic triggers)
- `model_topk_epoch_###.pt`: top-K checkpoints by success rate (saved when enabled)
- `obs_stats.json`: only when `data.normalize_obs` is enabled
- `rollout_videos/`: only when rollouts are enabled (training uses `val/epoch_###`,
  post-training defaults to `eval/`)

## Notes

- ACT/CNNMLP ignore observation normalization; `normalize_obs` is disabled
  automatically for these policies.
- Image normalization is applied in the dataset via `data.image_norm`.
- Validation loss is logged every `training.val_every` epochs; only the loss is tracked
  (no best-val selection or extra val stats). `split_indices.json` still records the
  train/val split from `data.train_ratio`/`data.val_ratio`.
- `rollout.init_states_dir` defaults to the LIBERO init_states path if unset.
- `rollout.env_horizon` defaults to 2000 for training-time rollouts.
- `training.save_ckpt_every` gates save logic: only epochs where `epoch % save_ckpt_every == 0` trigger saves.
- When save logic triggers, `model_last.pt` is always updated.
- Top-K saving uses `success_rate` from rollouts and only runs on save-trigger epochs.
  It keeps `training.save_topk` checkpoints (default 5) and writes `topk.json` with
  `epoch`, `success_rate`, and `path`.
- If you pass a `paths.save_dir` that already ends with `run_###`, it will be used
  as-is (no auto-increment).
- YAML configs are disabled; use CLI overrides and `train_config.json` instead.
- Policy configs are dataclass-backed; set `policy.name` and override
  `policy.act.model.*` / `policy.cnnmlp.model.*` as needed.
- `resume: true` loads `train_config.json` from `paths.save_dir` (or `saved_config_path`),
  and `saved_config_path` alone can be used to reproduce a run; both paths only allow
  overrides in the whitelist (device and wandb fields). If `train_config.json`
  is missing, it falls back to the legacy `config.json`.

## LR Scheduler (per policy)

Schedulers are optional and configured per policy. If `policy.<name>.scheduler.name` is `"none"`,
no scheduler is used.

Supported:
- `name`: `none` | `cosine` | `linear` | `constant`
- `warmup_steps`: number of warmup steps before decay
- `num_training_steps`: override total training steps (optional)
- `min_lr`: lower bound on learning rate (optional)

Example:

```bash
python standalone/train.py \
  --policy.name=dp \
  --policy.dp.scheduler.name=cosine \
  --policy.dp.scheduler.warmup_steps=500
```

## Image augmentation

Image augmentation is configured under `data.image_transforms` and is applied
only during training. Rollout/inference uses raw images, but still applies
`data.image_norm` to keep the input distribution consistent with training.

Available transforms (by `type`):

- `ColorJitter` (brightness/contrast/saturation/hue)
- `SharpnessJitter`
- `RandomAffine`
- `RandomCrop` (keeps original resolution when `size` is omitted)

Example (enable augmentation + imagenet normalization):

```bash
python standalone/train.py \
  --data.image_norm=imagenet \
  --data.image_transforms.enable=true \
  --data.image_transforms.max_num_transforms=3 \
  --data.image_transforms.tfs.random_crop.weight=1.0 \
  --data.image_transforms.tfs.random_crop.kwargs.padding=4
```

## Rollout usage (rollout_env.py)

You can run rollouts from a checkpoint and reuse the training config saved in
the checkpoint directory.

Typical command:

```bash
python standalone/rollout_env.py \
  ckpt=standalone/standalone_runs/train_cnnmlp_quickcheck/run_000/model_last.pt \
  steps=1000 n_rollouts=90 warmup_steps=10 use_mp=true num_procs=10 \
  save_videos=90 video_camera=agentview_rgb,eye_in_hand_rgb \
  video_dir=standalone/standalone_runs/train_cnnmlp_quickcheck/run_000/rollout_on_test_init
```

Notes:

- `ckpt` must point to the actual `run_###/model_last.pt` you want to evaluate.
- Use `model_topk_epoch_###.pt` for high-performing snapshots, and `model_last.pt`
  for the most recent snapshot.
- When `use_ckpt_config: true` (default), `data.*` and `policy.*` are filled
  from `train_config.json` in the checkpoint directory. Old checkpoints without
  this file still need explicit `data`/`policy` fields.
- `env_horizon` sets the environment max episode length; keep it >=
  `steps + warmup_steps` to avoid early termination errors. When
  `use_ckpt_config: true`, you can omit `env_horizon` to fall back to the
  checkpoint's `rollout.env_horizon` if present.
- `video_dir` is optional; if omitted, videos go under the checkpoint folder
  at `rollout_videos/eval/`.
- A `rollout_summary.json` file is written under `video_dir` (default path is
  used even when `save_videos=0`) with total success stats and per-anchor
  success rates when anchor metadata is available.

Rollout config fields:

- `ckpt`: checkpoint to load (required).
- `use_ckpt_config`: reuse `data` / `policy` from the checkpoint (default true).
- `init_states`: optional .pruned_init path (null means use HDF5 init states).
- `env_horizon`: environment max episode length (optional when using ckpt config).
- `steps`, `warmup_steps`: rollout length and warmup steps.
- `n_rollouts`, `sample_index`: number of rollouts and start index.
- `use_mp`, `num_procs`: multiprocessing settings.
- `save_videos`: number of videos to save (0 disables).
- `video_camera`, `video_fps`, `video_dir`: video settings.
- `device`: device string (e.g., "cuda:0").

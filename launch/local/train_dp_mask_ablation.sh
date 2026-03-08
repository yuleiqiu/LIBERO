#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

REPO_DIR=~/codes/LIBERO
cd "$REPO_DIR"

DEMO_SRC="${REPO_DIR}/libero/datasets/libero_single/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo_with_seg.hdf5"
EXP_ROOT="${REPO_DIR}/standalone/standalone_runs/train_dp/libero_single/alphabet_soup/mask_ablation"

OBS_HORIZON=2
MODEL_HORIZON=128
N_ACTION_STEPS=100
MASK_KEYS="agentview_segmentation_of_interest,eye_in_hand_segmentation_of_interest"
SEED=300
BATCH_SIZE=64
GPU_ID="${GPU_ID:-0}"
JOB_ID="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="seed${SEED}_bs${BATCH_SIZE}_to${OBS_HORIZON}_tp${MODEL_HORIZON}_ta${N_ACTION_STEPS}_mask_ablate"
SAVE_BASE_DIR="${EXP_ROOT}/${RUN_TAG}"
LOG_DIR="${SAVE_BASE_DIR}/local_logs/${JOB_ID}"

mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m standalone.train \
  --data.demo_file="${DEMO_SRC}" \
  --policy.name=dp \
  --data.image_norm=scale_0_1 \
  --data.obs_horizon="${OBS_HORIZON}" \
  --data.predict_horizon="${MODEL_HORIZON}" \
  --data.mask_keys="${MASK_KEYS}" \
  --policy.dp.encoder.image.type=dp_resnet \
  --policy.dp.encoder.image.use_separate_rgb_encoder_per_camera=true \
  --policy.dp.encoder.image.crop_randomizer.enable=true \
  --policy.dp.encoder.image.crop_randomizer.crop_height=115 \
  --policy.dp.encoder.image.crop_randomizer.crop_width=115 \
  --policy.dp.model.horizon="${MODEL_HORIZON}" \
  --policy.dp.model.n_obs_steps="${OBS_HORIZON}" \
  --policy.dp.model.n_action_steps="${N_ACTION_STEPS}" \
  --policy.dp.model.noise_scheduler_type=DDPM \
  --policy.dp.model.do_mask_loss_for_padding=true \
  --policy.dp.optimizer.lr=1e-4 \
  --policy.dp.optimizer.weight_decay=1e-6 \
  --training.batch_size="${BATCH_SIZE}" \
  --training.epochs=2000 \
  --training.val_every=1 \
  --training.save_ckpt_every=50 \
  --training.save_topk=5 \
  --rollout.every=50 \
  --rollout.warmup_steps=15 \
  --rollout.steps=1500 \
  --rollout.use_mp=true \
  --rollout.num_procs=6 \
  --paths.save_dir="${SAVE_BASE_DIR}" \
  --logging.use_wandb=true \
  --logging.wandb_project=libero_single \
  --logging.wandb_group="dp_mask_ablation" \
  --logging.wandb_tags="task:pick_up_alphabet_soup,policy:dp,exp:mask_ablation,to:${OBS_HORIZON},tp:${MODEL_HORIZON},ta:${N_ACTION_STEPS},seed:${SEED},bs:${BATCH_SIZE}" \
  --logging.experiment_name="dp_mask_on" \
  > "${LOG_DIR}/train.log" 2>&1

#!/bin/bash
#SBATCH --job-name=libero_dp_mask_compare
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=8G
#SBATCH --time=48:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
export PYTHONUNBUFFERED=1

# -----------------------
# Array IDs
# -----------------------
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

# -----------------------
# Log directory: <submit_dir>/<array_job_id>/<array_task_id>/
# -----------------------
LOG_DIR="$(pwd)/${ARRAY_JOB_ID}/${ARRAY_TASK_ID}"
mkdir -p "$LOG_DIR"
exec > "$LOG_DIR/slurm.out" 2> "$LOG_DIR/slurm.err"

# -----------------------
# Paths
# -----------------------
export REPO_DIR="${REPO_DIR:-$SCRATCH/work/LIBERO}"
EXP_ROOT="${EXP_ROOT:-$REPO_DIR/standalone/standalone_runs/dp/libero_single/alphabet_soup}"
mkdir -p "$EXP_ROOT"

TASK_SLUG="pick_up_alphabet_soup"
WANDB_PROJECT="${WANDB_PROJECT:-libero_single}"
DEMO_SRC_REL="libero/datasets/libero_single/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo_with_seg.hdf5"

# -----------------------
# Cache relocation (persistent): use $SCRATCH only
# -----------------------
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/.cache}"
export XDG_CACHE_HOME="$CACHE_ROOT"
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_DATASETS_CACHE="$CACHE_ROOT/huggingface/datasets"
export HF_HUB_CACHE="$CACHE_ROOT/huggingface/hub"
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"

# -----------------------
# Environment
# -----------------------
module load miniforge/24.7.1
CONDA_ENV="${CONDA_ENV:-libero}"
conda activate "$CONDA_ENV"
cd "$REPO_DIR"

# -----------------------
# Flash storage: copy demo to fast local storage for I/O
# Checkpoints and wandb still write to $SCRATCH
# -----------------------
DEMO_SRC="${REPO_DIR}/${DEMO_SRC_REL}"
if [[ ! -f "$DEMO_SRC" ]]; then
  echo "Demo file not found: $DEMO_SRC"
  exit 2
fi

LOCAL_SCRATCH_ROOT="${TMP_SHARED:-${TMPDIR:-/tmp}}"
DEMO_DST="${LOCAL_SCRATCH_ROOT}/$(basename "$DEMO_SRC")"
echo "Copying demo to local scratch: $DEMO_DST"
cp "$DEMO_SRC" "$DEMO_DST"
echo "Done: $DEMO_DST"
export LIBERO_RUNTIME_DEMO_FILE="$DEMO_DST"

# -----------------------
# Fixed experiment settings
# -----------------------
OBS_HORIZON=2
MODEL_HORIZON=128
ACTION_CHUNK=100
N_OBS_STEPS="${OBS_HORIZON}"
PREDICT_HORIZON="${MODEL_HORIZON}"
SEED=0
BATCH_SIZE=64
FIXED_LR=1e-4
FIXED_WEIGHT_DECAY=1e-6
MASK_KEYS_ALL="agentview_segmentation_of_interest,eye_in_hand_segmentation_of_interest"

# Note:
# In this codebase, "action_horizon=100" is most safely represented as
# policy.dp.model.n_action_steps=100. We keep the supervised/model trajectory
# length at 128 and do not set policy.dp.action_horizon here.

# -----------------------
# Two-way ablation table
# -----------------------
SWEEP_TABLE=(
  "variant=no_mask mask_mode=off mask_keys=__NONE__"
  "variant=with_mask mask_mode=on mask_keys=${MASK_KEYS_ALL}"
)

if [[ "$ARRAY_TASK_ID" -lt 0 || "$ARRAY_TASK_ID" -ge "${#SWEEP_TABLE[@]}" ]]; then
  echo "Invalid SLURM_ARRAY_TASK_ID=$ARRAY_TASK_ID (expected 0..$((${#SWEEP_TABLE[@]}-1)))"
  exit 2
fi

declare -A HP
for kv in ${SWEEP_TABLE[$ARRAY_TASK_ID]}; do
  key="${kv%%=*}"
  value="${kv#*=}"
  HP["$key"]="$value"
done

VARIANT="${HP[variant]}"
MASK_MODE="${HP[mask_mode]}"
MASK_KEYS_RAW="${HP[mask_keys]}"
MASK_KEYS=""
if [[ "$MASK_KEYS_RAW" != "__NONE__" ]]; then
  MASK_KEYS="$MASK_KEYS_RAW"
fi

# -----------------------
# System-managed settings (machine/runtime)
# -----------------------
ROLLOUT_NUM_PROCS=6
if [[ -n "${SLURM_CPUS_PER_TASK:-}" && "$SLURM_CPUS_PER_TASK" -lt "$ROLLOUT_NUM_PROCS" ]]; then
  ROLLOUT_NUM_PROCS="$SLURM_CPUS_PER_TASK"
fi

RUN_TAG="${VARIANT}_to${OBS_HORIZON}_tp${MODEL_HORIZON}_ta${ACTION_CHUNK}_seed${SEED}_bs${BATCH_SIZE}"
SAVE_BASE_DIR="${EXP_ROOT}/${RUN_TAG}"
mkdir -p "$SAVE_BASE_DIR"

export WANDB_DIR="${SAVE_BASE_DIR}/wandb/array_${ARRAY_JOB_ID}_task_${ARRAY_TASK_ID}"
export WANDB_CACHE_DIR="$CACHE_ROOT/wandb"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"

WANDB_GROUP_DEFAULT="${TASK_SLUG}_dp_mask_compare_${ARRAY_JOB_ID}"
WANDB_GROUP="${WANDB_GROUP:-$WANDB_GROUP_DEFAULT}"
WANDB_TAGS="${WANDB_TAGS:-suite:libero_single,task:${TASK_SLUG},policy:dp,exp:mask_compare,mask:${MASK_MODE},obs_horizon:${OBS_HORIZON},n_obs_steps:${N_OBS_STEPS},predict_horizon:${PREDICT_HORIZON},model_horizon:${MODEL_HORIZON},n_action_steps:${ACTION_CHUNK}}"

echo "Running array task ${ARRAY_TASK_ID}"
echo "VARIANT=${VARIANT}"
echo "RUN_TAG=${RUN_TAG}"
echo "SAVE_BASE_DIR=${SAVE_BASE_DIR}"
echo "MASK_KEYS=${MASK_KEYS:-<empty>}"

# -----------------------
# BASE_ARGS: experiment recipe (mostly fixed)
# -----------------------
BASE_ARGS=(
  --data.demo_file="${DEMO_SRC}"
  --policy.name=dp
  --data.image_norm=scale_0_1
  --data.obs_horizon="${OBS_HORIZON}"
  --data.predict_horizon="${PREDICT_HORIZON}"
  --policy.dp.encoder.image.type=dp_resnet
  --policy.dp.encoder.image.use_separate_rgb_encoder_per_camera=true
  --policy.dp.encoder.image.crop_randomizer.enable=true
  --policy.dp.encoder.image.crop_randomizer.crop_height=115
  --policy.dp.encoder.image.crop_randomizer.crop_width=115
  --policy.dp.model.horizon="${MODEL_HORIZON}"
  --policy.dp.model.n_obs_steps="${N_OBS_STEPS}"
  --policy.dp.model.n_action_steps="${ACTION_CHUNK}"
  --policy.dp.model.noise_scheduler_type=DDPM
  --policy.dp.model.do_mask_loss_for_padding=true
  --policy.dp.optimizer.lr="${FIXED_LR}"
  --policy.dp.optimizer.weight_decay="${FIXED_WEIGHT_DECAY}"
  --training.batch_size="${BATCH_SIZE}"
  --training.epochs=2000
  --training.val_every=1
  --training.save_ckpt_every=50
  --training.save_topk=5
  --rollout.every=50
  --rollout.warmup_steps=15
  --rollout.steps=1500
  --rollout.use_mp=true
  --logging.use_wandb=true
  --logging.wandb_project="${WANDB_PROJECT}"
)

if [[ -n "$MASK_KEYS" ]]; then
  BASE_ARGS+=(--data.mask_keys="${MASK_KEYS}")
fi

# -----------------------
# SYSTEM_ARGS: runtime + output naming
# -----------------------
SYSTEM_ARGS=(
  --data.seed="${SEED}"
  --rollout.num_procs="${ROLLOUT_NUM_PROCS}"
  --paths.save_dir="${SAVE_BASE_DIR}"
  --logging.experiment_name="dp_${TASK_SLUG}_${VARIANT}"
  --logging.wandb_group="${WANDB_GROUP}"
  --logging.wandb_tags="${WANDB_TAGS}"
)

python -m standalone.train \
  "${BASE_ARGS[@]}" \
  "${SYSTEM_ARGS[@]}" \
  > "$LOG_DIR/train.log" 2>&1

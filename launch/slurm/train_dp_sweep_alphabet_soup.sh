#!/bin/bash
#SBATCH --job-name=libero_dp_sweep
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=8G
#SBATCH --time=48:00:00
#SBATCH --array=0-0%1
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
export REPO_DIR="$SCRATCH/work/LIBERO"
EXP_ROOT="$SCRATCH/exp/libero/dp/libero_single/alphabet_soup"
mkdir -p "$EXP_ROOT"
TASK_SLUG="pick_up_alphabet_soup"
WANDB_PROJECT="libero_single"

# -----------------------
# Cache relocation (persistent): use $SCRATCH only
# -----------------------
CACHE_ROOT="$SCRATCH/.cache"
export XDG_CACHE_HOME="$CACHE_ROOT"
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_DATASETS_CACHE="$CACHE_ROOT/huggingface/datasets"
export HF_HUB_CACHE="$CACHE_ROOT/huggingface/hub"
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"

# -----------------------
# Environment
# -----------------------
module load miniforge/24.7.1
conda activate libero
cd "$REPO_DIR"

# -----------------------
# Flash storage: copy demo to NVMe for fast I/O
# Checkpoint/wandb still write to $SCRATCH to avoid timeout loss
# -----------------------
DEMO_SRC="./libero/datasets/libero_single/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo_with_seg.hdf5"
DEMO_DST="$TMP_SHARED/$(basename "$DEMO_SRC")"
echo "Copying demo to TMP_SHARED..."
cp "$DEMO_SRC" "$DEMO_DST"
echo "Done: $DEMO_DST"

# -----------------------
# Sweep table (one full config per array task)
# -----------------------
SWEEP_TABLE=(
  "predict_horizon=128 model_horizon=128 n_action_steps=100 seed=0 batch_size=64"
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

PREDICT_HORIZON="${HP[predict_horizon]}"
MODEL_HORIZON="${HP[model_horizon]}"
N_ACTION_STEPS="${HP[n_action_steps]}"
SEED="${HP[seed]}"
BATCH_SIZE="${HP[batch_size]}"

# Fixed training hyperparameters (not swept)
FIXED_LR="1e-4"
FIXED_WEIGHT_DECAY="1e-6"

# -----------------------
# System-managed settings (machine/runtime)
# -----------------------
ROLLOUT_NUM_PROCS=6
if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
  if [[ "$SLURM_CPUS_PER_TASK" -lt "$ROLLOUT_NUM_PROCS" ]]; then
    ROLLOUT_NUM_PROCS="$SLURM_CPUS_PER_TASK"
  fi
fi

RUN_TAG="ph${PREDICT_HORIZON}_h${MODEL_HORIZON}_na${N_ACTION_STEPS}_s${SEED}_bs${BATCH_SIZE}"
SAVE_BASE_DIR="${EXP_ROOT}/${RUN_TAG}"
mkdir -p "$SAVE_BASE_DIR"

export WANDB_DIR="${SAVE_BASE_DIR}/wandb/array_${ARRAY_JOB_ID}_task_${ARRAY_TASK_ID}"
export WANDB_CACHE_DIR="$CACHE_ROOT/wandb"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"
# Default wandb metadata (editable here). Can still override via environment if needed.
WANDB_GROUP_DEFAULT="${TASK_SLUG}_dp_sweep_${ARRAY_JOB_ID}"
WANDB_GROUP="${WANDB_GROUP:-$WANDB_GROUP_DEFAULT}"
WANDB_TAGS="${WANDB_TAGS:-suite:libero_single,task:${TASK_SLUG},policy:dp,type:sweep}"

echo "Running array task ${ARRAY_TASK_ID}"
echo "RUN_TAG=${RUN_TAG}"
echo "SAVE_BASE_DIR=${SAVE_BASE_DIR}"

# -----------------------
# BASE_ARGS: experiment recipe (mostly fixed)
# -----------------------
BASE_ARGS=(
  --data.demo_file="${DEMO_DST}"
  --policy.name=dp
  --data.image_norm=scale_0_1
  --data.obs_horizon=2
  --policy.dp.encoder.image.type=dp_resnet
  --policy.dp.encoder.image.use_separate_rgb_encoder_per_camera=true
  --policy.dp.encoder.image.crop_randomizer.enable=true
  --policy.dp.encoder.image.crop_randomizer.crop_height=115
  --policy.dp.encoder.image.crop_randomizer.crop_width=115
  --policy.dp.model.n_obs_steps=2
  --policy.dp.model.noise_scheduler_type=DDPM
  --policy.dp.model.do_mask_loss_for_padding=true
  --policy.dp.optimizer.lr="${FIXED_LR}"
  --policy.dp.optimizer.weight_decay="${FIXED_WEIGHT_DECAY}"
  --training.epochs=3000
  --training.val_every=1
  --training.save_ckpt_every=50
  --training.save_topk=5
  --rollout.every=50
  --rollout.warmup_steps=15
  --rollout.steps=1800
  --rollout.use_mp=true
  --logging.use_wandb=true
  --logging.wandb_project="${WANDB_PROJECT}"
)

# -----------------------
# SWEEP_ARGS: hyperparameters controlled by task id
# -----------------------
SWEEP_ARGS=(
  --data.seed="${SEED}"
  --training.batch_size="${BATCH_SIZE}"
  --data.predict_horizon="${PREDICT_HORIZON}"
  --policy.dp.model.horizon="${MODEL_HORIZON}"
  --policy.dp.model.n_action_steps="${N_ACTION_STEPS}"
)

# -----------------------
# SYSTEM_ARGS: runtime + output naming
# -----------------------
SYSTEM_ARGS=(
  --rollout.num_procs="${ROLLOUT_NUM_PROCS}"
  --paths.save_dir="${SAVE_BASE_DIR}"
  --logging.experiment_name="dp_single_alphabet_soup_${RUN_TAG}"
  --logging.wandb_group="${WANDB_GROUP}"
  --logging.wandb_tags="${WANDB_TAGS}"
)

python -m standalone.train \
  "${BASE_ARGS[@]}" \
  "${SWEEP_ARGS[@]}" \
  "${SYSTEM_ARGS[@]}" \
  > "$LOG_DIR/train.log" 2>&1

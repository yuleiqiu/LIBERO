#!/bin/bash
#SBATCH --job-name=libero_dp_eval_mask_compare
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8G
#SBATCH --time=1:00:00
#SBATCH --array=0-23
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
export PYTHONUNBUFFERED=1

# -----------------------
# Job / Task IDs
# -----------------------
JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-0}}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

# -----------------------
# Paths
# -----------------------
export REPO_DIR="${REPO_DIR:-$SCRATCH/work/LIBERO}"
EXP_ROOT="${EXP_ROOT:-$REPO_DIR/standalone/standalone_runs/dp/libero_single/alphabet_soup}"

# -----------------------
# Checkpoint naming config
# -----------------------
RUN_ID="${RUN_ID:-run_000}"
CKPT_NAME="${CKPT_NAME:-model_last.pt}"

PAD_TAG_RAW="${PAD_TAG:-actedge}"
PAD_TAG="${PAD_TAG_RAW//[^[:alnum:]_.-]/_}"

if GIT_TAG_DEFAULT="$(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null)"; then
  :
else
  GIT_TAG_DEFAULT="nogit"
fi
GIT_TAG_RAW="${GIT_TAG:-$GIT_TAG_DEFAULT}"
GIT_TAG="${GIT_TAG_RAW//[^[:alnum:]_.-]/_}"

SEED="${SEED:-300}"
BATCH_SIZE="${BATCH_SIZE:-64}"
OBS_HORIZON="${OBS_HORIZON:-2}"
MODEL_HORIZON="${MODEL_HORIZON:-128}"
ACTION_CHUNK="${ACTION_CHUNK:-100}"

MASK_VARIANTS=(
  no_mask
  with_mask
)

VARIANTS=()
for MASK_VARIANT in "${MASK_VARIANTS[@]}"; do
  VARIANTS+=("${MASK_VARIANT}_pad${PAD_TAG}_hash${GIT_TAG}_seed${SEED}_bs${BATCH_SIZE}_to${OBS_HORIZON}_tp${MODEL_HORIZON}_ta${ACTION_CHUNK}")
done

# -----------------------
# Eval seeds
# -----------------------
EVAL_SEEDS=(400 401 402)

# -----------------------
# Selected environments
# -----------------------
SELECTED_ENVS=(
  single
  distractor_1
  distractor_2
  distractor_3
)

NUM_VARIANTS=${#VARIANTS[@]}
NUM_ENVS=${#SELECTED_ENVS[@]}
NUM_SEEDS=${#EVAL_SEEDS[@]}
TOTAL=$((NUM_VARIANTS * NUM_ENVS * NUM_SEEDS))

if [ "$NUM_VARIANTS" -eq 0 ] || [ "$NUM_ENVS" -eq 0 ] || [ "$NUM_SEEDS" -eq 0 ]; then
  echo "VARIANTS / SELECTED_ENVS / EVAL_SEEDS must all be non-empty." >&2
  exit 1
fi

if [ "$TASK_ID" -lt 0 ] || [ "$TASK_ID" -ge "$TOTAL" ]; then
  echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID}, expect 0..$((TOTAL-1))" >&2
  exit 1
fi

PER_SEED=$((NUM_VARIANTS * NUM_ENVS))
SEED_IDX=$((TASK_ID / PER_SEED))
INNER_IDX=$((TASK_ID % PER_SEED))
VARIANT_IDX=$((INNER_IDX / NUM_ENVS))
ENV_IDX=$((INNER_IDX % NUM_ENVS))

EVAL_SEED="${EVAL_SEEDS[$SEED_IDX]}"
VARIANT="${VARIANTS[$VARIANT_IDX]}"
ENV_KEY="${SELECTED_ENVS[$ENV_IDX]}"

set_env_config() {
  case "$1" in
    single)
      SUITE="libero_single"
      REL_TASK="pick_up_the_alphabet_soup_and_place_it_in_the_basket"
      VIDEO_TAG="single"
      ;;
    distractor_1)
      SUITE="libero_multi"
      REL_TASK="pick_up_alphabet_soup/distractor_1/pick_up_the_alphabet_soup_and_place_it_in_the_basket"
      VIDEO_TAG="distractor_1"
      ;;
    distractor_2)
      SUITE="libero_multi"
      REL_TASK="pick_up_alphabet_soup/distractor_2/pick_up_the_alphabet_soup_and_place_it_in_the_basket"
      VIDEO_TAG="distractor_2"
      ;;
    distractor_3)
      SUITE="libero_multi"
      REL_TASK="pick_up_alphabet_soup/distractor_3/pick_up_the_alphabet_soup_and_place_it_in_the_basket"
      VIDEO_TAG="distractor_3"
      ;;
    *)
      echo "Unknown environment key: $1" >&2
      exit 1
      ;;
  esac
}

set_env_config "$ENV_KEY"

CKPT_DIR="${EXP_ROOT}/${VARIANT}/${RUN_ID}"
CKPT="${CKPT_DIR}/${CKPT_NAME}"
CKPT_STEM="${CKPT_NAME%.pt}"

# -----------------------
# Logs
# -----------------------
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
LOG_ROOT="${LOG_ROOT:-${SUBMIT_DIR}/results}"
LOG_DIR="${LOG_ROOT}/${JOB_ID}/${TASK_ID}"
mkdir -p "$LOG_DIR"
exec > "$LOG_DIR/slurm.out" 2> "$LOG_DIR/slurm.err"

# -----------------------
# Cache relocation
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

if [ ! -f "$CKPT" ]; then
  echo "Checkpoint not found: $CKPT" >&2
  exit 1
fi

# -----------------------
# Video output dir
# -----------------------
BASE_VIDEO_DIR="${CKPT_DIR}/rollout_videos/${CKPT_STEM}/eval/eval_seed_${EVAL_SEED}"

echo "Running TASK_ID=${TASK_ID} | variant=${VARIANT} | eval_seed=${EVAL_SEED} | env=${ENV_KEY} | tag=${VIDEO_TAG}"
echo "Checkpoint: ${CKPT}"
echo "EXP_ROOT: ${EXP_ROOT}"
echo "PAD_TAG: ${PAD_TAG}"
echo "GIT_TAG: ${GIT_TAG}"

python -m standalone.rollout_env \
  --ckpt="${CKPT}" \
  --data.seed="${EVAL_SEED}" \
  --data.demo_file="libero/datasets/libero_single/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo_with_seg.hdf5" \
  --bddl_file="libero/libero/bddl_files/${SUITE}/${REL_TASK}.bddl" \
  --init_states="libero/libero/init_files/${SUITE}/${REL_TASK}.pruned_init" \
  --steps=1500 \
  --warmup_steps=15 \
  --n_rollouts=60 \
  --use_mp=true \
  --num_procs=6 \
  --save_videos=60 \
  --video_dir="${BASE_VIDEO_DIR}/${VIDEO_TAG}"

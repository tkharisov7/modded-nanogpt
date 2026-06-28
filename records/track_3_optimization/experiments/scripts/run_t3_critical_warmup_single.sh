#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-0}"
SCHEDULE_KIND="${SCHEDULE_KIND:-baseline}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
WARMUP_TARGET="${WARMUP_TARGET:-muon_hidden_only}"
MUON_LR_MULT="${MUON_LR_MULT:-1.0}"
AUX_LR_MULT="${AUX_LR_MULT:-1.0}"
CRITICAL_LR_SEARCH_MAX="${CRITICAL_LR_SEARCH_MAX:-0.375}"
CRITICAL_LR_TOL_POWER="${CRITICAL_LR_TOL_POWER:-6}"
CRITICAL_LR_EXP_MAX_ITERS="${CRITICAL_LR_EXP_MAX_ITERS:-40}"
CRITICAL_LR_MAX_ITERS="${CRITICAL_LR_MAX_ITERS:-80}"
STOP_STEP="${STOP_STEP:-2720}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/fineweb10B}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-64}"
FINAL_SCHEDULE_STEPS="${FINAL_SCHEDULE_STEPS:-2900}"
FINAL_LR_POWER="${FINAL_LR_POWER:-1.2}"
COOLDOWN_START_OFFSET="${COOLDOWN_START_OFFSET:-0}"
WANDB_MODE="${WANDB_MODE:-disabled}"
WANDB_PROJECT="${WANDB_PROJECT:-modded-nanogpt-track3}"
WANDB_GROUP="${WANDB_GROUP:-T3_CRITICAL_WARMUP_LR_PUSH_V1}"
WANDB_NAME="${WANDB_NAME:-${SCHEDULE_KIND}_w${WARMUP_STEPS}_m${MUON_LR_MULT}_seed${SEED}}"

cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export WANDB_MODE
export WANDB_PROJECT
export WANDB_GROUP
export WANDB_NAME

torchrun --standalone --nproc_per_node=1 \
  records/track_3_optimization/experiments/train_gpt_critical_warmup.py \
  --seed "${SEED}" \
  --schedule-kind "${SCHEDULE_KIND}" \
  --warmup-steps "${WARMUP_STEPS}" \
  --warmup-target "${WARMUP_TARGET}" \
  --muon-lr-mult "${MUON_LR_MULT}" \
  --aux-lr-mult "${AUX_LR_MULT}" \
  --critical-lr-search-max "${CRITICAL_LR_SEARCH_MAX}" \
  --critical-lr-tol-power "${CRITICAL_LR_TOL_POWER}" \
  --critical-lr-exp-max-iters "${CRITICAL_LR_EXP_MAX_ITERS}" \
  --critical-lr-max-iters "${CRITICAL_LR_MAX_ITERS}" \
  --stop-step "${STOP_STEP}" \
  --data-dir "${DATA_DIR}" \
  --micro-batch-size "${MICRO_BATCH_SIZE}" \
  --final-schedule-steps "${FINAL_SCHEDULE_STEPS}" \
  --final-lr-power "${FINAL_LR_POWER}" \
  --cooldown-start-offset "${COOLDOWN_START_OFFSET}" \
  --wandb-mode "${WANDB_MODE}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-group "${WANDB_GROUP}" \
  --wandb-name "${WANDB_NAME}"

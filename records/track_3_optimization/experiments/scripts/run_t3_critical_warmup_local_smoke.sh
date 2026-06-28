#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/fineweb10B}"
STOP_STEP="${STOP_STEP:-2720}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-16}"
MUON_LR_MULT="${MUON_LR_MULT:-1.05}"
AUX_LR_MULT="${AUX_LR_MULT:-1.0}"
CRITICAL_LR_SEARCH_MAX="${CRITICAL_LR_SEARCH_MAX:-0.375}"
CRITICAL_LR_TOL_POWER="${CRITICAL_LR_TOL_POWER:-6}"
CRITICAL_LR_EXP_MAX_ITERS="${CRITICAL_LR_EXP_MAX_ITERS:-40}"
CRITICAL_LR_MAX_ITERS="${CRITICAL_LR_MAX_ITERS:-80}"
FINAL_SCHEDULE_STEPS="${FINAL_SCHEDULE_STEPS:-2900}"
FINAL_LR_POWER="${FINAL_LR_POWER:-1.2}"
COOLDOWN_START_OFFSET="${COOLDOWN_START_OFFSET:-0}"
WANDB_MODE="${WANDB_MODE:-disabled}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/records/track_3_optimization/experiments/logs/local_smoke}"
SEED="${SEED:-0}"
GPU_CSV="${GPU_CSV:-1,2,3,4,5}"

IFS=, read -r -a GPUS <<<"${GPU_CSV}"
if (( ${#GPUS[@]} < 2 )); then
  echo "GPU_CSV must contain at least two GPU ids." >&2
  exit 1
fi

mkdir -p "${LOG_ROOT}"
declare -a PIDS=()

run_one() {
  local gpu_id="$1"
  local schedule_kind="$2"
  local warmup_steps="$3"
  local muon_lr_mult="$4"
  local log_file="${LOG_ROOT}/${schedule_kind}_w${warmup_steps}_m${muon_lr_mult}_gpu${gpu_id}_seed${SEED}.log"
  echo "Launching ${schedule_kind} warmup_steps=${warmup_steps} muon_lr_mult=${muon_lr_mult} on GPU ${gpu_id}; log=${log_file}"
  GPU_ID="${gpu_id}" \
  SEED="${SEED}" \
  SCHEDULE_KIND="${schedule_kind}" \
  WARMUP_STEPS="${warmup_steps}" \
  MUON_LR_MULT="${muon_lr_mult}" \
  AUX_LR_MULT="${AUX_LR_MULT}" \
  CRITICAL_LR_SEARCH_MAX="${CRITICAL_LR_SEARCH_MAX}" \
  CRITICAL_LR_TOL_POWER="${CRITICAL_LR_TOL_POWER}" \
  CRITICAL_LR_EXP_MAX_ITERS="${CRITICAL_LR_EXP_MAX_ITERS}" \
  CRITICAL_LR_MAX_ITERS="${CRITICAL_LR_MAX_ITERS}" \
  FINAL_SCHEDULE_STEPS="${FINAL_SCHEDULE_STEPS}" \
  FINAL_LR_POWER="${FINAL_LR_POWER}" \
  COOLDOWN_START_OFFSET="${COOLDOWN_START_OFFSET}" \
  STOP_STEP="${STOP_STEP}" \
  DATA_DIR="${DATA_DIR}" \
  MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE}" \
  WANDB_MODE="${WANDB_MODE}" \
  WANDB_GROUP="${WANDB_GROUP:-T3_CRITICAL_WARMUP_LOCAL_SMOKE}" \
  bash "${REPO_ROOT}/records/track_3_optimization/experiments/scripts/run_t3_critical_warmup_single.sh" \
    >"${log_file}" 2>&1 &
  PIDS+=("$!")
}

run_one "${GPUS[0]}" baseline 0 1.0
run_one "${GPUS[1]}" adaptive_critical_lr 10 1.05
if (( ${#GPUS[@]} >= 3 )); then
  run_one "${GPUS[2]}" adaptive_critical_lr 20 1.05
fi
if (( ${#GPUS[@]} >= 4 )); then
  run_one "${GPUS[3]}" adaptive_critical_lr 10 1.10
fi
if (( ${#GPUS[@]} >= 5 )); then
  run_one "${GPUS[4]}" adaptive_critical_lr 10 1.15
fi

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"

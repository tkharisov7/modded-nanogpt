#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/fineweb10B}"
STOP_STEP="${STOP_STEP:-2720}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-16}"
MUON_LR_MULT="${MUON_LR_MULT:-1.05}"
AUX_LR_MULT="${AUX_LR_MULT:-1.0}"
WANDB_MODE="${WANDB_MODE:-disabled}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/records/track_3_optimization/experiments/logs/local_smoke}"
SEED="${SEED:-0}"

mkdir -p "${LOG_ROOT}"
declare -a PIDS=()

run_one() {
  local gpu_id="$1"
  local schedule_kind="$2"
  local warmup_steps="$3"
  local log_file="${LOG_ROOT}/${schedule_kind}_w${warmup_steps}_gpu${gpu_id}_seed${SEED}.log"
  echo "Launching ${schedule_kind} warmup_steps=${warmup_steps} on GPU ${gpu_id}; log=${log_file}"
  GPU_ID="${gpu_id}" \
  SEED="${SEED}" \
  SCHEDULE_KIND="${schedule_kind}" \
  WARMUP_STEPS="${warmup_steps}" \
  MUON_LR_MULT="${MUON_LR_MULT}" \
  AUX_LR_MULT="${AUX_LR_MULT}" \
  STOP_STEP="${STOP_STEP}" \
  DATA_DIR="${DATA_DIR}" \
  MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE}" \
  WANDB_MODE="${WANDB_MODE}" \
  WANDB_GROUP="${WANDB_GROUP:-T3_CRITICAL_WARMUP_LOCAL_SMOKE}" \
  bash "${REPO_ROOT}/records/track_3_optimization/experiments/scripts/run_t3_critical_warmup_single.sh" \
    >"${log_file}" 2>&1 &
  PIDS+=("$!")
}

run_one 2 baseline 0
run_one 3 linear 50
run_one 4 critical_table 50
run_one 5 catapult_proxy 50

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"

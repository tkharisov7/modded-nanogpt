#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
RUN_ROOT="${RUN_ROOT:-/capstor/store/cscs/uba/uba02/t3_critical_warmup}"
DATA_DIR="${DATA_DIR:-/capstor/store/cscs/uba/uba02/fineweb10B}"
STOP_STEP="${STOP_STEP:-2720}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-64}"
WARMUP_TARGET="${WARMUP_TARGET:-muon_hidden_only}"
AUX_LR_MULT="${AUX_LR_MULT:-1.0}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-modded-nanogpt-track3}"
WANDB_GROUP="${WANDB_GROUP:-T3_CRITICAL_WARMUP_GH200_${SLURM_ARRAY_JOB_ID:-manual}}"
CONTAINER_RUN_PREFIX="${CONTAINER_RUN_PREFIX:-}"
CANDIDATES_PER_NODE="${CANDIDATES_PER_NODE:-4}"
GRID_PRESET="${GRID_PRESET:-full_v1}"

if [[ "${WANDB_MODE}" == "online" && -z "${WANDB_API_KEY:-}" && -f "${HOME}/.netrc" ]]; then
  export WANDB_API_KEY
  WANDB_API_KEY="$(
    awk '
      $1 == "machine" { in_wandb = ($2 == "api.wandb.ai") }
      in_wandb {
        for (i = 1; i <= NF; i++) {
          if ($i == "password" && i < NF) {
            print $(i + 1)
            exit
          }
        }
      }
    ' "${HOME}/.netrc"
  )"
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID is required." >&2
  exit 1
fi

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/aggregate" "${RUN_ROOT}/wandb" "${RUN_ROOT}/wandb_cache"

declare -a CANDIDATES=()
add_candidate() {
  local schedule_kind="$1"
  local warmup_steps="$2"
  local muon_lr_mult="$3"
  local seed="$4"
  CANDIDATES+=("${schedule_kind},${warmup_steps},${muon_lr_mult},${seed}")
}

add_cell() {
  local schedule_kind="$1"
  local warmup_steps="$2"
  local muon_lr_mult="$3"
  for seed in 0 1 2 3 4 5 6 7; do
    add_candidate "${schedule_kind}" "${warmup_steps}" "${muon_lr_mult}" "${seed}"
  done
}

case "${GRID_PRESET}" in
  full_v1)
    for seed in 0 1 2 3 4 5 6 7; do
      add_candidate baseline 0 1.0 "${seed}"
    done

    for schedule_kind in linear critical_table catapult_proxy; do
      for warmup_steps in 25 50 100; do
        for muon_lr_mult in 1.0 1.05 1.1 1.15 1.2; do
          for seed in 0 1 2 3 4 5 6 7; do
            add_candidate "${schedule_kind}" "${warmup_steps}" "${muon_lr_mult}" "${seed}"
          done
        done
      done
    done
    ;;
  catapult_critical_high_v1)
    add_cell critical_table 25 1.20
    add_cell critical_table 25 1.25
    add_cell critical_table 25 1.30
    add_cell critical_table 25 1.35
    add_cell critical_table 50 1.20
    add_cell critical_table 50 1.25

    add_cell catapult_proxy 25 1.10
    add_cell catapult_proxy 25 1.15
    add_cell catapult_proxy 25 1.20
    add_cell catapult_proxy 25 1.25
    add_cell catapult_proxy 25 1.30
    add_cell catapult_proxy 50 1.25
    ;;
  *)
    echo "Unsupported GRID_PRESET=${GRID_PRESET}" >&2
    exit 1
    ;;
esac

START_INDEX=$((SLURM_ARRAY_TASK_ID * CANDIDATES_PER_NODE))
if (( START_INDEX < 0 || START_INDEX >= ${#CANDIDATES[@]} )); then
  echo "Unsupported SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}; start_index=${START_INDEX}; candidate_count=${#CANDIDATES[@]}" >&2
  exit 1
fi

echo "[$(date --iso-8601=seconds)] starting"
echo "repo_root=${REPO_ROOT}"
echo "data_dir=${DATA_DIR}"
echo "run_root=${RUN_ROOT}"
echo "warmup_target=${WARMUP_TARGET}"
echo "aux_lr_mult=${AUX_LR_MULT}"
echo "stop_step=${STOP_STEP}"
echo "micro_batch_size=${MICRO_BATCH_SIZE}"
echo "wandb_mode=${WANDB_MODE}"
echo "wandb_project=${WANDB_PROJECT}"
echo "wandb_group=${WANDB_GROUP}"
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  echo "wandb_api_key_configured=yes"
else
  echo "wandb_api_key_configured=no"
fi
echo "container_run_prefix=${CONTAINER_RUN_PREFIX}"
echo "candidates_per_node=${CANDIDATES_PER_NODE}"
echo "grid_preset=${GRID_PRESET}"
echo "candidate_count=${#CANDIDATES[@]}"
echo "start_index=${START_INDEX}"

run_candidate() {
  local candidate_index="$1"
  local gpu_slot="$2"
  IFS=, read -r schedule_kind warmup_steps muon_lr_mult seed <<<"${CANDIDATES[$candidate_index]}"

  local wandb_name="${schedule_kind}_w${warmup_steps}_m${muon_lr_mult}_seed${seed}_${SLURM_ARRAY_JOB_ID:-manual}_${candidate_index}"
  local log_file="${RUN_ROOT}/logs/${wandb_name}.log"
  local wandb_dir
  local wandb_cache_dir

  if [[ -n "${TMPDIR:-}" ]]; then
    wandb_dir="${TMPDIR}/wandb_${SLURM_ARRAY_JOB_ID:-manual}_${candidate_index}"
    wandb_cache_dir="${TMPDIR}/wandb_cache_${SLURM_ARRAY_JOB_ID:-manual}_${candidate_index}"
  else
    wandb_dir="${RUN_ROOT}/wandb/${SLURM_ARRAY_JOB_ID:-manual}_${candidate_index}"
    wandb_cache_dir="${RUN_ROOT}/wandb_cache/${SLURM_ARRAY_JOB_ID:-manual}_${candidate_index}"
  fi
  mkdir -p "${wandb_dir}" "${wandb_cache_dir}"

  echo "[$(date --iso-8601=seconds)] launching candidate_index=${candidate_index} gpu_slot=${gpu_slot} schedule_kind=${schedule_kind} warmup_steps=${warmup_steps} muon_lr_mult=${muon_lr_mult} seed=${seed} log_file=${log_file}"

  local train_cmd=(
    python -m torch.distributed.run --standalone --nproc_per_node=1
    records/track_3_optimization/experiments/train_gpt_critical_warmup.py
    --seed "${seed}"
    --schedule-kind "${schedule_kind}"
    --warmup-steps "${warmup_steps}"
    --warmup-target "${WARMUP_TARGET}"
    --muon-lr-mult "${muon_lr_mult}"
    --aux-lr-mult "${AUX_LR_MULT}"
    --stop-step "${STOP_STEP}"
    --data-dir "${DATA_DIR}"
    --micro-batch-size "${MICRO_BATCH_SIZE}"
    --wandb-mode "${WANDB_MODE}"
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${WANDB_GROUP}"
    --wandb-name "${wandb_name}"
  )

  if [[ -n "${CONTAINER_RUN_PREFIX}" ]]; then
    local quoted_cmd
    quoted_cmd="$(printf ' %q' "${train_cmd[@]}")"
    CUDA_VISIBLE_DEVICES="${gpu_slot}" WANDB_DIR="${wandb_dir}" WANDB_CACHE_DIR="${wandb_cache_dir}" \
      eval "${CONTAINER_RUN_PREFIX}${quoted_cmd}" 2>&1 | tee "${log_file}"
  else
    CUDA_VISIBLE_DEVICES="${gpu_slot}" WANDB_DIR="${wandb_dir}" WANDB_CACHE_DIR="${wandb_cache_dir}" \
      "${train_cmd[@]}" 2>&1 | tee "${log_file}"
  fi
}

cd "${REPO_ROOT}"

declare -a PIDS=()
for gpu_slot in $(seq 0 $((CANDIDATES_PER_NODE - 1))); do
  candidate_index=$((START_INDEX + gpu_slot))
  if (( candidate_index >= ${#CANDIDATES[@]} )); then
    break
  fi
  run_candidate "${candidate_index}" "${gpu_slot}" &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

echo "[$(date --iso-8601=seconds)] finished status=${status}"
exit "${status}"

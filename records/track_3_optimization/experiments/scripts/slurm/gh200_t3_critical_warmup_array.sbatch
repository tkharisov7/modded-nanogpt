#!/bin/bash

#SBATCH --job-name=t3-crit-warmup
#SBATCH --output=/capstor/store/cscs/uba/uba02/t3_critical_warmup/slurm/%x_%A_%a.out
#SBATCH --error=/capstor/store/cscs/uba/uba02/t3_critical_warmup/slurm/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --partition=normal
#SBATCH --account=uba02
#SBATCH --array=0-91%8

set -euo pipefail

mkdir -p /capstor/store/cscs/uba/uba02/t3_critical_warmup/slurm

export DATA_DIR="${DATA_DIR:-/capstor/store/cscs/uba/uba02/fineweb10B}"
export RUN_ROOT="${RUN_ROOT:-/capstor/store/cscs/uba/uba02/t3_critical_warmup}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-modded-nanogpt-track3}"
export WANDB_GROUP="${WANDB_GROUP:-T3_CRITICAL_WARMUP_GH200_${SLURM_ARRAY_JOB_ID}}"
export STOP_STEP="${STOP_STEP:-2720}"
export CANDIDATES_PER_NODE="${CANDIDATES_PER_NODE:-4}"
export REPO_ROOT="${REPO_ROOT:-/capstor/store/cscs/uba/uba02/modded-nanogpt}"

srun --environment=t3-nanogpt bash -lc '
  source /capstor/store/cscs/uba/uba02/venvs/t3-ngc-pt-25.06/bin/activate
  cd "${REPO_ROOT}"
  bash records/track_3_optimization/experiments/scripts/slurm/t3_critical_warmup_array_body.sh
'

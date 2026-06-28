#!/usr/bin/env bash
set -euo pipefail

# Reconstructed context for job 3761838.
# The exact original command was not fully recovered from shell history, but Slurm
# output confirms STOP_STEP=2720, CANDIDATES_PER_NODE=4, candidate_count=368,
# and the sparse full_v1 array task IDs listed below.

sbatch -A uba02 -p normal --time=04:00:00 --array=0,1,6,7,18,19,24,25,36,37,40,41,48,49,54,55,66,67,80,81,88,89 \
  --export=ALL,WANDB_API_KEY,STOP_STEP=2720,WANDB_MODE=online,CANDIDATES_PER_NODE=4,GRID_PRESET=full_v1 \
  records/track_3_optimization/experiments/scripts/slurm/gh200_t3_critical_warmup_array.sbatch


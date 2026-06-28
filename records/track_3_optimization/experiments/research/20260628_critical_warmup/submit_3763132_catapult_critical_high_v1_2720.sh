#!/usr/bin/env bash
set -euo pipefail

sbatch -A uba02 -p normal --time=04:00:00 --array=0-23%6 \
  --export=ALL,WANDB_API_KEY,STOP_STEP=2720,WANDB_MODE=online,CANDIDATES_PER_NODE=4,GRID_PRESET=catapult_critical_high_v1 \
  records/track_3_optimization/experiments/scripts/slurm/gh200_t3_critical_warmup_array.sbatch


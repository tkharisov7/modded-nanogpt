#!/usr/bin/env bash
set -euo pipefail

# Historical reproduction for the canceled no-cooldown probe.
# This was a bad direction: completed n=8 cells were far worse than WR #46.
# Keep this script for provenance, not as a recommended next run.

sbatch -A uba02 -p normal --time=04:00:00 --array=0-23%8 \
  --export=ALL,WANDB_API_KEY,STOP_STEP=2685,FINAL_SCHEDULE_STEPS=100000,FINAL_LR_POWER=1.0,WANDB_MODE=online,CANDIDATES_PER_NODE=4,GRID_PRESET=critical_short_nocool_v1 \
  records/track_3_optimization/experiments/scripts/slurm/gh200_t3_critical_warmup_array.sbatch


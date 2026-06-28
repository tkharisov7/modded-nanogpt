# Job 3763930: Short Critical Warmup Without Effective Cooldown

## Purpose

This job tested the user's question: can short critical warmup push a higher stable Muon LR if the final cooldown is removed or effectively delayed?

The run used `FINAL_SCHEDULE_STEPS=100000` and `FINAL_LR_POWER=1.0`, which effectively disabled the WR #46 PowerCool tail over the target step window. It also targeted `STOP_STEP=2685` to see whether a pre-WR pass was possible.

## Exact Submit Command

```bash
sbatch -A uba02 -p normal --time=04:00:00 --array=0-23%8 \
  --export=ALL,WANDB_API_KEY,STOP_STEP=2685,FINAL_SCHEDULE_STEPS=100000,FINAL_LR_POWER=1.0,WANDB_MODE=online,CANDIDATES_PER_NODE=4,GRID_PRESET=critical_short_nocool_v1 \
  records/track_3_optimization/experiments/scripts/slurm/gh200_t3_critical_warmup_array.sbatch
```

Returned Slurm job id:

```text
3763930
```

The job was canceled after partial completion when the completed cells showed the no-cooldown direction was much worse than WR #46.

Slurm stdout confirmed:

```text
stop_step=2685
candidates_per_node=4
grid_preset=critical_short_nocool_v1
candidate_count=96
```

## Grid

The preset is defined in:

```text
records/track_3_optimization/experiments/scripts/slurm/t3_critical_warmup_array_body.sh
```

It contains:

```text
critical_table: warmup_steps in {5, 10, 15, 20}
critical_table: muon_lr_mult in {1.20, 1.30, 1.40}
seeds: 0..7 for every cell
```

Total planned candidates:

```text
12 cells x 8 seeds = 96 runs
```

## Source Logs

Candidate logs:

```text
/capstor/store/cscs/uba/uba02/t3_critical_warmup/logs/*_3763930_*.log
```

Slurm logs:

```text
/capstor/store/cscs/uba/uba02/t3_critical_warmup/slurm/t3-crit-warmup_3763930_*.out
/capstor/store/cscs/uba/uba02/t3_critical_warmup/slurm/t3-crit-warmup_3763930_*.err
```

## Completion Before Cancellation

Parsed status before cancellation:

```text
logs: 64
SUMMARY_JSON records: 1056
final candidate logs: 64
Slurm outs: 16
Slurm status: {0: 8}
missing Slurm outs: 8
bad logs: 0
```

The queue was later verified clean after `scancel 3763930`.

## Completed Groups

| Step | schedule_kind | warmup_steps | muon_lr_mult | n | mean val_ema_loss | criterion | stable |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| 2685 | critical_table | 10 | 1.20 | 8 | 3.486661553 | -0.584527143 | 8/8 |
| 2685 | critical_table | 5 | 1.20 | 8 | 3.490278006 | -0.594756015 | 8/8 |
| 2685 | critical_table | 5 | 1.30 | 8 | 3.533496946 | -0.716997638 | 8/8 |
| 2685 | critical_table | 5 | 1.40 | 8 | 3.577174902 | -0.840537554 | 8/8 |
| 1375 | critical_table | 15 | 1.20 | 8 | 3.690671176 | -1.161553493 | 8/8 |
| 1375 | critical_table | 15 | 1.30 | 8 | 3.714640826 | -1.229349902 | 8/8 |
| 1375 | critical_table | 10 | 1.30 | 8 | 3.719105005 | -1.241976508 | 8/8 |
| 1375 | critical_table | 10 | 1.40 | 8 | 3.738708228 | -1.297422794 | 8/8 |

## Interpretation

This was a negative result. Removing or effectively delaying the final cooldown made the target-step loss far worse:

```text
best completed 2685 mean = 3.486661553
WR #46 2685 mean = 3.278630
```

The completed cells were stable in the sense that they produced finite summaries, but they were not competitive. The final cooldown should be treated as part of the WR #46 solution. The next attempt should layer a short critical warmup onto the accepted WR #46 tail schedule instead of replacing the tail.


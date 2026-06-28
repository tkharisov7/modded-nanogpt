# Job 3763132: Focused Catapult/Critical High-LR Grid

## Purpose

This job followed the sampled reference run with a focused high-LR grid around the most promising catapult/critical settings. It used `STOP_STEP=2720` and packed four single-GPU runs per GH200 node.

## Exact Submit Command

```bash
sbatch -A uba02 -p normal --time=04:00:00 --array=0-23%6 \
  --export=ALL,WANDB_API_KEY,STOP_STEP=2720,WANDB_MODE=online,CANDIDATES_PER_NODE=4,GRID_PRESET=catapult_critical_high_v1 \
  records/track_3_optimization/experiments/scripts/slurm/gh200_t3_critical_warmup_array.sbatch
```

Returned Slurm job id:

```text
3763132
```

Slurm stdout confirmed:

```text
stop_step=2720
candidates_per_node=4
grid_preset=catapult_critical_high_v1
candidate_count=96
```

## Grid

The preset is defined in:

```text
records/track_3_optimization/experiments/scripts/slurm/t3_critical_warmup_array_body.sh
```

It contains:

```text
critical_table: w25 m1.20, w25 m1.25, w25 m1.30, w25 m1.35, w50 m1.20, w50 m1.25
catapult_proxy: w25 m1.10, w25 m1.15, w25 m1.20, w25 m1.25, w25 m1.30, w50 m1.25
seeds: 0..7 for every cell
```

Total candidates:

```text
12 cells x 8 seeds = 96 runs
```

## Source Logs

Candidate logs:

```text
/capstor/store/cscs/uba/uba02/t3_critical_warmup/logs/*_3763132_*.log
```

Slurm logs:

```text
/capstor/store/cscs/uba/uba02/t3_critical_warmup/slurm/t3-crit-warmup_3763132_*.out
/capstor/store/cscs/uba/uba02/t3_critical_warmup/slurm/t3-crit-warmup_3763132_*.err
```

## Completion

Parsed status:

```text
logs: 96
SUMMARY_JSON records: 2496
final candidate logs: 96
Slurm outs: 24
Slurm status: {0: 24}
missing Slurm outs: 0
bad logs: 0
```

## Completed n=8 Groups at Step 2720

| Rank | schedule_kind | warmup_steps | muon_lr_mult | n | mean val_ema_loss | criterion | stable | min | max |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: |
| 1 | catapult_proxy | 25 | 1.15 | 8 | 3.276229233 | 0.010665341 | 8/8 | 3.274315596 | 3.277779341 |
| 2 | catapult_proxy | 25 | 1.30 | 8 | 3.276253819 | 0.010595799 | 8/8 | 3.274856329 | 3.278186560 |
| 3 | catapult_proxy | 25 | 1.20 | 8 | 3.276378095 | 0.010244294 | 8/8 | 3.274374485 | 3.277620077 |
| 4 | catapult_proxy | 25 | 1.10 | 8 | 3.276386797 | 0.010219680 | 8/8 | 3.275271177 | 3.278135300 |
| 5 | critical_table | 25 | 1.30 | 8 | 3.276713789 | 0.009294810 | 8/8 | 3.275025606 | 3.278897047 |
| 6 | catapult_proxy | 25 | 1.25 | 8 | 3.276771933 | 0.009130353 | 8/8 | 3.275256872 | 3.278212786 |
| 7 | critical_table | 25 | 1.20 | 8 | 3.276831716 | 0.008961260 | 8/8 | 3.274001837 | 3.278660536 |
| 8 | critical_table | 25 | 1.35 | 8 | 3.277015239 | 0.008442179 | 8/8 | 3.275124311 | 3.279289722 |
| 9 | critical_table | 25 | 1.25 | 8 | 3.277212203 | 0.007885082 | 8/8 | 3.275200605 | 3.278555632 |
| 10 | critical_table | 50 | 1.25 | 8 | 3.277522415 | 0.007007669 | 8/8 | 3.276179075 | 3.279073715 |
| 11 | catapult_proxy | 50 | 1.25 | 8 | 3.277798206 | 0.006227615 | 8/8 | 3.275898457 | 3.279150009 |
| 12 | critical_table | 50 | 1.20 | 8 | 3.277829856 | 0.006138095 | 8/8 | 3.275341749 | 3.280209303 |

## Best Cell Step Series

Best group at `2720`:

```text
catapult_proxy, warmup_steps=25, muon_lr_mult=1.15
```

Step series:

| Step | Mean val_ema_loss | Criterion | Pass |
| ---: | ---: | ---: | :--- |
| 2625 | 3.282448888 | -0.006926501 | no |
| 2700 | 3.277353942 | 0.007484181 | yes |
| 2705 | 3.277055860 | 0.008327287 | yes |
| 2710 | 3.276769757 | 0.009136506 | yes |
| 2715 | 3.276508361 | 0.009875846 | yes |
| 2720 | 3.276229233 | 0.010665341 | yes |

## Interpretation

This job found the best completed GH200-relative result so far:

```text
catapult_proxy w25 m1.15
mean = 3.276229233 at step 2720
```

It beat the job `3761838` GH200 baseline at the same `STOP_STEP=2720`:

```text
baseline mean = 3.276432037
delta = -0.000202805
```

It is not a new Track 3 WR because the accepted record #46 already passes at step `2690`, while this cell first formally passed at step `2700` in the recorded series.

This was a result-driven proxy success. It is useful evidence, but it is not the clean critical-LR-warmup thesis requested for the next iteration.


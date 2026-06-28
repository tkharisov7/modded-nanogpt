# Next Steps: Critical Warmup on Top of WR #46

## Main Decision

Do not continue the no-cooldown path. The data from job `3763930` says the WR #46 cooldown/tail dynamics are essential for being competitive near steps `2685..2690`.

The next search should keep WR #46 intact and add only a short hidden-Muon critical warmup.

## Proposed Implementation

Add a new deterministic schedule kind, for example:

```text
critical_ramp
```

Desired semantics:

```text
for step < warmup_steps:
  ramp hidden-Muon LR multiplier from low_start_mult to muon_lr_mult

for step >= warmup_steps:
  keep hidden-Muon LR multiplier at muon_lr_mult

then let the existing WR #46 global LR cooldown apply normally
```

Keep the final schedule:

```text
FINAL_SCHEDULE_STEPS=2900
FINAL_LR_POWER=1.2
```

This distinguishes the next experiment from the current `critical_table`, whose multiplier shape is non-monotone:

```text
0.12 -> 0.45 -> 1.15 -> 0.95 -> 1.05 -> 1.00
```

## First Grid

Use a small but fair grid:

```text
schedule_kind=critical_ramp
warmup_steps in {5, 10, 15, 20, 25}
muon_lr_mult in {1.10, 1.15, 1.20, 1.25, 1.30}
seeds 0..7
STOP_STEP=2685
```

This is `25` cells x `8` seeds = `200` candidate runs. With `CANDIDATES_PER_NODE=4`, this is `50` Slurm array tasks. A concurrency cap of `%8` or `%10` is reasonable if allocation pressure permits.

## Possible Narrow Grid

If queue budget is tight, start with:

```text
warmup_steps in {5, 10, 15}
muon_lr_mult in {1.15, 1.20, 1.25, 1.30}
seeds 0..7
STOP_STEP=2685
```

This is `12` cells x `8` seeds = `96` runs, matching the size of job `3763132`.

## Decision Rule

At `STOP_STEP=2685`, require:

```text
n=8 mean val_ema_loss <= 3.278585786
```

That would pass the Track 3 acceptance formula at step `2685`, beating WR #46's accepted step `2690`.

If no `2685` cell passes, inspect whether a cell materially improves the WR #46 2685 mean:

```text
WR #46 2685 mean = 3.278630
WR #46 2685 criterion = 0.00387
```

Any cell near or below this line deserves a `2690` confirmation. Cells that first pass only at `2700` are not record candidates.

## Suggested Submit Shape After Adding critical_ramp

```bash
sbatch -A uba02 -p normal --time=04:00:00 --array=0-49%8 \
  --export=ALL,WANDB_API_KEY,STOP_STEP=2685,WANDB_MODE=online,CANDIDATES_PER_NODE=4,GRID_PRESET=critical_ramp_wr46_v1 \
  records/track_3_optimization/experiments/scripts/slurm/gh200_t3_critical_warmup_array.sbatch
```

Do not add `FINAL_SCHEDULE_STEPS=100000` or `FINAL_LR_POWER=1.0`; those were the no-cooldown controls that failed in job `3763930`.


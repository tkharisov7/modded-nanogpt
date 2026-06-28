# 2026-06-28 Critical Warmup Daint Runs

This directory records the Daint experiments launched for the Track 3 critical-warmup search. The target was to find a new speedrun record by applying a short, deterministic Muon LR warmup only to hidden-matrix Muon parameter groups, while keeping the accepted Track 3 training stack intact unless explicitly tested otherwise.

## Record Target

Accepted Track 3 record #46 is in:

```text
records/track_3_optimization/results/20260619_cwd_rowfloor_tailema/
```

The accepted claim is:

```text
n=8 seeds 0-7
step 2690
mean val_ema_loss = 3.278329
(3.28 - mean) * sqrt(8) = 0.00473 >= 0.004
```

The same record reports:

| Step | Mean val_ema_loss | Criterion | Result |
| ---: | ---: | ---: | :--- |
| 2680 | 3.278941 | 0.00299 | fail |
| 2685 | 3.278630 | 0.00387 | fail |
| 2690 | 3.278329 | 0.00473 | pass |
| 2695 | 3.278055 | 0.00550 | pass |
| 2700 | 3.277787 | 0.00626 | pass |

The acceptance formula is:

```text
(3.28 - avg_loss) * sqrt(num_runs) >= 0.004
```

For `n=8`, this requires average validation EMA loss <= `3.278585786`.

## Baseline Stack To Preserve

The accepted solution stack is Tail-EMA readout, RowFloor, Cautious Weight Decay, SOAP-Muon, and PowerCool. The reference Muon LR is `0.0375`.

The clean critical-warmup thesis should preserve:

- Architecture.
- Dataset and validation.
- Global batch and micro-batch semantics.
- Tail-EMA semantics.
- RowFloor.
- CWD.
- SOAP implementation.
- Final benchmark rules.
- Deterministic fixed schedules; no online LR probing in a final PR.

The important lesson from the no-cooldown probe is that the WR #46 tail schedule is part of the record. Removing or effectively delaying the cooldown made the run much worse at the target step.

## Runtime Context

Repository on Daint:

```text
/capstor/store/cscs/uba/uba02/modded-nanogpt
```

Dataset:

```text
/capstor/store/cscs/uba/uba02/fineweb10B
```

Durable run logs:

```text
/capstor/store/cscs/uba/uba02/t3_critical_warmup/logs
```

Slurm stdout/stderr:

```text
/capstor/store/cscs/uba/uba02/t3_critical_warmup/slurm
```

Aggregates:

```text
/capstor/store/cscs/uba/uba02/t3_critical_warmup/aggregate
```

Container and runtime:

```text
EDF: t3-nanogpt
venv: /capstor/store/cscs/uba/uba02/venvs/t3-ngc-pt-25.06
Slurm account: uba02
partition: normal
packing: CANDIDATES_PER_NODE=4, one single-GPU run per GH200 GPU
W&B project: tkharisov7/modded-nanogpt-track3
```

W&B is monitoring only. CAPSTOR stdout logs and parsed `SUMMARY_JSON` records are source of truth.

## Launched Jobs

| Job | Preset | Status | Purpose | Outcome |
| ---: | :--- | :--- | :--- | :--- |
| 3761838 | sparse `full_v1` sample | complete | Reference sample: baseline plus selected linear, critical_table, and catapult_proxy cells at step 2720 | Baseline mean `3.276432037`; no critical-table cell beat it |
| 3763132 | `catapult_critical_high_v1` | complete | Focused catapult/critical high-LR grid at step 2720 | Best GH200-relative result: `catapult_proxy w25 m1.15`, mean `3.276229233`, but not a new WR |
| 3763930 | `critical_short_nocool_v1` | canceled after partial completion | Short critical warmups with the final cooldown effectively disabled | Bad direction; losses around `3.49` at step 2685 for completed cells |

## Current Conclusion

No launched job established a new accepted Track 3 WR.

The best completed result was:

```text
job 3763132
catapult_proxy, warmup_steps=25, muon_lr_mult=1.15
n=8 at step 2720
mean val_ema_loss = 3.276229233
criterion = 0.010665341
```

This beat the GH200 baseline from job `3761838` at the same `STOP_STEP=2720` by about `0.000202805`, but it was still too late for the official speedrun target. Its first formal pass in the recorded step series was step `2700`, whereas accepted WR #46 passes at step `2690`.

The no-cooldown experiment is strong negative evidence against removing PowerCool for the target-step search. The next search should put critical warmup on top of the accepted WR #46 tail schedule, not replace it.

## Recommended Next Direction

1. Keep the WR #46 tail intact:

```text
FINAL_SCHEDULE_STEPS=2900
FINAL_LR_POWER=1.2
MUON_LR=0.0375 base, varied by hidden-matrix Muon LR multiplier
Tail-EMA unchanged
RowFloor unchanged
CWD unchanged
```

2. Add a clean monotone critical warmup schedule for the thesis. The current `critical_table` has a non-monotone profile:

```text
0.12 -> 0.45 -> 1.15 -> 0.95 -> 1.05 -> 1.00
```

That shape is not the clean "short critical LR warmup, then hold higher stable LR" story. A better next implementation is a deterministic `critical_ramp` schedule:

```text
for steps 0..warmup_steps: ramp from low initial multiplier to target multiplier
after warmup: hold target multiplier until the existing WR #46 cooldown takes over
```

3. Test shorter warmups and aggressive but plausible Muon LR multipliers:

```text
warmup_steps: 5, 10, 15, 20, 25
muon_lr_mult: 1.10, 1.15, 1.20, 1.25, 1.30, optionally 1.35
STOP_STEP: 2685 for first pass, then 2690 confirmation
```

4. Compare only fair groups:

- Same step.
- Same seed set.
- Prefer full `n=8`.
- Use `SUMMARY_JSON` records from CAPSTOR logs.

5. WR bar:

To beat accepted WR #46, the candidate must pass at step `2685` or earlier, or produce a stronger accepted result at the same step under the final benchmark rules. A result that first passes at `2700` is useful evidence but not a new record.

## Useful Commands

Aggregate a job:

```bash
JOB=3763132
python records/track_3_optimization/experiments/tools/aggregate_t3_critical_warmup.py \
  --log-dir /capstor/store/cscs/uba/uba02/t3_critical_warmup/logs \
  --glob "*_${JOB}_*.log" \
  --out /capstor/store/cscs/uba/uba02/t3_critical_warmup/aggregate/${JOB}_aggregate.csv
```

Count completed summaries:

```bash
JOB=3763132
grep -h 'SUMMARY_JSON' /capstor/store/cscs/uba/uba02/t3_critical_warmup/logs/*_${JOB}_*.log | wc -l
```

Check failures:

```bash
JOB=3763132
grep -R 'Traceback\|RuntimeError\|nan\|inf' /capstor/store/cscs/uba/uba02/t3_critical_warmup/logs/*_${JOB}_*.log
```

Check queue:

```bash
squeue -u tkharisov7
```


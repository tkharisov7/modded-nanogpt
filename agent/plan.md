# Technical Plan: Track 3 Critical-Warmup LR Push

## Objective

Prepare a new Track 3 optimization PR under `records/track_3_optimization` that tests whether a critical-warmup-inspired schedule can safely push the stable learning rates of the current Track 3 algorithm above the accepted record lineage.

The working hypothesis is:

> The early trajectory is currently limited by stability, not by late optimization capacity. A warmup schedule shaped by the critical-warmup experiments can reach a higher effective stable LR without early divergence, giving the current SOAP-Muon / Tail-EMA / RowFloor / CWD stack a lower statistically valid crossing step.

This plan was approved for research-harness implementation. Do not create a final PR result directory until a candidate passes smoke and sweep validation.

## Implementation Status

- Phase 1: Research candidate script with Muon-only warmup knobs. Status: completed.
- Phase 2: W&B sweep schema and stdout aggregation tooling. Status: completed.
- Phase 3: Local GPU `2,3,4,5` launch scripts. Status: completed.
- Phase 4: GH200 Slurm/container templates. Status: completed.
- Phase 5: Static verification and dry-run checks. Status: completed.
- Phase 6: Real training smoke and GH200 sweeps. Status: partially completed; local one-step and step-125 smoke tests passed on GPUs `2,3,4,5` with `MICRO_BATCH_SIZE=16` and W&B offline. Full-length local sweeps are not started because RTX 4090 memory requires reduced microbatching and the step-125 smoke already takes about 10 minutes. CSCS access, account, partition, quota, and CAPSTOR data are resolved. GH200 sweeps remain blocked only on the CSCS Python/container launcher and W&B API-key configuration.

## Repo Context

Primary Track 3 target:

- Current accepted record in `records/track_3_optimization/README.md` is result `#46`.
- Result `#46` reaches 3.28 at `2690` steps with `n=8`.
- Source artifact: `records/track_3_optimization/results/20260619_cwd_rowfloor_tailema/train_gpt_cwd_SOTA.py`.
- Record stack:
  - SOAP-Muon on all hidden matrices, `precondition_frequency=1`, `SOAP_BETA2=0.90`.
  - Muon LR `MUON_LR=0.0375`.
  - Aux AdamW / Adam LR groups:
    - embedding `0.3`
    - projection head `1/320`
    - aux groups `0.01`
  - PowerCool LR schedule with `FINAL_SCHEDULE_STEPS=2900`, `FINAL_LR_POWER=1.2`.
  - Muon momentum warmup `0.85 -> 0.95` over `300` steps, cooldown over final `200` schedule steps.
  - Tail-EMA readout: `TAILEMA_TAU=150`, start `2400`, end `2900`, blend `0.6`, token embedding excluded.
  - RowFloor enabled, `TARGET_UW=0.3825`, `ROWFLOOR_RHO=1.0`.
  - Post-pin CWD `0.025`.

Critical warmup reference code lives in the parent repo at `../code`.

Important files studied:

- `../code/src/loss_landscape/experiments/warmup_threshold_runner.py`
- `../code/src/loss_landscape/experiments/train_runner.py`
- `../code/experiments/configs/I2_critical_lr_threshold_v1.json`
- `../code/experiments/configs/I2_critical_lr_threshold_minimal_3gpu_3seed_v1.json`
- `../code/experiments/configs/I2_effective_lr_threshold_3gpu_7seed_10k_v1.json`
- `../code/scripts/run_warmup_threshold_minimal_3gpu_3seed.sh`
- `../code/scripts/run_warmup_threshold_effective_lr_3gpu_7seed.sh`
- `../code/scripts/slurm/i2_max_stable_lr_full_15task_7seed_10k_v1_body.sh`
- `../code/scripts/slurm/rtx4090_i2_max_stable_lr_full_15task_7seed_10k_v1.sbatch`
- `../code/scripts/slurm/l40s_i2_max_stable_lr_full_15task_7seed_10k_v1.sbatch`
- `agent/critical_warmup.pdf`

Note: `agent/critical_warmup.pdf` is image-backed. Text extraction returns empty pages, so the report was reviewed by rendering the PDF pages to images. The numerical values below are approximate visual readings from the plots, not exported W&B tables.

Critical warmup PDF takeaways:

- The report's stated goal is to find the best post-warmup LR, not the largest non-diverging LR. This matters because gradient clipping can let very large LRs remain numerically stable while producing worse final loss.
- The search rule in the report tests candidate post-warmup LRs in increasing order, runs `7` seeds per candidate, scores by median final train loss, accepts a larger LR only if median loss improves, stops at the first non-improving/diverging LR, then binary-searches the bracket.
- In the 20k/10k views, critical warmup with a 500-step or 1000-step horizon supports visibly higher post-warmup efficient LR than linear/no-warmup baselines. The 10k LR plot shows `W_crit^500` near `0.62`, `W_crit^100` near `0.49`, and linear/no-warmup variants around `0.41..0.44`.
- Accuracy curves show critical warmup competitive with, and often ahead of, the linear/no-warmup baselines by the same budget. Very long linear warmup can lag early progress.
- In the 5k short-warmup view, `W_crit^25`, `W_crit^50`, and `W_crit^100` are all competitive in accuracy; linear `W=50/100` appears weaker in that slice. The catapult trace has high variance and should be treated as lower priority unless replicated.
- For Track 3, prioritize fixed critical-derived short warmups before catapult proxies. Search `W in {25, 50, 100}` locally, keep `{200, 300, 500}` for follow-up if the short schedules are too sharp or noisy.

## Key Constraint

Do not put online critical-LR probing into the final Track 3 submission.

The parent critical-warmup code estimates critical LR by extra loss evaluations / virtual steps. That is acceptable for research sweeps, but a Track 3 submission must keep the benchmark contract:

- fixed architecture
- fixed dataset and batch size
- one forward-backward pass per optimizer step
- no per-run validation-based early stopping
- all reproducibility code embedded in the submitted log/source

Therefore:

- Use critical-LR and persistent-catapult experiments only to derive candidate fixed schedules.
- The final PR script should contain deterministic closed-form or table-driven LR multipliers, not online critical-LR measurement.

## Critical Warmup Mechanics To Port Conceptually

The parent repo has two relevant warmup surfaces.

### I1 train-runner warmups

Implemented in `train_runner.py`:

- `linear`
- `critical_lr`
- `directional_sharpness`
- `top_eig`
- `grad_top10_rayleigh`
- `grad_top1_rayleigh`

The metric schedules resolve a candidate warmup LR from current metrics and fall back to the previous LR or `stable_lr / warmup_steps`.

Relevant behavior:

- `critical_lr`: uses `critical_sharpness`, then `lr = 2 / critical_sharpness`.
- `directional_sharpness`: uses `min(2 / directional_sharpness, critical_lr)`.
- `grad_top10_rayleigh`: uses `min(2 / grad_top10_rayleigh_contrib, critical_lr)`.
- `top_eig`: uses `2 / raw_top_eig_1`.

These are too expensive to run inside Track 3, but they provide schedule-shape candidates.

### I2 threshold runner

Implemented in `warmup_threshold_runner.py`.

Schedule kinds:

- `linear`
- `critical_lr`
- `critical_lr_half`
- `persistent_catapult`

Search objectives:

- `median_final_loss`
- `max_stable_lr`
- `max_stable_lr_avg_tail`
- `max_stable_lr_min_tail`
- `max_stable_lr_median_tail`
- `max_stable_lr_all`

Useful design choices to reuse:

- Search over schedule kind, warmup length, stable LR, and seeds.
- Decide stability from tail statistics, not from a single noisy endpoint.
- Compare linear warmup against critical warmup at the same post-warmup LR.
- Use smoke configs locally before full sweeps.
- For full Slurm sweeps, use array tasks split by schedule/objective.

## Proposed Track 3 Experiment Design

### What To Change

Start from `train_gpt_cwd_SOTA.py` and make only schedule/hyperparameter changes.

The minimal candidate PR should not alter:

- architecture
- dataset paths or format
- global batch size `8 * 64 * 1024`
- validation definition
- Tail-EMA readout semantics
- RowFloor implementation
- CWD implementation
- SOAP preconditioning implementation
- EMA-Nesterov implementation
- model initialization

Primary change surface:

- Introduce a deterministic warmup multiplier `warmup_mult(step)` that applies before the existing PowerCool schedule.
- Search higher base LRs for Muon and selected aux groups.
- Keep late schedule constants initially fixed at `FINAL_SCHEDULE_STEPS=2900`, `FINAL_LR_POWER=1.2`, so any gain is attributable to higher stable LR / early trajectory changes.

Suggested LR application:

```text
group_lr(step) = initial_lr * lr_family_mult * warmup_mult_family(step) * powercool_mult(step)
```

where `powercool_mult(step)` is equivalent to the current `_lr(step, initial_lr, power_c, power) / initial_lr`.

Apply the critical warmup only to Muon hidden-matrix groups in the first implementation. Keep embedding, projection head, and auxiliary Adam/AdamW groups on the baseline #46 schedule unless a later ablation explicitly tests aux warmup.

### Candidate Warmup Families

Evaluate these in order.

1. Baseline:
   - Current #46 schedule, no added warmup multiplier.
   - Needed to calibrate local hardware and code changes.

2. Linear warmup:
   - `warmup_mult = step / W` for `step <= W`, else `1`.
   - Test `W in {25, 50, 100}` first.
   - Keep `W in {200, 300, 500}` as follow-up values only if shorter warmups are noisy or unstable.
   - Purpose: isolate whether any ramp helps before using critical-inspired shapes.

3. Critical-derived table warmup:
   - Offline critical-warmup sweeps produce a median or conservative critical LR trace.
   - Convert that trace into a monotone table of LR multipliers for the first `W` steps.
   - Clip by a target max multiplier `M`.
   - Store as a hardcoded short table in the Track 3 research script, later simplify to a closed-form approximation if possible.
   - Highest initial priority: `W in {25, 50, 100}` with Muon LR multipliers around `1.05..1.20`.
   - Secondary priority: `W in {200, 300, 500}` if the best short critical schedule improves speed but has seed variance.

4. Persistent-catapult-inspired warmup:
   - Approximate the I2 `persistent_catapult` idea without online critical-LR probing.
   - Use a short high-LR plateau or staircase in the early phase, then settle to the stable higher LR.
   - Candidate shape:
     - bootstrap for `0..B`
     - aggressive plateau/ramp for `B..W`
     - fixed post-warmup LR after `W`
   - Must be fixed before each run and identical across seeds.
   - Lower priority than critical-derived warmup because the rendered report shows catapult-like runs with weaker or noisier accuracy.

5. Family-split warmup:
   - Default from the start.
   - Muon hidden matrices: critical-derived warmup and higher LR multiplier.
   - embedding/proj/aux Adam groups: baseline #46 LR schedule and baseline LR values.
   - Rationale: hidden matrix Muon path has the hypothesized stability headroom; aux groups may destabilize loss quickly.

### LR Search Axes

Treat LR multipliers as relative to current #46 values.

Initial local search:

- Muon LR multiplier: `{1.00, 1.05, 1.10, 1.15, 1.20}`
- Aux Adam shared multiplier: fixed at `1.00` for the first pass.
- Warmup length: `{25, 50, 100}` for the first pass; `{200, 300, 500}` only for follow-up.
- Schedule kind: `{baseline, linear, critical_table, catapult_proxy}`

Full sweep refinement:

- Muon LR multiplier around best local result in `0.025` or `0.05` increments.
- Aux multipliers only in a later ablation if Muon-only warmup is promising but appears bottlenecked by unchanged aux groups.
- Stop expanding once gains are below noise or divergence boundary is clear.

Keep `FINAL_SCHEDULE_STEPS=2900` during the first sweep. Only after finding a stable higher LR should we test whether schedule horizon can move earlier.

## Instrumentation Plan

The final PR should stay simple, but research runs need better observability.

Add research-only logging in the experimental script or wrapper:

- seed
- schedule kind
- warmup target, initially `muon_hidden_only`
- warmup length
- Muon LR multiplier
- aux LR multiplier, initially fixed at `1.0`
- current group LRs at each validation
- first non-finite train loss step, if any
- validation loss and Tail-EMA validation loss
- first per-seed crossing below `3.28`
- fixed candidate step statistics, especially `2680..2720`

Enable W&B in research and CSCS sweeps so runs can be monitored online. Keep stdout logs and parsed local artifacts as the source of truth for final validation and PR evidence.

## W&B Sweep Schema

For development and CSCS sweeps, create a W&B schema modeled on the parent I2 configs, but targeted to Track 3 stdout-based runs.

Proposed sweep file:

```text
records/track_3_optimization/experiments/sweeps/T3_critical_warmup_lr_push_v1.json
```

Logical schema:

```json
{
  "name": "T3_critical_warmup_lr_push_v1",
  "method": "grid",
  "metric": {
    "name": "val_ema_loss_at_claim_step",
    "goal": "minimize"
  },
  "parameters": {
    "study_id": {"values": ["T3_CRITICAL_WARMUP_LR_PUSH_V1"]},
    "study_claim": {
      "values": [
        "Fixed critical-warmup-derived schedules allow higher stable Track 3 learning rates without online critical-LR probing."
      ]
    },
    "base_record": {"values": ["20260619_cwd_rowfloor_tailema"]},
    "schedule_kind": {"values": ["baseline", "linear", "critical_table", "catapult_proxy"]},
    "warmup_target": {"values": ["muon_hidden_only"]},
    "warmup_steps": {"values": [25, 50, 100]},
    "muon_lr_mult": {"values": [1.0, 1.05, 1.1, 1.15, 1.2]},
    "aux_lr_mult": {"values": [1.0]},
    "final_schedule_steps": {"values": [2900]},
    "final_lr_power": {"values": [1.2]},
    "stop_step": {"values": [2720]},
    "seed": {"values": [0, 1, 2, 3, 4, 5, 6, 7]},
    "nproc_per_node": {"values": [1]}
  }
}
```

Use W&B for sweep metadata, dashboards, and live monitoring. Use stdout logs and aggregation artifacts as the final benchmark evidence.

If the short-warmup pass finds a stable but noisy improvement, run a second sweep with `warmup_steps` in `[200, 300, 500]` for the top schedule families only.

Expected logged summary metrics:

- `val_ema_loss_at_2680`
- `val_ema_loss_at_2685`
- `val_ema_loss_at_2690`
- `val_ema_loss_at_2695`
- `val_ema_loss_at_2700`
- `val_ema_loss_at_claim_step`
- `raw_val_loss_at_claim_step`
- `first_val_ema_below_3p28`
- `train_nonfinite_step`
- `completed_steps`
- `wall_seconds`
- `stable`

Aggregation script should compute:

```text
(3.28 - mean_loss) * sqrt(n)
```

for each fixed step and each non-cherry-picked seed set.

## Local Development Plan: GPUs 2-5

Use local GPUs `2,3,4,5` for smoke and small grids.

Assumptions:

- The Track 3 script can run with `torchrun --standalone --nproc_per_node=1` on one GPU.
- Single-GPU GH200/H100/A40 style runs are acceptable for research triage because Track 3 global batch is world-size independent, but final evidence should report hardware clearly.
- Local data exists at `data/fineweb10B` or the script is made configurable by environment variable in the research branch.

Local phase 0: reproduce baseline.

- Run seeds `0..3` with the unmodified #46 script.
- Stop at `STOP_STEP=2720` or `2750` for quick validation.
- Confirm local mean is within expected hardware offset.
- If local baseline is materially shifted, compare deltas only, not absolute record claims.

Local phase 1: smoke schedules.

- One seed, one GPU per schedule:
  - GPU 2: baseline
  - GPU 3: linear warmup
  - GPU 4: critical-table warmup
  - GPU 5: catapult-proxy warmup
- Use a conservative Muon LR multiplier first, e.g. `muon_lr_mult=1.05`, with `aux_lr_mult=1.00`.
- Stop early on non-finite train loss.

Local phase 2: 4-GPU grid.

- Run four independent single-GPU jobs at a time.
- Prefer breadth over seed count:
  - seed `0` for broad divergence map
  - seeds `0..3` for promising cells
  - seeds `0..7` only after a clear signal

Local acceptance to scale:

- No non-finite loss through `2720`.
- Mean delta on seeds `0..3` improves `val_ema` by at least `0.0004` around `2680..2700`, or reaches the same loss at least `10` steps earlier.
- No single seed is catastrophically worse relative to baseline.

## CSCS Daint GH200 Full Sweep Plan

Use CSCS Daint GH200s for full seed sweeps and final PR evidence.

Confirmed CSCS access/resource details:

- SSH alias `daint` works for user `tkharisov7`.
- Project/account: `uba02`.
- Relevant partitions from `scontrol show partitions`: `normal`, `low`, `debug`, `xfer`.
- Use `normal` for ordinary sweep jobs unless a short debug probe is needed.
- CAPSTOR storage path: `/capstor/store/cscs/uba/uba02`.
- FineWeb shard cache is installed at `/capstor/store/cscs/uba/uba02/fineweb10B`.
- Verified CAPSTOR dataset footprint: `21` `.bin` files, `3.9G`, with `fineweb_val_000000.bin` and train shards `fineweb_train_000001.bin` through `fineweb_train_000020.bin`.
- HPC quota shown in the portal: `950` node-hours quarterly, with `289.73` used at the time of planning. Treat the remaining budget as node-hour constrained, not GPU-hour constrained.

The parent Slurm pattern to copy:

- small `.sbatch` file sets partition/resources/array size
- body script maps `SLURM_ARRAY_TASK_ID` to one schedule/hyperparameter cell
- job writes stdout/stderr to a stable logs directory
- W&B dir/cache go under `$TMPDIR`
- Python/env setup is explicit
- array throttling controls concurrent jobs

Create GH200-specific scripts only after local smoke passes.

Proposed files:

```text
records/track_3_optimization/experiments/scripts/run_t3_critical_warmup_single.sh
records/track_3_optimization/experiments/scripts/slurm/gh200_t3_critical_warmup_array.sbatch
records/track_3_optimization/experiments/scripts/slurm/t3_critical_warmup_array_body.sh
```

CSCS software/data setup:

- Use a Docker/container image for the Python/PyTorch/runtime environment unless a native CSCS module stack is known to reproduce the local environment.
- Build the image from the repo/runtime requirements and publish it to a registry that CSCS can pull from, or use the CSCS-supported image import path. The image should contain code dependencies, not run outputs or the dataset.
- Do not bake `fineweb10B` into the Docker image. The dataset is large, should be persistent, and should live on CSCS storage.
- Use the CAPSTOR allocation:

```text
/capstor/store/cscs/uba/uba02
```

- Proposed dataset location:

```text
/capstor/store/cscs/uba/uba02/fineweb10B
```

- Proposed project/log location:

```text
/capstor/store/cscs/uba/uba02/t3_critical_warmup/
```

Data status:

- The first-pass `20` train shards plus validation shard are already present on CAPSTOR.
- Before launching sweeps, run a quick count/size sanity check on Daint:

```bash
find /capstor/store/cscs/uba/uba02/fineweb10B -maxdepth 1 -type f -name '*.bin' | wc -l
du -sh /capstor/store/cscs/uba/uba02/fineweb10B
```

- Mount or expose `/capstor/store/cscs/uba/uba02` inside the container so the training script sees `DATA_DIR=/capstor/store/cscs/uba/uba02/fineweb10B`.
- Keep W&B cache/temp directories under `$TMPDIR`; keep durable stdout logs and aggregation CSVs under `/capstor/store/cscs/uba/uba02/t3_critical_warmup/`.

CSCS body script responsibilities:

- launch the chosen Docker/container image
- set `REPO_ROOT`
- set `DATA_DIR=/capstor/store/cscs/uba/uba02/fineweb10B`
- enable W&B for online monitoring
- map array ID to:
  - schedule kind
  - warmup steps
  - warmup target, initially `muon_hidden_only`
  - Muon LR multiplier
  - seed
  - stop step
- run:

```bash
torchrun --standalone --nproc_per_node=1 records/track_3_optimization/experiments/train_gpt_critical_warmup.py --seed "${SEED}"
```

or the equivalent approved research script.

Slurm resource shape:

- one Daint node per array task
- four independent single-GPU training runs per node, one per visible GPU
- array over packed groups of four `(candidate cell, seed)` runs
- current first-pass grid has `368` runs, packed into `92` array tasks
- current template throttles to `8` concurrent nodes, i.e. up to `32` concurrent GPU runs
- reduce array range or throttle for smoke/debug submissions before launching the full grid
- write durable logs to `/capstor/store/cscs/uba/uba02/t3_critical_warmup/logs/${SLURM_ARRAY_JOB_ID}` and later copy the final accepted run logs into the PR result directory

Do not submit the full `0-91` array as the first CSCS test. Start with one or two packed array tasks, short `STOP_STEP`, and W&B online/offline behavior verified.

## Sweep Stages

### Stage A: Baseline Calibration

Runs:

- #46 baseline
- seeds `0..7`
- same stop/eval schedule as candidate runs

Outputs:

- baseline mean at `2680, 2685, 2690, 2695, 2700, 2705, 2710, 2720`
- local or GH200 hardware offset vs published #46
- parsed logs and aggregation CSV

Exit criteria:

- baseline reproduces within expected noise
- parser and stats code agree with README formula

### Stage B: Local Candidate Discovery

Runs:

- seeds `0` or `0..3`
- all schedule families
- conservative LR multipliers
- first-pass warmup lengths `25, 50, 100`

Outputs:

- divergence map
- promising LR ranges
- schedules to drop
- decision on whether longer `200, 300, 500` warmups deserve a local follow-up

Exit criteria:

- at least one candidate improves early target-zone loss without divergence
- or evidence shows hypothesis is weak and no GH200 full sweep should be spent

### Stage C: GH200 Medium Sweep

Runs:

- seeds `0..7`
- top 5-10 candidate cells from local discovery
- plus baseline

Outputs:

- fixed-step significance table
- per-seed crossing table
- paired deltas vs baseline

Exit criteria:

- candidate clears `(3.28 - mean) * sqrt(n) >= 0.004` at a step below `2690`.
- If no candidate is on track for a sub-`2690` record, do not spend the final sweep budget on merely improving the `2690` margin unless the result is needed as diagnostic evidence.

### Stage D: Final Refinement

Only if Stage C has signal.

Possible refinements:

- adjust `STOP_STEP` and dense validation steps around the new crossing
- small LR multiplier refinement
- simplify critical-table schedule into a short closed-form formula
- rerun `n=8`, then `n=10` or `n=16` if the margin is close

Do not change the stopping criterion after seeing per-run validation. The claimed step must be fixed for the whole seed set.

### Stage E: PR Packaging

Create a submission directory such as:

```text
records/track_3_optimization/results/YYYYMMDD_critical_warmup_lr_push/
```

Expected files:

- `README.md`
- `train_gpt_critical_warmup.py`
- `run.sh`
- logs for all seeds
- optional figures
- optional aggregation CSV

README must include:

- claim step
- seed set
- per-seed values at target-zone steps
- mean and significance formula
- exact schedule definition
- statement that schedule is fixed across seeds
- statement that no online critical-LR probing is used
- hardware and world size
- comparison against #46 baseline
- reproduction command

## Validation And Statistical Rules

Use the Track 3 acceptance formula:

```text
(3.28 - avg_loss) * sqrt(num_runs) >= 0.004
```

For reference:

- `n=8` requires average loss `<= 3.278586`
- `n=10` requires average loss `<= 3.278735`
- `n=16` requires average loss `<= 3.279000`
- `n=20` requires average loss `<= 3.279106`

Primary target:

- beat `2690` with `n >= 8`.

Secondary target:

- if no sub-`2690` record appears, preserve the best `2690` margin result as diagnostic evidence, but do not package it as the main objective of this PR.

Avoid p-hacking:

- decide seed set before running final evidence
- decide candidate step grid before final evidence
- report failed adjacent steps if claiming first pass
- do not choose per-seed stopping steps

## Code Change Plan After Approval

When implementation is approved, keep changes isolated.

1. Copy the current #46 script into a research candidate file.
2. Add environment/CLI controls for research only:
   - `--schedule-kind`
   - `--warmup-steps`
   - `--muon-lr-mult`
   - `--warmup-target`, defaulting to `muon_hidden_only`
   - `--aux-lr-mult`, fixed to `1.0` unless a later ablation explicitly enables aux LR tuning
   - `--stop-step`
   - optional `--data-dir`
3. Implement fixed warmup functions.
4. Add stdout lines that make parsing unambiguous.
5. Add a parser/aggregator script outside the final submission script.
6. Add local launch scripts for GPUs `2,3,4,5`.
7. Add GH200 Slurm scripts.
8. Run local smoke.
9. Run GH200 sweeps.
10. Freeze the winning schedule into a clean submission script with minimal knobs.

## Risks

Main scientific risk:

- Critical warmup may help full-batch GD in the parent repo but not the Track 3 SOAP-Muon stack, whose early dynamics are heavily shaped by SOAP, RowFloor, radius pinning, EMA-Nesterov, and PowerCool.

Benchmark risk:

- Online critical-LR measurement would likely violate the one-forward-backward-per-step spirit. Keep it out of the final PR.

Engineering risk:

- The #46 script has hardcoded paths and hyperparameters. Research knobs are useful for sweeps, but final PR code should be simplified after tuning.

Hardware risk:

- GH200, A40, H100, and local GPUs can have small validation offsets. Use paired baseline runs on the same hardware for decisions.

Statistical risk:

- A small apparent gain at `n=4` can vanish at `n=8+`. Scale only candidates with paired, consistent seed behavior.

Complexity risk:

- A table-driven critical schedule can look arbitrary. Prefer a simple formula in the final PR if performance is close.

## Resolved Decisions Before Implementation

- First warmup target: apply critical warmup only to Muon hidden-matrix LR groups. Keep aux Adam/AdamW groups on the baseline #46 LR schedule and values.
- Full-sweep objective: aim directly for a new Track 3 record below `2690`, not merely a safer margin at `2690`.
- CSCS runtime: use a Docker/container-based environment for GH200 runs unless a native CSCS module stack is later proven equivalent.
- CSCS storage: use CAPSTOR at `/capstor/store/cscs/uba/uba02`; dataset path `/capstor/store/cscs/uba/uba02/fineweb10B` is installed and verified.
- CSCS Slurm account/partition: use account `uba02` and partition `normal` for ordinary Daint sweep jobs.
- CSCS node-hour policy: pack four single-GPU runs per Daint node because the project is node-hour limited.
- CSCS runtime unresolved item: container launcher/image details still need to be finalized before full submission.
- W&B policy: enable W&B for CSCS sweep monitoring and dashboards. Use stdout logs plus parsed aggregation artifacts as the validation source of truth for the final PR.

## Immediate Next Step

Research harness implementation and local smoke are done. The next step is a CSCS runtime probe, not the full array.

- Set up or select the Docker/container image for Daint GH200.
- Confirm the container can see `/capstor/store/cscs/uba/uba02/fineweb10B`.
- Confirm `torchrun`, PyTorch CUDA, and W&B import inside the job environment.
- Submit one short debug/normal job with `STOP_STEP=1` and one packed array task.
- Submit a longer `STOP_STEP=125` CSCS probe only after the one-step job writes logs and W&B behaves as expected.

Do not create the final PR directory until a candidate passes local smoke and has a clear GH200 sweep plan.

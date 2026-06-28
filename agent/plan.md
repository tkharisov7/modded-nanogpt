# Plan: Track 3 Adaptive Critical-Warmup WR Search

## Goal

Beat the current Track 3 WR by using the earlier adaptive critical-warmup idea on the WR #46 SOAP-Muon stack, then searching the best post-warmup schedule scale with the page-8 exponential-bracket / binary-search rule.

Current target:

- Accepted WR source: `records/track_3_optimization/results/20260619_cwd_rowfloor_tailema/`
- WR accepted step: `2690`
- WR `n=8` mean at `2690`: `3.278329`
- WR `n=8` mean at `2685`: `3.278630`
- Track rule: `(3.28 - avg_loss) * sqrt(num_runs) >= 0.004`
- For `n=8`, passing mean at `2685` is `<= 3.278585786`

This phase is research/out-of-competition because adaptive critical warmup uses extra virtual-loss forwards during training. Legal fixed-schedule distillation is later work.

## Current Evidence

Research summaries:

```text
records/track_3_optimization/experiments/research/20260628_critical_warmup/
```

Completed Daint jobs:

- `3761838`: sparse sample at `STOP_STEP=2720`; baseline was best among completed cells.
- `3763132`: `catapult_proxy w=25 m=1.15` improved same-step late loss slightly, but did not beat WR crossing.
- `3763930`: removing/delaying cooldown was bad.

Takeaways:

- Linear warmup is not worth more budget now.
- Non-monotone warmup is not a bug. Critical warmup is expected to hit loss-increase boundaries and may temporarily worsen loss.
- The previous `critical_table` was only a proxy. The next implementation must use the real adaptive critical-LR rule.
- Cooldown matters. Retune timing for earlier target steps, but do not remove it.

## Core Understanding

There are two different searches, both using exponential bracket then binary search:

1. **Inner warmup-step critical LR search**
   - Happens during each adaptive critical-warmup step.
   - Uses the exact current training batch.
   - Searches the LR where the virtual one-step update crosses from loss-decreasing to loss-increasing.
   - This is the parent `../code` critical-LR rule adapted to the actual SOAP-Muon hidden-matrix update direction.

2. **Outer post-warmup scale search**
   - Happens across full training runs.
   - Searches the best post-warmup scale `s`, not just the largest stable value.
   - Candidate headline Muon LR is `s * 0.0375`.
   - Accept a larger `s` only if median target/final score improves.

The desired mechanism is:

```text
adaptive critical-LR warmup
  -> non-monotone edge-of-stability jumps during warmup
  -> higher best post-warmup SOAP-Muon schedule scale
  -> earlier 3.28 crossing
```

## Constraints

Keep WR #46 stack intact unless explicitly testing a named ablation:

- Architecture
- Dataset and validation
- Global batch semantics
- SOAP-Muon implementation
- RowFloor
- CWD
- Tail-EMA readout
- EMA-Nesterov
- WR #46 cooldown concept

Primary target is SOAP-Muon hidden matrices. Adam/scalar schedules should follow WR #46 unless an all-LR-scale ablation is explicitly requested.

## Inner Critical-LR Warmup Rule

Adapt the parent `../code` behavior:

- Parent run: `schedule_kind=critical_lr`, `warmup_steps=1000`, searched `stable_lr=0.4375`.
- At each warmup step, parent code calls `critical_lr_over_loader`.
- It computes current loss, builds a virtual update direction, exponentially brackets a safe and unsafe LR, then bisects.
- `_estimate_critical_lr` uses `0.5 * (lower + upper)` as the LR estimate.
- During warmup, this estimate is the actual training LR.
- After warmup, training switches to the separately searched stable/post-warmup LR.

Track 3 adaptation:

- Use the exact current optimizer-step training batch.
- Use the actual SOAP-Muon hidden-matrix update direction, not raw GD.
- Probe only hidden-matrix SOAP-Muon parameters first.
- Keep Adam/scalar params fixed during virtual probes for v1.
- Do not mutate real optimizer state during bracket/bisection probes.

Per warmup step:

```text
current_loss = loss(model, batch)
direction = SOAP-Muon hidden-matrix update direction for this batch/state

lower/safe:
  virtual loss is non-increasing vs current_loss

upper/unsafe:
  virtual loss strictly increases vs current_loss

exponential bracket:
  grow trial_lr until unsafe is found, up to critical_lr_search_max

binary search:
  bisect [lower, upper] until tolerance is reached

critical_lr_estimate:
  0.5 * (lower + upper)

actual warmup lr:
  critical_lr_estimate
```

Initial inner-search settings:

- `critical_lr_tol_power=6`
- `critical_lr_exp_max_iters=40`
- `critical_lr_search_max=10 * 0.0375 = 0.375`
- No extra safety multiplier in the primary plan. The midpoint estimate is the warmup LR.
- No additional runtime cap after the bracketed estimate, other than the search max used to find the bracket.

Log every adaptive warmup estimate:

```text
seed, step, current_loss,
lower_lr, upper_lr, midpoint_lr,
virtual_loss_lower, virtual_loss_upper,
search_hit_max,
post_warmup_scale
```

## SOAP-Muon Virtual Direction

Build the virtual trial from the exact WR #46 SOAP-Muon hidden-matrix update path. Do not use an approximate direction if it differs from the real optimizer.

- Run the normal forward/backward on the exact batch.
- For each SOAP-Muon hidden matrix, replay the same operations as WR #46:
  - use current grad
  - update/form Muon momentum consistently with `group["mu"]`
  - apply SOAP preconditioning where WR #46 does
  - apply attention trust gate/blend where WR #46 does
  - apply Muon orthogonalization
  - apply radial scaling
  - apply RowFloor / u-w floor
  - apply the WR #46 radius pin/rescale exactly as in the real step
  - apply WR #46 post-pin CWD exactly as in the real step

The virtual probe should answer: "what would the loss be if the hidden-matrix SOAP-Muon part of this exact WR #46 step used trial LR `lr`?"

Rules:

- Keep Adam/scalar params fixed during virtual loss probes in the primary experiment.
- Do not mutate real model or optimizer state during repeated bracket/bisection probes.
- Use temporary/cloned state for the trial update if the WR #46 update path mutates SOAP/Muon buffers.
- After choosing the critical LR, the real optimizer step must apply the selected LR once and commit optimizer state once.

## Outer Post-Warmup Scale Search

The outer candidate is a multiplicative scale `s` for post-warmup SOAP-Muon schedule.

```text
s = 1.0  -> headline Muon LR 0.0375
s = 2.0  -> headline Muon LR 0.075
s = 0.5  -> headline Muon LR 0.01875
```

Primary interpretation:

- Scale SOAP-Muon hidden-matrix LR schedule by `s`.
- Keep Adam/scalar schedules WR #46-compatible.

Optional ablation:

- Scale all LR groups by `s` to preserve whole-optimizer relative LR ratios.
- Label this separately because it is not the primary hidden-matrix-only hypothesis.

Outer search rule:

```text
start s = 1.0
run seeds 0,1,2
score by median target-zone/final val_ema_loss
try s * sqrt(2)
accept only if median score improves
continue upward until first non-improvement or divergence
binary-search the multiplicative bracket with geometric midpoints
promote the best few scales
```

Score metric:

- Tail-EMA enabled: use `val_ema_loss`.
- Tail-EMA disabled: use `val_loss`.
- Do not optimize for non-divergence only. Stable but worse LR is rejected.

Final selection:

- First pass: seeds `0,1,2`.
- Confirmation: seeds `0..6` for the page-8 median-over-seeds rule.
- Track-style final: seeds `0..7` because current WR report is `n=8`.

## Search Surface

Primary knobs:

- `schedule_kind=adaptive_critical_lr`
- `warmup_steps`
- `post_warmup_scale`
- `cooldown_end_step`
- `cooldown_start_offset`
- `cooldown_power`

Cooldown:

- Keep WR #46 PowerCool concept.
- Because target finish is earlier than `2900`, test cooldown ending earlier.
- Do not repeat no-cooldown except as a negative control.

Initial cooldown candidates:

```text
cooldown_end_step: 2685, 2690, 2720, 2900
cooldown_start_offset: 200, 300
cooldown_power: 1.2 first; then 1.0 and 1.4 near winners
```

Initial adaptive warmup candidates:

```text
warmup_steps: 5, 10, 15, 20, 25
critical_lr_search_max: 0.375
```

Use the outer `s` search inside each promising warmup/cooldown cell. For comparability with the earlier grid, the initial post-warmup scale candidates should cover the same range as the old `muon_lr_mult` search:

```text
post_warmup_scale rough values: 1.10, 1.15, 1.20, 1.25, 1.30, 1.35
```

The final scale selection still uses exponential bracket + log-space bisection, not a fixed grid only.

## State Reuse

Default: no state reuse between candidate scales.

Only reuse a prefix checkpoint if the entire prefix trajectory is identical. For adaptive critical warmup, this is usually false because `post_warmup_scale` may be part of logs/config but not the warmup trajectory; if warmup truly does not depend on `s`, reuse is possible only with full state:

```text
model weights
all optimizer state_dicts
SOAP/Muon state
EMA-Nesterov state
Tail-EMA state if active
RNG states
dataloader position
```

## Screening Protocol

Screening target:

```text
STOP_STEP=2685
```

Screen all hypothesis cells with seeds:

```text
0, 1, 2
```

Promote only cells that are:

- Stable for all three seeds
- Competitive at `2685`
- Better by median score, not merely stable
- Mechanistically plausible from logged critical-LR traces

Promotion thresholds:

- Strong promote: `n=3` mean below `3.2786`
- Possible promote: `n=3` mean below WR #46 `2685` mean `3.278630`
- Reject: unstable, clearly worse, or only improves late.

## First Round

Round A-small:

```text
schedule_kind: adaptive_critical_lr
warmup_steps: 10, 20
post_warmup_scale rough start: 1.05, 1.10, 1.15
cooldown_end_step: 2685, 2690
cooldown_start_offset: 200
cooldown_power: 1.2
outer post_warmup_scale: exponential bracket + log-space bisection
seeds: 0, 1, 2
```

Round A expansion:

```text
warmup_steps: 10, 15, 20
post_warmup_scale rough start: 1.05, 1.10, 1.15, 1.20, 1.25
cooldown_end_step: 2685, 2690, 2720
cooldown_start_offset: 200, 300
cooldown_power: 1.2
outer post_warmup_scale: exponential bracket + log-space bisection
seeds: 0, 1, 2
```

Round B:

- Add `cooldown_power=1.0, 1.4` near winners.
- Add `cooldown_end_step=2700, 2900` if early cooldown hurts.
- Add all-LR-scale ablation only after hidden-matrix scale has a signal.
- Promote best cells to `n=7`, then `n=8`.

## Analysis Rules

Source of truth:

```text
SUMMARY_JSON lines in CAPSTOR logs
```

For every job:

- Count completed summaries.
- Group by warmup steps, post-warmup scale, cooldown end, cooldown offset, cooldown power, and scale scope.
- Report `n`, mean, median, min, max, stable count.
- Compare equal seed counts.
- Report both target-zone `val_ema_loss` and first crossing.
- For final or near-final candidates, produce the same PR-style target-zone table cadence as WR #46: steps `2680, 2685, 2690, 2695, 2700`, with per-seed rows, mean row, and `(3.28 - mean) * sqrt(n)`.
- Summarize critical-LR traces: median warmup LR by step, max LR, search-hit-max count, and seed variability.

Final claim threshold:

```text
n=8 mean at 2685 <= 3.278585786
```

## Immediate Next Steps

1. Status: completed. Implement `adaptive_critical_lr` for SOAP-Muon hidden matrices.
2. Status: completed. Implement inner exponential bracket + binary search for each warmup step.
3. Status: completed. Add post-warmup scale `s` and outer search driver/logging.
4. Status: completed. Add cooldown end/start/power controls.
5. Status: completed. Add dense validation around `2680..2700`.
6. Status: completed. Run one-step and short local smoke tests.
7. Status: pending. Launch Round A-small at `STOP_STEP=2685` after local review/approval.
8. Status: pending. Promote only cells that improve median `val_ema_loss`.

Do not spend current effort on fixed-schedule distillation. The priority is to prove the adaptive critical-warmup mechanism can beat WR out of competition.

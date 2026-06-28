# Job 3761838: Sparse Full-Grid Reference Sample

## Purpose

This was the first larger GH200 run used to compare the Track 3 baseline against selected critical-warmup-inspired cells at `STOP_STEP=2720`.

It was not a complete full grid. It sampled selected array task IDs from the `full_v1` candidate ordering:

```text
baseline
linear
critical_table
catapult_proxy
warmup_steps in {25, 50, 100}
muon_lr_mult in {1.00, 1.05, 1.10, 1.15, 1.20}
seeds 0..7
```

Slurm stdout showed:

```text
stop_step=2720
candidates_per_node=4
candidate_count=368
wandb_group=T3_CRITICAL_WARMUP_GH200_3761838
```

The exact shell command used for submission was not fully recovered, but the launched array task IDs are visible from the Slurm output file names:

```text
0, 1, 6, 7, 18, 19, 24, 25, 36, 37, 40, 41, 48, 49, 54, 55, 66, 67, 80, 81, 88, 89
```

That produced `22` Slurm array tasks x `4` candidates per node = `88` completed candidate runs.

## Source Logs

Candidate logs:

```text
/capstor/store/cscs/uba/uba02/t3_critical_warmup/logs/*_3761838_*.log
```

Slurm logs:

```text
/capstor/store/cscs/uba/uba02/t3_critical_warmup/slurm/t3-crit-warmup_3761838_*.out
/capstor/store/cscs/uba/uba02/t3_critical_warmup/slurm/t3-crit-warmup_3761838_*.err
```

## Completion

Parsed status:

```text
logs: 88
SUMMARY_JSON records: 2288
final candidate logs: 88
Slurm outs: 22
Slurm status: {0: 22}
missing Slurm outs: 0
bad logs: 0
```

## Completed n=8 Groups at Step 2720

| Rank | schedule_kind | warmup_steps | muon_lr_mult | n | mean val_ema_loss | criterion | stable | min | max |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: |
| 1 | baseline | 0 | 1.00 | 8 | 3.276432037 | 0.010091722 | 8/8 | 3.274849415 | 3.278680563 |
| 2 | catapult_proxy | 25 | 1.10 | 8 | 3.276768923 | 0.009138866 | 8/8 | 3.275099993 | 3.279012680 |
| 3 | critical_table | 25 | 1.20 | 8 | 3.277219146 | 0.007865441 | 8/8 | 3.275227308 | 3.280037403 |
| 4 | catapult_proxy | 50 | 1.20 | 8 | 3.277321786 | 0.007575134 | 8/8 | 3.275609255 | 3.279839039 |
| 5 | critical_table | 25 | 1.10 | 8 | 3.277720720 | 0.006446779 | 8/8 | 3.275614500 | 3.279826403 |
| 6 | critical_table | 50 | 1.15 | 8 | 3.278087318 | 0.005409882 | 8/8 | 3.274966002 | 3.279961824 |
| 7 | catapult_proxy | 100 | 1.15 | 8 | 3.278532803 | 0.004149860 | 8/8 | 3.276514530 | 3.280315399 |
| 8 | critical_table | 100 | 1.05 | 8 | 3.278782099 | 0.003444743 | 8/8 | 3.276964188 | 3.280317307 |
| 9 | linear | 25 | 1.10 | 8 | 3.279379725 | 0.001754401 | 8/8 | - | - |
| 10 | linear | 50 | 1.15 | 8 | 3.280554116 | -0.001567276 | 8/8 | - | - |
| 11 | linear | 100 | 1.05 | 8 | 3.281806409 | -0.005109297 | 8/8 | - | - |

## Interpretation

The GH200 baseline was the best completed group in this sampled sweep at `STOP_STEP=2720`.

The useful result from this job is a fair same-step baseline:

```text
baseline, n=8, step 2720
mean val_ema_loss = 3.276432037
criterion = 0.010091722
```

This baseline is not identical to the accepted WR #46 formal claim because the accepted claim is about the earliest passing step, with `2690` as the record step. For any new speedrun record, compare against WR #46's `2690` pass and `2685` fail, not only against the GH200 `2720` baseline.


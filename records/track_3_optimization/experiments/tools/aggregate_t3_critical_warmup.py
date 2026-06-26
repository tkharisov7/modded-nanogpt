#!/usr/bin/env python3
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


GROUP_FIELDS = (
    "schedule_kind",
    "warmup_target",
    "warmup_steps",
    "muon_lr_mult",
    "aux_lr_mult",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--target-loss", type=float, default=3.28)
    parser.add_argument("--min-runs", type=int, default=1)
    return parser.parse_args()


def iter_log_files(paths):
    for path in paths:
        if path.is_dir():
            for child in sorted(path.rglob("*.txt")):
                yield child
            for child in sorted(path.rglob("*.log")):
                yield child
            for child in sorted(path.rglob("*.out")):
                yield child
        else:
            yield path


def iter_summary_events(path):
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            marker = "SUMMARY_JSON "
            if marker not in line:
                continue
            _, payload = line.split(marker, 1)
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            event["log_path"] = str(path)
            yield event


def config_key(event):
    return tuple(event.get(field) for field in GROUP_FIELDS)


def row_from_group(key, step, events, target_loss):
    losses = [float(event["val_ema_loss"]) for event in events]
    raw_losses = [float(event["val_loss"]) for event in events]
    n = len(losses)
    mean_loss = sum(losses) / n
    mean_raw_loss = sum(raw_losses) / n
    criterion = (target_loss - mean_loss) * math.sqrt(n)
    row = {field: value for field, value in zip(GROUP_FIELDS, key)}
    row.update(
        {
            "step": step,
            "n": n,
            "mean_val_ema_loss": mean_loss,
            "mean_val_loss": mean_raw_loss,
            "criterion": criterion,
            "passes_track3": criterion >= 0.004,
            "logs": ";".join(sorted(event["log_path"] for event in events)),
        }
    )
    return row


def main():
    args = parse_args()
    grouped = defaultdict(list)
    for path in iter_log_files(args.logs):
        for event in iter_summary_events(path):
            if event.get("event") != "validation":
                continue
            grouped[(config_key(event), int(event["step"]))].append(event)

    rows = []
    for (key, step), events in sorted(grouped.items()):
        if len(events) < args.min_runs:
            continue
        rows.append(row_from_group(key, step, events, args.target_loss))

    fieldnames = list(GROUP_FIELDS) + [
        "step",
        "n",
        "mean_val_ema_loss",
        "mean_val_loss",
        "criterion",
        "passes_track3",
        "logs",
    ]

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()

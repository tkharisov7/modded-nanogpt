#!/usr/bin/env python3
import argparse
import html
import re
import subprocess
from collections import defaultdict
from pathlib import Path


POINT_RE = re.compile(r"step:(\d+)/\d+\s+val_loss:([0-9.]+)")
NAME_RE = re.compile(
    r"adaptive_critical_lr_w(?P<warmup>\d+)_m(?P<mult>[0-9.]+)_c(?P<cooldown>\d+)_seed(?P<seed>\d+)_3769556_(?P<idx>\d+)\.log$"
)


COLORS = {
    (10, "1.05", "2685"): "#2ca02c",
    (10, "1.05", "2690"): "#9467bd",
    (10, "1.10", "2685"): "#d62728",
    (10, "1.10", "2690"): "#8c564b",
    (10, "1.15", "2685"): "#17becf",
    (10, "1.15", "2690"): "#e377c2",
    (20, "1.05", "2685"): "#1b9e77",
    (20, "1.05", "2690"): "#7570b3",
    (20, "1.10", "2685"): "#e7298a",
    (20, "1.10", "2690"): "#a6761d",
    (20, "1.15", "2685"): "#66a61e",
    (20, "1.15", "2690"): "#666666",
}


def parse_log(path):
    points = []
    with path.open() as f:
        for line in f:
            m = POINT_RE.search(line)
            if m:
                points.append((int(m.group(1)), float(m.group(2))))
    return points


def axis_mapper():
    # Pixel bounds of the existing figure_wr_vs_base.png plotting area.
    # Data limits are the visible axes in that image.
    left, right = 172.0, 1537.0
    top, bottom = 111.0, 1024.0
    x_min, x_max = 0.0, 3500.0
    y_min, y_max = 3.25, 4.0

    def xy(step, loss):
        x = left + (step - x_min) / (x_max - x_min) * (right - left)
        y = bottom - (loss - y_min) / (y_max - y_min) * (bottom - top)
        return x, y

    return xy


def polyline(points, xy):
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in (xy(step, loss) for step, loss in points))


def draw_run(points, color, opacity, width, radius, xy):
    if len(points) == 0:
        return []
    out = [
        f'<polyline points="{polyline(points, xy)}" fill="none" stroke="{color}" '
        f'stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round" opacity="{opacity}"/>'
    ]
    for step, loss in points:
        x, y = xy(step, loss)
        out.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" '
            f'stroke="white" stroke-width="1.2" opacity="{opacity}"/>'
        )
    return out


def draw_mean(points, color, opacity, width, radius, dash, xy):
    if len(points) == 0:
        return []
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    out = [
        f'<polyline points="{polyline(points, xy)}" fill="none" stroke="{color}" '
        f'stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round" opacity="{opacity}"{dash_attr}/>'
    ]
    for step, loss in points:
        x, y = xy(step, loss)
        out.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" '
            f'stroke="white" stroke-width="1.2" opacity="{opacity}"/>'
        )
    return out


def average_curves(runs):
    by_step = defaultdict(list)
    for points in runs:
        for step, loss in points:
            by_step[step].append(loss)
    return [(step, sum(vals) / len(vals)) for step, vals in sorted(by_step.items())]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="records/track_3_optimization/img/figure_wr_vs_base.png")
    parser.add_argument("--logs", default="/capstor/store/cscs/uba/uba02/t3_critical_warmup/logs")
    parser.add_argument("--out", default="records/track_3_optimization/img/figure_wr_vs_base_t3_critical_warmup_overlay.png")
    parser.add_argument("--svg", default="records/track_3_optimization/img/figure_wr_vs_base_t3_critical_warmup_overlay.svg")
    args = parser.parse_args()

    xy = axis_mapper()
    log_dir = Path(args.logs)
    grouped = defaultdict(list)
    no_validation = []
    parsed_runs = []

    for path in sorted(log_dir.glob("*3769556*.log")):
        m = NAME_RE.match(path.name)
        if not m:
            continue
        points = parse_log(path)
        meta = m.groupdict()
        key = (meta["warmup"], meta["mult"], meta["cooldown"])
        if points:
            grouped[key].append((path.name, points, meta))
            parsed_runs.append((path.name, points, meta))
        else:
            no_validation.append(path.name)

    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1620" height="1171" viewBox="0 0 1620 1171">',
        '<rect x="0" y="0" width="1620" height="1171" fill="none"/>',
    ]

    # Individual runs: same marker+line idiom as the benchmark, but translucent so the base curves remain readable.
    for key in sorted(grouped):
        color = COLORS.get(key, "#444444")
        for _name, points, _meta in grouped[key]:
            elements.extend(draw_run(points, color, opacity="0.30", width="2.4", radius="3.4", xy=xy))

    # Group means: thicker, fully opaque comparison curves.
    legend_rows = []
    for key in sorted(grouped):
        color = COLORS.get(key, "#444444")
        avg = average_curves([points for _name, points, _meta in grouped[key]])
        dash = "12 8" if key[0] == 20 else ""
        elements.extend(draw_mean(avg, color, opacity="0.95", width="5.0", radius="5.2", dash=dash, xy=xy))
        max_step = max(step for _name, points, _meta in grouped[key] for step, _loss in points)
        n = len(grouped[key])
        legend_rows.append((key, color, n, max_step))

    # Compact legend in the lower-left plot area, away from the original legend and main endpoint.
    legend_x, legend_y = 205, 760
    row_h = 24
    legend_h = 44 + row_h * len(legend_rows)
    elements.append(
        f'<rect x="{legend_x - 12}" y="{legend_y - 28}" width="500" height="{legend_h}" '
        'rx="8" fill="white" fill-opacity="0.82" stroke="#bdbdbd" stroke-width="1.5"/>'
    )
    elements.append(
        f'<text x="{legend_x}" y="{legend_y}" font-family="DejaVu Sans, Arial, sans-serif" '
        'font-size="24" fill="#222">T3 critical warmup cohort 3769556</text>'
    )
    for i, (key, color, n, max_step) in enumerate(legend_rows, start=1):
        y = legend_y + 10 + i * row_h
        label = f"w={key[0]}, m={key[1]}, c={key[2]}: mean + {n} runs, latest {max_step}"
        dash_attr = ' stroke-dasharray="12 8"' if key[0] == 20 else ""
        elements.append(f'<line x1="{legend_x}" y1="{y - 7}" x2="{legend_x + 50}" y2="{y - 7}" stroke="{color}" stroke-width="5" stroke-linecap="round"{dash_attr}/>')
        elements.append(f'<circle cx="{legend_x + 25}" cy="{y - 7}" r="5.2" fill="{color}" stroke="white" stroke-width="1.2"/>')
        elements.append(
            f'<text x="{legend_x + 64}" y="{y}" font-family="DejaVu Sans, Arial, sans-serif" '
            f'font-size="19" fill="#222">{html.escape(label)}</text>'
        )

    elements.append("</svg>")
    svg_path = Path(args.svg)
    svg_path.write_text("\n".join(elements))

    out_path = Path(args.out)
    subprocess.run(
        [
            "magick",
            str(Path(args.base)),
            "(",
            "-background",
            "none",
            str(svg_path),
            ")",
            "-composite",
            str(out_path),
        ],
        check=True,
    )

    print(f"wrote {out_path}")
    print(f"wrote {svg_path}")
    print(f"plotted_runs={len(parsed_runs)}")
    for key in sorted(grouped):
        latest = max(step for _name, points, _meta in grouped[key] for step, _loss in points)
        print(f"group w={key[0]} m={key[1]} c={key[2]} runs={len(grouped[key])} latest_step={latest}")
    if no_validation:
        print("no_validation_yet=" + ",".join(no_validation))


if __name__ == "__main__":
    main()

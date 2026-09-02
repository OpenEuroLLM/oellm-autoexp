#!/usr/bin/env python3
"""Build the self-contained Dense 32B checkpoint weight-analysis slide deck."""

from __future__ import annotations

import csv
import html
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRIOR = (
    ROOT
    / "e/project1/e-sta-openeurollm/luukkonen1/oellm-autoexp"
    / "checkpoint_scan_prior_20260902_171101_526/artifacts/comparison"
)
LATE = (
    ROOT
    / "e/home/jusers/luukkonen1/jupiter/e-sta-workdir/oellm-autoexp"
    / "checkpoint_scan_20260902_162252_497/artifacts/model/comparison"
)
OUTPUT = ROOT / "docs/fp8-loss-turn/checkpoint-weight-analysis-by-block.html"

COLORS = {
    "attention": "#f97316",
    "mlp": "#14b8a6",
    "norm": "#a78bfa",
    "embed": "#60a5fa",
    "fp8": "#fb7185",
    "bf16": "#38bdf8",
    "ink": "#e8edf6",
    "muted": "#9aa7b8",
    "grid": "#334155",
    "good": "#4ade80",
}

GROUPS = ["Attention", "MLP", "Normalization", "Embedding + head"]
GROUP_COLOR = {
    "Attention": COLORS["attention"],
    "MLP": COLORS["mlp"],
    "Normalization": COLORS["norm"],
    "Embedding + head": COLORS["embed"],
}

LABELS = {
    "decoder.layers.self_attention.linear_proj.weight": "Attn output",
    "decoder.layers.self_attention.linear_qkv.weight": "QKV",
    "decoder.layers.mlp.linear_fc1.weight": "MLP FC1",
    "decoder.layers.mlp.linear_fc2.weight": "MLP FC2",
    "embedding.word_embeddings.weight": "Embedding",
    "output_layer.weight": "Output head",
    "decoder.layers.self_attention.linear_qkv.layer_norm_weight": "Attn input norm",
    "decoder.layers.mlp.linear_fc1.layer_norm_weight": "Pre-MLP norm",
    "decoder.layers.self_attention.q_layernorm.weight": "Q norm",
    "decoder.layers.self_attention.k_layernorm.weight": "K norm",
    "decoder.final_layernorm.weight": "Final norm",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def group_for(tensor: str) -> str:
    if "embedding" in tensor or "output_layer" in tensor:
        return "Embedding + head"
    if "linear_qkv.weight" in tensor or "linear_proj.weight" in tensor:
        return "Attention"
    if "mlp.linear_fc1.weight" in tensor or "mlp.linear_fc2.weight" in tensor:
        return "MLP"
    return "Normalization"


def weighted_group_rms(rows: list[dict[str, str]], run: str) -> dict[tuple[int, str], float]:
    buckets: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["run"] == run:
            buckets[(int(row["iteration"]), group_for(row["tensor"]))].append(row)
    result = {}
    for key, values in buckets.items():
        count = sum(int(v["numel"]) for v in values)
        result[key] = math.sqrt(
            sum(int(v["numel"]) * float(v["global_rms"]) ** 2 for v in values) / count
        )
    return result


def svg_line_chart(
    series: list[tuple[str, str, list[tuple[float, float]]]],
    *,
    width: int = 1040,
    height: int = 340,
    y_min: float | None = None,
    y_max: float | None = None,
    turn: float | None = 66625,
    y_label: str = "",
    x_ticks: list[int] | None = None,
    percent: bool = False,
) -> str:
    ml, mr, mt, mb = 72, 28, 24, 48
    pw, ph = width - ml - mr, height - mt - mb
    points = [p for _, _, vals in series for p in vals]
    xmin, xmax = min(x for x, _ in points), max(x for x, _ in points)
    ymin = min(y for _, y in points) if y_min is None else y_min
    ymax = max(y for _, y in points) if y_max is None else y_max
    pad = (ymax - ymin) * 0.08 or 0.1
    if y_min is None:
        ymin -= pad
    if y_max is None:
        ymax += pad

    def sx(x: float) -> float:
        return ml + (x - xmin) / (xmax - xmin) * pw

    def sy(y: float) -> float:
        return mt + (ymax - y) / (ymax - ymin) * ph

    out = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img">']
    for i in range(5):
        y = ymin + (ymax - ymin) * i / 4
        py = sy(y)
        label = f"{y:.1f}%" if percent else f"{y:.2f}×"
        out.append(f'<line x1="{ml}" y1="{py:.1f}" x2="{width-mr}" y2="{py:.1f}" class="grid"/>')
        out.append(f'<text x="{ml-12}" y="{py+5:.1f}" text-anchor="end" class="tick">{label}</text>')
    ticks = x_ticks or [round(xmin), round((xmin + xmax) / 2), round(xmax)]
    for x in ticks:
        px = sx(x)
        out.append(f'<text x="{px:.1f}" y="{height-15}" text-anchor="middle" class="tick">{x/1000:g}k</text>')
    if turn is not None and xmin <= turn <= xmax:
        px = sx(turn)
        out.append(f'<line x1="{px:.1f}" y1="{mt}" x2="{px:.1f}" y2="{mt+ph}" class="turn"/>')
        out.append(f'<text x="{px+7:.1f}" y="{mt+14}" class="turn-label">loss turn</text>')
    for label, color, values in series:
        coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in values)
        out.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/>')
        for x, y in values:
            out.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.3" fill="{color}"/>')
    if y_label:
        out.append(f'<text x="17" y="{mt+ph/2:.1f}" transform="rotate(-90 17 {mt+ph/2:.1f})" text-anchor="middle" class="axis-label">{html.escape(y_label)}</text>')
    out.append("</svg>")
    return "".join(out)


def mini_line_chart(label: str, color: str, values: list[tuple[int, float]], domain: tuple[float, float]) -> str:
    svg = svg_line_chart(
        [(label, color, values)],
        width=510,
        height=235,
        y_min=domain[0],
        y_max=domain[1],
        x_ticks=[4000, 40000, 75126],
    )
    return f'<div class="mini"><div class="mini-title"><span class="dot" style="background:{color}"></span>{label}</div>{svg}</div>'


def rate_bars(rates: list[tuple[str, str, float, float]]) -> str:
    width, height = 1110, 380
    ml, mr, mt, mb = 160, 48, 20, 54
    pw, ph = width - ml - mr, height - mt - mb
    vmax = max(max(fp8, bf16) for _, _, fp8, bf16 in rates) * 1.18
    vmin = min(0, min(min(fp8, bf16) for _, _, fp8, bf16 in rates))

    def sy(y: float) -> float:
        return mt + (vmax - y) / (vmax - vmin) * ph

    zero = sy(0)
    slot = pw / len(rates)
    bar_w = 22
    out = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img">']
    for val in [0, 0.1, 0.2, 0.3, 0.4]:
        if val <= vmax:
            y = sy(val)
            out.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{width-mr}" y2="{y:.1f}" class="grid"/>')
            out.append(f'<text x="{ml-12}" y="{y+5:.1f}" text-anchor="end" class="tick">{val:.1f}%</text>')
    for i, (label, group, fp8, bf16) in enumerate(rates):
        cx = ml + slot * (i + 0.5)
        for value, color, dx in [(fp8, COLORS["fp8"], -13), (bf16, COLORS["bf16"], 13)]:
            top = sy(value)
            y = min(top, zero)
            h = abs(zero - top)
            out.append(f'<rect x="{cx+dx-bar_w/2:.1f}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" rx="4" fill="{color}"/>')
        out.append(f'<text x="{cx:.1f}" y="{height-28}" text-anchor="middle" class="tick strong">{html.escape(label)}</text>')
        out.append(f'<rect x="{cx-31:.1f}" y="{height-17}" width="62" height="4" rx="2" fill="{GROUP_COLOR[group]}"/>')
    out.append(f'<line x1="{ml}" y1="{zero:.1f}" x2="{width-mr}" y2="{zero:.1f}" class="axis"/>')
    out.append(f'<text x="22" y="{mt+ph/2:.1f}" transform="rotate(-90 22 {mt+ph/2:.1f})" text-anchor="middle" class="axis-label">RMS growth per 1k steps</text>')
    out.append("</svg>")
    return "".join(out)


def mix(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> str:
    t = max(0.0, min(1.0, t))
    return "#%02x%02x%02x" % tuple(round(a + (b - a) * t) for a, b in zip(c1, c2))


def layer_color(position: int, count: int) -> str:
    """Continuous cyan→violet→orange color scale for model depth."""
    t = position / max(count - 1, 1)
    if t <= 0.5:
        return mix((34, 211, 238), (167, 139, 250), t * 2)
    return mix((167, 139, 250), (249, 115, 22), (t - 0.5) * 2)


def heatmap(rows: list[tuple[str, str, list[float]]]) -> str:
    width, height = 1120, 310
    left, top, cell_w, cell_h, gap = 154, 42, 13.5, 42, 3
    out = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img">']
    for idx, (label, group, values) in enumerate(rows):
        y = top + idx * (cell_h + 18)
        out.append(f'<text x="{left-14}" y="{y+cell_h/2+5:.1f}" text-anchor="end" class="tick strong">{html.escape(label)}</text>')
        for layer, value in enumerate(values):
            if value >= 0:
                color = mix((37, 49, 66), (249, 115, 22), min(value / 1.4, 1))
            else:
                color = mix((37, 49, 66), (56, 189, 248), min(abs(value) / 1.4, 1))
            x = left + layer * cell_w
            out.append(f'<rect x="{x:.1f}" y="{y}" width="{cell_w-gap:.1f}" height="{cell_h}" rx="2" fill="{color}"><title>layer {layer}: {value:+.2f}%</title></rect>')
        positive = sum(v > 0 for v in values)
        out.append(f'<text x="{left+64*cell_w+12:.1f}" y="{y+cell_h/2+5:.1f}" class="tick" fill="{GROUP_COLOR[group]}">{positive}/64 positive</text>')
    for layer in [0, 15, 31, 47, 63]:
        x = left + layer * cell_w + (cell_w-gap)/2
        out.append(f'<text x="{x:.1f}" y="{height-24}" text-anchor="middle" class="tick">{layer}</text>')
    out.append(f'<text x="{left+32*cell_w:.1f}" y="{height-3}" text-anchor="middle" class="axis-label">transformer layer →</text>')
    out.append("</svg>")
    return "".join(out)


def norm_history_chart(values: dict[tuple[int, int], float]) -> str:
    """Plot every layer for one norm family across the complete flagship run."""
    width, height = 1120, 430
    ml, mr, mt, mb = 76, 34, 28, 54
    pw, ph = width - ml - mr, height - mt - mb
    iterations = sorted({iteration for iteration, _ in values})
    xmin, xmax, ymin, ymax = iterations[0], iterations[-1], 0.0, 1.8

    def sx(x: float) -> float:
        return ml + (x - xmin) / (xmax - xmin) * pw

    def sy(y: float) -> float:
        return mt + (ymax - y) / (ymax - ymin) * ph

    def coords(points: list[tuple[int, float]]) -> str:
        return " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)

    layers = sorted({layer for _, layer in values})
    out = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img">']
    for y in [0, 0.5, 1.0, 1.5]:
        py = sy(y)
        out.append(f'<line x1="{ml}" y1="{py:.1f}" x2="{width-mr}" y2="{py:.1f}" class="grid"/>')
        out.append(f'<text x="{ml-12}" y="{py+5:.1f}" text-anchor="end" class="tick">{y:.1f}</text>')
    for x in [4000, 20000, 40000, 60000, 75126]:
        out.append(f'<text x="{sx(x):.1f}" y="{height-18}" text-anchor="middle" class="tick">{x/1000:g}k</text>')
    for x, label, klass in [(34455, "stack swap", "event"), (66625, "loss turn", "turn")]:
        px = sx(x)
        out.append(f'<line x1="{px:.1f}" y1="{mt}" x2="{px:.1f}" y2="{mt+ph}" class="{klass}"/>')
        out.append(f'<text x="{px+7:.1f}" y="{mt+14}" class="{klass}-label">{label}</text>')
    for position, layer in enumerate(layers):
        points = [(iteration, values[(iteration, layer)]) for iteration in iterations]
        color = layer_color(position, len(layers))
        out.append(f'<polyline class="layer-line" data-layer="{layer}" points="{coords(points)}" fill="none" stroke="{color}" stroke-opacity=".72" stroke-width="1.5"><title>layer {layer}</title></polyline>')
    medians = [
        (iteration, sorted(values[(iteration, layer)] for layer in layers))
        for iteration in iterations
    ]
    median_points = []
    for iteration, ordered in medians:
        middle = len(ordered) // 2
        median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
        median_points.append((iteration, median))
    out.append(f'<polyline class="layer-line median-line" data-layer="median" points="{coords(median_points)}" fill="none" stroke="#f8fafc" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"><title>cross-layer median</title></polyline>')
    out.append(f'<text x="18" y="{mt+ph/2:.1f}" transform="rotate(-90 18 {mt+ph/2:.1f})" text-anchor="middle" class="axis-label">absolute RMS of gain vector</text>')
    out.append("</svg>")
    return "".join(out)


def stacked_share(shares: list[tuple[str, float]]) -> str:
    width, height = 1040, 132
    x, y, w, h = 25, 22, 990, 58
    cursor = x
    out = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img">']
    for group, share in shares:
        sw = w * share / 100
        out.append(f'<rect x="{cursor:.2f}" y="{y}" width="{max(sw, 1):.2f}" height="{h}" fill="{GROUP_COLOR[group]}"><title>{group}: {share:.3f}%</title></rect>')
        if sw > 80:
            out.append(f'<text x="{cursor+sw/2:.1f}" y="{y+35}" text-anchor="middle" class="bar-label">{share:.1f}%</text>')
        cursor += sw
    legend_x = 25
    for group, share in shares:
        out.append(f'<circle cx="{legend_x+5}" cy="111" r="5" fill="{GROUP_COLOR[group]}"/>')
        out.append(f'<text x="{legend_x+16}" y="116" class="tick">{html.escape(group)} {share:.3f}%</text>')
        legend_x += 240
    out.append("</svg>")
    return "".join(out)


def build() -> None:
    prior_family = read_csv(PRIOR / "family_summary.csv")
    late_family = read_csv(LATE / "family_summary.csv")
    late_tensors = read_csv(LATE / "tensor_trajectories.csv")
    late_channels = read_csv(LATE / "channel_trajectories.csv")

    # Parameter shares at the first complete checkpoint.
    first = [r for r in prior_family if int(r["iteration"]) == 4000]
    total = sum(int(r["numel"]) for r in first)
    shares = [
        (group, 100 * sum(int(r["numel"]) for r in first if group_for(r["tensor"]) == group) / total)
        for group in ["MLP", "Attention", "Embedding + head", "Normalization"]
    ]

    # Stitch actual-rms history across the early and late scans, then index at 4k.
    early = weighted_group_rms(prior_family, "flagship_prior")
    late_fp8 = weighted_group_rms(late_family, "flagship")
    full = dict(early)
    full.update(late_fp8)
    domains = {
        "Attention": (0.95, 2.7),
        "MLP": (0.95, 2.7),
        "Normalization": (0.95, 1.12),
        "Embedding + head": (0.95, 1.24),
    }
    mini_charts = []
    for group in GROUPS:
        baseline = full[(4000, group)]
        values = sorted((iteration, value / baseline) for (iteration, g), value in full.items() if g == group)
        mini_charts.append(mini_line_chart(group, GROUP_COLOR[group], values, domains[group]))

    # Controlled 60k->68k rates.
    family_index = {
        (r["run"], int(r["iteration"]), r["tensor"]): float(r["global_rms"])
        for r in late_family
    }
    rate_specs = [
        ("QKV", "Attention", "decoder.layers.self_attention.linear_qkv.weight"),
        ("Attn out", "Attention", "decoder.layers.self_attention.linear_proj.weight"),
        ("FC1", "MLP", "decoder.layers.mlp.linear_fc1.weight"),
        ("FC2", "MLP", "decoder.layers.mlp.linear_fc2.weight"),
        ("Attn input norm", "Normalization", "decoder.layers.self_attention.linear_qkv.layer_norm_weight"),
        ("Embedding", "Embedding + head", "embedding.word_embeddings.weight"),
        ("Head", "Embedding + head", "output_layer.weight"),
    ]
    rates = []
    for label, group, tensor in rate_specs:
        vals = []
        for run in ["flagship", "bf16"]:
            rate = (
                family_index[(run, 68000, tensor)] / family_index[(run, 60000, tensor)] - 1
            ) / 8 * 100
            vals.append(rate)
        rates.append((label, group, vals[0], vals[1]))

    # Layerwise FP8 excess at 68k relative to the shared 60k checkpoint.
    tensor_index = {
        (r["run"], int(r["iteration"]), r["tensor"], int(r["layer"])): float(r["rms_ratio"])
        for r in late_tensors
    }
    heat_specs = [
        ("QKV", "Attention", "decoder.layers.self_attention.linear_qkv.weight"),
        ("Attention output", "Attention", "decoder.layers.self_attention.linear_proj.weight"),
        ("MLP FC1", "MLP", "decoder.layers.mlp.linear_fc1.weight"),
        ("MLP FC2", "MLP", "decoder.layers.mlp.linear_fc2.weight"),
    ]
    heat_rows = []
    heat_stats = {}
    for label, group, tensor in heat_specs:
        values = [
            100
            * (
                tensor_index[("flagship", 68000, tensor, layer)]
                / tensor_index[("bf16", 68000, tensor, layer)]
                - 1
            )
            for layer in range(64)
        ]
        heat_rows.append((label, group, values))
        heat_stats[label] = (min(values), sum(values) / len(values), max(values), sum(v > 0 for v in values))

    # Dead norm-gain elements in the late flagship checkpoints.
    dead_iters = [60000, 64000, 68000, 72000, 75126]
    dead_specs = [
        ("Pre-MLP", COLORS["norm"], "decoder.layers.mlp.linear_fc1.layer_norm_weight"),
        ("Q norm", "#f472b6", "decoder.layers.self_attention.q_layernorm.weight"),
        ("K norm", "#c084fc", "decoder.layers.self_attention.k_layernorm.weight"),
        ("Attention input", "#fde047", "decoder.layers.self_attention.linear_qkv.layer_norm_weight"),
    ]
    dead_series = []
    for label, color, tensor in dead_specs:
        values = []
        for iteration in dead_iters:
            rows = [
                r
                for r in late_channels
                if r["run"] == "flagship"
                and int(r["iteration"]) == iteration
                and r["tensor"] == tensor
                and r["axis"] == "element"
            ]
            count = sum(round(float(r["frac_below_dead_ratio"]) * int(r["channels"])) for r in rows)
            values.append((iteration, count))
        dead_series.append((label, color, values))
    dead_chart = svg_line_chart(
        dead_series,
        width=780,
        height=330,
        y_min=0,
        y_max=40,
        y_label="gain elements below 1% of peers",
        x_ticks=dead_iters,
        percent=True,
    ).replace("0.0%", "0").replace("10.0%", "10").replace("20.0%", "20").replace("30.0%", "30").replace("40.0%", "40")

    # Norm layer-0 trajectory, used as a compact factual callout.
    combined_tensors = read_csv(PRIOR / "tensor_trajectories.csv") + late_tensors
    layer0 = {
        (r["run"], int(r["iteration"])): float(r["rms"])
        for r in combined_tensors
        if r["tensor"] == "decoder.layers.self_attention.linear_qkv.layer_norm_weight"
        and int(r["layer"]) == 0
    }
    norm_history_specs = [
        ("attention-input", "Attention-input norm", "decoder.layers.self_attention.linear_qkv.layer_norm_weight"),
        ("pre-mlp", "Pre-MLP norm", "decoder.layers.mlp.linear_fc1.layer_norm_weight"),
        ("q-norm", "Q norm", "decoder.layers.self_attention.q_layernorm.weight"),
        ("k-norm", "K norm", "decoder.layers.self_attention.k_layernorm.weight"),
        ("final", "Final norm", "decoder.final_layernorm.weight"),
    ]
    norm_options = []
    norm_panels = []
    for index, (slug, label, tensor) in enumerate(norm_history_specs):
        values = {
            (int(r["iteration"]), int(r["layer"])): float(r["rms"])
            for r in combined_tensors
            if r["tensor"] == tensor and r["run"] in {"flagship_prior", "flagship"}
        }
        iterations = sorted({iteration for iteration, _ in values})
        layers = sorted({layer for _, layer in values})
        highlighted = 0 if 0 in layers else layers[0]

        def median_at(iteration: int) -> float:
            ordered = sorted(values[(iteration, layer)] for layer in layers)
            middle = len(ordered) // 2
            return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2

        first, last = iterations[0], iterations[-1]
        display = "" if index == 0 else " hidden"
        highlight_label = "layer-0 RMS" if highlighted == 0 else "only-vector RMS"
        count_label = f"{len(layers)} layer vectors" if len(layers) > 1 else "single final-norm vector"
        norm_options.append(f'<option value="{slug}">{label}</option>')
        norm_panels.append(
            f'<div class="norm-panel" data-norm="{slug}"{display}>'
            f'<div class="card norm-chart">{norm_history_chart(values)}</div>'
            f'<div class="grid3 norm-kpis">'
            f'<div class="card"><div class="kpi cyan">{values[(first, highlighted)]:.3f} → {values[(last, highlighted)]:.3f}</div><div class="kpi-label">{highlight_label}, {first/1000:g}k → {last/1000:g}k</div></div>'
            f'<div class="card"><div class="kpi">{median_at(first):.3f} → {median_at(last):.3f}</div><div class="kpi-label">cross-layer median RMS</div></div>'
            f'<div class="card"><div class="kpi green">{len(iterations)}</div><div class="kpi-label">checkpoints · {count_label}</div></div>'
            f'</div></div>'
        )

    deck = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dense 32B checkpoint weights — block view</title>
<style>
:root{{--bg:#071018;--panel:#0d1925;--panel2:#122232;--ink:{COLORS['ink']};--muted:{COLORS['muted']};--grid:{COLORS['grid']};--attention:{COLORS['attention']};--mlp:{COLORS['mlp']};--norm:{COLORS['norm']};--embed:{COLORS['embed']};--fp8:{COLORS['fp8']};--bf16:{COLORS['bf16']};}}
*{{box-sizing:border-box}} html{{scroll-snap-type:y mandatory;background:var(--bg)}} body{{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}} .slide{{min-height:100vh;scroll-snap-align:start;padding:5.5vh 6vw 5vh;display:flex;flex-direction:column;position:relative;overflow:hidden;background:radial-gradient(circle at 88% 12%,rgba(96,165,250,.11),transparent 28%),linear-gradient(145deg,#071018,#09131d 52%,#071018)}} .slide:after{{content:attr(data-n);position:absolute;right:3.3vw;bottom:2.5vh;color:#506176;font-size:13px;letter-spacing:.12em}} h1{{font-size:clamp(54px,7.2vw,104px);line-height:.94;letter-spacing:-.055em;margin:0;max-width:1080px}} h2{{font-size:clamp(34px,4.2vw,62px);line-height:1.02;letter-spacing:-.035em;margin:0 0 2.5vh}} h3{{font-size:22px;margin:0 0 10px}} p{{font-size:clamp(17px,1.55vw,24px);line-height:1.42;color:var(--muted);margin:.5em 0}} .eyebrow{{font-size:14px;letter-spacing:.16em;text-transform:uppercase;color:#7dd3fc;font-weight:750;margin-bottom:3vh}} .sub{{font-size:clamp(22px,2.3vw,34px);line-height:1.25;max-width:1000px;color:#c6d1df;margin-top:3vh}} .accent{{color:#fb923c}} .cyan{{color:#67e8f9}} .violet{{color:#c4b5fd}} .green{{color:#86efac}} .grid2{{display:grid;grid-template-columns:1fr 1fr;gap:22px;flex:1;min-height:0}} .grid3{{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}} .card{{background:linear-gradient(145deg,rgba(18,34,50,.95),rgba(12,25,38,.92));border:1px solid rgba(148,163,184,.18);border-radius:18px;padding:22px;box-shadow:0 18px 50px rgba(0,0,0,.18)}} .kpi{{font-size:clamp(34px,4vw,60px);font-weight:780;letter-spacing:-.04em}} .kpi-label{{font-size:15px;line-height:1.35;color:var(--muted);margin-top:8px}} .hero-bottom{{margin-top:auto}} .verdict{{border-left:5px solid var(--attention);padding:8px 0 8px 22px;font-size:clamp(24px,2.6vw,40px);line-height:1.25;max-width:1180px}} .pill{{display:inline-flex;align-items:center;gap:8px;padding:7px 12px;border-radius:999px;background:#14263a;color:#cbd5e1;font-size:13px;margin:4px 6px 4px 0}} .dot{{display:inline-block;width:10px;height:10px;border-radius:50%}} .chart{{display:block;width:100%;height:auto;overflow:visible}} .grid{{stroke:var(--grid);stroke-width:1}} .axis{{stroke:#64748b;stroke-width:1.2}} .event{{stroke:#64748b;stroke-width:1.5;stroke-dasharray:5 5}} .event-label{{fill:#94a3b8;font-size:12px;font-weight:700}} .turn{{stroke:#facc15;stroke-width:2;stroke-dasharray:6 5}} .turn-label{{fill:#fde047;font-size:12px;font-weight:700}} .tick{{fill:#9aa7b8;font-size:13px}} .strong{{font-weight:700;fill:#d7e0ea}} .axis-label{{fill:#94a3b8;font-size:13px}} .bar-label{{fill:#071018;font-weight:800;font-size:15px}} .mini-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;flex:1;min-height:0}} .mini{{background:rgba(13,25,37,.8);border:1px solid rgba(148,163,184,.14);border-radius:16px;padding:12px 14px 4px;min-height:0}} .mini-title{{font-size:16px;font-weight:750;display:flex;align-items:center;gap:9px}} .takeaway{{font-size:clamp(18px,1.7vw,27px);color:#dbe6f3;line-height:1.35}} .small{{font-size:14px;color:#8392a7;line-height:1.45}} .block-list{{display:grid;gap:13px}} .block{{display:grid;grid-template-columns:130px 1fr auto;gap:16px;align-items:center;border-left:5px solid var(--c);background:#0e1d2a;border-radius:10px;padding:15px 18px}} .block b{{font-size:19px;color:var(--c)}} .block span{{color:#aebaca;font-size:15px}} .block strong{{font-variant-numeric:tabular-nums;font-size:17px}} .legend{{display:flex;gap:22px;align-items:center;flex-wrap:wrap;color:#aab8c8;font-size:14px}} .legend i{{display:inline-block;width:25px;height:5px;border-radius:5px;margin-right:7px;vertical-align:middle}} .heat-stat{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:10px}} .heat-stat .card{{padding:15px}} .heat-stat b{{font-size:23px}} .split{{display:grid;grid-template-columns:1.55fr .75fr;gap:22px;flex:1;align-items:center}} .bigcall{{font-size:58px;font-weight:780;letter-spacing:-.045em;line-height:1;color:#d8b4fe}} .arrow{{color:#66788d;padding:0 5px}} .decision-grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}} .decision{{padding:20px 22px;border-radius:14px;background:#0e1d2a;border-top:4px solid var(--c)}} .decision h3{{color:var(--c)}} .decision p{{font-size:17px}} .sources{{font-size:13px;line-height:1.5;color:#738399}} code{{color:#bad4f5;background:#102033;border-radius:5px;padding:2px 5px}} .nav{{position:fixed;right:18px;top:50%;transform:translateY(-50%);z-index:10;display:grid;gap:8px}} .nav a{{width:8px;height:8px;border-radius:50%;background:#526174;display:block}} .nav a:hover{{background:#f8fafc;transform:scale(1.35)}} @media(max-width:850px){{.slide{{padding:5vh 5vw;overflow:auto}}.grid2,.grid3,.mini-grid,.split,.decision-grid{{grid-template-columns:1fr}}.nav{{display:none}}h1{{font-size:52px}}.block{{grid-template-columns:1fr}}}}
.norm-toolbar{{display:flex;align-items:center;justify-content:space-between;gap:20px;margin-bottom:14px}} .norm-toolbar label{{font-size:15px;color:#aab8c8}} select{{margin-left:10px;padding:10px 38px 10px 13px;border:1px solid #526174;border-radius:9px;background:#102033;color:#e8edf6;font:inherit;font-weight:700}} .norm-panel[hidden]{{display:none}} .norm-chart{{padding:10px 18px 4px}} .norm-kpis{{margin-top:14px}} .norm-kpis .card{{padding:14px 18px}} .norm-kpis .kpi{{font-size:clamp(28px,3vw,45px)}}
.layer-scale{{display:inline-flex;align-items:center;gap:8px}} .layer-gradient{{display:inline-block;width:150px;height:7px;border-radius:5px;background:linear-gradient(90deg,#22d3ee,#a78bfa,#f97316)}}
.layer-line{{cursor:crosshair;transition:stroke-width .08s,stroke-opacity .08s}} .layer-line:hover{{stroke-width:6;stroke-opacity:1}} .median-line:hover{{stroke-width:7}} .hover-readout{{min-width:112px;color:#f8fafc;font-size:15px;font-weight:800;text-align:center;padding:8px 12px;border:1px solid #526174;border-radius:9px;background:#102033}}
@media print{{html{{scroll-snap-type:none}}.slide{{height:7.5in;min-height:7.5in;width:13.333in;break-after:page;padding:.5in .65in}}.nav{{display:none}}}}
</style></head><body>
<nav class="nav" aria-label="Slides">{''.join(f'<a href="#s{i}" title="Slide {i}"></a>' for i in range(1,10))}</nav>

<section class="slide" id="s1" data-n="01 / 09">
  <div class="eyebrow">OELLM · dense 32B · checkpoint state scan</div>
  <h1>Weights are stable.<br><span class="accent">Attention drifts fastest.</span></h1>
  <p class="sub">A block-by-block view of 4k–75k training history and the controlled FP8/BF16 fork.</p>
  <div class="grid3 hero-bottom">
    <div class="card"><div class="kpi green">0</div><div class="kpi-label">non-finite tensors, skipped tensors, or exact-zero logical tensors</div></div>
    <div class="card"><div class="kpi cyan">64 / 64</div><div class="kpi-label">layers show excess FP8 QKV growth by 68k</div></div>
    <div class="card"><div class="kpi violet">+0.13 pp</div><div class="kpi-label">attention-output RMS-rate excess vs BF16, per 1k steps</div></div>
  </div>
</section>

<section class="slide" id="s2" data-n="02 / 09">
  <div class="eyebrow">Model anatomy</div><h2>Group the evidence by what the block does</h2>
  <div class="grid2">
    <div class="card">
      <div class="block-list">
        <div class="block" style="--c:var(--attention)"><b>Attention</b><span>Fused QKV + output projection · 64 layers</span><strong>17.82%</strong></div>
        <div class="block" style="--c:var(--mlp)"><b>MLP</b><span>SwiGLU FC1 + FC2 · 64 layers</span><strong>74.26%</strong></div>
        <div class="block" style="--c:var(--norm)"><b>Norms</b><span>Input, pre-MLP, Q/K, final gains</span><strong>0.002%</strong></div>
        <div class="block" style="--c:var(--embed)"><b>Edges</b><span>Token embedding + output head</span><strong>7.92%</strong></div>
      </div>
    </div>
    <div class="card">
      <h3>33.89B named parameters</h3>
      {stacked_share(shares)}
      <p class="takeaway">MLPs dominate parameter count, but the clearest controlled divergence is in <span class="accent">attention</span>. Norms are tiny by count and still useful as tail diagnostics.</p>
    </div>
  </div>
</section>

<section class="slide" id="s3" data-n="03 / 09">
  <div class="eyebrow">Full trajectory · flagship · indexed to 4k</div><h2>No catastrophic knee in any block</h2>
  <div class="mini-grid">{''.join(mini_charts)}</div>
  <p class="small">Parameter-weighted group RMS. The vertical line marks iteration 66,625. Matrix growth is large over the full run but smooth; the late acceleration is shallow and concentrated in attention.</p>
</section>

<section class="slide" id="s4" data-n="04 / 09">
  <div class="eyebrow">Controlled fork · same weights at 60k · measured at 68k</div><h2>FP8 adds drift mainly inside attention</h2>
  <div class="legend"><span><i style="background:var(--fp8)"></i>flagship FP8</span><span><i style="background:var(--bf16)"></i>BF16 control</span><span>colored underline = block type</span></div>
  {rate_bars(rates)}
  <div class="grid3">
    <div class="card"><div class="kpi accent">+0.126 pp</div><div class="kpi-label">attention output growth-rate excess</div></div>
    <div class="card"><div class="kpi accent">+0.091 pp</div><div class="kpi-label">QKV growth-rate excess</div></div>
    <div class="card"><div class="kpi" style="color:var(--mlp)">≈ +0.02 pp</div><div class="kpi-label">FC1 / FC2 excess — much smaller</div></div>
  </div>
</section>

<section class="slide" id="s5" data-n="05 / 09">
  <div class="eyebrow">Layer localization · FP8 excess change at 68k vs BF16</div><h2>The attention signal spans depth</h2>
  <p class="small">Each cell is one transformer layer. Orange = FP8 grew more since the shared 60k checkpoint; blue = BF16 grew more. Hover cells for exact values.</p>
  {heatmap(heat_rows)}
  <div class="heat-stat">
    <div class="card"><b class="accent">64 / 64</b><div class="kpi-label">QKV layers positive · mean {heat_stats['QKV'][1]:+.2f}%</div></div>
    <div class="card"><b class="accent">64 / 64</b><div class="kpi-label">attention-output layers positive · mean {heat_stats['Attention output'][1]:+.2f}%</div></div>
    <div class="card"><b style="color:var(--mlp)">62 / 64</b><div class="kpi-label">FC1 positive · mean {heat_stats['MLP FC1'][1]:+.2f}%</div></div>
    <div class="card"><b style="color:var(--mlp)">56 / 64</b><div class="kpi-label">FC2 positive · mean {heat_stats['MLP FC2'][1]:+.2f}%</div></div>
  </div>
</section>

<section class="slide" id="s6" data-n="06 / 09">
  <div class="eyebrow">Normalization blocks · tail behavior</div><h2>Gain tails decay; matrices do not die</h2>
  <div class="split">
    <div class="card">
      <div class="legend"><span><i style="background:var(--norm)"></i>Pre-MLP norm</span><span><i style="background:#f472b6"></i>Q norm</span><span><i style="background:#c084fc"></i>K norm</span><span><i style="background:#fde047"></i>Attention-input norm</span></div>
      {dead_chart}
      <p class="small">Each line counts individual gain parameters—not layers or whole tensors—below 1% of the median gain in the same norm vector, summed across all 64 layers.</p>
    </div>
    <div>
      <div class="card"><div class="bigcall">{layer0[('flagship_prior',4000)]:.3f}<span class="arrow">→</span>{layer0[('flagship',60000)]:.3f}<span class="arrow">→</span>{layer0[('flagship',75126)]:.3f}</div><p>Layer-0 attention-input norm RMS, 4k → 60k → 75k.</p><p class="small">Dramatic, but smooth. The BF16 continuation reaches {layer0[('bf16',68000)]:.3f} by 68k; because it inherits the 60k state, this rules out only continued FP8—not latent earlier damage.</p></div>
      <div class="card" style="margin-top:18px"><div class="kpi green">0</div><div class="kpi-label">dead rows or columns in any 2-D matrix from 4k–75k</div></div>
    </div>
  </div>
  <p class="small">“Dead” = channel RMS below 1% of its peer-channel median. First crossings occur well before the loss turn: K norm 16k, pre-MLP 24k, Q norm 28k.</p>
</section>

<section class="slide" id="s7" data-n="07 / 09">
  <div class="eyebrow">Normalization · complete flagship history</div><h2>Inspect every norm family</h2>
  <div class="norm-toolbar">
    <div class="legend"><span class="layer-scale">layer 0 <i class="layer-gradient"></i> layer 63</span><span><i style="background:#f8fafc"></i>cross-layer median</span></div>
    <div><span id="layer-hover" class="hover-readout">Hover: —</span><label for="norm-select">Norm family <select id="norm-select">{''.join(norm_options)}</select></label></div>
  </div>
  {''.join(norm_panels)}
</section>

<section class="slide" id="s8" data-n="08 / 09">
  <div class="eyebrow">Interpretation</div><h2>What the checkpoint weights do—and do not—say</h2>
  <div class="decision-grid">
    <div class="decision" style="--c:var(--good)"><h3>Ruled down</h3><p>Catastrophic corruption, non-finite weights, exact-zero tensors, and dead matrix rows/columns.</p></div>
    <div class="decision" style="--c:var(--attention)"><h3>Lead weight signature</h3><p>Mild FP8-specific growth-rate inflection in QKV and attention output projections.</p></div>
    <div class="decision" style="--c:var(--norm)"><h3>Likely symptom</h3><p>Norm-tail collapse starts early and accelerates late. Its continuation under BF16 does not exclude damage inherited at the 60k fork.</p></div>
    <div class="decision" style="--c:var(--embed)"><h3>Still unresolved</h3><p>The scan is correlational. It does not establish why the loss turns at 66,625.</p></div>
  </div>
  <div class="verdict hero-bottom"><strong>Bottom line:</strong> the weights remain globally coherent. Attention carries the strongest differential signal, but it is a candidate marker—not a causal explanation.</div>
</section>

<section class="slide" id="s9" data-n="09 / 09">
  <div class="eyebrow">Decision path</div><h2>Test the attention signature against function</h2>
  <div class="grid3">
    <div class="card"><div class="kpi accent">01</div><h3>Fixed-data loss</h3><p>Score 60k–72k checkpoints on identical held-out batches and correlate the turn with blockwise drift.</p></div>
    <div class="card"><div class="kpi cyan">02</div><h3>Updates + optimizer</h3><p>Measure update-to-weight ratios and optimizer-state movement. Bucket-only attribution is the current limitation.</p></div>
    <div class="card"><div class="kpi violet">03</div><h3>Attention internals</h3><p>Join QKV/output drift to activation amax, gradients, and FP8 scales by layer.</p></div>
  </div>
  <div class="card hero-bottom">
    <h3>Evidence base</h3>
    <p class="sources">Complete scans of 515 logical tensors and 773 channel groups per checkpoint; 33,890,653,184 named parameters. Flagship: 4k–75,126. BF16 control: 60k, 64k, 68k. Statistics are exact for RMS/extrema and deterministic samples for directional drift. Generated from <code>family_summary.csv</code>, <code>tensor_trajectories.csv</code>, and <code>channel_trajectories.csv</code> on 2026-09-02.</p>
    <div class="legend"><span><i style="background:var(--attention)"></i>Attention</span><span><i style="background:var(--mlp)"></i>MLP</span><span><i style="background:var(--norm)"></i>Normalization</span><span><i style="background:var(--embed)"></i>Embedding + head</span></div>
  </div>
</section>
<script>
const slides=[...document.querySelectorAll('.slide')];
const normSelect=document.getElementById('norm-select');
normSelect.addEventListener('change',()=>{{document.querySelectorAll('.norm-panel').forEach(panel=>panel.hidden=panel.dataset.norm!==normSelect.value)}});
const layerHover=document.getElementById('layer-hover');
document.querySelectorAll('.layer-line').forEach(line=>{{line.addEventListener('pointerenter',()=>{{layerHover.textContent=line.dataset.layer==='median'?'Cross-layer median':`Layer ${{line.dataset.layer}}`}});line.addEventListener('pointerleave',()=>{{layerHover.textContent='Hover: —'}})}});
document.addEventListener('keydown',e=>{{if(e.target.tagName==='SELECT')return;let i=slides.findIndex(s=>Math.abs(s.getBoundingClientRect().top)<innerHeight/2);if(['ArrowDown','ArrowRight',' ','PageDown'].includes(e.key)){{e.preventDefault();slides[Math.min(i+1,slides.length-1)].scrollIntoView({{behavior:'smooth'}})}}if(['ArrowUp','ArrowLeft','PageUp'].includes(e.key)){{e.preventDefault();slides[Math.max(i-1,0)].scrollIntoView({{behavior:'smooth'}})}}}});
</script></body></html>"""
    OUTPUT.write_text(deck)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    build()

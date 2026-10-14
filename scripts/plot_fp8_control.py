#!/usr/bin/env python3
"""The controlled test: does the loss turn need FP8?

cont4b forks from the flagship at iteration 63,125 and runs the identical
schedule and data in bf16. Anything that happens in BOTH runs is not caused by
FP8. Three measures, one panel each.

    python3 scripts/plot_fp8_control.py    # -> docs/64k-debug/fp8_control.png
"""

import collections
import csv
import math
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = "docs/64k-debug/data"
BREAK, FORK, LO, HI = 66625, 63125, 64000, 68000
FLAG_C, BF16_C = "#2a78d6", "#eb6834"  # categorical slots 1 and 2
INK, INK2, MUTED, SURF = "#0b0b0b", "#52514e", "#b5b3ad", "#fcfcfb"
TENSORS = ["linear_qkv", "linear_fc1", "q_layernorm", "k_layernorm"]
WTENSORS = [
    "self_attention.linear_qkv",
    "self_attention.linear_proj",
    "mlp.linear_fc1",
    "mlp.linear_fc2",
]


def style(ax):
    ax.grid(True, color=MUTED, alpha=0.35, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.set_facecolor(SURF)


def loss(path, width=125):
    b = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        i = int(r["iter"])
        if FORK <= i <= HI:
            b[i // width * width + width // 2].append(float(r["lm_loss"]))
    return sorted((k, st.median(v)) for k, v in b.items())


def gain_floor(path):
    """{tensor: {iter: median over layers of the min channel gain}}"""
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(path)):
        acc[r["tensor"]][int(r["iter"])].append(float(r["min"]))
    return {t: {i: st.median(v) for i, v in d.items()} for t, d in acc.items()}


def wstats(path):
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(path)):
        acc[(r["run"], r["tensor"])][int(r["iter"])].append(float(r["wmax"]))
    return {k: {i: st.median(v) for i, v in d.items()} for k, d in acc.items()}


fig, axes = plt.subplots(
    1, 3, figsize=(15.5, 5.2), facecolor=SURF, gridspec_kw={"width_ratios": [1.25, 1, 1]}
)

# --- 1. the loss ----------------------------------------------------------
ax = axes[0]
for path, label, color in [
    (f"{D}/loss.csv", "flagship (FP8)", FLAG_C),
    (f"{D}/loss_cont4b.csv", "cont4b (bf16)", BF16_C),
]:
    xs, ys = zip(*loss(path))
    ax.plot(xs, ys, color=color, lw=1.8, label=label, zorder=3)
ax.axvline(BREAK, color=MUTED, lw=1.4, zorder=1)
ax.axvline(FORK, color=MUTED, lw=1.2, ls=(0, (5, 4)), zorder=1)
style(ax)
ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower right")
ax.set_xlabel("iteration", color=INK2, fontsize=9.5)
ax.set_ylabel("lm loss", color=INK2, fontsize=9.5)
ax.set_title(
    "1. Both runs turn up at 66,625 — FP8 just turns harder",
    color=INK,
    fontsize=10.5,
    loc="left",
    pad=8,
)
ax.annotate(
    "fork\n63,125", (FORK + 80, ax.get_ylim()[1]), color=INK2, fontsize=8, ha="left", va="top"
)
ax.annotate(
    "loss turns\n66,625", (BREAK, ax.get_ylim()[1]), color=INK2, fontsize=8, ha="center", va="top"
)

# --- 2. gain floor --------------------------------------------------------
gf, gc = gain_floor(f"{D}/norm_gains.csv"), gain_floor(f"{D}/norm_gains_cont4b.csv")
ax = axes[1]
x = np.arange(len(TENSORS))
w = 0.36
for k, (d, label, color) in enumerate(
    [(gf, "flagship (FP8)", FLAG_C), (gc, "cont4b (bf16)", BF16_C)]
):
    # log-decay per 1k iters, negative = the floor is falling
    v = [math.log(d[t][HI] / d[t][LO]) / ((HI - LO) / 1000) for t in TENSORS]
    bars = ax.bar(x + (k - 0.5) * w, v, w * 0.92, color=color, label=label, zorder=3, lw=0)
    ax.bar_label(bars, fmt="%.3f", fontsize=7.5, color=INK2, padding=2)
ax.set_xticks(x)
ax.set_xticklabels(TENSORS, fontsize=8, rotation=12, ha="right")
ax.axhline(0, color=INK2, lw=1, zorder=2)
style(ax)
ax.set_ylabel(
    "min-gain decay per 1k iters\n(more negative = collapsing faster)", color=INK2, fontsize=9
)
ax.set_title(
    "2. Gain floor: bf16 collapses FASTER on 3 of 4", color=INK, fontsize=10.5, loc="left", pad=8
)

# --- 3. weights -----------------------------------------------------------
ws = wstats(f"{D}/weight_stats.csv")
ax = axes[2]
x = np.arange(len(WTENSORS))
for k, (run, label, color) in enumerate(
    [("flagship", "flagship (FP8)", FLAG_C), ("cont4b", "cont4b (bf16)", BF16_C)]
):
    v = [(ws[(run, t)][HI] / ws[(run, t)][LO] - 1) * 100 for t in WTENSORS]
    bars = ax.bar(x + (k - 0.5) * w, v, w * 0.92, color=color, label=label, zorder=3, lw=0)
    ax.bar_label(bars, fmt="%+.1f%%", fontsize=7.5, color=INK2, padding=2)
ax.set_xticks(x)
ax.set_xticklabels([t.split(".")[-1] for t in WTENSORS], fontsize=8, rotation=12, ha="right")
style(ax)
ax.set_ylim(0, max(b.get_height() for c in ax.containers for b in c) * 1.22)
ax.set_ylabel("max|W| growth 64k->68k (%)", color=INK2, fontsize=9)
ax.set_title("3. Weight outlier growth: same in both", color=INK, fontsize=10.5, loc="left", pad=8)

fig.suptitle(
    "The controlled test: cont4b runs the same data in bf16 from iteration 63,125",
    color=INK,
    fontsize=13.5,
    x=0.032,
    ha="left",
    y=0.975,
)
fig.text(
    0.032,
    0.915,
    "Anything that happens in BOTH runs is not caused by FP8. "
    "All three measures happen in both. Window 64,000 -> 68,000 spans the break.",
    color=INK2,
    fontsize=9,
)
fig.tight_layout(rect=(0, 0, 1, 0.88))
fig.savefig("docs/64k-debug/fp8_control.png", dpi=150, facecolor=SURF)
print("wrote docs/64k-debug/fp8_control.png")

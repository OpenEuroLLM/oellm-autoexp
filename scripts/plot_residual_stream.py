#!/usr/bin/env python3
"""Per-layer residual-stream budget across checkpoints.

Reads docs/64k-debug/data/residual_stream.csv (scan_residual_stream.py)
and draws the four quantities per layer -- attention gain, attention
branch RMS, residual stream RMS -- plus the two gauge-invariant
branch/stream ratios and their trend over training.

Read the profiles across checkpoints, never one at a time: the
branch/stream ratio falls with depth in every trained transformer, so
the shape is the baseline and only a CHANGE in it over training means
anything.

python3 scripts/plot_residual_stream.py
"""

import csv
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURF, INK, MUTE = "#fcfcfb", "#0b0b0b", "#b5b3ad"
GREY, GREEN, ORANGE = "#52514e", "#1baf7a", "#eb6834"
# Sequential single-hue ramp: checkpoints are ordered in time, so lightness
# carries the ordering and no reader has to decode nine arbitrary hues.
RAMP = [
    "#cfe0f7",
    "#a9c8ef",
    "#82afe6",
    "#5b96dd",
    "#2a78d6",
    "#2361ad",
    "#1c4a85",
    "#14345e",
    "#0d2340",
]
ONSET = 60500

rows = defaultdict(dict)
for r in csv.DictReader(open("docs/64k-debug/data/residual_stream.csv")):
    rows[int(r["iter"])][int(r["layer"])] = {
        k: (float(v) if v else np.nan) for k, v in r.items() if k not in ("iter", "layer")
    }
its = sorted(rows)
cols = [RAMP[round(i * (len(RAMP) - 1) / max(len(its) - 1, 1))] for i in range(len(its))]


def prof(it, key):
    d = rows[it]
    return np.array([d[L][key] for L in sorted(d)])


L = np.arange(len(rows[its[0]]))
fig, ax = plt.subplots(2, 3, figsize=(15.5, 8.2), facecolor=SURF)
ax = ax.ravel()
for a in ax:
    a.set_facecolor(SURF)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        a.spines[s].set_color(MUTE)
    a.tick_params(colors=GREY, labelsize=9)
    a.grid(True, color=MUTE, alpha=0.35, lw=0.6)
    a.set_axisbelow(True)

panels = [
    ("attn_gain_rms", "rms(gain), attention input-norm", "Attention gain", True),
    ("attn_rms", "rms of the attention branch output", "Attention branch, absolute size", True),
    ("h_out_rms", "rms(h) leaving the layer", "Residual stream", True),
    ("attn_ratio", "rms(attn) / rms(h)", "Attention share of the stream", True),
    ("mlp_ratio", "rms(mlp) / rms(h)", "MLP share of the stream", True),
]
for a, (key, ylab, title, logy) in zip(ax, panels):
    for it, c in zip(its, cols):
        a.plot(L, prof(it, key), "-", color=c, lw=1.8, label=f"{it:,}")
    if logy:
        a.set_yscale("log")
    a.set_xlabel("layer", color=GREY, fontsize=9)
    a.set_ylabel(ylab, color=GREY, fontsize=9)
    a.set_title(title, color=INK, fontsize=11, loc="left", fontweight="bold")

# Trend panel: the profiles are overlaid, so the movement over training needs
# its own axes.
a = ax[5]
mid = [L for L in sorted(rows[its[0]]) if 2 <= L <= 61]
for key, col, mk, lab in (
    ("attn_ratio", ORANGE, "o", "attention"),
    ("mlp_ratio", GREY, "s", "MLP"),
):
    y = [np.median([rows[i][L][key] for L in mid]) for i in its]
    a.plot(
        its,
        np.array(y) / y[0],
        mk + "-",
        color=col,
        lw=2.0,
        ms=5,
        label=f"{lab}, median of layers 2-61",
    )
y0 = [rows[i][0]["attn_ratio"] for i in its]
a.plot(
    its,
    np.array(y0) / y0[0],
    "^--",
    color=ORANGE,
    lw=1.6,
    ms=6,
    mfc=SURF,
    label="attention, layer 0",
)
a.axhline(1.0, color=MUTE, lw=1.0)
a.axvline(ONSET, color=GREEN, ls=":", lw=1.8)
a.annotate(
    f"onset ~{ONSET:,}",
    (ONSET, a.get_ylim()[0]),
    color=GREEN,
    fontsize=8.5,
    ha="right",
    va="bottom",
    rotation=90,
    xytext=(ONSET - 1200, a.get_ylim()[0]),
)
a.set_xlabel("iteration", color=GREY, fontsize=9)
a.set_ylabel(f"branch share, relative to {its[0]:,}", color=GREY, fontsize=9)
a.set_title("Trend: is any of it moving?", color=INK, fontsize=11, loc="left", fontweight="bold")
a.legend(frameon=False, fontsize=8.5, labelcolor=GREY, loc="best")

h, lb = ax[0].get_legend_handles_labels()
fig.legend(
    h,
    lb,
    frameon=False,
    fontsize=8.5,
    labelcolor=GREY,
    ncol=len(its),
    loc="lower center",
    bbox_to_anchor=(0.5, -0.005),
    title="iteration",
    title_fontproperties={"size": 8.5},
)
fig.tight_layout(rect=(0, 0.045, 1, 1))
fig.savefig("docs/64k-debug/residual_stream.png", dpi=150, facecolor=SURF)
print("wrote docs/64k-debug/residual_stream.png")

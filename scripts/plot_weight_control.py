#!/usr/bin/env python3
"""Is the flagship's weight growth anomalous, and is it bulk or outliers?

`fp8_amax.csv`'s weight amax comes from TE's delayed-scaling `_extra_state`, so
it exists only in FP8 runs and the bf16 control cannot be compared against it.
This plots max|W| and rms|W| recomputed offline from the weight tensors
(scan_weight_stats.py), which every run has.

    python3 scripts/plot_weight_control.py     # -> weight_control.png

Colour encodes the RUN in every panel; the row encodes the measure. Left column
is the trajectory, right column the controlled 60k->64k comparison -- a few
percent over 4,000 steps is invisible on a 75,000-step axis, so it gets bars.
"""

import collections
import csv
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BREAK, FORK, BASE = 66625, 60000, 8000
RUNS = [
    ("flagship", "flagship (FP8 hybrid)", "#2a78d6"),  # categorical slot 1
    ("cont4bf16", "cont4 (bf16)", "#eb6834"),
]  # slot 2
INK, INK2, MUTED, SURF = "#0b0b0b", "#52514e", "#b5b3ad", "#fcfcfb"
TENSORS = [
    "self_attention.linear_qkv",
    "self_attention.linear_proj",
    "mlp.linear_fc1",
    "mlp.linear_fc2",
]


def load(path="docs/fp8-loss-turn/data/weight_stats.csv"):
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(path)):
        for m in ("wmax", "wrms"):
            acc[(r["run"], r["tensor"], m)][int(r["iter"])].append(float(r[m]))
    return {k: {i: st.median(v) for i, v in d.items()} for k, d in acc.items()}


def style(ax):
    ax.grid(True, color=MUTED, alpha=0.35, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.set_facecolor(SURF)


d = load()
fig, axes = plt.subplots(
    2, 2, figsize=(12.5, 8.4), facecolor=SURF, gridspec_kw={"width_ratios": [1, 1.15]}
)

for row, (m, mlab) in enumerate([("wmax", "max|W|"), ("wrms", "rms|W|")]):
    # --- left: trajectory, median across the four tensors -------------------
    ax = axes[row][0]
    for run, label, color in RUNS:
        # Index each tensor to the FLAGSHIP baseline before taking the median,
        # so cont4's fork point lands on the flagship line iff they really do
        # share that checkpoint.
        per = collections.defaultdict(list)
        for t in TENSORS:
            s = d.get((run, t, m))
            if not s:
                continue
            base = d[("flagship", t, m)][BASE]
            for i, v in s.items():
                per[i].append(v / base)
        if not per:
            continue
        xs = sorted(per)
        ax.plot(
            xs,
            [st.median(per[x]) for x in xs],
            color=color,
            lw=2,
            marker="o",
            ms=6,
            label=label,
            zorder=3,
        )
    ax.axvline(FORK, color=MUTED, lw=1.2, ls=(0, (5, 4)), zorder=1)
    ax.axvline(BREAK, color=MUTED, lw=1.4, zorder=1)
    ax.axhline(1.0, color=INK2, lw=1, ls=":", zorder=2)
    style(ax)
    ax.set_ylabel(f"{mlab}   (x vs step {BASE:,})", color=INK2, fontsize=9.5)
    ax.set_title(
        f"{mlab}: median across the four GEMMs", color=INK, fontsize=10.5, loc="left", pad=8
    )
    if row == 1:
        ax.set_xlabel("iteration", color=INK2, fontsize=9.5)

    # --- right: the controlled window ---------------------------------------
    ax = axes[row][1]
    x = np.arange(len(TENSORS))
    w = 0.36
    for k, (run, label, color) in enumerate(RUNS):
        vals = []
        for t in TENSORS:
            s = d.get((run, t, m), {})
            vals.append((s[64000] / s[60000] - 1) * 100 if 60000 in s and 64000 in s else np.nan)
        bars = ax.bar(
            x + (k - 0.5) * w, vals, w * 0.92, color=color, label=label, zorder=3, linewidth=0
        )
        ax.bar_label(bars, fmt="%+.2f%%", fontsize=7.5, color=INK2, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels([t.split(".")[-1] for t in TENSORS], fontsize=8.5)
    style(ax)
    ax.set_ylabel(f"{mlab} growth, 60k->64k (%)", color=INK2, fontsize=9.5)
    ax.set_title(
        f"{mlab}: same 4,000 steps from identical weights",
        color=INK,
        fontsize=10.5,
        loc="left",
        pad=8,
    )
    ax.set_ylim(0, max(1e-9, np.nanmax([b.get_height() for c in ax.containers for b in c])) * 1.28)

handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    frameon=False,
    fontsize=9.5,
    labelcolor=INK2,
    ncol=2,
    loc="upper left",
    bbox_to_anchor=(0.055, 0.905),
    handlelength=1.6,
)
fig.suptitle(
    "Weight growth is in the outliers - but the bf16 control grows just as fast",
    color=INK,
    fontsize=13.5,
    x=0.055,
    ha="left",
    y=0.975,
)
fig.text(
    0.055,
    0.932,
    "Recomputed from the weight tensors, so both runs are measurable. cont4 forks from the "
    "flagship at 60,000 (dashed); the loss turns at 66,625 (solid).",
    color=INK2,
    fontsize=9,
)
fig.tight_layout(rect=(0, 0, 1, 0.89))
fig.savefig("docs/fp8-loss-turn/weight_control.png", dpi=150, facecolor=SURF)
print("wrote docs/fp8-loss-turn/weight_control.png")


# The fork is a free correctness check: cont4 continues from the flagship's own
# 60,000 checkpoint, so at that iteration the two runs must agree exactly. If
# they do not, the scan is wrong.
print("\nfork check at 60,000 -- flagship vs cont4bf16 must be identical:")
_ok = True
for _t in TENSORS:
    for _m in ("wmax", "wrms"):
        _a = d.get(("flagship", _t, _m), {}).get(FORK)
        _b = d.get(("cont4bf16", _t, _m), {}).get(FORK)
        if _a is None or _b is None:
            continue
        _same = _a == _b
        _ok &= _same
        print(f"  {_t:<32} {_m}  {_a:.6g} vs {_b:.6g}  {'MATCH' if _same else '*** DIFFER ***'}")
print("all identical:", _ok)

#!/usr/bin/env python3
"""Five panels on why the norm-gain and activation-growth theories do not
explain the loss turn -- and the one statistic that does move at it.

    python3 scripts/plot_drift.py          # -> drift_evidence.png

Inputs: norm_gains.csv (scan_norm_gains.py), fp8_amax.csv (scan_fp8_amax.py),
loss.csv (extract_loss.py).
"""

import collections
import csv
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SWAP = 34455  # stack swap: mcore 0.19 + TE 2.18 + document masking
BREAK = 66625  # where the loss turned around
GAP = (64000, 68000)  # BREAK falls in here and no checkpoint does

# Categorical slots 1-5 of the reference palette, in fixed order.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, MUTED, SURF = "#0b0b0b", "#52514e", "#b5b3ad", "#fcfcfb"


def series(path, key, value):
    """{name: {iter: median over layers}}"""
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(path)):
        out[r[key]][int(r["iter"])].append(float(r[value]))
    return {k: {i: st.median(v) for i, v in d.items()} for k, d in out.items()}


def binned_loss(path, width=250, first=4000):
    """Median lm loss per `width` iterations -- 15k raw points is a smear.

    Starts at `first` so the warmup plunge does not flatten the rest, and so
    the x-axis matches the checkpoint-derived panels.
    """
    b = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        if int(r["iter"]) < first:
            continue
        b[int(r["iter"]) // width * width + width // 2].append(float(r["lm_loss"]))
    return sorted((k, st.median(v)) for k, v in b.items())


def markers(ax, gap=True, top=1.0):
    """The two event lines, plus the blind spot around the break.

    `top` clips the lines in axes fraction, to keep them out of an inset.
    """
    if gap:
        ax.axvspan(*GAP, color=MUTED, alpha=0.16, lw=0, zorder=0)
    for x, style in [(SWAP, (0, (6, 4))), (BREAK, "-")]:
        ax.axvline(x, ymax=top, color=MUTED, lw=1.4, ls=style, zorder=1)


def style(ax):
    ax.grid(True, color=MUTED, alpha=0.35, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_facecolor(SURF)


def draw(ax, data, order, labels, logy=False, ncol=4, gap=True):
    for slot, name in enumerate(order):
        xs = sorted(data[name])
        ax.plot(
            xs,
            [data[name][x] for x in xs],
            color=C[slot],
            lw=2,
            marker="o",
            ms=4,
            label=labels[name],
            zorder=3,
        )
    if logy:
        ax.set_yscale("log")
    markers(ax, gap)
    style(ax)
    # Legend sits in the gap above the axes: these panels are too dense to
    # give up plot area to a legend box.
    ax.legend(
        frameon=False,
        fontsize=8.5,
        labelcolor=INK2,
        ncol=ncol,
        loc="lower left",
        bbox_to_anchor=(0, 1.005),
        handlelength=1.6,
        columnspacing=1.6,
        borderpad=0,
    )


def title(ax, txt):
    ax.set_title(txt, color=INK, fontsize=11, loc="left", pad=26)


gains = series("docs/64k-debug/data/norm_gains.csv", "tensor", "mean")
floor = series("docs/64k-debug/data/norm_gains.csv", "tensor", "min")
amax = series("docs/64k-debug/data/fp8_amax.csv", "layer", "amax_act")
wgt = series("docs/64k-debug/data/fp8_amax.csv", "layer", "amax_wgt")
loss = binned_loss("docs/64k-debug/data/loss.csv")

fig, axes = plt.subplots(
    5,
    1,
    figsize=(9, 16.5),
    sharex=True,
    facecolor=SURF,
    gridspec_kw={"height_ratios": [0.85, 1, 1, 1, 1]},
)

g_order = ["linear_qkv", "linear_fc1", "q_layernorm", "k_layernorm", "final_layernorm"]
g_lab = {
    "linear_qkv": "input_layernorm",
    "linear_fc1": "pre_mlp_layernorm",
    "q_layernorm": "q_layernorm",
    "k_layernorm": "k_layernorm",
    "final_layernorm": "final_layernorm",
}

# --- 0. the thing being explained ----------------------------------------
xs, ys = zip(*loss)
axes[0].plot(xs, ys, color=INK, lw=1.6, zorder=3)
markers(axes[0], gap=False, top=0.38)  # logged every step: no blind spot here
style(axes[0])
title(axes[0], "The loss falls to 1.539 at 66,625, then climbs for the rest of the run")
axes[0].set_ylabel("lm loss", color=INK2, fontsize=9.5)

# The turn is 0.045 on a curve that spans 0.7: show it zoomed where the
# full-range curve leaves the panel empty.
ins = axes[0].inset_axes([0.52, 0.40, 0.46, 0.56], zorder=6)
zoom = [(x, y) for x, y in loss if 56000 <= x <= 76000]
ins.plot(*zip(*zoom), color=INK, lw=1.5, zorder=3)
ins.axvline(BREAK, color=MUTED, lw=1.4, zorder=1)
style(ins)
ins.tick_params(labelsize=7.5)
ins.set_title("zoom: 56k-75k", color=INK2, fontsize=8, loc="left", pad=3)

# --- 1. norm gains, mean --------------------------------------------------
draw(axes[1], gains, g_order, g_lab, ncol=5)
axes[1].axhline(1.0, color=INK2, lw=1, ls=":", zorder=2)
axes[1].set_ylim(0.72, 2.05)
axes[1].annotate("init = 1.0", (44000, 1.03), color=INK2, fontsize=8.5)
title(axes[1], "Mean norm gains drift smoothly - and slow down, rather than kink, at the turn")
axes[1].set_ylabel("mean gain", color=INK2, fontsize=9.5)

# --- 2. norm gains, floor -------------------------------------------------
# The mean is held up by a max that never moves; what actually happens is the
# smallest channel in each tensor collapsing. Median over layers of per-layer min.
draw(axes[2], floor, g_order, g_lab, logy=True, ncol=5)
axes[2].axhline(1.0, color=INK2, lw=1, ls=":", zorder=2)
title(
    axes[2],
    "But the FLOOR collapses - and after the break pre_mlp_layernorm's decay re-accelerates ~7x",
)
axes[2].set_ylabel("min gain over channels (log)", color=INK2, fontsize=9.5)
axes[2].annotate(
    "after the break, not at it:\nreads as symptom, not cause",
    (9000, 0.013),
    color=INK2,
    fontsize=8.5,
    ha="left",
)

# --- 3. activation amax ---------------------------------------------------
a_order = [
    "self_attention.linear_qkv",
    "mlp.linear_fc1",
    "self_attention.linear_proj",
    "mlp.linear_fc2",
]
a_lab = {
    "self_attention.linear_qkv": "linear_qkv in  (after norm)",
    "mlp.linear_fc1": "linear_fc1 in  (after norm)",
    "self_attention.linear_proj": "linear_proj in  (attention out)",
    "mlp.linear_fc2": "linear_fc2 in  (post-SwiGLU)",
}
draw(axes[3], amax, a_order, a_lab, logy=True, ncol=4)
title(axes[3], "Only the two inputs NOT bounded by a norm grow - and they grow from step 4,000")
axes[3].set_ylabel("activation amax (log)", color=INK2, fontsize=9.5)


# --- 4. indexed growth: weights vs activations -----------------------------
# Two measures on very different scales, so index both to iteration 8,000
# rather than reaching for a second y-axis.
def indexed(d):
    per = collections.defaultdict(list)
    for mod in d:
        for i, v in d[mod].items():
            per[i].append(v)
    med = {i: st.median(v) for i, v in per.items()}
    base = med[8000]
    return {i: v / base for i, v in med.items()}


idx = {"weights": indexed(wgt), "activations": indexed(amax)}
draw(
    axes[4],
    idx,
    ["weights", "activations"],
    {"weights": "weight amax", "activations": "activation amax"},
    ncol=2,
)
axes[4].axhline(1.0, color=INK2, lw=1, ls=":", zorder=2)
title(axes[4], "The thing that actually grows is the WEIGHT amax, not the activations")
axes[4].set_ylabel("growth vs step 8,000 (x)", color=INK2, fontsize=9.5)
axes[4].set_xlabel("iteration", color=INK2, fontsize=9.5)

for x, txt in [(SWAP, "stack swap\n34,455"), (BREAK, "loss turns\n66,625")]:
    axes[1].annotate(txt, (x, 1.97), color=INK2, fontsize=8.5, ha="center", va="top")
axes[1].annotate(
    "no checkpoint in here",
    (sum(GAP) / 2, 1.13),
    color=INK2,
    fontsize=7.5,
    ha="center",
    va="center",
    rotation=90,
)

fig.suptitle(
    "Nothing kinks at iteration 66,625 - except the norm-gain floor, and only after it",
    color=INK,
    fontsize=13.5,
    x=0.06,
    ha="left",
    y=0.989,
)
fig.tight_layout(rect=(0, 0, 1, 0.972))
fig.savefig("docs/64k-debug/drift_evidence.png", dpi=150, facecolor=SURF)
print("wrote docs/64k-debug/drift_evidence.png")

#!/usr/bin/env python3
"""Adam second moment against adam_eps, over the run and across the control.

python3 scripts/plot_optimizer_state.py  # ->
docs/64k-debug/optimizer_state.png
"""

import collections
import csv
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "docs/64k-debug/data"
EPS, BREAK = 1e-8, 66625
C1, C2 = "#2a78d6", "#eb6834"  # categorical slots 1-2
INK, INK2, MUTED, SURF = "#0b0b0b", "#52514e", "#b5b3ad", "#fcfcfb"


def style(ax):
    ax.grid(True, color=MUTED, alpha=0.35, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.set_facecolor(SURF)


def load(path):
    by = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        by[int(r["iter"])].append(r)
    out = {}
    for i, g in by.items():
        out[i] = {
            k: st.median([float(r[k]) for r in g])
            for k in ("sqrt_v_p50", "step_p50", "frac_sqrtv_below_10eps")
        }
    return out


f = load(f"{D}/optimizer_state.csv")
c = load(f"{D}/optimizer_state_cont4b.csv")
its = sorted(f)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), facecolor=SURF)

ax = axes[0]
ax.plot(
    its,
    [f[i]["sqrt_v_p50"] / EPS for i in its],
    color=C1,
    lw=2,
    marker="o",
    ms=5,
    zorder=3,
    label="flagship (FP8)",
)
ck = sorted(c)
ax.plot(
    ck,
    [c[i]["sqrt_v_p50"] / EPS for i in ck],
    color=C2,
    lw=2,
    marker="o",
    ms=7,
    zorder=4,
    label="cont4b (bf16)",
)
ax.axhline(1, color=INK2, lw=1, ls=":", zorder=2)
ax.axvline(BREAK, color=MUTED, lw=1.4, zorder=1)
style(ax)
ax.set_yscale("log")
ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper right")
ax.set_xlabel("iteration", color=INK2, fontsize=9.5)
ax.set_ylabel(r"median sqrt(v) / adam_eps", color=INK2, fontsize=9.5)
ax.set_title(
    "1. The second moment falls toward eps all run", color=INK, fontsize=10.5, loc="left", pad=8
)
ax.annotate("eps", (its[0], 1.15), color=INK2, fontsize=8)

ax = axes[1]
ax.plot(
    its,
    [f[i]["frac_sqrtv_below_10eps"] * 100 for i in its],
    color=C1,
    lw=2,
    marker="o",
    ms=5,
    zorder=3,
)
ax.plot(
    ck,
    [c[i]["frac_sqrtv_below_10eps"] * 100 for i in ck],
    color=C2,
    lw=2,
    marker="o",
    ms=7,
    zorder=4,
)
ax.axvline(BREAK, color=MUTED, lw=1.4, zorder=1)
style(ax)
ax.set_xlabel("iteration", color=INK2, fontsize=9.5)
ax.set_ylabel("% of params within 10x of eps", color=INK2, fontsize=9.5)
ax.set_title("2. Most of the model ends up eps-damped", color=INK, fontsize=10.5, loc="left", pad=8)

ax = axes[2]
ax.plot(its, [f[i]["step_p50"] for i in its], color=C1, lw=2, marker="o", ms=5, zorder=3)
ax.plot(ck, [c[i]["step_p50"] for i in ck], color=C2, lw=2, marker="o", ms=7, zorder=4)
ax.axvline(BREAK, color=MUTED, lw=1.4, zorder=1)
style(ax)
ax.set_xlabel("iteration", color=INK2, fontsize=9.5)
ax.set_ylabel(r"median |m| / (sqrt(v) + eps)", color=INK2, fontsize=9.5)
ax.set_title("3. Effective step bottoms out at 64,000", color=INK, fontsize=10.5, loc="left", pad=8)

fig.suptitle(
    "Adam state: a driver that moves one way for 50,000 iterations, then reverses at the break",
    color=INK,
    fontsize=13,
    x=0.033,
    ha="left",
    y=0.975,
)
fig.tight_layout(rect=(0, 0, 1, 0.91))
fig.savefig("docs/64k-debug/optimizer_state.png", dpi=150, facecolor=SURF)
print("wrote docs/64k-debug/optimizer_state.png")

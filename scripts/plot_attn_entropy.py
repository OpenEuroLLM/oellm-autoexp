#!/usr/bin/env python3
"""Attention entropy across the break -- the test of the QK-norm hypothesis.

python3 scripts/plot_attn_entropy.py   # ->
docs/64k-debug/attn_entropy.png
"""

import collections
import csv
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "docs/64k-debug/data/attn_entropy.csv"
BREAK = 66625
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


rows = list(csv.DictReader(open(SRC)))
by_iter = collections.defaultdict(list)
for r in rows:
    by_iter[int(r["iter"])].append((int(r["layer"]), float(r["entropy_nats"])))
its = sorted(by_iter)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), facecolor=SURF)

ax = axes[0]
ax.plot(
    its,
    [st.mean([e for _, e in by_iter[i]]) for i in its],
    color=C1,
    lw=2,
    marker="o",
    ms=6,
    label="mean over layers",
    zorder=3,
)
ax.plot(
    its,
    [min(e for _, e in by_iter[i]) for i in its],
    color=C2,
    lw=2,
    marker="o",
    ms=6,
    label="most concentrated layer",
    zorder=3,
)
ax.axvline(BREAK, color=MUTED, lw=1.4, zorder=1)
style(ax)
ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="center left")
ax.set_xlabel("iteration", color=INK2, fontsize=9.5)
ax.set_ylabel("attention entropy (nats)", color=INK2, fontsize=9.5)
ax.set_title(
    "Entropy RISES after the break — no collapse", color=INK, fontsize=10.5, loc="left", pad=8
)
ax.annotate(
    "loss turns\n66,625", (BREAK, ax.get_ylim()[1]), color=INK2, fontsize=8, ha="center", va="top"
)

ax = axes[1]
for it, color, lw in [(64000, C1, 2), (75126, C2, 2)]:
    pts = sorted(by_iter[it])
    ax.plot(
        [lay for lay, _ in pts],
        [e for _, e in pts],
        color=color,
        lw=lw,
        label=f"iteration {it:,}",
        zorder=3,
    )
style(ax)
ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")
ax.set_xlabel("layer", color=INK2, fontsize=9.5)
ax.set_ylabel("attention entropy (nats)", color=INK2, fontsize=9.5)
ax.set_title("Per layer: best checkpoint vs the end", color=INK, fontsize=10.5, loc="left", pad=8)

fig.suptitle(
    "Attention entropy: the QK-norm hypothesis, refuted",
    color=INK,
    fontsize=13,
    x=0.04,
    ha="left",
    y=0.975,
)
fig.tight_layout(rect=(0, 0, 1, 0.90))
fig.savefig("docs/64k-debug/attn_entropy.png", dpi=150, facecolor=SURF)
print("wrote docs/64k-debug/attn_entropy.png")

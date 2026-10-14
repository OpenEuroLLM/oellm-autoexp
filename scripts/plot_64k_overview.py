#!/usr/bin/env python3
"""Three pictures of the 64k problem: the loss turn, the held-out flip, z-loss.

python3 scripts/plot_64k_overview.py   # -> docs/64k-debug/issue_overview.png
"""

import collections
import csv
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOSS = "docs/64k-debug/data/loss.csv"
ZLOSS = "docs/64k-debug/data/zloss.csv"
BREAK, SWAP = 66625, 34455
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3
INK, INK2, MUTED, SURF = "#0b0b0b", "#52514e", "#b5b3ad", "#fcfcfb"

# Held-out scores (lower = better); jobs in DEBUG.md.
EVAL = {60000: 1.500890, 64000: 1.495481, 68000: 1.511213, 72000: 1.519173}


def style(ax):
    ax.grid(True, color=MUTED, alpha=0.35, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.set_facecolor(SURF)


def binned(path, col, width=250, first=4000):
    b = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        i = int(r["iter"])
        if i >= first:
            b[i // width * width + width // 2].append(float(r[col]))
    return sorted((k, st.median(v)) for k, v in b.items())


fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURF)

# --- 1. the loss, against the trend it was on -----------------------------
ax = axes[0]
pts = binned(LOSS, "lm_loss")
xs, ys = zip(*pts)
ax.plot(xs, ys, color=C1, lw=1.7, zorder=3, label="training loss")
# A local linear trend over the last healthy stretch. Deliberately NOT the
# log-extrapolation from 12k: over 40k iterations that form over-diverges and
# would overstate the gap.
fit = [(x, y) for x, y in pts if 50000 <= x <= 64000]
fx = [x / 1000 for x, _ in fit]
fy = [y for _, y in fit]
n = len(fx)
mx, my = sum(fx) / n, sum(fy) / n
sl = sum((a - mx) * (b - my) for a, b in zip(fx, fy)) / sum((a - mx) ** 2 for a in fx)
ic = my - sl * mx
ex = [x for x, _ in pts if x >= 50000]
ax.plot(
    ex,
    [ic + sl * (x / 1000) for x in ex],
    color=INK2,
    lw=1.3,
    ls=(0, (5, 4)),
    zorder=2,
    label="local trend, 50k-64k",
)
ax.axvline(BREAK, color=MUTED, lw=1.4, zorder=1)
ax.axvline(SWAP, color=MUTED, lw=1.2, ls=(0, (6, 4)), zorder=1)
style(ax)
ax.set_ylim(1.52, 1.66)
ax.set_xlim(40000, 77000)
ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower left")
ax.set_xlabel("iteration", color=INK2, fontsize=9.5)
ax.set_ylabel("lm loss", color=INK2, fontsize=9.5)
ax.set_title("1. The loss leaves its trend and climbs", color=INK, fontsize=10.5, loc="left", pad=8)
ax.annotate("turn\n66,625", (BREAK, 1.652), color=INK2, fontsize=8, ha="center", va="top")

# --- 2. held-out scores ---------------------------------------------------
ax = axes[1]
ks = sorted(EVAL)
ax.plot(ks, [EVAL[k] for k in ks], color=C2, lw=2, marker="o", ms=8, zorder=3)
best = min(EVAL, key=EVAL.get)
ax.annotate(
    f"best checkpoint: {best:,}",
    (best, EVAL[best]),
    color=INK2,
    fontsize=8.5,
    ha="left",
    va="center",
    xytext=(12, -4),
    textcoords="offset points",
)
for k in ks:
    ax.annotate(
        f"{EVAL[k]:.4f}",
        (k, EVAL[k]),
        color=INK2,
        fontsize=7.5,
        ha="center",
        va="bottom",
        xytext=(0, 7),
        textcoords="offset points",
    )
ax.axvline(BREAK, color=MUTED, lw=1.4, zorder=1)
style(ax)
ax.set_xlabel("checkpoint iteration", color=INK2, fontsize=9.5)
ax.set_ylabel("held-out loss (lower = better)", color=INK2, fontsize=9.5)
ax.set_title(
    "2. The model itself gets worse after 64,000", color=INK, fontsize=10.5, loc="left", pad=8
)

# --- 3. z-loss diagnostic -------------------------------------------------
ax = axes[2]
zs = binned(ZLOSS, "z", width=500, first=4000)
zx, zy = zip(*zs)
ax.plot(zx, zy, color=C3, lw=1.7, zorder=3)
ax.axvline(BREAK, color=MUTED, lw=1.4, zorder=1)
ax.axvline(SWAP, color=MUTED, lw=1.2, ls=(0, (6, 4)), zorder=1)
style(ax)
ax.set_xlabel("iteration", color=INK2, fontsize=9.5)
ax.set_ylabel(r"mean(log$Z$$^2$)", color=INK2, fontsize=9.5)
ax.set_title(
    "3. z-loss is flat — it does nothing at the turn", color=INK, fontsize=10.5, loc="left", pad=8
)

fig.suptitle(
    "The 64k problem in three pictures", color=INK, fontsize=13.5, x=0.033, ha="left", y=0.975
)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig("docs/64k-debug/issue_overview.png", dpi=150, facecolor=SURF)
print("wrote docs/64k-debug/issue_overview.png")

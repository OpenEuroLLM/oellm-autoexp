#!/usr/bin/env python3
"""Adam's eps damping saturates long before the loss departs, and unwinds
after.

The physically meaningful quantity is not sqrt(v) but the fraction of each Adam
step that eps eats:  step = lr*m/(sqrt(v)+eps), so suppression = eps/(sqrt(v)+eps).
Left panel: it climbs over the first ~24,000 iterations, then sits flat at ~15-16%
straight through the onset of the loss departure (~60,500), and only *falls* after
64,000 -- i.e. the damping relaxes once the model is already degrading. Right
panel: the median step itself, at two independent coverages, showing the same.

All series are size-weighted across optimizer shards (an earlier version took an
unweighted median across shards, which let a 135M-parameter shard speak for 33.9B).

  python3 scripts/plot_optimizer_turn.py
"""

import csv
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURF, INK, MUTE = "#fcfcfb", "#0b0b0b", "#b5b3ad"
BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
ONSET, EPS = 60500, 1e-8
D = "docs/64k-debug/data/"


def load(fname):
    """Size-weighted sqrt(v) median and step median per iteration."""
    by = defaultdict(list)
    for r in csv.DictReader(open(D + fname)):
        by[int(r["iter"])].append(r)
    its = np.array(sorted(by), float)
    out = {}
    for key in ("sqrt_v_p50", "step_p50"):
        out[key] = np.array(
            [
                sum(float(r[key]) * int(r["numel"]) for r in by[int(i)])
                / sum(int(r["numel"]) for r in by[int(i)])
                for i in its
            ]
        )
    out["numel"] = np.array([sum(int(r["numel"]) for r in by[int(i)]) for i in its])
    return its, out


early_i, early = load("optimizer_state.csv")  # 4.4% coverage, 8,000 onward
sprd_i, sprd = load("optimizer_state_spread.csv")  # 9.5% coverage, 44,000 onward
wide_i, wide = load("optimizer_state_wide.csv")  # 35% coverage, verification


def supp(d):
    return 100 * EPS / (d["sqrt_v_p50"] + EPS)


early_i = early_i[early_i < 44000]
early = {k: v[: len(early_i)] for k, v in early.items()}

fig, ax = plt.subplots(1, 2, figsize=(12, 4.3), facecolor=SURF)
for a in ax:
    a.set_facecolor(SURF)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        a.spines[s].set_color(MUTE)
    a.tick_params(colors=GREY, labelsize=9)
    a.grid(True, color=MUTE, alpha=0.35, lw=0.6)
    a.set_axisbelow(True)
    a.axvline(ONSET, color=GREEN, ls=":", lw=1.8)
    a.set_xlabel("iteration", color=GREY, fontsize=9)

ax[0].plot(
    early_i,
    supp(early),
    "o--",
    color=BLUE,
    lw=1.6,
    ms=5,
    mfc=SURF,
    label="8 buckets, 4.4% of model",
)
ax[0].plot(sprd_i, supp(sprd), "o-", color=BLUE, lw=2.0, ms=5, label="8 buckets spread, 9.5%")
ax[0].plot(wide_i, supp(wide), "D", color=ORANGE, ms=7, label="28 buckets, 35% (verification)")
ax[0].set_ylabel("% of each Adam step eaten by eps", color=GREY, fontsize=9)
ax[0].set_ylim(0, 20)
ax[0].set_title(
    "eps damping saturates by 24,000 and is flat through the onset",
    color=INK,
    fontsize=11,
    loc="left",
    fontweight="bold",
)
ax[0].legend(frameon=False, fontsize=8.5, labelcolor=GREY, loc="lower left")
ax[0].annotate(
    "onset of the loss\ndeparture (~60,500)",
    (ONSET, 3.0),
    color=GREEN,
    fontsize=8.5,
    ha="left",
    va="bottom",
    xytext=(ONSET + 1200, 3.0),
)

ax[1].plot(sprd_i, sprd["step_p50"], "o-", color=BLUE, lw=2.0, ms=5, label="9.5% of model")
ax[1].plot(wide_i, wide["step_p50"], "D", color=ORANGE, ms=7, label="35% of model")
ax[1].set_ylabel(r"median $|m|/(\sqrt{v}+\epsilon)$  (before lr)", color=GREY, fontsize=9)
ax[1].set_xlim(42000, 77000)
ax[1].set_title(
    "Median step: scatter only through the onset, rises after 64,000",
    color=INK,
    fontsize=11,
    loc="left",
    fontweight="bold",
)
ax[1].legend(frameon=False, fontsize=8.5, labelcolor=GREY, loc="upper left")

fig.tight_layout()
fig.savefig("docs/64k-debug/optimizer_turn.png", dpi=150, facecolor=SURF)
print("wrote docs/64k-debug/optimizer_turn.png")

#!/usr/bin/env python3
"""The single most important figure: 66,625 is a crossing point, not an event.

Left  -- the loss against the scaling law fitted on the healthy stretch. The run
         tracks the law, peels away, and only crosses back upward at 66,625.
Right -- the excess over that law. It grows smoothly as (t-t0)^2 from an onset
         near 60,500. Nothing discrete happens at 66,625; that is simply where
         the growing damage overtakes the still-improving trend.

  python3 scripts/plot_onset.py
"""

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURF, INK, MUTE = "#fcfcfb", "#0b0b0b", "#b5b3ad"
BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
FIT_LO, FIT_HI = 36000, 60000
BREAK, ONSET = 66625, 60500

it, ls = [], []
with open("docs/64k-debug/data/loss.csv") as f:
    for r in csv.DictReader(f):
        it.append(int(r["iter"]))
        ls.append(float(r["lm_loss"]))
x, y = np.array(it, float), np.array(ls, float)

m = (x >= FIT_LO) & (x <= FIT_HI)
best = None
for a in np.arange(0.05, 2.0, 0.002):
    D = np.column_stack([np.ones(m.sum()), x[m] ** (-a)])
    b, *_ = np.linalg.lstsq(D, y[m], rcond=None)
    rss = float(((y[m] - D @ b) ** 2).sum())
    if best is None or rss < best[0]:
        best = (rss, a, b)
_, alpha, (Linf, A) = best


def law(t):
    return Linf + A * t ** (-alpha)


def binned(lo, hi, step=500):
    xs, ys = [], []
    for s in range(lo, hi, step):
        w = (x >= s) & (x < s + step)
        if w.sum() >= 20:
            xs.append(s + step / 2)
            ys.append(y[w].mean())
    return np.array(xs), np.array(ys)


bx, by = binned(FIT_LO, 75200)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=SURF)
for a in ax:
    a.set_facecolor(SURF)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        a.spines[s].set_color(MUTE)
    a.tick_params(colors=GREY, labelsize=9)
    a.grid(True, color=MUTE, alpha=0.35, lw=0.6)
    a.set_axisbelow(True)
    a.axvline(BREAK, color=GREY, ls="--", lw=1.4)
    a.axvline(ONSET, color=GREEN, ls=":", lw=1.8)

ax[0].plot(bx, by, color=BLUE, lw=2.0, label="training loss (500-iter mean)")
tt = np.linspace(FIT_LO, 75200, 400)
ax[0].plot(
    tt,
    law(tt),
    color=ORANGE,
    lw=2.0,
    ls="--",
    label=f"scaling law fitted on {FIT_LO // 1000}k-{FIT_HI // 1000}k",
)
ax[0].set_xlabel("iteration", color=GREY, fontsize=9)
ax[0].set_ylabel("lm loss", color=GREY, fontsize=9)
ax[0].set_title(
    "The run leaves its scaling law", color=INK, fontsize=11, loc="left", fontweight="bold"
)
ax[0].legend(frameon=False, fontsize=8.5, labelcolor=GREY, loc="upper center")

ex = by - law(bx)
ax[1].axhline(0, color=MUTE, lw=1.0)
ax[1].plot(bx, ex, color=BLUE, lw=2.0, label="excess loss over the law")
w = bx > ONSET
ax[1].plot(
    bx[w],
    np.polyfit(bx[w] - ONSET, ex[w], 2)[0] * (bx[w] - ONSET) ** 2,
    color=ORANGE,
    lw=2.0,
    ls="--",
    label="(t - t0)$^2$, t0 = 60,500",
)
ax[1].set_xlabel("iteration", color=GREY, fontsize=9)
ax[1].set_ylabel("loss above trend", color=GREY, fontsize=9)
ax[1].set_title(
    "The damage grows smoothly, from well before the 'break'",
    color=INK,
    fontsize=11,
    loc="left",
    fontweight="bold",
)
ax[1].legend(frameon=False, fontsize=8.5, labelcolor=GREY, loc="upper left")

ax[1].annotate(
    "onset ~60,500",
    (ONSET, ax[1].get_ylim()[1] * 0.42),
    color=GREEN,
    fontsize=8.5,
    ha="right",
    rotation=90,
    va="top",
)
ax[1].annotate(
    "66,625 -- where the curve\nturns upward (a crossing,\nnot an event)",
    (BREAK, ax[1].get_ylim()[1] * 0.05),
    color=GREY,
    fontsize=8.5,
    ha="left",
    va="bottom",
    xytext=(BREAK + 900, ax[1].get_ylim()[1] * 0.05),
)

fig.tight_layout()
fig.savefig("docs/64k-debug/onset.png", dpi=150, facecolor=SURF)
print(f"alpha={alpha:.3f} Linf={Linf:.4f}  excess at 75k = {ex[-1]:+.4f}")
print("wrote docs/64k-debug/onset.png")

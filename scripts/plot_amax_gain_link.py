#!/usr/bin/env python3
"""Is the growing weight amax the mirror image of the decaying layer-0 norm
gain?

In a pre-norm transformer the function depends only on the product
gain (*) W_qkv, so a shrinking gain paired with a growing weight leaves the
activations unchanged. That trade is real -- but only in layers 0-1. Everywhere
else the gain grows too, and the flat activation amax has a structural cause
(RMSNorm pins its output to unit RMS), not a compensatory one. What the amax plot
is really tracking is a growing outlier tail: max|W| outruns rms|W| by 2x in qkv
and 8x in fc2, identically in the bf16 control.

  python3 scripts/plot_amax_gain_link.py
"""

import csv
from collections import defaultdict
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURF, INK, MUTE = "#fcfcfb", "#0b0b0b", "#b5b3ad"
BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
ONSET = 60500


def read(path, **cast):
    for r in csv.DictReader(open(path)):
        for k, f in cast.items():
            r[k] = f(r[k])
        yield r


# --- panel A: qkv input-norm gain, layer 0 vs the other 63 ---------------------
g = defaultdict(dict)
for r in read("docs/64k-debug/data/gain_distribution.csv", iter=int, layer=int, median_abs=float):
    if r["tensor"] == "linear_qkv":
        g[r["iter"]][r["layer"]] = r["median_abs"]
gi = np.array(sorted(g), float)
g0 = np.array([g[int(i)][0] for i in gi])
g1 = np.array([g[int(i)][1] for i in gi])
gm = np.array([median(v for L, v in g[int(i)].items() if L > 1) for i in gi])

# --- panel B: layer-0 activation amax into qkv vs the model median -------------
a = defaultdict(list)
for r in read("fp8_amax_history.csv", iter=int, shard=int, amax_act=float, amax_wgt=float):
    if r["layer"] == "self_attention.linear_qkv":
        a[r["iter"]].append(r)
# Skip the warm-up: the amax history is still settling before ~16,000.
ai = np.array([i for i in sorted(a) if i >= 16000], float)
a0 = np.array([next(r["amax_act"] for r in a[int(i)] if r["shard"] == 0) for i in ai])
am = np.array([median(r["amax_act"] for r in a[int(i)]) for i in ai])
wm = np.array([median(r["amax_wgt"] for r in a[int(i)]) for i in ai])

# --- panel C: max|W| vs rms|W| -- outlier tail, flagship and bf16 control ------
ws = defaultdict(list)
for r in read("docs/64k-debug/data/weight_stats.csv", iter=int, wmax=float, wrms=float):
    ws[(r["run"], r["tensor"])].append(r)


def ratio(run, tensor):
    rows = ws[(run, tensor)]
    its = sorted({r["iter"] for r in rows})
    return np.array(its, float), np.array(
        [median(r["wmax"] / r["wrms"] for r in rows if r["iter"] == i) for i in its]
    )


fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3), facecolor=SURF)
for x in ax:
    x.set_facecolor(SURF)
    for s in ("top", "right"):
        x.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        x.spines[s].set_color(MUTE)
    x.tick_params(colors=GREY, labelsize=9)
    x.grid(True, color=MUTE, alpha=0.35, lw=0.6)
    x.set_axisbelow(True)
    x.axvline(ONSET, color=GREEN, ls=":", lw=1.8)
    x.set_xlabel("iteration", color=GREY, fontsize=9)

ax[0].semilogy(gi, g0, "o-", color=ORANGE, lw=2.0, ms=5, label="layer 0")
ax[0].semilogy(gi, g1, "s-", color=GREY, lw=1.6, ms=4, label="layer 1")
ax[0].semilogy(gi, gm, "o-", color=BLUE, lw=2.0, ms=5, label="median, layers 2-63")
ax[0].set_ylabel(r"median $|g|$, qkv input-norm gain", color=GREY, fontsize=9)
ax[0].set_title(
    "Only layers 0-1 decay; 50 of 64 grow", color=INK, fontsize=11, loc="left", fontweight="bold"
)
ax[0].legend(frameon=False, fontsize=8.5, labelcolor=GREY, loc="lower left")

ax[1].plot(ai, a0 / a0[0], "o-", color=ORANGE, lw=2.0, ms=5, label="layer 0")
ax[1].plot(ai, am / am[0], "o-", color=BLUE, lw=2.0, ms=5, label="median of 64 layers")
ax[1].plot(ai, wm / wm[0], "^--", color=BLUE, lw=1.6, ms=5, mfc=SURF, label="median weight amax")
ax[1].axhline(1.0, color=MUTE, lw=1.0)
ax[1].set_ylabel("amax, relative to iteration 16,000", color=GREY, fontsize=9)
ax[1].set_title(
    "Layer 0 is the one place activations are not flat",
    color=INK,
    fontsize=11,
    loc="left",
    fontweight="bold",
)
ax[1].legend(frameon=False, fontsize=8.5, labelcolor=GREY, loc="upper left")

for tensor, col, lab in (
    ("mlp.linear_fc2", ORANGE, "mlp.linear_fc2"),
    ("self_attention.linear_qkv", BLUE, "self_attention.linear_qkv"),
):
    i, y = ratio("flagship", tensor)
    ax[2].plot(i, y, "o-", color=col, lw=2.0, ms=5, label=lab + "  (FP8)")
    for run, mk in (("cont4bf16", "D"), ("cont4b", "s")):
        if (run, tensor) in ws:
            i2, y2 = ratio(run, tensor)
            ax[2].plot(i2, y2, mk, color=col, ms=7, mfc=SURF, mew=1.6)
ax[2].plot([], [], "D", color=GREY, mfc=SURF, mew=1.6, label="bf16 controls")
ax[2].set_ylabel(r"$\max|W| \,/\, \mathrm{rms}|W|$  (outlier tail)", color=GREY, fontsize=9)
ax[2].set_title(
    "Weight amax growth is an outlier tail; bf16 matches",
    color=INK,
    fontsize=11,
    loc="left",
    fontweight="bold",
)
ax[2].legend(frameon=False, fontsize=8.5, labelcolor=GREY, loc="center left")

fig.tight_layout()
fig.savefig("docs/64k-debug/amax_gain_link.png", dpi=150, facecolor=SURF)
print("wrote docs/64k-debug/amax_gain_link.png")

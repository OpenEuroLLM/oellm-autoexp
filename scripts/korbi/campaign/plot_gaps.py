"""Loss curves and paired gaps with their ln-t trend, as PNG (gap_trend.py's numbers, drawn).

    python scripts/korbi/campaign/plot_gaps.py out.png --ref <suite dir>:<arm> \
        --arm <suite dir>:<arm>[=label] [--arm ...] [--title ...]

Left: the loss per 250-step window, on log steps. Right: each arm's gap to the reference per
window (same data order), the fit a + b ln t from --fit-start on and its extrapolation to the
--predict-tokens marks.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from gap_trend import runs  # noqa: E402


def _arm(spec):
    path, _, rest = spec.partition(":")
    arm, _, label = rest.partition("=")
    return Path(path), arm, label or arm


def windows(d, ref, w, start=0):
    common = sorted(i for i in d if ref is None or i in ref)
    xs, ys = [], []
    for lo in range(start, common[-1] - w + 1, w):
        vals = [d[i] - (ref[i] if ref is not None else 0) for i in common if lo < i <= lo + w]
        if len(vals) >= 5:
            xs.append(lo + w / 2)
            ys.append(np.mean(vals))
    return np.array(xs), np.array(ys)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--ref", required=True)
    p.add_argument("--arm", action="append", required=True)
    p.add_argument("--title", default="")
    p.add_argument("--window", type=int, default=250)
    p.add_argument("--fit-start", type=int, default=1000)
    p.add_argument("--tokens-per-step", type=float, default=64 * 8192)
    p.add_argument("--predict-tokens", type=float, nargs="+", default=[4e9, 1e10])
    a = p.parse_args()
    cache = {}

    def load(path, arm):
        if path not in cache:
            cache[path] = runs(path)
        return cache[path][arm]

    rpath, rarm, rlabel = _arm(a.ref)
    ref = load(rpath, rarm)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x, y = windows(ref, None, a.window)
    ax1.plot(x, y, "k-", lw=2, label=f"{rlabel} (reference)")
    t_max = max(a.predict_tokens) / a.tokens_per_step
    for i, spec in enumerate(a.arm):
        path, arm, label = _arm(spec)
        d = load(path, arm)
        c = f"C{i}"
        x, y = windows(d, None, a.window)
        ax1.plot(x, y, "-", color=c, label=label)
        x, g = windows(d, ref, a.window)
        ax2.plot(x, g, "o", color=c, ms=3, label=label)
        fit = x >= a.fit_start
        if fit.sum() >= 3:
            b, c0 = np.polyfit(np.log(x[fit]), g[fit], 1)
            t = np.geomspace(a.fit_start, t_max, 100)
            ax2.plot(t, c0 + b * np.log(t), "--", color=c, lw=1)
            for tok in a.predict_tokens:
                s = tok / a.tokens_per_step
                ax2.annotate(f"{c0 + b * np.log(s):+.3f}", (s, c0 + b * np.log(s)), color=c, fontsize=8,
                             textcoords="offset points", xytext=(3, 3))
    ax2.set_ylim(-0.12, 0.25)
    for tok in a.predict_tokens:
        ax2.axvline(tok / a.tokens_per_step, color="0.8", lw=0.8)
        ax2.text(tok / a.tokens_per_step, 0.24, f"{tok:.0e} tok", fontsize=7, ha="center", va="top", color="0.5")
    ax2.axhline(0, color="k", lw=0.8)
    ax1.set(xscale="log", xlabel="step", ylabel=f"loss ({a.window}-step windows)", title="loss")
    ax1.set_ylim(top=min(ax1.get_ylim()[1], 4.0))
    ax2.set(xscale="log", xlabel="step", ylabel=f"loss - {rlabel} (paired)",
            title=f"gap, fit a + b ln t from step {a.fit_start}")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle(a.title)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

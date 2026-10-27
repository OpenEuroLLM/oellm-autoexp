"""Per-arm loss-curve fits of the GQA rematch, for comparisons ACROSS data orders (runs at another
node count see other batches: the cyclic sampler buckets per data-parallel rank, so equal-step
gaps are batch noise there). Each arm's training loss is averaged over 100-step windows and fitted
on the stable phase (from step 500 to the cooldown at 3200, or to its last step) with
L(t) = E + A t^-a; arms are compared by their fitted loss at common steps, never step by step.
A window mean of 100 logged steps averages out most of the batch-to-batch noise (~0.05 per
batch); the fit's residual is printed as a check.

    python scripts/korbi/campaign/rematch_fit.py <local copy dir> [s2|s3 ...]
"""

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rematch_analysis as ra  # noqa: E402

WINDOW, START, COOL = 100, 500, 3200
AT = (2000, 3000, 3200)


def law(t, e, a, p):
    return e + a * np.power(t, -p)


def fit(series):
    steps = sorted(i for i in series if START < i <= COOL)
    if not steps or steps[-1] < 1500:
        return None
    centers, means = [], []
    for lo in range(START, steps[-1] - WINDOW + 1, WINDOW):
        w = [series[i] for i in steps if lo < i <= lo + WINDOW]
        if len(w) >= 5:
            centers.append(lo + WINDOW / 2)
            means.append(np.mean(w))
    t, y = np.array(centers), np.array(means)
    (e, a, p), _ = curve_fit(law, t, y, p0=(1.5, 50.0, 0.5), maxfev=20000)
    resid = float(np.sqrt(np.mean((law(t, e, a, p) - y) ** 2)))
    return {"E": e, "A": a, "p": p, "resid": resid, "last": steps[-1], "pred": {s: law(s, e, a, p) for s in AT}}


def main():
    root = sys.argv[1]
    ra.ROOT = Path(root)
    sizes = sys.argv[2:] or ["s2", "s3"]
    for size in sizes:
        train, _ = ra.load(size)
        fits = {a: fit(train[a]) for a in ra.ARMS + ["ldm_lr025", "ldm_lr0125"] if a in train}
        fits = {a: f for a, f in fits.items() if f}
        if "mhc4" not in fits:
            continue
        ref = fits["mhc4"]["pred"]
        print(f"\n## {size}: fitted stable-phase training loss L(t) = E + A t^-a (fit from {START} to "
              f"min(last, {COOL})); gaps vs mHC-4 x0.25 at the same step\n")
        print("| arm | fitted to | resid | " + " | ".join(f"L({s})" for s in AT) + " | "
              + " | ".join(f"gap @{s}" for s in AT) + " | E | a |")
        print("|---|---:|---:|" + "---:|" * (2 * len(AT) + 2))
        for a, f in fits.items():
            name = ra.NAMES.get(a, {"ldm_lr025": "LDM x0.25", "ldm_lr0125": "LDM x0.125"}.get(a, a))
            print(f"| {name} | {f['last']} | {f['resid']:.4f} | "
                  + " | ".join(f"{f['pred'][s]:.4f}" for s in AT) + " | "
                  + " | ".join(f"{f['pred'][s] - ref[s]:+.4f}" for s in AT)
                  + f" | {f['E']:.3f} | {f['p']:.2f} |")


if __name__ == "__main__":
    main()

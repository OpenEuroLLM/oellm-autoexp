#!/usr/bin/env python3
"""When does each per-iteration signal leave its own trend, and does it
accelerate?

Background. The training loss departs from its scaling law *smoothly*, growing
like (t-t0)^2 from an onset well before iteration 66,625; the "breakpoint" is
only where that growing damage overtakes the still-improving trend. So the
question for the other signals is not "does it kink at 66,000" -- nothing does --
but "does it leave its own early trend, and when".

Scope, and why it is narrow. This runs only on signals logged every iteration
(lm loss, grad norm, z-loss). The per-checkpoint scans (norm gains, FP8 amax)
cannot support the test: checkpoints were saved every 4,000 iterations, so only
seven exist between the software swap at 34,455 and the onset region, and
extrapolating a trend 20,000 iterations from seven points is not measurable.
Two earlier versions of this script tried it anyway and both failed the negative
control below -- scoring checkpoints we believe are healthy at up to 27 sd, then
8 sd. Densifying is not an option: 22 of the 23 saved checkpoints are already
scanned.

Method, per series:

  1. fit a smooth 'no special time' law on a clean post-swap window,
  2. extrapolate forward, score the departure against the *prediction* standard
     error se = s*sqrt(1 + x0'(X'X)^-1 x0), which grows with distance,
  3. if the departure is real, fit r(t) = c + C*max(t-t0,0)^p for the onset t0
     and the growth exponent p.

--control repeats step 1-2 at a matched extrapolation distance over a stretch we
believe is healthy. A series whose control does not come back small is reported
as UNUSABLE rather than interpreted: for that series the trend model is
misspecified badly enough that departures mean nothing.

No GPU and no checkpoint reads -- pure re-analysis of the logged series.

  python3 scripts/fit_onset.py --control
  python3 scripts/fit_onset.py
"""

import argparse
import csv

import numpy as np

CONTROL_MAX = 3.0  # |z| a healthy stretch must stay under for the series
# to be interpretable at all


def design(x, kind):
    lt = np.log(x)
    if kind == "log":
        return np.column_stack([np.ones(len(x)), lt])
    if kind == "log2":
        return np.column_stack([np.ones(len(x)), lt, lt**2])
    if kind == "log3":
        return np.column_stack([np.ones(len(x)), lt, lt**2, lt**3])
    raise ValueError(kind)


def fit_trend(x, y, kinds=("log", "log2", "log3")):
    """Pick the trend by BIC on the fit window; return a scorer for later
    points.

    Only models linear in their parameters, so the prediction standard
    error is exact rather than a linearisation.
    """
    n, best = len(x), None
    for kind in kinds:
        D = design(x, kind)
        p = D.shape[1]
        if n - p < 3:
            continue
        beta, *_ = np.linalg.lstsq(D, y, rcond=None)
        rss = float(((y - D @ beta) ** 2).sum())
        if rss <= 0:
            continue
        bic = n * np.log(rss / n) + p * np.log(n)
        if best is None or bic < best[0]:
            best = (bic, kind, beta, np.linalg.pinv(D.T @ D), np.sqrt(rss / (n - p)), p)
    if best is None:
        return None
    _, kind, beta, XtXi, s, p = best

    def predict(x0):
        return design(np.atleast_1d(np.asarray(x0, float)), kind) @ beta

    def score(x0, y0):
        D0 = design(np.atleast_1d(np.asarray(x0, float)), kind)
        lev = np.einsum("ij,jk,ik->i", D0, XtXi, D0)
        return (np.atleast_1d(y0) - D0 @ beta) / (s * np.sqrt(1.0 + lev))

    return predict, score, kind, n - p


def binned(x, y, edges):
    """Mean of y over each bin -- averages out per-step noise before
    scoring."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() >= 20:
            out.append((0.5 * (lo + hi), float(y[m].mean()), int(m.sum())))
    return out


def load():
    it, loss, gn = [], [], []
    with open("docs/64k-debug/data/loss.csv") as f:
        for r in csv.DictReader(f):
            it.append(int(r["iter"]))
            loss.append(float(r["lm_loss"]))
            gn.append(float(r["grad_norm"]))
    zmap = {}
    with open("docs/64k-debug/data/zloss.csv") as f:
        for r in csv.DictReader(f):
            try:
                zmap[int(r["iter"])] = float(r["z"])
            except ValueError:
                continue
    x = np.array(it, float)
    zs = np.array([zmap.get(int(i), np.nan) for i in x])
    return [
        ("lm loss", x, np.array(loss, float)),
        ("grad norm", x, np.array(gn, float)),
        ("z-loss", x, zs),
    ]


def run(fit_lo, fit_hi, test_edges, title):
    print(f"\n{title}")
    print(f"fit {fit_lo:,}-{fit_hi:,}; departure / prediction se, binned means\n")
    centres = [0.5 * (a + b) for a, b in zip(test_edges[:-1], test_edges[1:])]
    hdr = f"{'series':<12} {'trend':<6} {'dof':>6} " + " ".join(
        f"{c / 1000:>7.1f}k" for c in centres
    )
    print(hdr)
    print("-" * len(hdr))
    res = {}
    for label, x, y in load():
        ok = np.isfinite(y)
        xf, yf = x[ok], y[ok]
        m = (xf >= fit_lo) & (xf <= fit_hi)
        if m.sum() < 50:
            continue
        got = fit_trend(xf[m], yf[m])
        if got is None:
            continue
        predict, score, kind, dof = got
        cells, worst = [], 0.0
        for c, mu, _ in binned(xf, yf, test_edges):
            zsc = float(score(c, mu)[0])
            worst = max(worst, abs(zsc))
            cells.append(f"{zsc:+8.1f}")
        print(f"{label:<12} {kind:<6} {dof:>6} " + "".join(cells))
        res[label] = (worst, predict, xf, yf)
    return res


def onset(label, predict, x, y, lo=50000):
    """Fit r(t) = c + C*max(t-t0,0)^p to the residual; report the (t0,p)
    ridge."""
    w = x >= lo
    xr, rr = x[w], y[w] - predict(x[w])
    best = None
    for t0 in range(lo, 70000, 250):
        for p in np.arange(0.5, 4.01, 0.05):
            f = np.where(xr > t0, np.maximum(xr - t0, 0.0) ** p, 0.0)
            D = np.column_stack([np.ones(len(xr)), f])
            b, *_ = np.linalg.lstsq(D, rr, rcond=None)
            rss = float(((rr - D @ b) ** 2).sum())
            if best is None or rss < best[0]:
                best = (rss, t0, p)
    rss0, t0, p = best
    print(f"\n  {label}: best onset t0 = {t0:,}, exponent p = {p:.2f}")
    print(f"    {'forced t0':>10} {'best p':>7} {'RSS / best':>11}")
    for tf in range(52000, 70000, 2000):
        b2 = None
        for pp in np.arange(0.5, 4.01, 0.05):
            f = np.where(xr > tf, np.maximum(xr - tf, 0.0) ** pp, 0.0)
            D = np.column_stack([np.ones(len(xr)), f])
            bt, *_ = np.linalg.lstsq(D, rr, rcond=None)
            r2 = float(((rr - D @ bt) ** 2).sum())
            if b2 is None or r2 < b2[0]:
                b2 = (r2, pp)
        print(f"    {tf:10,} {b2[1]:7.2f} {b2[0] / rss0:11.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", action="store_true")
    args = ap.parse_args()

    if args.control:
        res = run(
            36000,
            48000,
            list(range(48000, 60001, 2000)),
            "NEGATIVE CONTROL -- this stretch is believed healthy",
        )
        print()
        for label, (worst, *_) in res.items():
            verdict = "usable" if worst < CONTROL_MAX else "UNUSABLE"
            print(f"  {label:<12} largest |z| = {worst:5.1f}   {verdict}")
        return

    ctl = run(
        36000,
        48000,
        list(range(48000, 60001, 2000)),
        "NEGATIVE CONTROL -- this stretch is believed healthy",
    )
    usable = {k for k, (w, *_) in ctl.items() if w < CONTROL_MAX}
    print(
        "\n  usable:",
        ", ".join(sorted(usable)) or "none",
        "| unusable:",
        ", ".join(sorted(set(ctl) - usable)) or "none",
    )

    res = run(
        36000, 60000, list(range(60000, 76001, 2000)), "MAIN -- departure from the pre-onset trend"
    )
    print("\nOnset fits (usable series only)")
    for label, (_, predict, x, y) in res.items():
        if label in usable:
            onset(label, predict, x, y)
        else:
            print(f"\n  {label}: skipped -- failed its negative control")


if __name__ == "__main__":
    main()

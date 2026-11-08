#!/usr/bin/env python3
"""Tail extrapolation of the ambitious v4_2ph training loss (raw and recalibrated to the q35 data mix)
against the conservative q35 run.

Input is the output directory of scripts/korbi/production/recalibrated_loss.py (recal_per_step.csv,
recal_eval_steps.csv and the W&B cache conservative_q35_main_gqa.csv). Lessons from the q35 modelling
(2026-10-29 .. 11-01): fits over the full range are dominated by the early transient and flatten too early,
so only the tail (D >= --tail-from, default 250B tokens) is fitted; every form is checked out of sample
(fit tail .. --oos-cut, predict the rest).

Forms: E + A*D^-a (free), E + A*D^-0.3, A*D^-a (pure power law), a - b*ln D (log-linear). The band in the
plot spans the forms whose out-of-sample error stays within --oos-tol mnat; the line is the best by BIC.

    python scripts/korbi/campaign/loss_extrapolation.py --recal-dir <dir> --out <dir> [--theme light|dark|both]

Writes extrapolation.json (fits, out-of-sample errors, predictions) and loss_extrapolation_<theme>.png.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

V4_TOKENS_PER_STEP = 8192 * 8192
Q35_TOKENS_PER_STEP = 2048 * 8192
PHASE1_END_STEP = 23691   # 8k -> 64k curriculum switch of v4_2ph (other data afterwards)

FORMS = {
    "E+PL": (lambda x, p: p[0] + abs(p[1]) * x ** -abs(p[2]), 3),
    "E+PL a=0.3": (lambda x, p: p[0] + abs(p[1]) * x ** -0.3, 2),
    "pure PL": (lambda x, p: abs(p[0]) * x ** -abs(p[1]), 2),
    "log-linear": (lambda x, p: p[0] - abs(p[1]) * np.log(x), 2),
}

THEMES = {   # dataviz reference palette: slot 1 blue (v4), slot 2 orange (q35)
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3dc",
              "v4": "#2a78d6", "q35": "#eb6834"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "muted": "#c3c2b7", "grid": "#33322f",
             "v4": "#3987e5", "q35": "#d95926"},
}


def bins(x, y, size):
    n = len(y) // size
    return x[: n * size].reshape(n, size).mean(1), y[: n * size].reshape(n, size).mean(1)


def fit(form, x, y, sigma, seed=0):
    f, k = FORMS[form]
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(60):
        p0 = np.r_[rng.uniform(0.8, 1.6), rng.uniform(0.01, 2, k - 1)]
        try:
            r = least_squares(lambda p: (f(x / 100, p) - y) / sigma, p0, max_nfev=6000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best


def model_series(name, x, y, a, grid):
    """Tail fits of one series: out-of-sample error, BIC, predictions on `grid` (B tokens)."""
    tail = x >= a.tail_from
    lx = np.log(x[tail])
    sigma = np.std(y[tail] - np.polyval(np.polyfit(lx, y[tail], 2), lx))
    old, new = tail & (x <= a.oos_cut), x > a.oos_cut
    res = {}
    for form, (f, k) in FORMS.items():
        r_old = fit(form, x[old], y[old], sigma)
        oos = (f(x[new] / 100, r_old.x) - y[new]) * 1000 if new.any() else np.array([np.nan])
        r = fit(form, x[tail], y[tail], sigma)
        res[form] = {"params": r.x.tolist(), "bic": float(2 * r.cost + k * np.log(tail.sum())),
                     "oos_mean_mnat": float(np.mean(oos)), "oos_last_mnat": float(oos[-1]),
                     "pred": f(grid / 100, r.x)}
    ok = [m for m in res if abs(res[m]["oos_last_mnat"]) <= a.oos_tol] or list(res)
    best = min(ok, key=lambda m: res[m]["bic"])
    band = np.array([res[m]["pred"] for m in ok])
    print(f"{name}: noise {sigma * 1000:.1f} mnat; accepted {ok}; best {best}")
    for m, v in res.items():
        print(f"   {m:11s} OOS {v['oos_mean_mnat']:+6.1f}/{v['oos_last_mnat']:+6.1f} mnat  BIC {v['bic']:8.1f}")
    return {"forms": res, "accepted": ok, "best": best, "lo": band.min(0), "hi": band.max(0),
            "line": res[best]["pred"], "sigma": float(sigma)}


def plot(theme, a, data, fits, grid, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = THEMES[theme]
    fig, ax = plt.subplots(figsize=(9.6, 5.6), dpi=160)
    fig.patch.set_facecolor(c["surface"])
    ax.set_facecolor(c["surface"])
    ext = grid >= data["v4_x"].max()
    qext = grid >= data["q35_x"].max()
    # measured (smoothed), then extrapolation bands + best-fit lines
    ax.plot(data["q35_x"], data["q35_y"], color=c["q35"], lw=2, label="q35 conservative (flag-v1), measured")
    ax.plot(data["v4_x"], data["v4_raw"], color=c["v4"], lw=1.5, ls=(0, (4, 3)), alpha=0.6,
            label="v4_2ph raw (opt81 mix)")
    ax.plot(data["v4_x"], data["v4_recal"], color=c["v4"], lw=2, label="v4_2ph recalibrated to flag-v1")
    for key, color, mask in (("q35", c["q35"], qext), ("raw", c["v4"], ext), ("recal", c["v4"], ext)):
        fz = fits[key]
        ax.fill_between(grid[mask], fz["lo"][mask], fz["hi"][mask], color=color, alpha=0.13, lw=0)
        ax.plot(grid[mask], fz["line"][mask], color=color, lw=1.5, ls=(0, (1.5, 2.5)),
                alpha=0.6 if key == "raw" else 1.0)
    ax.scatter(data["ev_x"], data["ev_recal"], s=26, color=c["v4"], zorder=3, edgecolor=c["surface"], lw=2)
    p1 = PHASE1_END_STEP * V4_TOKENS_PER_STEP / 1e9
    ax.axvline(p1, color=c["muted"], lw=1, ls=(0, (2, 3)))
    ax.text(p1 * 1.03, 1.93, "v4 phase 2 (64k data)\nstarts at 1.59T", color=c["muted"], fontsize=8.5, va="top")
    # direct labels at the right edge of the measured data
    for key, color, y in (("q35", c["q35"], fits["q35"]["line"][-1]), ("recal", c["v4"], fits["recal"]["line"][-1]),
                          ("raw", c["v4"], fits["raw"]["line"][-1])):
        lab = {"q35": f"q35  {y:.3f}", "recal": f"v4 recal  {y:.3f}", "raw": f"v4 raw  {y:.3f}"}[key]
        ax.text(grid[-1] * 1.02, y, lab, color=c["ink"], fontsize=8.5, va="center")
    ax.set_xscale("log")
    ax.set_xlim(40, grid[-1])
    ax.set_ylim(1.0, 1.95)
    ax.set_xticks([50, 100, 200, 500, 1000, 2000, a.extrapolate_to])
    ax.set_xticklabels(["50B", "100B", "200B", "500B", "1T", "2T", f"{a.extrapolate_to / 1000:g}T"])
    ax.minorticks_off()
    ax.set_xlabel("training tokens", color=c["muted"])
    ax.set_ylabel("training loss (8k context)", color=c["muted"])
    ax.grid(True, color=c["grid"], lw=0.6)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=c["muted"], length=0)
    leg = ax.legend(frameon=False, loc="lower left", fontsize=9)
    for t in leg.get_texts():
        t.set_color(c["ink"])
    fig.tight_layout()
    fig.subplots_adjust(right=0.84)
    path = os.path.join(out, f"loss_extrapolation_{theme}.png")
    fig.savefig(path, facecolor=c["surface"])
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recal-dir", required=True, help="output dir of recalibrated_loss.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tail-from", type=float, default=250.0, help="B tokens")
    ap.add_argument("--oos-cut", type=float, default=600.0, help="B tokens")
    ap.add_argument("--oos-tol", type=float, default=9.0, help="mnat, out-of-sample error at the last bin")
    ap.add_argument("--extrapolate-to", type=float, default=3000.0, help="B tokens")
    ap.add_argument("--theme", choices=["light", "dark", "both"], default="both")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    per = pd.read_csv(os.path.join(a.recal_dir, "recal_per_step.csv"), index_col=0)
    ev = pd.read_csv(os.path.join(a.recal_dir, "recal_eval_steps.csv"), index_col=0)
    q = pd.read_csv(os.path.join(a.recal_dir, "conservative_q35_main_gqa.csv"), index_col=0)["lm loss"].dropna()

    x = per.tokens_B.values
    xb, raw_b = bins(x, per.train.values, 200)
    _, recal_b = bins(x, per.recal_train.values, 200)
    qx, qy = bins(q.index.values * Q35_TOKENS_PER_STEP / 1e9, q.values, 800)
    grid = np.geomspace(40, a.extrapolate_to, 400)

    fits = {"raw": model_series("v4 raw", xb, raw_b, a, grid),
            "recal": model_series("v4 recal", xb, recal_b, a, grid),
            "q35": model_series("q35", qx, qy, a, grid)}

    marks = [1000, PHASE1_END_STEP * V4_TOKENS_PER_STEP / 1e9, 3000]
    summary = {"v4_last_step": int(per.index.max()), "v4_last_tokens_B": float(x.max()),
               "tail_from_B": a.tail_from, "oos_cut_B": a.oos_cut, "marks_B": marks, "series": {}}
    for key, fz in fits.items():
        at = lambda arr: [float(np.interp(np.log(m), np.log(grid), arr)) for m in marks]  # noqa: E731
        summary["series"][key] = {
            "best": fz["best"], "accepted": fz["accepted"], "noise_mnat": fz["sigma"] * 1000,
            "line_at_marks": at(fz["line"]), "lo_at_marks": at(fz["lo"]), "hi_at_marks": at(fz["hi"]),
            "forms": {m: {k: v for k, v in d.items() if k != "pred"} for m, d in fz["forms"].items()}}
    summary["eval_steps"] = ev.reset_index().to_dict(orient="records")
    with open(os.path.join(a.out, "extrapolation.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print("\nat 1T / 1.59T / 3T (best line [band]):")
    for key, s in summary["series"].items():
        print(f"   {key:6s} " + "  ".join(f"{l:.4f} [{lo:.4f}-{hi:.4f}]" for l, lo, hi in
                                         zip(s["line_at_marks"], s["lo_at_marks"], s["hi_at_marks"])))

    smooth = lambda s: s.rolling(100, min_periods=20, center=True).mean().values  # noqa: E731
    data = {"v4_x": x, "v4_raw": smooth(per.train), "v4_recal": smooth(per.recal_train),
            "q35_x": q.index.values * Q35_TOKENS_PER_STEP / 1e9, "q35_y": smooth(q),
            "ev_x": ev.tokens_B.values, "ev_recal": ev.recal_train.values}
    for theme in (["light", "dark"] if a.theme == "both" else [a.theme]):
        print("wrote", plot(theme, a, data, fits, grid, a.out))


if __name__ == "__main__":
    main()

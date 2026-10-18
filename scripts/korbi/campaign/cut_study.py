"""Where can a run be cut? From the existing long curves (docs/hybrid_moe_3b_campaign.md §9.10).

    uv run --project ../loss_prediction python scripts/korbi/campaign/cut_study.py \
        <suite dir> [...] --target 1.5728e9 --cuts 0.3e9 0.4e9 ... [--reference attn]

For each cut D_cut (seen tokens): fit every arm of a suite on its curve up to D_cut with
joint_fit.py (own E, A, alpha per run, weak exponent prior), predict the stable-phase loss at
D_target, and compare with the measured one. Per suite and cut:
- mean error: the common offset (all arms), %;
- gap error: RMS over arms of (predicted - measured) gap to the reference arm, nats -- what an
  architecture decision depends on;
- ranking: Spearman of predicted vs measured;
- the same gap error / ranking for the raw loss at the cut (the no-fit baseline);
- the gap trend: each arm's gap to the reference, g(D) = (block-mean loss - reference's) on
  200-step windows from step 1000 to D_cut, extrapolated as g(D) = g_inf + c D^-beta (linear
  least squares, beta fixed, ``--gap-beta``, default 0.5). Under angular MuOwn some gaps shrink
  steadily (the Qwen3.5-like GDN hybrid's lead over attention: -0.030 at 1k steps, +0.001 at
  7.6k), which the per-run fits miss.
The earliest cut whose gap error is below the arm differences that matter (≈ 0.003-0.005 nats)
is where runs can stop.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import parse_log, run_logs  # noqa: E402
from joint_fit import joint_fit, load, model, tau_tok  # noqa: E402


def raw_at(suite, cut_tokens):
    """Mean training loss over the 200 steps before D_cut, per arm (the no-fit baseline)."""
    from predict import schedule_from_log

    out = {}
    for arm, log in run_logs(suite).items():
        d = parse_log(log)
        if not d:
            continue
        _, meta = schedule_from_log(log.read_text(errors="replace"), int(d["total"]))
        c = cut_tokens / (meta["gbs"] * meta["seq"])
        t, y = d["it"], d["loss"]
        w = (t > c - 200) & (t <= c)
        if w.sum() >= 20:
            out[arm] = float(np.mean(y[w]))
    return out


def gap_trend(suite, ref, cut_tokens, target_tokens, beta):
    """Per arm: the gap to ``ref`` extrapolated from its trajectory up to D_cut (docstring)."""
    from predict import schedule_from_log

    curves, ks = {}, {}
    for arm, log in run_logs(suite).items():
        d = parse_log(log)
        if not d:
            continue
        _, meta = schedule_from_log(log.read_text(errors="replace"), int(d["total"]))
        ks[arm] = meta["gbs"] * meta["seq"]
        curves[arm] = (d["it"].astype(float), d["loss"])
    if ref not in curves:
        return {}
    out = {}
    for arm, (t, y) in curves.items():
        if arm == ref or ks[arm] != ks[ref]:
            continue
        c = cut_tokens / ks[arm]
        tr, yr = curves[ref]
        pts = []
        for e in np.arange(1200.0, c + 1, 200.0):
            w, wr = (t > e - 200) & (t <= e), (tr > e - 200) & (tr <= e)
            if w.sum() >= 5 and wr.sum() >= 5:
                pts.append((e * ks[arm], float(np.mean(y[w]) - np.mean(yr[wr]))))
        if len(pts) < 5:
            continue
        D, g = np.array(pts).T
        X = np.column_stack([np.ones_like(D), D ** -beta])
        (g_inf, cc), *_ = np.linalg.lstsq(X, g, rcond=None)
        out[arm] = float(g_inf + cc * target_tokens ** -beta)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suites", nargs="+", type=Path)
    ap.add_argument("--target", type=float, required=True, help="seen tokens to predict")
    ap.add_argument("--cuts", type=float, nargs="+", required=True, help="cut points, seen tokens")
    ap.add_argument("--reference", default="attn")
    ap.add_argument("--alpha-prior", type=float, default=1.0)
    ap.add_argument("--block", type=int, default=20)
    ap.add_argument("--fit-start", type=float, default=200.0, help="first fitted step: max(warmup, this)")
    ap.add_argument("--gap-beta", type=float, default=0.5, help="gap-trend exponent")
    a = ap.parse_args()
    print("| suite | D_cut (BT) | fraction of target | mean error | gap error (nats) | ranking | raw: gap error | raw: ranking | gap trend: gap error |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for suite in a.suites:
        for cut in a.cuts:
            arms = load(suite, cut, a.target, a.block, a.fit_start)
            arms = {n: v for n, v in arms.items() if v["measured"] is not None}
            if len(arms) < 3 or a.reference not in arms:
                continue
            try:
                _, _, per = joint_fit(arms, a.alpha_prior)
            except Exception:  # noqa: BLE001
                continue
            pred = {n: float(np.exp(model(per[n][2], per[n][0], per[n][1], arms[n]["L0"],
                                          tau_tok(arms[n], [arms[n]["target_steps"]])))[0]) for n in arms}
            meas = {n: arms[n]["measured"] for n in arms}
            names = sorted(arms)
            p, m = np.array([pred[n] for n in names]), np.array([meas[n] for n in names])
            ref = names.index(a.reference)
            gp, gm = p - p[ref], m - m[ref]
            gap_err = float(np.sqrt(np.mean(np.delete(gp - gm, ref) ** 2)))
            rho = spearmanr(p, m).correlation
            raw = raw_at(suite, cut)
            rr = np.array([raw.get(n, np.nan) for n in names])
            if np.isfinite(rr).all():
                gr = rr - rr[ref]
                raw_gap = float(np.sqrt(np.mean(np.delete(gr - gm, ref) ** 2)))
                raw_rho = spearmanr(rr, m).correlation
            else:
                raw_gap = raw_rho = float("nan")
            gt = gap_trend(suite, a.reference, cut, a.target, a.gap_beta)
            ge = [gt[n] - gm[i] for i, n in enumerate(names) if n in gt]
            trend_gap = float(np.sqrt(np.mean(np.square(ge)))) if ge else float("nan")
            print(f"| {suite.name} | {cut / 1e9:.2f} | {cut / a.target:.2f} | {100 * np.mean(p / m - 1):+.2f} % "
                  f"| {gap_err:.4f} | {rho:.3f} | {raw_gap:.4f} | {raw_rho:.3f} | {trend_gap:.4f} |")


if __name__ == "__main__":
    main()

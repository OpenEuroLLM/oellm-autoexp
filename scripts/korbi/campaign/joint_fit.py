"""Learning-curve fits of a screen's arms from SHORT runs (docs/hybrid_moe_3b_campaign.md §9.6):
every run its own three parameters (E, A, alpha), fitted jointly per suite with a weak prior
tying the exponents together; the arms are then compared on the predicted loss at a later stage
instead of training them out. Default --alpha-prior 1.0 (per-run alpha, the prior only
regularises): from a 4000-step cut it predicts step 6000 to +0.1-0.25 % per arm, with an
arm-to-arm error spread of 0.0006-0.0017 nats (rank correlation 0.98 / 1.00), better than a
shared alpha (0.0027-0.0035) and than the raw loss at the cut (0.0013-0.0020).

    uv run --project ../loss_prediction python scripts/korbi/campaign/joint_fit.py \
        <suite dir> [...] --cut 2.1e9 --target 4e9 [--validate] [--csv out.csv]

Time is seen tokens D = steps x gbs x seq_len throughout: ``--cut`` / ``--target`` are token
counts, tau is the LR area in tokens, so runs of different batch sizes are compared (and their
A / t0 are expressed) at equal D -- for an architecture decision N *and* D are matched.

Model per arm i (LR-area time tau, anchored at the arm's initial loss L0_i): three parameters
    L_i(t) = E_i + A_i (tau(t) + t0_i)^-alpha_i,   t0_i = (A_i / (L0_i - E_i))^(1/alpha_i)
with ``--alpha-prior 0`` alpha_i = alpha shared by the suite (per size and optimizer); with
``--alpha-prior s`` every arm has its own alpha_i, shrunk towards the suite's alpha (fitted) by a
Gaussian prior of width s -- the hierarchical middle between independent per-run fits (s large)
and one shared exponent (s -> 0). Fitted on the training loss blocks between max(warmup, 200) and ``--cut``
(steps), weighted by the block noise. A single short run cannot separate E from alpha (it trades
one for the other); sharing alpha across arms is what identifies it.

Output per arm: E, A, the predicted loss at ``--target`` (steps, the same stable schedule
continued) and, with ``--validate``, the measured stable-phase training loss at the target (mean
of the 200 steps before it) and the prediction error. Per suite: the shared alpha, and the
Spearman correlation of predicted vs measured ranking.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import LP, parse_log, run_logs  # noqa: E402
from predict import schedule_from_log  # noqa: E402

sys.path.insert(0, str(LP / "src"))
from losspred.fitting.blocks import block_curve  # noqa: E402
from losspred.schedule import WSDScheduleConfig  # noqa: E402


def load(suite, cut_tokens, target_tokens, block, min_start=200.0):
    """``cut_tokens`` / ``target_tokens``: seen tokens D (= steps x gbs x seq_len), converted to
    each run's steps -- runs are compared at equal D, not at equal steps."""
    arms = {}
    for arm, log in sorted(run_logs(suite).items()):
        d = parse_log(log)
        if not d or len(d["it"]) < 50:
            continue
        sched, meta = schedule_from_log(log.read_text(errors="replace"), int(d["total"]))
        k = float(meta["gbs"] * meta["seq"])  # tokens per step
        cut, target = cut_tokens / k, target_tokens / k
        t, y = d["it"].astype(float), d["loss"]
        ok = np.isfinite(y) & (y > 0)
        t, y = t[ok], y[ok]
        start = max(float(sched.warmup), min_start)
        if (t <= cut).sum() < 50 or t[-1] < min(cut, start + 300):
            continue
        sel = t <= cut
        c = block_curve(arm, t[sel], y[sel], block=block, start=start, max_steps=200.0, rel_width=0.05)
        if len(c.t) < 5:
            continue
        stable = WSDScheduleConfig(total=int(max(target, sched.total)), warmup=sched.warmup, decay=0)
        meas = None
        win = (t > target - 200) & (t <= target)
        decay_start = sched.decay_start if sched.decay > 0 else sched.total
        if win.sum() >= 20 and target <= decay_start:
            meas = float(np.mean(y[win]))
        arms[arm] = dict(curve=c, sched=stable, L0=float(y[0]), measured=meas, last=int(t[-1]), k=k,
                         target_steps=target)
    return arms


def tau_tok(arm, steps):
    """LR-area time in tokens: (cumulative relative LR in steps) x tokens per step."""
    return arm["sched"].effective_time(np.asarray(steps, float), 1.0) * arm["k"]


def model(alpha, E, logA, L0, tau):
    A = np.exp(logA)
    t0 = (A / max(L0 - E, 1e-6)) ** (1.0 / alpha)
    return np.log(E + A * (tau + t0) ** -alpha)


def joint_fit(arms, alpha_prior=0.0):
    """alpha_prior = 0: one alpha for all arms. > 0: every arm its own alpha_i, shrunk towards
    the suite's alpha (also fitted) by a Gaussian prior of width alpha_prior (a penalty
    (alpha_i - alpha) / alpha_prior added to the residuals). Returns (alpha, alpha_se, per) with
    per[arm] = (E, logA, alpha_i)."""
    names = sorted(arms)
    taus = {n: tau_tok(arms[n], arms[n]["curve"].t) for n in names}
    zmin = {n: float(arms[n]["curve"].z.min()) for n in names}
    k = 3 if alpha_prior > 0 else 2

    def unpack(x):
        per = {}
        for i, n in enumerate(names):
            E, logA = x[1 + k * i], x[2 + k * i]
            per[n] = (E, logA, x[3 + k * i] if k == 3 else x[0])
        return x[0], per

    def resid(x):
        alpha, per = unpack(x)
        out = []
        for n in names:
            c = arms[n]["curve"]
            E, logA, ai = per[n]
            out.append((model(ai, E, logA, arms[n]["L0"], taus[n]) - c.z) / np.sqrt(c.var))
            if k == 3:
                out.append(np.array([(ai - alpha) / alpha_prior]))
        return np.concatenate(out)

    best = None
    for a0 in (0.2, 0.4, 0.8):
        for ef in (0.6, 0.85):
            x0, lo, hi = [a0], [0.01], [3.0]
            for n in names:
                ehi = float(np.exp(zmin[n]))
                x0 += [ef * ehi, np.log(30.0) + a0 * np.log(arms[n]["k"])] + ([a0] if k == 3 else [])
                lo += [0.3, -5.0] + ([0.01] if k == 3 else [])
                hi += [ehi * 0.999, 60.0] + ([3.0] if k == 3 else [])
            x0 = np.clip(x0, np.array(lo) + 1e-6, np.array(hi) - 1e-6)
            r = least_squares(resid, x0, bounds=(lo, hi), loss="soft_l1", f_scale=3.0, max_nfev=8000)
            if best is None or r.cost < best.cost:
                best = r
    alpha, per = unpack(best.x)
    # Gauss-Newton standard error of alpha (noise-normalised residuals)
    try:
        cov = np.linalg.inv(best.jac.T @ best.jac)
        alpha_se = float(np.sqrt(cov[0, 0]))
    except np.linalg.LinAlgError:
        alpha_se = float("nan")
    return alpha, alpha_se, per


def fit_one(arm, alpha_center, alpha_prior, z=None, x0=None):
    """One run's (E, logA, alpha) on its blocks (or the resampled log-losses ``z``), with the
    exponent prior centred on the suite's alpha."""
    c, L0 = arm["curve"], arm["L0"]
    tau = tau_tok(arm, c.t)
    zz = c.z if z is None else z
    ehi = float(np.exp(min(c.z.min(), zz.min())))

    def resid(x):
        r = (model(x[2], x[0], x[1], L0, tau) - zz) / np.sqrt(c.var)
        return np.concatenate([r, [(x[2] - alpha_center) / max(alpha_prior, 1e-3)]])

    lo, hi = [0.3, -5.0, 0.01], [ehi * 0.999, 60.0, 3.0]
    starts = [x0] if x0 is not None else [[f * ehi, np.log(30.0) + alpha_center * np.log(arm["k"]), alpha_center]
                                         for f in (0.6, 0.85)]
    best = None
    for s0 in starts:
        s0 = np.clip(s0, np.array(lo) + 1e-6, np.array(hi) - 1e-6)
        r = least_squares(resid, s0, bounds=(lo, hi), loss="soft_l1", f_scale=3.0, max_nfev=3000)
        if best is None or r.cost < best.cost:
            best = r
    return best.x


def bootstrap(arm, params, alpha_center, alpha_prior, target, n_boot=50, block_len=5, seed=0):
    """Moving-block bootstrap of the run's block residuals (keeps their autocorrelation): refit
    ``n_boot`` times; returns the 5 / 95 % percentiles of E, A, alpha and the prediction."""
    rng = np.random.default_rng(seed)
    c = arm["curve"]
    tau = tau_tok(arm, c.t)
    E, logA, ai = params
    fit = model(ai, E, logA, arm["L0"], tau)
    res = c.z - fit
    n = len(res)
    tau_t = tau_tok(arm, [arm["target_steps"]])
    out = []
    for _ in range(n_boot):
        idx = np.concatenate([np.arange(s, s + block_len) for s in
                              rng.integers(0, max(n - block_len, 1), size=n // block_len + 1)])[:n]
        idx = np.clip(idx, 0, n - 1)
        x = fit_one(arm, alpha_center, alpha_prior, z=fit + res[idx], x0=[E, logA, ai])
        out.append([x[0], np.exp(x[1]), x[2], float(np.exp(model(x[2], x[0], x[1], arm["L0"], tau_t))[0])])
    out = np.array(out)
    ci = {k: (float(np.percentile(out[:, j], 5)), float(np.percentile(out[:, j], 95)))
          for j, k in enumerate(("E", "A", "alpha", "pred"))}
    ci["_pred_reps"] = out[:, 3]
    return ci


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suites", nargs="+", type=Path)
    ap.add_argument("--cut", type=float, default=2.1e9,
                    help="use the curve up to this many seen tokens D (steps x gbs x seq_len)")
    ap.add_argument("--target", type=float, default=4.0e9, help="predict the loss at this many seen tokens")
    ap.add_argument("--block", type=int, default=20)
    ap.add_argument("--fit-start", type=float, default=200.0,
                    help="first fitted step: max(warmup, this); runs without warmup start inside the "
                         "early transient otherwise")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--alpha-prior", type=float, default=1.0,
                    help="0: one alpha per suite; > 0: per-arm alpha with a Gaussian shrinkage prior of "
                         "this width towards the suite alpha (each run its own E, A, alpha)")
    ap.add_argument("--bootstrap", type=int, default=0, help="moving-block bootstrap replicas per run "
                                                                     "(5-95 %% intervals of E, A, alpha, prediction)")
    ap.add_argument("--reference", default="attn", help="arm the paired gaps are taken against")
    ap.add_argument("--csv", type=Path)
    a = ap.parse_args()
    rows = []
    for suite in a.suites:
        arms = load(suite, a.cut, a.target, a.block, a.fit_start)
        if len(arms) < 2:
            print(f"{suite.name}: fewer than 2 arms past {a.cut / 1e9:.2f}BT")
            continue
        alpha, alpha_se, per = joint_fit(arms, a.alpha_prior)
        print(f"\n### {suite.name}: fit to D = {a.cut / 1e9:.2f}BT, predict D = {a.target / 1e9:.2f}BT; "
              f"shared alpha = {alpha:.3f} ± {alpha_se:.3f}\n")
        print(f"| arm | E [5-95 %] | A [5-95 %] | alpha [5-95 %] | predicted [5-95 %] | vs {a.reference} [5-95 %, paired] | measured | error |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        pred, meas = [], []
        def predict(n):
            E, logA, ai = per[n]
            return float(np.exp(model(ai, E, logA, arms[n]["L0"], tau_tok(arms[n], [arms[n]["target_steps"]])))[0])

        # paired bootstrap: every arm uses the same resampling indices (same seed, same number of
        # blocks), so the replicas of two arms are paired and their difference has its own interval
        cis = {n: bootstrap(arms[n], per[n], alpha, a.alpha_prior, a.target, a.bootstrap) if a.bootstrap else {}
               for n in arms}
        ref = a.reference if a.reference in arms else None
        for n in sorted(arms, key=predict):
            E, logA, ai = per[n]
            p = predict(n)
            m = arms[n]["measured"] if a.validate else None
            err = f"{100 * (p / m - 1):+.2f} %" if m else ""
            ci = cis[n]
            gap = ""
            if ref and n != ref:
                g = p - predict(ref)
                gap = f"{g:+.4f}"
                ra, rb = ci.get("_pred_reps"), cis[ref].get("_pred_reps")
                if ra is not None and rb is not None and len(ra) == len(rb):
                    d = ra - rb
                    gap += f" [{np.percentile(d, 5):+.4f}, {np.percentile(d, 95):+.4f}]"

            def cell(v, k, fmt):
                return f"{v:{fmt}} [{ci[k][0]:{fmt}}, {ci[k][1]:{fmt}}]" if k in ci else f"{v:{fmt}}"

            print(f"| {n} | {cell(E, 'E', '.3f')} | {cell(np.exp(logA), 'A', '.1f')} | {cell(ai, 'alpha', '.3f')} "
                  f"| {cell(p, 'pred', '.4f')} | {gap} | {f'{m:.4f}' if m else ''} | {err} |")
            row = dict(suite=suite.name, arm=n, cut=a.cut, target=a.target, alpha_suite=alpha,
                       alpha_se=alpha_se, alpha=ai, E=E, A=float(np.exp(logA)), predicted=p,
                       measured=m, alpha_prior=a.alpha_prior)
            for k, v in ci.items():
                if not k.startswith("_"):
                    row[f"{k}_lo"], row[f"{k}_hi"] = v
            rows.append(row)
            if m:
                pred.append(p)
                meas.append(m)
        if len(meas) >= 3:
            rho = spearmanr(pred, meas).correlation
            d = np.array(pred) - np.array(meas)
            # the arm comparison only needs the differences: their error after removing the
            # suite-wide offset
            print(f"\nranking Spearman {rho:.3f}; mean error {100 * np.mean(d / np.array(meas)):+.2f} %, "
                  f"spread of the error across arms (std) {np.std(d):.4f} nats")
    if a.csv and rows:
        with a.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()

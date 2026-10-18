"""Curve parameters and health from hyperparameters (docs/hybrid_moe_3b_campaign.md §8.3, §9.8).

    uv run --project ../loss_prediction python scripts/korbi/campaign/hp_model.py \
        <suite dir> [...] [--cut <tokens>] [--csv runs.csv]

1. Per run: its own curve parameters (E, A, alpha; L = E + A (tau + t0)^-alpha, t0 anchored at
   the initial loss, tau = LR area), fitted per suite with joint_fit.py's weak exponent prior
   (width 1.0: each run its own alpha, the prior only regularises short fits), on the stable
   part up to ``--cut`` seen tokens (time in tokens: A comparable across batch sizes). Its
   hyperparameters are read from Megatron's argument dump: lr,
   gbs, warmup, optimizer (adam / muown / angular_muown / adam + ngpt), the parameter count N
   (non-embedding: total - padded vocab x hidden), and health from losspred's spike detector
   (max spike height over the whole run, number of spikes).
2. Parameter model: per optimizer, OLS of log E, log A and alpha on
   [1, log lr, log gbs, log N, log(1 + warmup)] (features with no variation in a group are
   dropped), with R^2 and the coefficients.
3. Health model, a linear threshold: unhealthy (max spike > ``--max-spike``) iff
   w . [log lr, log gbs, log N, log(1 + warmup), optimizer one-hot] > 0, fitted by an
   L2-regularised logistic regression (the decision boundary is the threshold); plus an OLS of
   log(max spike + 1e-3) on the same features (the continuous spikiness).
Architecture is not a feature: the parameter model is for hyperparameters at a fixed
architecture (the arm is in the CSV for per-arch models once enough runs per arch exist).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import detector, parse_log, run_logs  # noqa: E402
from joint_fit import joint_fit, load  # noqa: E402

VOCAB = 262144


def hparams(log: Path) -> dict:
    text = log.read_text(errors="replace")

    def arg(name, cast=str):
        m = re.search(rf"^\S*\s*{name} \.+ (\S+)", text, re.M)
        return cast(m.group(1)) if m and m.group(1) != "None" else None

    m = re.search(r"number of parameters on \([a-z_, ]+\) model parallel rank \([0-9, ]+\): (\d+)", text)
    total = int(m.group(1)) if m else None
    hidden = arg("hidden_size", int)
    opt = arg("optimizer") or "adam"
    if arg("ngpt") == "True":
        opt = "ngpt"
    return dict(lr=arg("lr", float), gbs=arg("global_batch_size", int),
                warmup=arg("lr_warmup_iters", int) or 0, optimizer=opt,
                n_total=total, n=(total - VOCAB * hidden) if total and hidden else None,
                hidden=hidden, seed=arg("seed"))


def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    r2 = 1 - res.var() / y.var() if y.var() > 0 else float("nan")
    return beta, r2, float(np.sqrt(np.mean(res**2)))


def logistic(X, y, l2=1e-2):
    """L2-regularised logistic regression (intercept unpenalised); returns w."""
    def loss(w):
        z = X @ w
        return np.mean(np.logaddexp(0, z) - y * z) + l2 * np.sum(w[1:] ** 2)

    def grad(w):
        p = 1 / (1 + np.exp(-(X @ w)))
        g = X.T @ (p - y) / len(y)
        g[1:] += 2 * l2 * w[1:]
        return g

    return minimize(loss, np.zeros(X.shape[1]), jac=grad, method="L-BFGS-B").x


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suites", nargs="+", type=Path)
    ap.add_argument("--cut", type=float, default=1e15,
                    help="fit only up to this many seen tokens D = steps x gbs x seq_len (default: all)")
    ap.add_argument("--max-spike", type=float, default=0.05)
    ap.add_argument("--block", type=int, default=20)
    ap.add_argument("--csv", type=Path)
    a = ap.parse_args()
    det = detector()
    rows = []
    for suite in a.suites:
        arms = load(suite, a.cut, a.cut, a.block)
        if not arms:
            continue
        try:
            _, _, per = joint_fit(arms, 1.0) if len(arms) > 1 else (None, None, None)
        except Exception as e:  # noqa: BLE001
            print(f"{suite.name}: fit failed ({e})")
            continue
        logs = run_logs(suite)
        for arm in arms:
            if per is None or arm not in per:
                continue
            E, logA, alpha = per[arm]
            hp = hparams(logs[arm])
            d = parse_log(logs[arm])
            t, y = d["it"].astype(float), d["loss"]
            ok = np.isfinite(y) & (y > 0) & (t >= 20)
            sp, rg = det(t[ok], y[ok])
            ev = [e for e in sp.events if e.end < ok.sum()]
            rows.append(dict(suite=suite.name, arm=arm, E=float(E), A=float(np.exp(logA)), alpha=float(alpha),
                             steps=int(t[-1]), spikes=len(ev),
                             max_spike=max((e.height for e in ev), default=0.0),
                             growth=sum(e.kind == "growth" for e in rg.events), **hp))
    rows = [r for r in rows if r["lr"] and r["gbs"] and r["n"]]
    print(f"{len(rows)} runs with fits and hyperparameters")
    feats = ["log_lr", "log_gbs", "log_N", "log1p_warmup"]

    def fx(r):
        return [np.log(r["lr"]), np.log(r["gbs"]), np.log(r["n"]), np.log1p(r["warmup"])]

    # parameter model per optimizer
    print("\n## Curve parameters from hyperparameters (OLS per optimizer)\n")
    print("| optimizer | target | n | R^2 | rmse | " + " | ".join(["intercept"] + feats) + " |")
    print("|---|---|---:|---:|---:|" + "---:|" * (len(feats) + 1))
    for opt in sorted({r["optimizer"] for r in rows}):
        g = [r for r in rows if r["optimizer"] == opt]
        if len(g) < 4:
            continue
        F = np.array([fx(r) for r in g])
        keep = [j for j in range(F.shape[1]) if np.ptp(F[:, j]) > 1e-9]
        X = np.column_stack([np.ones(len(g))] + [F[:, j] for j in keep])
        for target, yv in (("log E", np.log([r["E"] for r in g])), ("log A", np.log([r["A"] for r in g])),
                           ("alpha", np.array([r["alpha"] for r in g]))):
            if len(g) <= X.shape[1]:
                continue
            beta, r2, rmse = ols(X, yv)
            coef = ["-"] * len(feats)
            for k, j in enumerate(keep):
                coef[j] = f"{beta[k + 1]:+.3f}"
            print(f"| {opt} | {target} | {len(g)} | {r2:.2f} | {rmse:.3f} | {beta[0]:+.3f} | " + " | ".join(coef) + " |")
    # health: linear threshold over hyperparameters
    opts = sorted({r["optimizer"] for r in rows})
    F = np.array([fx(r) + [1.0 if r["optimizer"] == o else 0.0 for o in opts[1:]] for r in rows])
    mu, sd = F.mean(0), F.std(0) + 1e-9
    Z = np.column_stack([np.ones(len(rows)), (F - mu) / sd])
    yb = np.array([1.0 if (r["max_spike"] > a.max_spike or r["growth"] > 0) else 0.0 for r in rows])
    names = feats + [f"opt={o}" for o in opts[1:]]
    print(f"\n## Health: linear threshold (unhealthy = spike > {a.max_spike} or growth)\n")
    print(f"unhealthy runs: {int(yb.sum())} / {len(rows)}: " +
          ", ".join(f"{r['suite']}/{r['arm']}" for r, b in zip(rows, yb) if b))
    if 0 < yb.sum() < len(rows):
        w = logistic(Z, yb)
        acc = np.mean(((Z @ w) > 0) == (yb > 0.5))
        print(f"score s = {w[0]:+.2f} + " + " + ".join(f"{w[k + 1]:+.2f} z({n})" for k, n in enumerate(names)) +
              f"; unhealthy iff s > 0 (z = standardised feature); training accuracy {acc:.2f}")
        if yb.sum() < 5:
            print("(too few unhealthy runs for a reliable boundary: a targeted lr x warmup x gbs grid is needed)")
    sp = np.log(np.array([r["max_spike"] for r in rows]) + 1e-3)
    beta, r2, rmse = ols(Z, sp)
    print("\nspikiness log(max spike + 1e-3) = " + f"{beta[0]:+.2f} + " +
          " + ".join(f"{beta[k + 1]:+.2f} z({n})" for k, n in enumerate(names)) + f"  (R^2 {r2:.2f})")
    if a.csv:
        with a.csv.open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)


if __name__ == "__main__":
    main()

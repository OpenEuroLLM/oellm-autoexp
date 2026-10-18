"""Sequential design: which runs to launch next.

Acquisition = expected information gain about the *decision quantities* Q per
GPU-hour, under the linearised (Laplace) posterior:

    Q(theta) = [ log L*_a(N_T, r N_T) - log L*_ref(N_T, r N_T)   for live arms a, ratios r
                 log lr*_a(N_T, r N_T)                            (optional) ]

    IG(x) = -1/2 log(1 - v^T A^-1 v / s),   A = G S G^T,  v = G S j_x,  s = 1 + j_x^T S j_x

where G = dQ/dtheta, S the posterior covariance and j_x the whitened Jacobian of the
candidate's predicted log loss. Because the model is parametric and power-law
shaped, candidates at the *ends* of the affordable (N, D) range and LR probes where
the optimum extrapolates worst come out on top automatically.

Greedy batches condition S on already chosen (and on pending) runs -- in the
linear-Gaussian approximation this is exactly the "Kriging believer" fantasy,
because the covariance update does not depend on the (unknown) outcome.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from . import data as _data
from .forms import LOGB0, features
from .model import FitResult, _Rows, prepare

SEQ_LEN = 4096


def make_Q(fit: FitResult, archs: list[str], reference: str = "fullattn",
           N_T: float = 3e9, ratios=(100.0, 300.0, 1000.0), include_lr: bool = True,
           study: str | None = None) -> tuple[Callable[[np.ndarray], np.ndarray], list[str]]:
    study = study or fit.layout.reference_study
    all_archs = list(dict.fromkeys([reference] + list(archs)))
    rows = pd.DataFrame([{"arch": a, "study": study, "N": N_T, "D": r * N_T, "y": 0.0}
                         for a in all_archs for r in ratios])
    R = _Rows(fit.layout, prepare(rows))
    idx = {(a, r): i for i, (a, r) in enumerate((a, r) for a in all_archs for r in ratios)}
    form = fit.layout.form
    labels = [f"gap[{a}|r={r:g}]" for a in archs if a != reference for r in ratios]
    if include_lr:
        labels += [f"loglr*[{a}|r={r:g}]" for a in archs for r in ratios]

    def Q(theta: np.ndarray) -> np.ndarray:
        env = R.predict(theta, "envelope")
        out = [env[idx[(a, r)]] - env[idx[(reference, r)]]
               for a in archs if a != reference for r in ratios]
        if include_lr:
            opt = form.optimum(R.params(theta), R.X)
            out += [opt["loglr"][idx[(a, r)]] for a in archs for r in ratios]
        return np.asarray(out)

    return Q, labels


def q_gradient(Q, theta: np.ndarray, h: float = 1e-4) -> np.ndarray:
    q0 = Q(theta)
    G = np.zeros((len(q0), len(theta)))
    for j in range(len(theta)):
        e = np.zeros_like(theta)
        e[j] = h
        G[:, j] = (Q(theta + e) - Q(theta - e)) / (2 * h)
    return G


def candidate_grid(fit: FitResult, archs: list[str], sizes: dict[str, float],
                   budgets_bt=(6, 12, 20, 30, 50, 80, 120, 200, 300),
                   lr_mults=(0.5, 1.0, 2.0), study: str | None = None,
                   max_ratio: float = 5000.0, trust: float = 2.0) -> pd.DataFrame:
    """Runs around the *current* posterior-mean HP optimum (batch rounded to a power
    of two in sequences). Cost is the measured-throughput GPU-hours.

    Trust region: lr and batch are clipped to the observed range widened by ``trust``
    (one factor-2 step by default) -- the HP law is least reliable exactly where it
    extrapolates, so exploration moves outward step by step instead of jumping."""
    study = study or fit.layout.reference_study
    form = fit.layout.form
    obs = fit.df[fit.active & fit.df.lr.notna()]
    lr_lo, lr_hi = obs.lr.min() / trust, obs.lr.max() * trust
    gbs_lo = np.log2(obs.B.min() / SEQ_LEN / trust)
    gbs_hi = np.log2(obs.B.max() / SEQ_LEN * trust)
    out = []
    for a in archs:
        for sname, N in sizes.items():
            for d in budgets_bt:
                D = d * 1e9
                if D / N > max_ratio:
                    continue
                probe = prepare(pd.DataFrame([{"arch": a, "study": study, "N": N, "D": D, "y": 0}]))
                R = _Rows(fit.layout, probe)
                opt = form.optimum(R.params(fit.theta), R.X)
                if not opt:
                    gbs, lr0 = 256, 1e-3
                else:
                    B_tok = np.exp(opt["logB"][0] + LOGB0)
                    gbs = int(2 ** np.clip(np.round(np.log2(B_tok / SEQ_LEN)), gbs_lo, gbs_hi))
                    Xb = features(N, D, None, gbs * SEQ_LEN)
                    lr0 = float(np.exp(form.optimum(R.params(fit.theta), Xb,
                                                    logB=Xb["logB"])["loglr"][0]))
                lr0 = float(np.clip(lr0, lr_lo, lr_hi))
                for m in lr_mults:
                    lr = float(np.clip(lr0 * m, lr_lo, lr_hi))
                    if m != 1.0 and np.isclose(lr, lr0):
                        continue  # clipped onto the centre point: duplicate
                    out.append({"arch": a, "study": study, "size": sname, "N": N, "D": D,
                                "lr": lr, "lr_mult": m, "B": gbs * SEQ_LEN, "gbs": gbs,
                                "y": 0.0, "cost": float(_data.gpu_hours(a, N, D))})
    return pd.DataFrame(out)


def _whitened_jac(fit: FitResult, cand: pd.DataFrame, noise_inflation: float) -> tuple:
    c = prepare(cand)
    R = _Rows(fit.layout, c)
    J = R.jac(fit.theta)
    sd = fit.noise_sd(c["study"].values) * noise_inflation
    return J / sd[:, None], R.predict(fit.theta)


def condition(cov: np.ndarray, Jw: np.ndarray) -> np.ndarray:
    """Posterior covariance after observing rows with whitened Jacobian Jw."""
    if len(Jw) == 0:
        return cov
    SJ = cov @ Jw.T
    Sm = Jw @ SJ + np.eye(len(Jw))
    return cov - SJ @ np.linalg.solve(Sm, SJ.T)


def select_batch(fit: FitResult, cand: pd.DataFrame, Q, k: int = 4, cost_power: float = 1.0,
                 pending: pd.DataFrame | None = None, noise_inflation: float = 1.0,
                 cov: np.ndarray | None = None, theta: np.ndarray | None = None) -> pd.DataFrame:
    """Greedy IG/cost batch of size k (conditioning on pending runs first)."""
    theta = fit.theta if theta is None else theta
    S = (fit.cov if cov is None else cov).copy()
    G = q_gradient(Q, theta)
    if pending is not None and len(pending):
        Jp, _ = _whitened_jac(fit, pending, noise_inflation)
        S = condition(S, Jp)
    J, mu = _whitened_jac(fit, cand, noise_inflation)
    cost = cand["cost"].values
    chosen, rows = [], []
    for step in range(min(k, len(cand))):
        A = G @ S @ G.T + 1e-12 * np.eye(len(G))
        V = G @ S @ J.T  # q x C
        s = 1.0 + np.einsum("ij,jk,ik->i", J, S, J)
        quad = np.einsum("ic,ic->c", V, np.linalg.solve(A, V))
        ig = -0.5 * np.log(np.clip(1.0 - quad / s, 1e-12, None))
        acq = ig / cost**cost_power
        acq[chosen] = -np.inf
        i = int(np.argmax(acq))
        chosen.append(i)
        rows.append(dict(cand.iloc[i].to_dict(), IG=float(ig[i]), acq=float(acq[i]),
                         pred_logloss=float(mu[i]), step=step))
        S = condition(S, J[i:i + 1])
    out = pd.DataFrame(rows)
    out["cand_index"] = chosen
    return out


def total_uncertainty(fit: FitResult, Q, cov: np.ndarray | None = None) -> float:
    """1/2 log det Cov(Q): the quantity each launch is supposed to shrink."""
    G = q_gradient(Q, fit.theta)
    A = G @ (fit.cov if cov is None else cov) @ G.T
    return 0.5 * float(np.linalg.slogdet(A + 1e-12 * np.eye(len(A)))[1])

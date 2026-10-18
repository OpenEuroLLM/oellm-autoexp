"""Sequential vs parallel: launch now, or wait for the pending runs?

For every run x the planner would like to launch *now*, simulate K plausible
outcomes of the runs that are still pending (posterior-predictive fantasies),
update the posterior with each, and re-run the planner:

    P_wanted(x) = fraction of fantasies in which x (or a near-equivalent run) is
                  still selected and its architecture still live
    W(x)        = cost(x) * (1 - P_wanted(x))          expected commitment waste [GPU-h]
    gain(x)     = lambda * dt * kappa                  value of the saved wall-clock [GPU-h]

    launch now  iff  W(x) < gain(x)

lambda  [GPU-h per hour]  price of wall-clock (deadline pressure; ~0 if the queue is the
                          bottleneck anyway)
dt      [h]               time until the *informative* pending results arrive (partial
                          curves count: see curves.py -- often 20-30% of a trunk)
kappa   [0, 1]            probability that x is on the critical path of the decision

Idle capacity that would otherwise be forfeited can be priced by ``idle_discount``
(multiplies cost(x): 0 = free).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from . import decision, design
from .model import FitResult, _chol, prepare, _Rows


@dataclass
class Clock:
    lam: float = 0.0  # GPU-h per hour of wall-clock
    dt_hours: float = 24.0
    kappa: float = 1.0
    idle_discount: float = 1.0


def _kalman(fit: FitResult, pending: pd.DataFrame, y: np.ndarray, noise_inflation=1.0):
    """Posterior (theta, cov) after observing y at the pending rows (linearised)."""
    p = prepare(pending)
    R = _Rows(fit.layout, p)
    J = R.jac(fit.theta)
    f = R.predict(fit.theta)
    sd = fit.noise_sd(p["study"].values) * noise_inflation
    Jw = J / sd[:, None]
    SJ = fit.cov @ Jw.T
    Sm = Jw @ SJ + np.eye(len(Jw))
    K = SJ @ np.linalg.inv(Sm)
    theta = fit.theta + K @ ((y - f) / sd)
    cov = fit.cov - K @ SJ.T
    return theta, cov


def _near(a: pd.Series, b: pd.DataFrame, tol: float = 1.5) -> bool:
    m = ((b.arch == a.arch) & np.isclose(b.N, a.N)
         & (np.maximum(b.D / a.D, a.D / b.D) <= tol)
         & (np.maximum(b.lr / a.lr, a.lr / b.lr) <= tol))
    return bool(m.any())


def plan(fit: FitResult, make_candidates, make_Q, k: int, pending: pd.DataFrame | None,
         regime=decision.Regime(), rule=decision.EliminationRule(), reference: str = "fullattn"):
    """One planner pass: eliminate, then select a batch among the arms still worth
    data (live, review, and promoted -- a promoted arm still needs its HP laws)."""
    mult = decision.compute_multipliers(fit, regime=regime, S=200, reference=reference)
    status = decision.eliminate(mult, rule, decision.stability_flags(fit))
    keep = status.loc[status.status != "drop", "arch"].tolist()
    arms = [reference] + keep
    cand = make_candidates(fit, arms)
    if cand is None or not len(cand):
        return pd.DataFrame(columns=["arch", "N", "D", "lr", "cost", "IG", "acq"]), status
    Q, _ = make_Q(fit, arms)
    batch = design.select_batch(fit, cand, Q, k=k, pending=pending)
    return batch, status


def launch_decisions(fit: FitResult, make_candidates, make_Q, pending: pd.DataFrame,
                     k: int = 3, K: int = 16, clock: Clock = Clock(), seed: int = 0,
                     regime=decision.Regime(), rule=decision.EliminationRule(),
                     reference: str = "fullattn") -> pd.DataFrame:
    """Evaluate the launch-now rule for the batch the planner proposes *given* pending."""
    batch_now, _ = plan(fit, make_candidates, make_Q, k, pending, regime, rule, reference)
    rng = np.random.default_rng(seed)
    p = prepare(pending)
    R = _Rows(fit.layout, p)
    L = _chol(fit.cov)
    wanted = np.zeros(len(batch_now))
    for _ in range(K):
        th = fit.theta + L @ rng.standard_normal(len(fit.theta))
        y = R.predict(th) + fit.noise_sd(p["study"].values) * rng.standard_normal(len(p))
        theta_k, cov_k = _kalman(fit, p, y)
        fit_k = replace(fit, theta=theta_k, cov=cov_k)
        batch_k, _ = plan(fit_k, make_candidates, make_Q, k, None, regime, rule, reference)
        for i, (_, x) in enumerate(batch_now.iterrows()):
            wanted[i] += _near(x, batch_k)
    P = wanted / K
    out = batch_now[["arch", "size", "D", "lr", "gbs", "cost", "IG", "acq"]].copy()
    out["P_wanted_after_pending"] = P
    out["waste_gpuh"] = out["cost"] * clock.idle_discount * (1 - P)
    out["wallclock_gain_gpuh"] = clock.lam * clock.dt_hours * clock.kappa
    out["launch_now"] = out["waste_gpuh"] < out["wallclock_gain_gpuh"]
    return out

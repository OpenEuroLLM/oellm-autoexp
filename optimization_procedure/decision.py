"""Architecture decisions: compute multipliers in the target regime, drop / keep / promote.

For architecture ``a`` vs a reference ``b`` at a GPU-hour budget C and token ratio
r = D/N:

    N_a = argsolve cost_a(N, rN) = C,   L_a = L_a(N_a, r N_a)  (+ HP regret)
    M_a = C_b(L_a) / C    with   L_b(N_b(C_b), r N_b(C_b)) = L_a

M_a > 1: the reference needs M_a times the GPU-hours to match ``a``. Both sides pay
their *HP regret*: the loss at the HPs we would actually pick (posterior-mean
optimum) under each posterior sample -- an arm whose lr optimum is still unknown
is penalised for that, which is what makes cheap LR probes valuable.

Everything is computed on posterior samples (Laplace), vectorised; the bisection
for C_b runs on all samples at once.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import data as _data
from .forms import features
from .model import FitResult, _Rows


@dataclass
class Regime:
    """Target regime: reference model sizes and D/N ratios (cost is fixed by the
    reference arch at that (N, r))."""

    N_ref: tuple[float, ...] = (3e9,)
    ratios: tuple[float, ...] = (100.0, 300.0, 1000.0)


def _arch_params(fit: FitResult, arch: str, study: str, theta: np.ndarray) -> dict:
    df = pd.DataFrame({"arch": [arch], "study": [study], "N": [1e8], "D": [1e10], "y": [0.0]})
    r = _Rows(fit.layout, df.assign(lr=np.nan, B=np.nan))
    p = r.params(theta)  # (1,) or (1, S)
    return {k: v[0] for k, v in p.items()}


def _loss_at(fit: FitResult, p: dict, p_hat: dict, N, D, with_regret: bool) -> np.ndarray:
    """log L at (N, D) for sample params p (arrays (S,)), using the HPs that the
    posterior mean (p_hat, scalars) would choose."""
    form = fit.layout.form
    X = features(N, D)
    env = form.envelope(p, X)
    if not with_regret:
        return env
    opt_hat = form.optimum(p_hat, X)
    if not opt_hat:
        return env
    X_hp = dict(X, loglr=opt_hat["loglr"], logB=opt_hat["logB"])
    return env + form.bowl(p, X_hp)


def n_for_cost(arch: str, C: float, r: float) -> float:
    s0, sig = _data.throughput(arch)
    # C = r N * N^sig / (3600 s0)
    return (C * 3600.0 * s0 / r) ** (1.0 / (1.0 + sig))


def compute_multipliers(
    fit: FitResult,
    archs: list[str] | None = None,
    reference: str = "fullattn",
    regime: Regime = Regime(),
    study: str | None = None,
    S: int = 400,
    with_regret: bool = True,
    theta_samples: np.ndarray | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    study = study or fit.layout.reference_study
    archs = archs or [a for a in fit.layout.arch_tags if a != reference]
    TH = fit.samples(S, seed) if theta_samples is None else theta_samples
    p_b = _arch_params(fit, reference, study, TH)
    p_b_hat = _arch_params(fit, reference, study, fit.theta)
    out = []
    for N_ref in regime.N_ref:
        for r in regime.ratios:
            C = float(_data.gpu_hours(reference, N_ref, r * N_ref))
            for a in archs:
                p_a = _arch_params(fit, a, study, TH)
                p_a_hat = _arch_params(fit, a, study, fit.theta)
                Na = n_for_cost(a, C, r)
                La = _loss_at(fit, p_a, p_a_hat, Na, r * Na, with_regret)
                # bisection on log C_b for all samples at once
                s0_b, sig_b = _data.throughput(reference)
                lo = np.full(TH.shape[1], np.log(C) - 8.0)
                hi = np.full(TH.shape[1], np.log(C) + 8.0)
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    Nb = (np.exp(mid) * 3600.0 * s0_b / r) ** (1.0 / (1.0 + sig_b))
                    Lb = _loss_at(fit, p_b, p_b_hat, Nb, r * Nb, with_regret)
                    hi = np.where(Lb <= La, mid, hi)  # reference already reaches L_a
                    lo = np.where(Lb <= La, lo, mid)
                logM = 0.5 * (lo + hi) - np.log(C)
                out.append({
                    "arch": a, "N_ref": N_ref, "ratio": r, "gpu_h": C, "N_arch": Na,
                    "M_median": float(np.exp(np.median(logM))),
                    "M_q05": float(np.exp(np.quantile(logM, 0.05))),
                    "M_q95": float(np.exp(np.quantile(logM, 0.95))),
                    "logM_samples": logM,
                })
    return pd.DataFrame(out)


@dataclass
class EliminationRule:
    delta: float = 0.05  # "within 5% compute of the reference" counts as competitive
    p_drop: float = 0.05


def stability_flags(fit: FitResult) -> dict[str, list[str]]:
    """Per-arch reasons that veto automatic promotion: diverged / QA-failed runs and
    structural misfit of the functional form (see model.FitResult.arch_misfit)."""
    flags: dict[str, list[str]] = {}
    for h in fit.held_out:
        if h.reason.startswith("run QA"):
            flags.setdefault(h.arch, []).append(f"{h.reason}: {h.obs_id}")
    for _, r in fit.arch_misfit.iterrows():
        if r.structural_misfit:
            flags.setdefault(r.arch, []).append(f"form misfit (LOO rms z={r.rms_z:.2f})")
    return flags


def eliminate(mult: pd.DataFrame, rule: EliminationRule = EliminationRule(),
              flags: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """drop:    P(M > 1 - delta) < p_drop at *every* regime point (conservative: an arm
                that might win anywhere in the regime survives)
       promote: P(M > 1 + delta) > 1 - p_drop at every regime point
       review:  would be promoted, but has stability / misfit flags (never auto-promote
                an arm that diverged or that the form does not describe)
       live:    otherwise."""
    flags = flags or {}
    rows = []
    for a, g in mult.groupby("arch"):
        p_comp = [float(np.mean(s > np.log(1 - rule.delta))) for s in g.logM_samples]
        p_win = [float(np.mean(s > np.log(1 + rule.delta))) for s in g.logM_samples]
        if max(p_comp) < rule.p_drop:
            status = "drop"
        elif min(p_win) > 1 - rule.p_drop:
            status = "promote"
        else:
            status = "live"
        if status == "promote" and flags.get(a):
            status = "review"
        rows.append({"arch": a, "status": status, "P_competitive_max": max(p_comp),
                     "P_better_min": min(p_win), "M_median_range":
                     f"{g.M_median.min():.2f}-{g.M_median.max():.2f}",
                     "flags": "; ".join(flags.get(a, []))})
    return pd.DataFrame(rows)

"""Out-of-sample checks: the only admissible evidence that a form extrapolates.

  holdout_validation   refit without a subset (e.g. the largest size), predict it
  leave_one_scale_out  holdout_validation for each size in turn (largest first)
  compare_forms        the same, for several forms -> weights for model averaging
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .model import HierarchicalModel


def holdout_validation(model: HierarchicalModel, df: pd.DataFrame, mask: np.ndarray,
                       taus: dict | None = None) -> pd.DataFrame:
    """Refit on ``~mask`` (taus fixed if given), predict ``mask`` rows with predictive
    sd (parameter + noise). Rows with weight 0 are skipped."""
    mask = np.asarray(mask, bool)
    train, test = df[~mask], df[mask & (df.get("weight", 1.0) > 0)]
    fit = model.fit(train, learn_taus=taus is None, taus=taus)
    mu, sd = fit.predict(test, noise=True)
    out = test[["obs_id", "arch", "study", "N", "D", "lr", "B", "y"]].copy()
    out["pred"] = mu
    out["sd"] = sd
    out["z"] = (out.y - mu) / sd
    out["rel_err_pct"] = 100 * (np.exp(out.y - mu) - 1)
    return out


def summarize(v: pd.DataFrame) -> dict:
    return {"n": len(v), "mean_rel_err_pct": float(v.rel_err_pct.mean()),
            "mae_rel_err_pct": float(v.rel_err_pct.abs().mean()),
            "coverage_90": float(np.mean(np.abs(v.z) < 1.645)),
            "rms_z": float(np.sqrt(np.mean(v.z**2)))}


def leave_one_scale_out(model: HierarchicalModel, df: pd.DataFrame, taus: dict | None = None,
                        study: str | None = None) -> pd.DataFrame:
    rows = []
    for N in sorted(df.N.unique(), reverse=True):
        m = (df.N == N).values
        if study is not None:
            m &= (df.study == study).values
        v = holdout_validation(model, df, m, taus)
        rows.append({"held_out_N": N, **summarize(v)})
    return pd.DataFrame(rows)


def compare_forms(models: dict[str, HierarchicalModel], df: pd.DataFrame,
                  mask: np.ndarray) -> pd.DataFrame:
    """Held-out log predictive density per form -> softmax weights."""
    rows = []
    for name, m in models.items():
        v = holdout_validation(m, df, mask)
        lpd = float(np.sum(-0.5 * v.z**2 - np.log(v.sd) - 0.5 * np.log(2 * np.pi)))
        rows.append({"form": name, "lpd": lpd, **summarize(v)})
    out = pd.DataFrame(rows)
    w = np.exp(out.lpd - out.lpd.max())
    out["weight"] = w / w.sum()
    return out

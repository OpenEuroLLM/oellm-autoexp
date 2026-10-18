"""Pick the best (gbs, lr) per cell BEFORE fitting the scaling curve.

A scaling law is a statement about the *envelope* L*(N, D) = min over training
hyperparameters. Feeding it raw sweep points instead fits a cloud that mixes
well- and badly-tuned runs, and the fitted exponent then absorbs the tuning
error. So the selection is a required step, not a convenience.

Modes
-----
``per-arch`` (default for fitting)
    argmin val_loss over (gbs, lr) within each (arch, size, budget). Each
    architecture is allowed its own optimum — the honest envelope, since the
    optimal lr genuinely differs between attention and gated-linear layers.
``shared`` (required for the head-to-head comparison)
    Within each (size, budget) pick the ONE (gbs, lr) minimising the mean loss
    across architectures, then keep that cell for every architecture. Costs a
    little envelope tightness, buys exactly matched pairs — the arch difference
    is then measured at identical data order, batch and schedule, so every
    nuisance cancels in the difference.
``none``
    Passthrough.

Note on the current sweep: ``main_scaling_all.yaml`` runs exactly ONE (gbs, lr)
per (size, budget) — the tiers are pre-picked "near-winner" combos from an
earlier baseline sweep — so selection is a no-op today and `report_coverage`
says so. It stops being a no-op the moment an lr/bs point is added, and the
tier-boundary warning below is live already.
"""
from __future__ import annotations

import pandas as pd

CELL = ["size", "budget_bt"]


def report_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """How many distinct (gbs, lr) each (arch, size, budget) cell actually has."""
    cov = (df.groupby(["arch"] + CELL)
             .apply(lambda g: len(set(zip(g["gbs"], g["lr"]))), include_groups=False)
             .rename("n_hp").reset_index())
    n_choice = int((cov["n_hp"] > 1).sum())
    print(f"[select] {len(cov)} (arch,size,budget) cells; {n_choice} have >1 (gbs,lr) to choose from")
    if n_choice == 0:
        print("[select] NOTE: one hyperparameter setting per cell — selection is a no-op. "
              "The scaling fit is only as good as the pre-picked tiers; it is NOT an envelope.")
    return cov


def report_tier_switches(df: pd.DataFrame) -> pd.DataFrame:
    """Budgets at which a size changes (gbs, lr) — the discontinuities in L(D).

    Within a size the sweep switches tier as the budget grows (0.1B gbs128->256
    at 80BT, 0.2B gbs128/lr1e-3 -> gbs512/lr2e-3 at 50BT). A single L(D) fit
    across a switch splices two hyperparameter regimes and mis-reads the slope,
    which then propagates straight into the iso-loss compute multiplier.
    """
    rows = []
    for size, g in df.groupby("size"):
        tiers = (g.groupby(["gbs", "lr"])["budget_bt"]
                  .agg(["min", "max"]).reset_index().sort_values("min"))
        if len(tiers) > 1:
            for _, t in tiers.iterrows():
                rows.append({"size": size, "gbs": t["gbs"], "lr": t["lr"],
                             "budget_lo": t["min"], "budget_hi": t["max"]})
    out = pd.DataFrame(rows)
    if not out.empty:
        print("[select] sizes spanning multiple (gbs,lr) tiers — do not fit L(D) across a switch:")
        print(out.to_string(index=False))
    return out


def select_best(df: pd.DataFrame, mode: str = "per-arch", verbose: bool = True) -> pd.DataFrame:
    """Reduce to one row per (arch, size, budget) under the chosen selection rule."""
    if mode == "none":
        return df
    if df.empty:
        return df

    if mode == "per-arch":
        keep = df.loc[df.groupby(["arch"] + CELL)["val_loss"].idxmin()]

    elif mode == "shared":
        # Rank hyperparameter cells by mean loss across the archs that ran them,
        # restricted to the HP settings every arch in the cell actually has, so
        # the winner is not decided by a partially-populated setting.
        winners = []
        for cell, g in df.groupby(CELL):
            n_arch = g["arch"].nunique()
            hp = g.groupby(["gbs", "lr"]).agg(n=("arch", "nunique"), m=("val_loss", "mean"))
            full = hp[hp["n"] == n_arch]
            best = (full if len(full) else hp)["m"].idxmin()
            winners.append({"size": cell[0], "budget_bt": cell[1], "gbs": best[0], "lr": best[1]})
        keep = df.merge(pd.DataFrame(winners), on=CELL + ["gbs", "lr"], how="inner")

    else:
        raise ValueError(f"unknown selection mode: {mode}")

    if verbose:
        print(f"[select] mode={mode}: {len(df)} -> {len(keep)} points")
    return keep.sort_values(["arch", "size", "budget_bt"]).reset_index(drop=True)

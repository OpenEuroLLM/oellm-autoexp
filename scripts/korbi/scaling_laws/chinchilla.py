"""Chinchilla / Kaplan-style scaling-law analysis, per architecture.

Fits the Hoffmann et al. parametric form

    L(N, D) = E + A * N**-alpha + B * D**-beta

and derives the compute-optimal frontier from it. Two cost accountings, because
they give materially different answers here:

  FLOPs      C = 6ND                          -> N_opt ∝ C**(beta/(alpha+beta))
  GPU-hours  C_h = D / (s0 * N**-sigma * 3600) -> N_opt ∝ C_h**(beta/(alpha+sigma*beta))

sigma ≈ 0.68 (see `costs.throughput_powerlaw`), so an extra parameter costs less
real machine time than 6ND implies and the GPU-hour-optimal model is bigger than
the FLOP-optimal one. Since sigma also differs slightly per architecture, the two
rankings need not agree — which is the point of computing both.

Method follows Hoffmann et al. Approach 3: minimise a Huber loss (delta=1e-3) on
log-space residuals with a LogSumExp parametrisation, from a grid of starts.
Uncertainty comes from a size-stratified bootstrap, which is not optional here:
this sweep spans 12x in N against Chinchilla's ~1000x, so the exponents are
weakly identified and a point estimate on its own would badly overstate what the
data supports.

Approach 2 (IsoFLOP profiles) needs >=3 model sizes sharing a FLOP budget;
`isoflop_feasibility` reports whether that has become possible yet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import logsumexp

from costs import throughput_powerlaw

# Hoffmann et al. Appendix D.2 initialisation grid.
INIT_LA = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0)
INIT_LB = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0)
INIT_LE = (-1.0, -0.5, 0.0, 0.5, 1.0)
INIT_AL = (0.0, 0.5, 1.0, 1.5, 2.0)
INIT_BE = (0.0, 0.5, 1.0, 1.5, 2.0)
HUBER_DELTA = 1e-3

PARAMS = ("le", "la", "alpha", "lb", "beta")


def _residuals(p, ln_n, ln_d, ln_l):
    """log L_pred - log L_obs, with L_pred = E + A N^-alpha + B D^-beta."""
    le, la, alpha, lb, beta = p
    stack = np.vstack([np.full_like(ln_n, le), la - alpha * ln_n, lb - beta * ln_d])
    return logsumexp(stack, axis=0) - ln_l


def _huber_cost(r, delta=HUBER_DELTA):
    a = np.abs(r)
    return float(np.sum(np.where(a <= delta, 0.5 * r**2, delta * (a - 0.5 * delta))))


def fit_parametric(sub: pd.DataFrame, full_grid: bool = True, x0=None, n_refine: int = 40) -> dict:
    """Fit L(N,D)=E+A*N^-alpha+B*D^-beta on one architecture's points.

    Multistart over the Hoffmann grid, but the grid is SCREENED first: the
    objective is cheap to evaluate and the optimiser is not, so all 4500 starts
    are scored directly and only the best `n_refine` are handed to least_squares.
    Same optimum, ~100x less work than optimising from every start.
    """
    ln_n = np.log(sub["N"].values.astype(float))
    ln_d = np.log(sub["D"].values.astype(float))
    ln_l = np.log(sub["val_loss"].values.astype(float))
    if len(ln_l) < 6:
        return {}
    lo = [-3.0, -10.0, 0.01, -10.0, 0.01]
    hi = [float(np.log(sub["val_loss"].min())), 40.0, 2.0, 40.0, 2.0]

    if x0 is not None:
        starts = [np.clip(np.asarray(x0, float), lo, hi)]
    elif full_grid:
        grid = [
            np.clip(np.array([le, la, al, lb, be]), lo, hi)
            for le in INIT_LE
            for la in INIT_LA
            for al in INIT_AL
            for lb in INIT_LB
            for be in INIT_BE
        ]
        scored = sorted(grid, key=lambda s: _huber_cost(_residuals(s, ln_n, ln_d, ln_l)))
        starts = scored[:n_refine]
    else:
        starts = [np.array([0.0, 10.0, 0.5, 10.0, 0.5])]

    best, best_obj = None, np.inf
    for s in starts:
        try:
            r = least_squares(
                _residuals,
                np.clip(s, lo, hi),
                bounds=(lo, hi),
                args=(ln_n, ln_d, ln_l),
                loss="huber",
                f_scale=HUBER_DELTA,
                max_nfev=2000,
            )
        except Exception:
            continue
        if r.cost < best_obj:
            best, best_obj = r.x, r.cost
    if best is None:
        return {}
    le, la, alpha, lb, beta = best
    resid = _residuals(best, ln_n, ln_d, ln_l)
    return {
        "E": float(np.exp(le)),
        "A": float(np.exp(la)),
        "alpha": float(alpha),
        "B": float(np.exp(lb)),
        "beta": float(beta),
        "n": int(len(ln_l)),
        "rmse_mn": float(1000 * np.sqrt(np.mean((np.expm1(resid) * sub["val_loss"].values) ** 2))),
        "_x": best,
    }


def compute_optimal(fp: dict, sigma: float | None = None) -> dict:
    """Exponents and prefactor of the compute-optimal frontier.

    sigma=None  -> FLOPs accounting, C = 6ND. sigma given -> GPU-hour
    accounting, cost ∝ D * N**sigma (see module docstring).

    Minimising L subject to the cost constraint gives     N_opt = G *
    u**(beta/(alpha+sigma*beta))     with u the cost in natural units
    D_opt = u / N_opt**sigma and sigma=1 recovers the FLOPs case
    exactly.
    """
    if not fp:
        return {}
    a, b, A, B = fp["alpha"], fp["beta"], fp["A"], fp["B"]
    s = 1.0 if sigma is None else float(sigma)
    denom = a + s * b
    if denom <= 0:
        return {}
    n_exp = b / denom  # N_opt ∝ cost^n_exp
    d_exp = 1.0 - s * n_exp  # D_opt ∝ cost^d_exp
    G = ((a * A) / (s * b * B)) ** (1.0 / denom)
    return {"n_exp": float(n_exp), "d_exp": float(d_exp), "G": float(G), "sigma": s}


def frontier(fp: dict, co: dict, cost: np.ndarray, sigma_units: float = 6.0):
    """N_opt, D_opt and L_opt along the compute-optimal frontier.

    `cost` is FLOPs (with sigma_units=6, so u = C/6) or GPU-hours * throughput
    constant, whichever `co` was built for.
    """
    u = np.asarray(cost, float) / sigma_units
    n = co["G"] * u ** co["n_exp"]
    d = u / n ** co["sigma"]
    loss = fp["E"] + fp["A"] * n ** -fp["alpha"] + fp["B"] * d ** -fp["beta"]
    return n, d, loss


def bootstrap(
    sub: pd.DataFrame, fp: dict, n_boot: int = 200, seed: int = 0, sigma: float | None = None
) -> dict:
    """Size-stratified bootstrap CIs on the exponents and the frontier slope.

    Stratified by model size on purpose: with only four distinct N, an
    unstratified resample regularly drops a whole size and the fit degenerates,
    which would inflate the interval for the wrong reason.
    """
    if not fp:
        return {}
    rng = np.random.default_rng(seed)
    groups = [g.index.values for _, g in sub.groupby("size")]
    out = {"alpha": [], "beta": [], "n_exp": [], "E": []}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        f = fit_parametric(sub.loc[idx], full_grid=False, x0=fp["_x"])
        if not f:
            continue
        co = compute_optimal(f, sigma)
        if not co:
            continue
        out["alpha"].append(f["alpha"])
        out["beta"].append(f["beta"])
        out["E"].append(f["E"])
        out["n_exp"].append(co["n_exp"])
    ci = {}
    for k, v in out.items():
        if len(v) >= 20:
            ci[f"{k}_lo"], ci[f"{k}_hi"] = (float(x) for x in np.percentile(v, [5, 95]))
        else:
            ci[f"{k}_lo"] = ci[f"{k}_hi"] = np.nan
    ci["n_boot_ok"] = len(out["alpha"])
    return ci


def isoflop_feasibility(df: pd.DataFrame, bins_per_decade: int = 3) -> pd.DataFrame:
    """Can Approach 2 (IsoFLOP parabolas) be run yet?

    Needs >=3 sizes per budget.
    """
    d = df.copy()
    d["flop_bin"] = (np.round(np.log10(d["flops"]) * bins_per_decade) / bins_per_decade).round(3)
    cov = d.groupby(["arch", "flop_bin"])["size"].nunique().rename("n_sizes").reset_index()
    usable = cov[cov["n_sizes"] >= 3]
    best = int(cov["n_sizes"].max()) if len(cov) else 0
    print(
        f"[chinchilla] IsoFLOP (Approach 2): {len(usable)} of {len(cov)} "
        f"(arch, FLOP-bin) cells have >=3 model sizes; best cell has {best}"
    )
    # A parabola has 3 parameters, so 3 sizes locate a minimum with zero degrees
    # of freedom and no error bar — Hoffmann et al. used ~10 sizes per budget.
    # Report the coverage but do not run it: a 3-point minimum would look like a
    # measurement and is not one.
    print(
        "[chinchilla] NOT run — a 3-point parabola has no residual DOF, so its "
        "minimum carries no uncertainty. Approach 3 (parametric) only. Enable it "
        "by running >=4 model sizes at a shared token budget."
    )
    return usable


def analyse(
    df: pd.DataFrame, n_boot: int = 200, verbose: bool = True, n_mode: str = "measured"
) -> pd.DataFrame:
    """Per-architecture parametric fit + both compute-optimal frontiers."""
    from fit import ARCH_ORDER

    rows = []
    # ONE reference budget for every architecture — the top of the range they all
    # cover. Evaluating each at its own max FLOPs would rank them by who happens
    # to own the biggest run, not by who is better.
    c_ref = float(df.groupby("arch")["flops"].max().min())
    for arch in [a for a in ARCH_ORDER if a in set(df["arch"])]:
        sub = df[df["arch"] == arch]
        fp = fit_parametric(sub)
        if not fp:
            if verbose:
                print(f"[chinchilla] {arch}: too few points ({len(sub)}) — skipped")
            continue
        _, sigma = throughput_powerlaw(arch, n_mode)
        co_f = compute_optimal(fp)  # FLOPs, C = 6ND
        co_h = compute_optimal(fp, sigma=sigma)  # GPU-hours, cost ∝ D*N^sigma
        ci = bootstrap(sub, fp, n_boot=n_boot)
        # tokens-per-parameter is only constant in C when alpha == beta, so it
        # too is quoted at the shared reference budget.
        n_o, d_o, l_o = frontier(fp, co_f, np.array([c_ref]))
        extrap = c_ref > float(sub["flops"].max()) * 1.05
        rows.append(
            {
                "arch": arch,
                "n": fp["n"],
                "rmse_mn": fp["rmse_mn"],
                "E": fp["E"],
                "A": fp["A"],
                "B": fp["B"],
                "alpha": fp["alpha"],
                "beta": fp["beta"],
                "N_exp_flops": co_f["n_exp"],
                "D_exp_flops": co_f["d_exp"],
                "sigma": sigma,
                "N_exp_gpuh": co_h["n_exp"],
                "D_exp_gpuh": co_h["d_exp"],
                "C_ref": c_ref,
                "N_opt_ref": float(n_o[0]),
                "D_opt_ref": float(d_o[0]),
                "tok_per_param_ref": float(d_o[0] / n_o[0]),
                "L_opt_ref": float(l_o[0]),
                "L_opt_extrapolated": bool(extrap),
                **ci,
            }
        )
    out = pd.DataFrame(rows)
    if verbose and not out.empty:
        wide = (out["alpha_hi"] - out["alpha_lo"]) > out["alpha"]
        if wide.any():
            print(
                f"[chinchilla] WARNING: alpha CI wider than the estimate for "
                f"{', '.join(out.loc[wide, 'arch'])} — N spans only "
                f"{df['N'].max() / df['N'].min():.0f}x here vs ~1000x in Hoffmann et al. "
                f"Treat the exponents as indicative, not measured."
            )
    return out


# ---------------------------------------------------------------- plotting ---
def plot_loglog(
    df: pd.DataFrame,
    res: pd.DataFrame,
    out_png: str,
    xcol: str = "gpu_hours",
    xlabel: str = "training GPU-hours",
):
    """Kaplan-style log-log: cost vs final loss, and vs REDUCIBLE loss.

    Left panel is the literal request — log cost against log final loss. It is
    almost straight but nearly flat, and that is the honest picture: loss is
    bounded below by E ~ 1.6, so over this sweep log L spans only ~0.09 decades
    and the log axis buys little over a linear one.

    Right panel is where the power law actually lives. Subtracting the fitted
    per-architecture E leaves the reducible loss, which spans ~0.26 decades and
    IS a straight line in log-log — its slope is the compute exponent. Reading a
    slope off the left panel instead would systematically understate it, because
    the irreducible floor flattens the curve as loss falls.

    Both panels connect budgets within one model size only, same as the ratio
    plot: joining points across sizes would draw a path no run ever took.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    from fit import (
        ARCH_COLORS,
        ARCH_DASHES,
        ARCH_ORDER,
        INK,
        INK_2,
        SIZE_MARKER,
        envelope,
        legends,
        style_axes,
    )

    archs = [a for a in ARCH_ORDER if a in set(df["arch"])]
    e_by_arch = {r["arch"]: r["E"] for _, r in res.iterrows()} if not res.empty else {}
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))
    slopes = {}
    for arch in archs:
        sub = df[df["arch"] == arch]
        col, dash = ARCH_COLORS[arch], ARCH_DASHES[arch]
        e = e_by_arch.get(arch, np.nan)
        red = sub["val_loss"] - e
        for ax, y, ok in [
            (axes[0], sub["val_loss"], sub["val_loss"] > 0),
            (axes[1], red, red > 1e-4),
        ]:
            s = sub[ok]
            yy = y[ok]
            for size, g in s.groupby("size"):
                idx = g.sort_values(xcol).index
                if len(idx) >= 2:
                    ax.plot(
                        df.loc[idx, xcol],
                        yy.loc[idx],
                        color=col,
                        lw=1.8,
                        dashes=dash,
                        alpha=0.7,
                        zorder=3,
                    )
                ax.scatter(
                    df.loc[g.index, xcol],
                    yy.loc[g.index],
                    color=col,
                    marker=SIZE_MARKER.get(size, "o"),
                    s=46,
                    zorder=4,
                    linewidths=1.1,
                    edgecolors="#fcfcfb",
                )
        # Slope on the compute-optimal FRONTIER, not through the cloud: a line
        # through all (N, D) cells measures the shape of the sampling grid as
        # much as the architecture, which is the same trap `fit.envelope` exists
        # to avoid.
        if np.isfinite(e):
            front = envelope(sub, xcol)
            fred = front["val_loss"] - e
            m = fred > 1e-4
            if m.sum() >= 3:
                slope, icpt = np.polyfit(np.log10(front.loc[m, xcol]), np.log10(fred[m]), 1)
                slopes[arch] = float(slope)
                xs = np.logspace(
                    np.log10(front.loc[m, xcol].min()), np.log10(front.loc[m, xcol].max()), 40
                )
                axes[1].plot(
                    xs,
                    10 ** (icpt + slope * np.log10(xs)),
                    color=col,
                    lw=2.6,
                    dashes=dash,
                    alpha=0.95,
                    zorder=6,
                )

    for ax, ylab, title in [
        (axes[0], "validation loss (log)", "Final loss vs compute"),
        (axes[1], "reducible loss  L − E  (log)", "Reducible loss — slope is the compute exponent"),
    ]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10.5, color=INK)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.tick_params(axis="y", which="minor", labelsize=7)
        style_axes(ax)
    fig.subplots_adjust(left=0.07, right=0.79, top=0.90, bottom=0.17, wspace=0.26)
    legends(
        axes[1],
        archs,
        df["size"],
        suffix={a: (f"  slope {slopes[a]:+.3f}" if a in slopes else "") for a in archs},
    )
    fig.text(
        0.07,
        0.035,
        "thin lines join budgets within one model size · heavy line and "
        "quoted slope are the compute-optimal frontier",
        fontsize=8,
        color=INK_2,
    )
    fig.text(
        0.07,
        0.008,
        "CAVEAT: slopes are conditional on the fitted per-arch E — a larger "
        "E mechanically steepens the slope, and E is weakly identified here, so rank "
        "architectures by the paired ratio, not by these slopes.",
        fontsize=8,
        color=INK_2,
    )
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_frontier(df: pd.DataFrame, res: pd.DataFrame, out_png: str):
    """Compute-optimal frontier per arch: N_opt(C) and the resulting L_opt(C)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from fit import ARCH_COLORS, ARCH_DASHES, INK, INK_2, legends, style_axes

    archs = list(res["arch"])
    lo, hi = df["flops"].min(), df["flops"].max()
    cs = np.logspace(np.log10(lo), np.log10(hi), 60)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.9))
    for arch in archs:
        r = res[res["arch"] == arch].iloc[0]
        fp = {k: float(r[k]) for k in ("E", "A", "alpha", "B", "beta")}
        col, dash = ARCH_COLORS[arch], ARCH_DASHES[arch]
        sub = df[df["arch"] == arch]
        co = compute_optimal(fp)
        n_o, d_o, l_o = frontier(fp, co, cs)
        axes[0].plot(cs, n_o, color=col, lw=2.0, dashes=dash)
        axes[1].plot(cs, l_o, color=col, lw=2.0, dashes=dash)
        axes[1].scatter(
            sub["flops"], sub["val_loss"], color=col, s=16, alpha=0.30, linewidths=0, zorder=2
        )
    for ax, ylab, title in [
        (axes[0], "compute-optimal N (non-emb params)", "Optimal model size"),
        (axes[1], "validation loss", "Compute-optimal frontier vs observed points"),
    ]:
        ax.set_xscale("log")
        ax.set_xlabel("training FLOPs (6ND)")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10.5, color=INK)
        style_axes(ax)
    axes[0].set_yscale("log")
    axes[0].text(
        0.02,
        0.94,
        "slope = N_opt exponent\n(Chinchilla 0.50, Kaplan 0.73)",
        transform=axes[0].transAxes,
        fontsize=8,
        color=INK_2,
        va="top",
    )
    fig.subplots_adjust(left=0.075, right=0.80, top=0.90, bottom=0.13, wspace=0.28)
    legends(
        axes[1],
        archs,
        suffix={a: f"  N∝C^{res.loc[res.arch == a, 'N_exp_flops'].iloc[0]:.2f}" for a in archs},
    )
    fig.text(
        0.075,
        0.015,
        "exponents are weakly identified over this N range — read the "
        "bootstrap CIs in chinchilla_fits.csv before quoting them",
        fontsize=8,
        color=INK_2,
    )
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

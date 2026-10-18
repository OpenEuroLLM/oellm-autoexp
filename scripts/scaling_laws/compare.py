"""Head-to-head: does the hybrid advantage over full attention survive scale?

This is the gate the scaling fit has to pass first. Three questions, in order:

1. **Is the hybrid better everywhere?**  `matched_pairs` differences each hybrid
   against fullattn *within* a (size, budget, gbs, lr) cell. The sweep runs
   every architecture on the identical data order, batch and schedule, so the
   difference cancels those nuisances exactly — but NOT model size: the hybrids
   carry +9-24% non-embedding params over fullattn at the same size label.
   `param_adjusted` removes that part using fullattn's own fitted N-term, so
   "architecture effect" and "it is simply bigger" can be told apart.

2. **Does the distance keep similar?**  `trend_test` regresses that difference on
   ln N and ln D. `dDelta/dlnD > 0` means the advantage ERODES as you overtrain —
   the one outcome that would kill extrapolation. Errors are reported, because
   "the gap is flat" and "the gap is unmeasurable" look identical without them.

3. **Is it worth the throughput it costs?**  A constant loss offset is not a
   constant compute saving. `iso_loss_multiplier` converts Delta into the factor
   by which fullattn compute must grow to reach the hybrid's loss, and then into
   GPU-hours via the measured throughput. This is the decision metric: at
   SEQ_LEN 4096 every hybrid is 3-18% SLOWER per token, so a quality win only
   pays if it exceeds that.

The conversion in (3) uses the local slope dL/dln(D) of the fullattn curve inside
the same (gbs, lr) tier. That slope FLATTENS as D/N grows, which is why the same
Delta buys a larger compute multiplier the deeper into overtraining you go.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from costs import tok_s_ratio

BASELINE = "fullattn"
CELL = ["size", "budget_bt", "gbs", "lr"]


def matched_pairs(df: pd.DataFrame, baseline: str = BASELINE) -> pd.DataFrame:
    """Per-cell loss difference of each arch vs `baseline` (negative = better)."""
    base = (df[df["arch"] == baseline]
            .set_index(CELL)[["val_loss", "gpu_hours", "flops", "N"]]
            .rename(columns={"val_loss": "base_loss", "gpu_hours": "base_gpu_hours",
                             "flops": "base_flops", "N": "base_N"}))
    other = df[df["arch"] != baseline].set_index(CELL)
    j = other.join(base, how="inner").reset_index()
    j["delta"] = j["val_loss"] - j["base_loss"]
    j["delta_mn"] = 1000.0 * j["delta"]          # milli-nats, the readable unit
    # Loss RATIO. A constant ratio is the scale-free statement: it says the arch
    # is a fixed percentage better at any (N, D), so the claim transfers to
    # arbitrary scale without knowing where the loss curve ends up. A constant
    # DIFFERENCE is a different claim, and the two are not compatible — over this
    # sweep L falls ~23%, so a fixed 25 mn gap is a ratio drifting by ~0.2 pp,
    # and a fixed ratio is a gap shrinking from 25 to 20 mn. `trend_test` reports
    # both so the data picks.
    j["ratio"] = j["val_loss"] / j["base_loss"]
    j["ratio_pct"] = 100.0 * (j["ratio"] - 1.0)  # negative = hybrid better
    return j.sort_values(["arch", "size", "budget_bt"]).reset_index(drop=True)


def _ols(y, lnn, lnd):
    """Regress y on centred ln N and ln D; return coefficients and their SEs."""
    X = np.c_[np.ones(len(y)), lnn - lnn.mean(), lnd - lnd.mean()]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = float(resid @ resid) / max(1, len(y) - X.shape[1])
    return beta, np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * s2), float(np.sqrt(s2))


def trend_test(pairs: pd.DataFrame) -> pd.DataFrame:
    """Does the advantage hold at arbitrary scale — as a difference or as a ratio?

    Runs the same regression on both readings of "the distance stays similar":

      difference  Delta   = L_arch - L_base            (milli-nats)
      ratio       R       = 100*(L_arch/L_base - 1)    (percent)

    and reports each one's drift with ln N and ln D. The invariant to trust is
    whichever has a scale trend indistinguishable from zero: that is the one
    that transfers to scales outside the sweep. `better_invariant` names it by
    comparing |t| on the ln D slope, since overtraining is the axis in question.

    Centring makes each intercept the mean advantage over the covered range
    rather than an extrapolation to N=D=1.
    """
    out = []
    for arch, g in pairs.groupby("arch"):
        if len(g) < 4:
            out.append({"arch": arch, "n": len(g),
                        "mean_delta_mn": g["delta_mn"].mean(),
                        "mean_ratio_pct": g["ratio_pct"].mean()})
            continue
        lnn = np.log(g["N"].values.astype(float))
        lnd = np.log(g["D"].values.astype(float))
        bd, sd, rd = _ols(g["delta_mn"].values, lnn, lnd)
        br, sr, rr = _ols(g["ratio_pct"].values, lnn, lnd)
        t_d = abs(bd[2] / sd[2]) if sd[2] else np.inf
        t_r = abs(br[2] / sr[2]) if sr[2] else np.inf
        out.append({
            "arch": arch, "n": len(g),
            "mean_delta_mn": bd[0], "se_mean": sd[0],
            "dDelta_dlnN": bd[1], "se_dlnN": sd[1],
            "dDelta_dlnD": bd[2], "se_dlnD": sd[2], "resid_mn": rd,
            "mean_ratio_pct": br[0], "se_ratio": sr[0],
            "dRatio_dlnN": br[1], "se_rlnN": sr[1],
            "dRatio_dlnD": br[2], "se_rlnD": sr[2], "resid_ratio_pct": rr,
            "t_delta_lnD": t_d, "t_ratio_lnD": t_r,
            "better_invariant": "ratio" if t_r < t_d else "difference",
        })
    return pd.DataFrame(out)


def pairwise(df: pd.DataFrame, archs=None) -> pd.DataFrame:
    """Every architecture against every other, paired on the cells both ran.

    Answers "which hybrid is best" directly, without routing through a fit. The
    pairing is exact (same data order, batch, schedule), so the paired SE is far
    tighter than either architecture's own spread and small margins are still
    resolvable.
    """
    import itertools
    p = df.pivot_table(index=CELL, columns="arch", values="val_loss")
    archs = list(archs or p.columns)
    rows = []
    for a, b in itertools.combinations(archs, 2):
        if a not in p or b not in p:
            continue
        q = p[[a, b]].dropna()
        if len(q) < 3:
            continue
        r = 100.0 * (q[a] / q[b] - 1.0)
        se = float(r.std(ddof=1) / np.sqrt(len(r)))
        rows.append({"arch_a": a, "arch_b": b, "n": len(r), "mean_pct": float(r.mean()),
                     "se": se, "t": float(r.mean() / se) if se else np.inf})
    return pd.DataFrame(rows)


def _n_credit(A: float, alpha: float, n, n_ref):
    """Loss an arch gains over `n_ref` from its extra parameters alone (>0 = bigger)."""
    return A * (np.asarray(n_ref, float) ** -alpha - np.asarray(n, float) ** -alpha)


def param_adjusted(df: pd.DataFrame, baseline: str = BASELINE, n_boot: int = 200,
                   seed: int = 0) -> tuple[pd.DataFrame, dict, list]:
    """Re-express every loss at the BASELINE's parameter count of its size.

    Fits L = E + A*N^-alpha + B*D^-beta on the baseline alone (where N is not in
    question), then moves each point along that N-term from its own measured N
    to the baseline's N at the same size and D:

        val_loss_adj = val_loss + A * (N_base^-alpha - N^-alpha)

    so a hybrid that is only better because it is bigger comes out even. This
    assumes a mixer parameter is worth what a dense-attention/MLP parameter is
    worth to fullattn — the natural null, but an assumption: if hybrid params are
    worth less, the adjustment over-corrects. alpha is weakly identified (N spans
    12x), so the credit is bootstrapped (size-stratified, as `chinchilla.bootstrap`)
    and returned as `n_credit_lo/hi_mn`; `samples` holds the (A, alpha) draws for
    `pairwise(..., credit_samples=...)`.
    """
    from chinchilla import fit_parametric
    base = df[df["arch"] == baseline]
    fp = fit_parametric(base)
    if not fp:
        return pd.DataFrame(), {}, []
    n_ref = df["size"].map(base.groupby("size")["N"].first())
    out = df.copy()
    out["N_ref"] = n_ref
    credit = _n_credit(fp["A"], fp["alpha"], out["N"], n_ref)
    out["n_credit_mn"] = 1000.0 * credit
    out["val_loss_raw"] = out["val_loss"]
    out["val_loss_adj"] = out["val_loss"] + credit

    rng = np.random.default_rng(seed)
    groups = [g.index.values for _, g in base.groupby("size")]
    samples = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        f = fit_parametric(base.loc[idx], full_grid=False, x0=fp["_x"])
        if f:
            samples.append((f["A"], f["alpha"]))
    if len(samples) >= 20:
        c = np.array([1000.0 * _n_credit(A, a, out["N"], n_ref) for A, a in samples])
        out["n_credit_lo_mn"], out["n_credit_hi_mn"] = np.percentile(c, [5, 95], axis=0)
    return out, fp, samples


def with_adjusted_loss(adj: pd.DataFrame, A: float | None = None,
                       alpha: float | None = None) -> pd.DataFrame:
    """`adj` with val_loss swapped for the param-adjusted loss (optionally re-derived
    for another (A, alpha) draw), so every existing comparison runs on it unchanged."""
    out = adj.copy()
    if A is None:
        out["val_loss"] = out["val_loss_adj"]
    else:
        out["val_loss"] = out["val_loss_raw"] + _n_credit(A, alpha, out["N"], out["N_ref"])
    return out


def pairwise_adjusted(adj: pd.DataFrame, samples: list, archs=None) -> pd.DataFrame:
    """`pairwise` on param-adjusted losses, plus a 90% band on each mean from the
    bootstrap (A, alpha) draws — the fit uncertainty the paired SE cannot see."""
    pw = pairwise(with_adjusted_loss(adj), archs)
    if pw.empty or len(samples) < 20:
        return pw
    draws = [pairwise(with_adjusted_loss(adj, A, a), archs).set_index(["arch_a", "arch_b"])
             ["mean_pct"] for A, a in samples]
    d = pd.concat(draws, axis=1)
    lo, hi = np.nanpercentile(d.values, [5, 95], axis=1)
    band = pd.DataFrame({"fit_lo_pct": lo, "fit_hi_pct": hi}, index=d.index).reset_index()
    return pw.merge(band, on=["arch_a", "arch_b"], how="left")


def ranking_by_tier(df: pd.DataFrame, reference: str = "gdn7_nope") -> pd.DataFrame:
    """Loss ratio vs `reference` per (gbs, lr) tier — does the ranking hold up?

    A ranking averaged over tiers is only meaningful if it is the same ranking in
    each of them. Where it is not, the difference is telling you about
    hyperparameter sensitivity, not about the architecture, and the average is an
    artifact of which tiers happen to be populated.
    """
    p = df.pivot_table(index=CELL + ["tier"], columns="arch", values="val_loss").reset_index()
    if reference not in p:
        return pd.DataFrame()
    rows = []
    for tier, g in p.groupby("tier"):
        if g[reference].isna().all():
            continue
        for a in [c for c in p.columns if c not in CELL + ["tier"]]:
            if a == reference or g[a].isna().all():
                continue
            r = 100.0 * (g[a] / g[reference] - 1.0)
            rows.append({"tier": tier, "arch": a, "reference": reference,
                         "n": int(r.notna().sum()), "mean_pct": float(r.mean())})
    return pd.DataFrame(rows)


def _tier_slope(g: pd.DataFrame) -> tuple[float, float] | None:
    """Local log-linear fit L = c + b*ln(D) of the baseline inside one tier."""
    if len(g) < 2:
        return None
    b, c = np.polyfit(np.log(g["D"].values.astype(float)), g["val_loss"].values, 1)
    return float(b), float(c)


def iso_loss_multiplier(df: pd.DataFrame, baseline: str = BASELINE) -> pd.DataFrame:
    """Compute factor by which `baseline` must grow to match each arch's loss.

    `tok_mult`  > 1 : the arch reaches that loss on fewer tokens.
    `gpuh_mult` > 1 : the arch reaches that loss on fewer GPU-hours  <- decision metric.
    `extrapolated`  : the required baseline D lies outside the tier's fitted range,
                      so the multiplier leans on the local slope beyond its support.
    """
    rows = []
    for tier, g in df.groupby("tier"):
        bg = g[g["arch"] == baseline]
        fitres = _tier_slope(bg)
        if fitres is None:
            continue
        b, c = fitres
        if b >= 0:                       # non-decreasing baseline: slope is noise
            continue
        lo, hi = np.log(bg["D"].min()), np.log(bg["D"].max())
        size = bg["size"].iloc[0]
        for _, r in g[g["arch"] != baseline].iterrows():
            ln_need = (r["val_loss"] - c) / b        # baseline ln D at this loss
            tok = float(np.exp(ln_need) / r["D"])
            rows.append({
                "tier": tier, "arch": r["arch"], "size": size,
                "budget_bt": r["budget_bt"], "d_over_n": r["d_over_n"],
                "delta_mn": 1000.0 * (r["val_loss"] - (c + b * np.log(r["D"]))),
                "baseline_slope": b, "n_baseline": len(bg),
                "tok_mult": tok,
                "gpuh_mult": tok * tok_s_ratio(r["arch"], size, baseline),
                "extrapolated": not (lo <= ln_need <= hi),
            })
    return pd.DataFrame(rows).sort_values(["arch", "size", "budget_bt"]).reset_index(drop=True)


# ---------------------------------------------------------------- plotting ---
def _scale_plot(df, ycol, ref, ylabel, title, note, out_png, hollow_col=None,
                xcol="d_over_n",
                xlabel="overtraining ratio  D / N  (tokens per non-embedding parameter)"):
    """Shared body of the comparison plots: y against a log cost/scale axis.

    Lines connect points WITHIN one model size only. A line across sizes would
    draw a trajectory that no run ever took — the x ordering interleaves the size
    ladders, so joining them implies a continuity the data does not have.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from fit import ARCH_COLORS, ARCH_DASHES, ARCH_LABEL, SIZE_MARKER, label_ends, legends, style_axes

    archs = [a for a in ARCH_COLORS if a in set(df["arch"])]
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    ax.axhline(ref, color="#52514e", lw=1.2, zorder=1)
    ends = []
    for arch in archs:
        sub = df[df["arch"] == arch]
        col = ARCH_COLORS[arch]
        for size, g in sub.groupby("size"):
            g = g.sort_values(xcol)
            if len(g) >= 2:                       # ladder within one model size
                ax.plot(g[xcol], g[ycol], color=col, lw=1.8,
                        dashes=ARCH_DASHES[arch], alpha=0.7, zorder=3)
            filled = g if hollow_col is None else g[~g[hollow_col]]
            ax.scatter(filled[xcol], filled[ycol], color=col,
                       marker=SIZE_MARKER.get(size, "o"), s=52, zorder=4,
                       linewidths=1.2, edgecolors="#fcfcfb")
            if hollow_col is not None:            # hollow = extrapolated, lower trust
                h = g[g[hollow_col]]
                ax.scatter(h[xcol], h[ycol], facecolors="none", edgecolors=col,
                           marker=SIZE_MARKER.get(size, "o"), s=52, zorder=4, linewidths=1.4)
        if len(sub):
            last = sub.loc[sub[xcol].idxmax()]
            ends.append((last[xcol], last[ycol], ARCH_LABEL.get(arch, arch), col))
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, color="#0b0b0b")
    ax.margins(x=0.16)
    style_axes(ax)
    ax.text(0.005, -0.135, note, transform=ax.transAxes, fontsize=8, color="#52514e")
    fig.subplots_adjust(left=0.085, right=0.70, top=0.91, bottom=0.14)
    legends(ax, archs, df["size"])
    label_ends(ax, ends)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_deltas(pairs: pd.DataFrame, out_png: str):
    """Delta vs D/N — the 'does the distance keep similar' plot."""
    _scale_plot(pairs, "delta_mn", 0.0,
                "Δ validation loss vs full attention  (milli-nats)",
                "Hybrid advantage over full attention — matched (size, budget, gbs, lr)",
                "below 0 = hybrid better · lines join budgets within one model size",
                out_png)


def plot_ratios(pairs: pd.DataFrame, out_png: str):
    """Loss RATIO vs D/N — flat here means the advantage holds at arbitrary scale."""
    _scale_plot(pairs, "ratio_pct", 0.0,
                "validation loss vs full attention  (%, negative = better)",
                "Relative hybrid advantage — a flat line extrapolates to any scale",
                "flat = constant percentage gain, independent of N and D",
                out_png)


def plot_ratio_vs_compute(pairs: pd.DataFrame, out_png: str, xcol: str = "gpu_hours"):
    """Loss ratio against COMPUTE — the scaling reading of the same quantity.

    Same y-axis as `plot_ratios`, but the x-axis is total training cost instead
    of D/N, which puts all four size ladders on one common scale: a flat band
    then says the relative advantage is the same at every compute budget, which
    is the statement you extrapolate with.

    Axis choice matters and is not symmetric between the two cost measures:
      flops     — identical for both architectures at a matched cell (6ND with N
                  and D matched), so the x position is unambiguous.
      gpu_hours — the HYBRID's own cost, since throughput differs. Points are
                  therefore at matched (N, D) but plotted at what the hybrid
                  actually spends; full attention would sit at `base_gpu_hours`,
                  slightly to the left for every hybrid except SWA.
    """
    lab = {"gpu_hours": "training GPU-hours (this architecture's own cost)",
           "flops": "training FLOPs (6ND — identical for both at a matched cell)"}
    note = {"gpu_hours": "flat = same relative gain at every compute budget · "
                         "x is the hybrid's own cost; full attention sits slightly left",
            "flops": "flat = same relative gain at every compute budget · "
                     "x is identical for both architectures at a matched cell"}
    _scale_plot(pairs, "ratio_pct", 0.0,
                "validation loss vs full attention  (%, negative = better)",
                "Relative hybrid advantage vs compute — the scaling view",
                note[xcol], out_png, xcol=xcol, xlabel=lab[xcol])


def plot_multipliers(mult: pd.DataFrame, out_png: str):
    """Iso-loss GPU-hour multiplier vs D/N — the 'is it compute optimal' plot."""
    _scale_plot(mult, "gpuh_mult", 1.0,
                "GPU-hours full attention needs for the same loss  (×)",
                "Iso-loss compute multiplier vs full attention (measured throughput, seq 4096)",
                "above 1 = hybrid is compute-cheaper · hollow = extrapolated beyond the fitted range",
                out_png, hollow_col="extrapolated")


def plot_param_adjusted(adj: pd.DataFrame, out_png: str, baseline: str = BASELINE):
    """Every architecture vs full attention, before and after pricing out parameters.

    Top row, shared y: the per-cell loss ratio vs `baseline` against training
    compute — left RAW, right PARAM-ADJUSTED (`param_adjusted`). Reading across
    the pair shows how much of each hybrid's apparent gain was only model size.
    The adjusted points carry the 90% bootstrap band of the fullattn N-term fit
    as error bars (fit uncertainty only, not the equal-worth-per-parameter
    assumption). x is the BASELINE cell's 6ND FLOPs, so all archs of one cell
    share an x; lines join one (size, gbs, lr) ladder only, never across a tier
    switch or a size.

    Bottom: the "who wins at each size" summary — mean adjusted ratio +- SE per
    (arch, size), with the raw mean as a hollow marker and a thin connector so
    the size of the parameter correction is visible per point.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from fit import (ARCH_COLORS, ARCH_DASHES, ARCH_LABEL, INK, INK_2, SIZE_MARKER, SURFACE,
                     label_ends, legends, style_axes)

    raw = matched_pairs(adj, baseline)
    fixed = matched_pairs(with_adjusted_loss(adj), baseline)
    if raw.empty:
        return
    # The baseline's credit is 0, so its loss is unchanged: the band on the ratio
    # is just the credit band divided by the baseline loss.
    if "n_credit_lo_mn" in fixed:
        fixed["ratio_lo"] = fixed["ratio_pct"] + 0.1 * (fixed["n_credit_lo_mn"]
                                                        - fixed["n_credit_mn"]) / fixed["base_loss"]
        fixed["ratio_hi"] = fixed["ratio_pct"] + 0.1 * (fixed["n_credit_hi_mn"]
                                                        - fixed["n_credit_mn"]) / fixed["base_loss"]
    archs = [a for a in ARCH_COLORS if a in set(adj["arch"])]
    hybrids = [a for a in archs if a != baseline]
    x = "base_flops"

    fig = plt.figure(figsize=(13.0, 9.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.42, wspace=0.07,
                          left=0.065, right=0.83, top=0.9, bottom=0.08)
    ax_raw = fig.add_subplot(gs[0, 0])
    ax_adj = fig.add_subplot(gs[0, 1], sharey=ax_raw)
    ax_sum = fig.add_subplot(gs[1, :])

    ends = []
    for ax, df, title in [(ax_raw, raw, "raw — as trained (hybrids are 7-24% larger in N_ne)"),
                          (ax_adj, fixed, "param-adjusted — every arch moved to full attention's N")]:
        ax.axhline(0.0, color=ARCH_COLORS[baseline], lw=2.0, zorder=2)
        for arch in hybrids:
            sub = df[df["arch"] == arch]
            col = ARCH_COLORS[arch]
            for (size, _tier), g in sub.groupby(["size", "tier"]):
                g = g.sort_values(x)
                if len(g) >= 2:
                    ax.plot(g[x], g["ratio_pct"], color=col, lw=1.8, dashes=ARCH_DASHES[arch],
                            alpha=0.75, zorder=3)
                if ax is ax_adj and "ratio_lo" in g:
                    ax.errorbar(g[x], g["ratio_pct"],
                                yerr=[g["ratio_pct"] - g["ratio_lo"], g["ratio_hi"] - g["ratio_pct"]],
                                fmt="none", ecolor=col, elinewidth=1.0, alpha=0.45, capsize=0,
                                zorder=3)
                ax.scatter(g[x], g["ratio_pct"], color=col, marker=SIZE_MARKER.get(size, "o"),
                           s=40, zorder=4, linewidths=1.0, edgecolors=SURFACE)
            if ax is ax_adj and len(sub):
                last = sub.loc[sub[x].idxmax()]
                ends.append((last[x], last["ratio_pct"], ARCH_LABEL.get(arch, arch), col))
        ax.set_xscale("log")
        ax.set_xlabel("training FLOPs of the cell (6ND, full-attention N)")
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        style_axes(ax)
    ax_raw.set_ylabel("loss ratio vs full attention  (%)")
    plt.setp(ax_adj.get_yticklabels(), visible=False)
    # Room inside the axes for the direct labels, same on both panels so their
    # x ranges stay identical.
    for ax in (ax_raw, ax_adj):
        ax.margins(x=0.2)
    for ax in (ax_raw, ax_adj):
        ax.text(0.01, 0.02, "below 0 = better than full attention", transform=ax.transAxes,
                fontsize=8, color=INK_2)
    legends(ax_adj, archs, adj["size"])
    label_ends(ax_adj, ends)
    label_ends(ax_raw, [(r[x], r["ratio_pct"], ARCH_LABEL.get(a, a), ARCH_COLORS[a])
                        for a in hybrids
                        for r in [raw[raw["arch"] == a].loc[lambda d: d[x].idxmax()]]])

    # ---- bottom: per-(arch, size) means, raw -> adjusted ----
    sizes = [s for s in SIZE_MARKER if s in set(adj["size"])]
    n = len(hybrids)
    width = 0.8 / max(n, 1)
    ax_sum.axhline(0.0, color=ARCH_COLORS[baseline], lw=2.0, zorder=1)
    for j, arch in enumerate(hybrids):
        col = ARCH_COLORS[arch]
        for i, size in enumerate(sizes):
            r = raw[(raw["arch"] == arch) & (raw["size"] == size)]["ratio_pct"]
            f = fixed[(fixed["arch"] == arch) & (fixed["size"] == size)]["ratio_pct"]
            if f.empty:
                continue
            xp = i - 0.4 + width * (j + 0.5)
            se = f.std(ddof=1) / np.sqrt(len(f)) if len(f) > 1 else 0.0
            ax_sum.plot([xp, xp], [r.mean(), f.mean()], color=col, lw=1.0, alpha=0.5, zorder=2)
            ax_sum.scatter([xp], [r.mean()], facecolors=SURFACE, edgecolors=col,
                           marker=SIZE_MARKER[size], s=34, linewidths=1.2, zorder=3)
            ax_sum.errorbar([xp], [f.mean()], yerr=[se], fmt=SIZE_MARKER[size], color=col,
                            markersize=7, markeredgecolor=SURFACE, elinewidth=1.4, capsize=0,
                            zorder=4)
    for i in range(1, len(sizes)):
        ax_sum.axvline(i - 0.5, color="#e6e5e1", lw=1.0, zorder=0)
    ax_sum.set_xticks(range(len(sizes)))
    ax_sum.set_xticklabels([f"{s}  (N_ne {adj[adj['size'] == s]['N_ref'].iloc[0] / 1e6:.0f}M)"
                            for s in sizes])
    ax_sum.set_xlim(-0.5, len(sizes) - 0.5)
    ax_sum.set_ylabel("mean loss ratio vs full attention  (%)")
    ax_sum.set_title("who wins at each size — filled = param-adjusted mean ± SE over the "
                     "size's cells, hollow = raw mean", fontsize=10, color=INK, loc="left")
    style_axes(ax_sum)
    ax_sum.grid(False, axis="x")
    # Colour is not the only identity cue here: the legend repeats the fixed
    # left-to-right order, and each size group reads in that same order.
    from matplotlib.lines import Line2D
    from fit import _legend
    _legend(ax_sum, [Line2D([0], [0], color=ARCH_COLORS[a], marker="o", linestyle="",
                            markersize=6, label=f"{j + 1}. {ARCH_LABEL.get(a, a)}")
                     for j, a in enumerate(hybrids)]
            + [Line2D([0], [0], color=INK_2, marker="o", linestyle="", markersize=6,
                      label="param-adjusted"),
               Line2D([0], [0], color=INK_2, marker="o", linestyle="", markersize=6,
                      markerfacecolor=SURFACE, label="raw")],
            "left → right in each size", (1.01, 1.0))
    ax_sum.text(0.005, 0.03, "below 0 = better than full attention", transform=ax_sum.transAxes,
                fontsize=8, color=INK_2)

    fig.suptitle("Architecture vs full attention, with and without the parameter difference",
                 fontsize=12.5, color=INK, x=0.065, ha="left", y=0.975)
    fig.text(0.065, 0.935,
             "adjustment: L + A·(N_fa^-α − N^-α) from full attention's own L(N,D) fit; error bars = "
             "90% bootstrap band of that fit. Assumes a mixer parameter is worth a dense one.",
             fontsize=8.5, color=INK_2)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

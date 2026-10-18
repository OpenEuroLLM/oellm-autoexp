"""Fit and plot per-architecture scaling curves.

Primary model: the compute power law
    L(C) = E + A * C**(-alpha)
where C is training FLOPs or GPU-hours, E the irreducible loss. Fit per
architecture with scipy; also expose a Chinchilla L(N,D) = E + A/N**a + B/D**b fit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

ARCH_ORDER = ["fullattn", "swa7_rope100k", "gdn7_nope", "gdn3_nope",
              "gdn3_qwen35", "mlstm7_nope", "mamba2_nope"]

# Categorical palette, assigned in fixed order and never cycled. Validated
# all-pairs on a light surface (the dataviz validator, 7 slots): worst CVD
# ΔE 6.7 (protan, #e34948 vs #8a5a00) and worst normal-vision ΔE 16.3
# (#4a3aa7 vs #2a78d6). A CVD ΔE in the 6-8 band is only legal WITH secondary
# encoding, so architecture identity is carried three ways — colour, dash
# pattern, and a direct label at the end of each curve — and never by colour
# alone. Two slots sit below 3:1 contrast on the surface, which the direct
# labels plus the emitted CSV tables discharge.
#
# The gdn3_qwen35 slot (#a3327c) was picked by scanning candidate hues against
# the all-pairs check rather than by eye: at this lightness band the 6-hue space
# was effectively saturated, and every nearer candidate (magenta #c2419a, teal
# #0e8ea8, violet #9c6ff0, olive #7a8c1f...) pushed some pair BELOW the
# normal-vision floor of 15. #a3327c leaves both worst-case pairs exactly where
# the 6-colour palette already had them, i.e. it costs nothing. If an 8th arch
# is ever added, do NOT invent another hue — fold it into small multiples.
ARCH_COLORS = {
    "fullattn":      "#2a78d6",   # blue    — the baseline everything is read against
    "swa7_rope100k": "#eda100",   # yellow
    "gdn7_nope":     "#1baf7a",   # aqua
    "gdn3_nope":     "#8a5a00",   # brown -- distinct from the aqua 7:1 GDN
    "gdn3_qwen35":   "#a3327c",   # magenta -- the Qwen3.5 block at the same 3:1
    "mlstm7_nope":   "#4a3aa7",   # violet
    "mamba2_nope":   "#e34948",   # red
}
ARCH_DASHES = {
    "fullattn":      (1, 0),          # solid — baseline
    "swa7_rope100k": (5, 2),
    "gdn7_nope":     (1.5, 1.5),
    "gdn3_nope":     (4, 1.5, 1.5, 1.5),
    # Same dash FAMILY as gdn3_nope (long dash + short ticks) with an extra tick:
    # the two 3:1 arms read as a pair at a glance, which is the comparison the
    # qwen35 arm exists to support.
    "gdn3_qwen35":   (4, 1.5, 1.5, 1.5, 1.5, 1.5),
    "mlstm7_nope":   (7, 2, 1.5, 2),
    "mamba2_nope":   (3, 1.5, 1.5, 1.5),
}
ARCH_LABEL = {
    "fullattn": "full attn", "swa7_rope100k": "SWA 7:1", "gdn7_nope": "GDN 7:1", "gdn3_nope": "GDN 3:1",
    "gdn3_qwen35": "Qwen3.5 3:1",
    "mlstm7_nope": "mLSTM 7:1", "mamba2_nope": "Mamba2 7:1",
}
SIZE_MARKER = {"0.1B": "o", "0.2B": "s", "0.4B": "^", "0.9B": "D"}

INK, INK_2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def style_axes(ax):
    """Recessive grid and axes so the data marks carry the chart."""
    ax.grid(True, which="major", color="#e6e5e1", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c3")
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def _legend(ax, handles, title, anchor):
    leg = ax.legend(handles=handles, title=title, loc="upper left", bbox_to_anchor=anchor,
                    fontsize=8.5, title_fontsize=8.5, framealpha=0, edgecolor="none")
    leg.get_title().set_color(INK_2)
    leg.get_title().set_ha("left")
    for t in leg.get_texts():
        t.set_color(INK)
    ax.add_artist(leg)
    return leg


def legends(ax, archs, sizes=(), suffix=None):
    """Architecture (colour+dash) and model-size (marker) legends, outside right.

    Both are attached with `add_artist`: a second `ax.legend(...)` call REPLACES
    the first, which is what used to make the architecture colour key vanish.
    Placing them outside the axes keeps them off the data at every zoom level.
    """
    from matplotlib.lines import Line2D
    suffix = suffix or {}
    _legend(ax, [Line2D([0], [0], color=ARCH_COLORS[a], lw=2.2, dashes=ARCH_DASHES[a],
                        marker="o", markersize=5, markeredgecolor=SURFACE,
                        label=ARCH_LABEL.get(a, a) + suffix.get(a, "")) for a in archs],
            "architecture", (1.01, 1.0))
    present = [s for s in SIZE_MARKER if s in set(sizes)]
    if present:
        _legend(ax, [Line2D([0], [0], marker=SIZE_MARKER[s], color=INK_2, linestyle="",
                            markersize=6, label=s) for s in present],
                "model size", (1.01, 1.0 - 0.085 * (len(archs) + 1.6)))


def label_ends(ax, items, min_gap_px=12.0):
    """Direct-label each series at its right edge, nudged apart so they stay legible.

    Identity is carried by colour, dash pattern AND this label — the palette's
    worst CVD pair sits in the 6-8 band, which is only legal with such secondary
    encoding.
    """
    if not items:
        return
    ax.figure.canvas.draw()
    pos = [(ax.transData.transform((x, y)), t, c) for x, y, t, c in items]
    order = sorted(range(len(pos)), key=lambda i: pos[i][0][1])
    ys = [pos[i][0][1] for i in order]
    for k in range(1, len(ys)):
        ys[k] = max(ys[k], ys[k - 1] + min_gap_px)
    inv = ax.transData.inverted()
    for k, i in enumerate(order):
        (px, _), text, col = pos[i]
        x, y = inv.transform((px, ys[k]))
        ax.annotate(text, (x, y), xytext=(7, 0), textcoords="offset points", color=col,
                    fontsize=8.5, va="center", fontweight="medium", annotation_clip=False)


def _power_law(c, E, A, alpha):
    return E + A * np.power(c, -alpha)


def fit_power_law(c: np.ndarray, loss: np.ndarray) -> dict:
    """Fit L = E + A*C^-alpha. Returns params + R^2. Robust to few points."""
    c = np.asarray(c, float)
    loss = np.asarray(loss, float)
    nan = {"E": np.nan, "A": np.nan, "alpha": np.nan, "se_alpha": np.nan,
           "r2": np.nan, "n": len(c)}
    if len(c) < 3:
        return nan
    E0 = max(loss.min() - 0.1, 0.1)
    # A0 from two extreme points on a log scale
    lo, hi = np.argmin(c), np.argmax(c)
    alpha0 = 0.05
    A0 = max((loss[lo] - E0), 1e-6) * c[lo] ** alpha0
    try:
        popt, pcov = curve_fit(
            _power_law, c, loss, p0=[E0, A0, alpha0],
            bounds=([0.0, 0.0, 1e-4], [loss.min(), np.inf, 2.0]), maxfev=200000,
        )
    except Exception:
        return nan
    resid = loss - _power_law(c, *popt)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((loss - loss.mean()) ** 2)) or np.nan
    # E and alpha are strongly correlated over a short lever arm; the covariance
    # is what tells you the exponent is unidentified, so keep its standard error.
    se = float(np.sqrt(pcov[2, 2])) if np.all(np.isfinite(pcov)) else np.inf
    return {"E": popt[0], "A": popt[1], "alpha": popt[2], "se_alpha": se,
            "r2": 1 - ss_res / ss_tot if ss_tot else np.nan, "n": len(c)}


def envelope(sub: pd.DataFrame, xcol: str) -> pd.DataFrame:
    """Lower-left Pareto frontier: the compute-optimal points at each cost.

    L(C) is a statement about the BEST loss reachable for compute C, so it must
    be fitted to the frontier. Fitting the raw cloud instead regresses through
    every (N, D) cell the sweep happens to contain, and the result is dominated
    by which cells exist rather than by the architecture — that is what pinned E
    to its bound and made whichever arch owned the largest single run look
    steepest.
    """
    keep, best = [], np.inf
    for i, r in sub.sort_values(xcol).iterrows():
        if r["val_loss"] < best - 1e-12:
            keep.append(i)
            best = r["val_loss"]
    return sub.loc[keep]


def common_range(df: pd.DataFrame, xcol: str) -> tuple[float, float]:
    """Cost range every architecture actually covers — fits outside it compare
    an arch against nothing and are pure extrapolation."""
    lo = df.groupby("arch")[xcol].min().max()
    hi = df.groupby("arch")[xcol].max().min()
    return float(lo), float(hi)


def fit_all(df: pd.DataFrame, xcol: str, use_envelope: bool = True,
            verbose: bool = True) -> pd.DataFrame:
    """Per-arch power-law fit against `xcol` (flops or gpu_hours) on the frontier."""
    lo, hi = common_range(df, xcol)
    if verbose:
        cov = df.groupby("arch")[xcol].agg(["min", "max", "count"])
        span = (cov["max"] / cov["min"]).round(1)
        if (cov["max"] / hi > 1.25).any() or (cov["min"] / lo < 0.8).any():
            print(f"[fit] WARNING: unequal {xcol} coverage — archs are not compared "
                  f"over the same range. Common range {lo:.3g}-{hi:.3g}; per-arch:")
            print(cov.assign(span=span).to_string())
    out = []
    for arch in [a for a in ARCH_ORDER if a in set(df["arch"])]:
        sub = df[df["arch"] == arch]
        pts = envelope(sub, xcol) if use_envelope else sub
        fit = fit_power_law(pts[xcol].values, pts["val_loss"].values)
        # E and alpha trade off badly over a short lever arm: with few frontier
        # points the fit can run away to a large alpha with E absorbing it and
        # still score a high R^2. The tell is the exponent's own standard error,
        # not the residual — so flag on se/alpha rather than on goodness of fit.
        span = pts[xcol].max() / pts[xcol].min() if len(pts) else np.nan
        rel_se = fit["se_alpha"] / fit["alpha"] if fit["alpha"] else np.inf
        fit["low_conf"] = bool(fit["n"] < 5 or not np.isfinite(span) or span < 4
                               or not np.isfinite(rel_se) or rel_se > 0.5
                               or not (0.02 < fit["alpha"] < 0.5))
        out.append({"arch": arch, "axis": xcol, "x_lo": lo, "x_hi": hi,
                    "n_all": len(sub), "span": span, **fit})
    df_out = pd.DataFrame(out)
    if verbose and df_out["low_conf"].any():
        bad = ", ".join(df_out.loc[df_out["low_conf"], "arch"])
        print(f"[fit] {xcol}: alpha NOT identified (se/alpha > 0.5 or out of range) for "
              f"{bad}. The fitted curve still interpolates its range; only the exponent "
              f"— and therefore any extrapolation — is untrustworthy. Compare "
              f"architectures with compare.iso_loss_multiplier, not with these alphas.")
    return df_out


def _chinchilla(X, E, A, a, B, b):
    N, D = X
    return E + A * np.power(N, -a) + B * np.power(D, -b)


def fit_chinchilla(df: pd.DataFrame) -> dict:
    """Fit L(N,D)=E+A/N^a+B/D^b across all points (all archs pooled by default)."""
    N = df["N"].values.astype(float)
    D = df["D"].values.astype(float)
    L = df["val_loss"].values.astype(float)
    if len(L) < 6:
        return {}
    try:
        popt, _ = curve_fit(
            _chinchilla, (N, D), L,
            p0=[1.0, 1e3, 0.3, 1e3, 0.3],
            bounds=([0, 0, 1e-3, 0, 1e-3], [L.min(), np.inf, 1.5, np.inf, 1.5]),
            maxfev=400000,
        )
    except Exception:
        return {}
    return dict(zip(["E", "A", "a", "B", "b"], popt))


def fit_hypotheses(df: pd.DataFrame, baseline: str = "fullattn") -> pd.DataFrame:
    """How does the architecture advantage enter L(N, D)? Nested model comparison.

    All four share the Chinchilla backbone L = E + A*N^-a + B*D^-b and differ only
    in how the architecture term enters — which is exactly the question, because
    they extrapolate differently:

      H0  none              no architecture effect (the null)
      H1  loss offset       L_arch = L_base - delta        (constant Δ forever)
      H2  effective size    N_eff  = kappa * N             (arch buys parameters)
      H2d effective data    D_eff  = kappa * D             (arch buys tokens)

    H1 and H2/H2d agree over the fitted range and diverge outside it: a constant
    offset keeps its full value at any scale, an effective-size/data factor keeps
    a constant *compute* saving instead. Reported by RMSE and AIC; with the lever
    arm this sweep has, expect them to be close — that closeness is the finding,
    not a defect, and it is why `compare.trend_test` on matched pairs is the
    primary evidence and this table the secondary.
    """
    from scipy.optimize import least_squares

    archs = [a for a in ARCH_ORDER if a in set(df["arch"])]
    if baseline not in archs or len(df) < 12:
        return pd.DataFrame()
    ai = {a: i for i, a in enumerate(archs)}
    k = len(archs)
    idx = df["arch"].map(ai).values
    N = df["N"].values.astype(float)
    D = df["D"].values.astype(float)
    L = df["val_loss"].values.astype(float)
    b0 = ai[baseline]

    def resid(kind):
        def f(p):
            E, lA, a, lB, b = p[:5]
            t = p[5:5 + k] - p[5 + b0]          # arch term, pinned to 0 at baseline
            nn = N * np.exp(t[idx]) if kind == "N" else N
            dd = D * np.exp(t[idx]) if kind == "D" else D
            add = t[idx] if kind == "add" else 0.0
            return E + np.exp(lA) * nn ** -a + np.exp(lB) * dd ** -b + add - L
        return f

    p0 = [2.0, np.log(1e3), 0.3, np.log(1e3), 0.3] + [0.0] * k
    lo = [0.0, -5, 0.05, -5, 0.05] + [-2.0] * k
    hi = [float(L.min()), 25, 1.0, 25, 1.0] + [2.0] * k
    rows = []
    for kind, name in [(None, "H0 none"), ("add", "H1 loss offset"),
                       ("N", "H2 effective size"), ("D", "H2d effective data")]:
        npar = 5 if kind is None else 5 + k - 1
        f = (lambda p: (lambda E, lA, a, lB, b: E + np.exp(lA) * N ** -a
                        + np.exp(lB) * D ** -b - L)(*p[:5])) if kind is None else resid(kind)
        r = least_squares(f, p0, bounds=(lo, hi), max_nfev=200000)
        mse = float(np.mean(r.fun ** 2))
        row = {"model": name, "k": npar, "rmse_mn": 1000 * np.sqrt(mse),
               "aic": len(L) * np.log(mse) + 2 * npar,
               "a": r.x[2], "b": r.x[4], "E": r.x[0]}
        if kind is not None:
            t = r.x[5:5 + k] - r.x[5 + b0]
            for a in archs:
                row[a] = 1000 * t[ai[a]] if kind == "add" else float(np.exp(t[ai[a]]))
        rows.append(row)
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, fits: pd.DataFrame, xcol: str, xlabel: str, out_png: str,
         title: str = "Multilingual scaling — cooldown checkpoints"):
    """Log-x scatter + fitted curve per arch; colour+dash = architecture, marker = size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    archs = [a for a in ARCH_ORDER if a in set(df["arch"])]
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    suffix = {}
    for arch in archs:
        sub = df[df["arch"] == arch]
        col = ARCH_COLORS[arch]
        front = set(envelope(sub, xcol).index)
        for size, g in sub.groupby("size"):
            on = g[g.index.isin(front)]
            off = g[~g.index.isin(front)]
            # off-frontier points are context, not evidence for L(C): recede them
            ax.scatter(off[xcol], off["val_loss"], color=col, marker=SIZE_MARKER.get(size, "o"),
                       s=30, zorder=3, alpha=0.28, linewidths=0)
            ax.scatter(on[xcol], on["val_loss"], color=col, marker=SIZE_MARKER.get(size, "o"),
                       s=52, zorder=4, linewidths=1.1, edgecolors=SURFACE)
        f = fits[(fits.arch == arch) & (fits.axis == xcol)]
        if len(f) and np.isfinite(f.iloc[0]["alpha"]):
            r = f.iloc[0]
            # only over the range every arch covers — beyond it the curves are
            # extrapolating into cells the other architectures never ran
            xs = np.logspace(np.log10(r["x_lo"]), np.log10(r["x_hi"]), 100)
            ys = _power_law(xs, r["E"], r["A"], r["alpha"])
            ax.plot(xs, ys, color=col, lw=2.0, dashes=ARCH_DASHES[arch], alpha=0.9, zorder=5)
            # alpha lives in the legend, not on the curve: the frontiers converge
            # at the right edge, so end labels would land on top of each other
            # and on the data.
            suffix[arch] = ("  α unidentified" if r["low_conf"]
                            else f"  α={r['alpha']:.3f}±{r['se_alpha']:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("validation loss")
    ax.set_title(title, fontsize=11, color=INK)
    ax.text(0.005, -0.135, "solid = compute-optimal frontier (fitted) · faded = dominated points"
            " · curves interpolate; α is not identified well enough to extrapolate",
            transform=ax.transAxes, fontsize=8, color=INK_2)
    ax.margins(x=0.16)
    style_axes(ax)
    fig.subplots_adjust(left=0.085, right=0.68, top=0.91, bottom=0.14)
    legends(ax, archs, df["size"], suffix=suffix)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

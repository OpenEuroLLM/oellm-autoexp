#!/usr/bin/env python3
r"""The three scaling regimes on one figure: strong, weak-batch, weak-model.

WHY A GRID AND NOT THREE SEPARATE PLOTS
---------------------------------------
"Does it scale?" has no answer until you say what is held fixed, and the three
answers disagree. At 512 nodes the same 32B model reads 417.9 TFLOP/s under
strong scaling, 329.1 under weak-batch and 359.8 under weak-model — a 27% spread
that is entirely an artefact of the question, not the machine. Putting the three
regimes side by side on a SHARED y-axis is what stops a reader picking the
flattering one by accident.

  column 1  STRONG       model fixed, GBS fixed, nodes rise.
                         Work per GPU FALLS, so a declining curve is correct and
                         expected. This is the honest worst case.
  column 2  WEAK-BATCH   model fixed, GBS proportional to nodes.
                         Work per GPU is CONSTANT, so ideal is a FLAT line and
                         the slope is the pure cost of a wider world.
  column 3  WEAK-MODEL   model size AND batch grow with nodes, parallelism tuned
                         per size. This is the regime a compute-budget estimate
                         actually lives in.

ENCODING
--------
Colour is the MODEL FAMILY and nothing else, held identical across columns 1 and
2, so a reader who learns "orange is 7B" keeps it. Column 3 has no single model
- every point is a different one - so it gets its own hue and labels each point
with the size instead of pretending to be a fifth family.

The palette is the validated categorical order (slots 1-4); aqua and yellow sit
below 3:1 on a light surface, so the relief rule applies and every series is
directly labelled at its endpoint AND written to the companion CSV.

LIKE-FOR-LIKE, OR THE CURVE IS FICTION
--------------------------------------
Two families are deliberately truncated rather than drawn to their last measured
arm, because those arms changed parallelism to run at all:

  * 1.7B stops at 256 nodes (strong) and 128 nodes (weak-batch). The 512- and
    256-node arms run TP=2 where the rest of the family runs TP=1; the drop at
    those points is substantially the parallelism change, not a scaling result.
    See `fp8_small_1b7_fix.yaml` and the asterisk section of RESULTS.md.
  * 7B starts at 32 nodes: it does not fit at TP=1 with fp32 grad accumulation,
    so the whole family runs TP=2 + sequence-parallel. That IS internally
    consistent, it just has no 8- or 16-node arm.

USAGE
-----
    python scripts/korbi/scaling_curves.py --from-csv dump/all_runs_speed_20260830.csv \
        --peak-tflops 989.4 --out dump/final_20260830/fig_scaling_regimes.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from speed_scaling_plots import classify, load_from_csv, table  # noqa: E402

# Validated categorical slots 1-4 + slot 7. Do not re-order: the ordering is the
# CVD-safety mechanism, checked with the palette validator, not a taste call.
#   node scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a,#eda100" --mode light
#   -> PASS lightness / chroma / CVD (worst adjacent dE 9.1) / normal-vision (22.9)
#      WARN contrast for aqua+yellow -> relief = direct labels + table view
MODEL_COLOR = {
    "32B": "#2a78d6",  # slot 1 blue    - the flagship, most important series
    "7B": "#eb6834",  # slot 2 orange
    "1.7B": "#1baf7a",  # slot 3 aqua
    "0.4B": "#eda100",  # slot 4 yellow
}
LADDER_COLOR = "#4a3aa7"  # slot 7 violet - a different KIND of series, not a 5th family
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d8d7d2"

# Order matters: it is the legend order and the z-order, largest model first.
STRONG = [
    ("32B", r"speedscale_fp8strong-n(\d+)_"),
    ("7B", r"speedscale_fp8small-strong-7b-n(\d+)_"),
    ("1.7B", r"speedscale_fp8small-strong-1b7-n(8|16|32|64|128|256)_"),
    ("0.4B", r"speedscale_fp8small-strong-400m-n(\d+)_"),
]
WEAKBATCH = [
    ("32B", r"speedscale_fp8weakbatch-n(\d+)_"),
    ("7B", r"speedscale_fp8small-weakbs-7b-n(\d+)_"),
    ("1.7B", r"speedscale_fp8small-weakbs-1b7-n(8|16|32|64|128)_"),
    ("0.4B", r"speedscale_fp8small-weakbs-400m-n(\d+)_"),
]
WEAKMODEL = [("1.9B to 32B", r"speedscale_fp8weakmodel-.*-n(\d+)")]

# Model running at each node count of the weak-model ladder, for the point labels.
LADDER_MODEL = {64: "1.9B", 128: "3.5B", 256: "7B", 512: "17B", 1024: "32B"}

COLUMNS = [
    ("Strong scaling", "model and global batch fixed\nwork per GPU falls", STRONG),
    ("Weak scaling — batch axis", "GBS proportional to nodes\nwork per GPU constant", WEAKBATCH),
    (
        "Weak scaling — model axis",
        "model grows with the machine\nparallelism tuned per size",
        WEAKMODEL,
    ),
]

RC = {
    "font.size": 10.5,
    "axes.titlesize": 11.5,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 10,
    "axes.linewidth": 0.8,
    "lines.linewidth": 2.0,
    "lines.markersize": 6.5,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",  # solid hairline; dashed grid reads as a threshold
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 170,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "legend.frameon": False,
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
}


def collect(csv_path: Path, specs, filters):
    """-> [(label, rows, kind)] for one column, skipping series with no data."""
    out = []
    for label, pattern in specs:
        runs = load_from_csv(csv_path, label, pattern, filters)
        if not runs:
            print(f"  ! no runs matched {pattern!r}", file=sys.stderr)
            continue
        kind = classify(runs)
        out.append((label, table(runs, kind), kind))
    return out


def _node_axis(ax, xs):
    """Actual node counts on the ticks — a log2 axis otherwise prints 2^3."""
    xs = sorted(set(xs))
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(v) for v in xs])
    ax.minorticks_off()
    ax.tick_params(length=0)


def draw(all_cols, out: Path, peak: float | None, csv_out: Path | None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context(RC):
        fig, axes = plt.subplots(
            2, 3, figsize=(14.4, 7.6), sharey="row", gridspec_kw=dict(hspace=0.34, wspace=0.10)
        )

        for col, ((title, subtitle, _), series) in enumerate(zip(COLUMNS, all_cols)):
            top, bot = axes[0][col], axes[1][col]
            xs_all = []
            col_max_x = max((r["nodes"] for _, rows, _ in series for r in rows), default=0)
            # Endpoint labels are placed against the OTHER series, not blindly:
            # a series that stops short has neighbours running through the space
            # beside it, and "above" is only free if it is not the lowest there.
            at_top = {lab: {r["nodes"]: r["median"] for r in rs} for lab, rs, _ in series}
            at_bot = {
                lab: {r["nodes"]: 100.0 * r["efficiency"] for r in rs} for lab, rs, _ in series
            }

            def place(label, xe, ye, book):
                if xe >= col_max_x:
                    return (7, -3), "left", "center"  # panel edge: room to the right
                others = [v[xe] for lab, v in book.items() if lab != label and xe in v]
                below_all = all(ye <= o for o in others) if others else False
                return ((0, -13), "center", "top") if below_all else ((0, 12), "center", "bottom")

            for label, rows, kind in series:
                color = MODEL_COLOR.get(label, LADDER_COLOR)
                x = [r["nodes"] for r in rows]
                y = [r["median"] for r in rows]
                xs_all += x
                lo = [r["median"] - r["p10"] for r in rows]
                hi = [r["p90"] - r["median"] for r in rows]
                # p10-p90 across iterations. It is NOT an uncertainty interval on
                # the median, and it is invisible for most arms on purpose: the
                # spread is a median of 0.67% of the rate, i.e. thinner than the
                # marker. Where a whisker IS visible it means one or two
                # iterations stalled - 1.7B@128 dips to 39.9 TFLOP/s on a 480
                # median - so read a long whisker as "this arm hiccuped", not as
                # "this measurement is uncertain". The plateau confirms the bulk
                # rate in every such case (drift < 0.7%).
                top.errorbar(
                    x,
                    y,
                    yerr=[lo, hi],
                    fmt="none",
                    ecolor=color,
                    elinewidth=0.9,
                    capsize=2.5,
                    capthick=0.9,
                    alpha=0.55,
                    zorder=2,
                )
                # 2px surface ring on the markers so overlapping series stay legible
                top.plot(
                    x,
                    y,
                    marker="o",
                    color=color,
                    label=label,
                    zorder=3,
                    markeredgecolor="#fcfcfb",
                    markeredgewidth=1.4,
                )
                # Relief for the low-contrast slots + selective direct labelling:
                # the ENDPOINT only, never a number on every point.
                # A series that stops short of the panel edge has other lines
                # running through the space to its right, so its label goes ABOVE
                # the point instead of into them.
                off, ha, va = place(label, x[-1], y[-1], at_top)
                top.annotate(
                    f"{y[-1]:.0f}",
                    (x[-1], y[-1]),
                    textcoords="offset points",
                    xytext=off,
                    ha=ha,
                    va=va,
                    fontsize=9,
                    color=color,
                    fontweight="bold",
                    zorder=4,
                )
                bot.plot(
                    x,
                    [100.0 * r["efficiency"] for r in rows],
                    marker="o",
                    color=color,
                    zorder=3,
                    markeredgecolor="#fcfcfb",
                    markeredgewidth=1.4,
                    label=label,
                )
                ye = 100.0 * rows[-1]["efficiency"]
                boff, bha, bva = place(label, x[-1], ye, at_bot)
                bot.annotate(
                    f"{ye:.0f}%",
                    (x[-1], ye),
                    textcoords="offset points",
                    xytext=boff,
                    ha=bha,
                    va=bva,
                    fontsize=9,
                    color=color,
                    fontweight="bold",
                    zorder=4,
                )
                # Ideal for a WEAK regime is a flat line at the reference rate;
                # for STRONG the per-GPU rate is expected to fall, so drawing a
                # flat reference there would imply a target that does not exist.
                if kind == "weak":
                    top.axhline(y[0], ls=(0, (4, 3)), lw=1.0, color=color, alpha=0.45, zorder=1)
                # Column 3 has a different model at every point; that identity is
                # the whole content of the series, so it is labelled per point.
                if label.startswith("1.9B"):
                    for xi, yi in zip(x, y):
                        top.annotate(
                            LADDER_MODEL.get(xi, ""),
                            (xi, yi),
                            textcoords="offset points",
                            xytext=(0, 11),
                            ha="center",
                            fontsize=8.5,
                            color=INK_MUTED,
                        )

            # Title and subtitle are placed by hand: set_title() gives one type
            # size, and stacking a second text on top of it collides.
            top.text(
                0.5,
                1.20,
                title,
                transform=top.transAxes,
                ha="center",
                va="bottom",
                fontsize=11.5,
                fontweight="bold",
                color=INK,
            )
            top.text(
                0.5,
                1.02,
                subtitle,
                transform=top.transAxes,
                ha="center",
                va="bottom",
                fontsize=9,
                color=INK_MUTED,
                linespacing=1.4,
            )
            bot.axhline(100.0, ls=(0, (4, 3)), lw=1.0, color=INK_MUTED, alpha=0.5, zorder=1)
            for ax in (top, bot):
                _node_axis(ax, xs_all)
            # Only the bottom row carries the x label; six identical labels is chrome.
            bot.set_xlabel("nodes (4 GH200 each)", color=INK_MUTED)
            top.set_ylim(0, 620)
            bot.set_ylim(0, 118)

        axes[0][0].set_ylabel("TFLOP/s per GPU")
        axes[1][0].set_ylabel("parallel efficiency (%)")
        if peak:
            # Same measure, exact linear re-expression (v / peak) — a unit axis,
            # not a second scale carrying a second variable.
            sec = axes[0][2].secondary_yaxis(
                "right", functions=(lambda v: 100.0 * v / peak, lambda m: m * peak / 100.0)
            )
            sec.set_ylabel(f"MFU (% of {peak:g} TFLOP/s peak)", color=INK_MUTED)
            sec.tick_params(colors=INK_MUTED, length=0)

        handles = [
            plt.Line2D(
                [],
                [],
                color=c,
                lw=2.4,
                marker="o",
                ms=6.5,
                markeredgecolor="#fcfcfb",
                markeredgewidth=1.4,
                label=m,
            )
            for m, c in MODEL_COLOR.items()
        ]
        handles.append(
            plt.Line2D(
                [],
                [],
                color=LADDER_COLOR,
                lw=2.4,
                marker="o",
                ms=6.5,
                markeredgecolor="#fcfcfb",
                markeredgewidth=1.4,
                label="weak-model ladder (1.9B→32B)",
            )
        )
        handles.append(
            plt.Line2D(
                [],
                [],
                color=INK_MUTED,
                lw=1.0,
                ls=(0, (4, 3)),
                label="ideal (flat / no scaling loss)",
            )
        )
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=6,
            bbox_to_anchor=(0.5, -0.055),
            columnspacing=1.6,
        )
        # Efficiency is not one quantity: the two regimes define it differently,
        # and a reader comparing 57% against 75% without knowing that is being
        # misled by the axis label.
        fig.text(
            0.5,
            -0.105,
            "Parallel efficiency is defined per regime: STRONG = speedup / (N/N$_{ref}$)  ·  "
            "WEAK = per-GPU rate / reference per-GPU rate.\n"
            "Whiskers are p10–p90 ACROSS ITERATIONS (median 0.67% of the rate — thinner "
            "than the marker for 44 of 58 arms). Run-to-run spread of an identical config "
            "is 0.81% CV (21 repeated configs).\n"
            "Both are an order of magnitude below every difference this figure shows. A "
            "long whisker means one iteration stalled, not an uncertain measurement — the "
            "last-quartile plateau agrees with the median to <0.7% throughout.\n"
            "Strong-scaling GBS is 4096 for 32B and 16384 for the small families. "
            "1.7B stops where it would switch to TP=2; 7B has no arm below 32 nodes "
            "(does not fit at TP=1). FP8 hybrid + delayed throughout.\n"
            "The weak-model ladder exceeding 100% is NOT superlinear scaling: it is "
            "referenced to its 64-node arm, which runs TP=PP=1 at M=2 — a weak "
            "operating point that flatters everything measured against it.",
            ha="center",
            va="top",
            fontsize=8.5,
            color=INK_MUTED,
            linespacing=1.5,
        )
        fig.suptitle(
            "Megatron-LM FP8 on JUPITER GH200 — three scaling regimes, 8 to 1024 nodes",
            fontsize=13.5,
            y=1.045,
            color=INK,
        )

        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        fig.savefig(out.with_suffix(".pdf"))
        plt.close(fig)
    print(f"  plot written: {out}")
    print(f"  plot written: {out.with_suffix('.pdf')}")

    if csv_out:
        flat = []
        for (title, _, _), series in zip(COLUMNS, all_cols):
            for label, rows, kind in series:
                for r in rows:
                    flat.append(dict(regime=title, kind=kind, **r))
        with csv_out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(flat[0].keys()))
            w.writeheader()
            w.writerows(flat)
        print(f"  csv written : {csv_out}   ({len(flat)} rows — this is the table view)")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--from-csv", type=Path, default=Path("dump/all_runs_speed_20260830.csv"))
    ap.add_argument("--filter", action="append", default=[], metavar="COL=REGEX")
    ap.add_argument(
        "--peak-tflops",
        type=float,
        default=None,
        help="add an MFU axis using this per-GPU peak (GH200 BF16 dense "
        "is 989.4; the FP8 peak is 2x that, so say which you mean)",
    )
    ap.add_argument("--out", type=Path, default=Path("dump/final_20260830/fig_scaling_regimes.png"))
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    if not args.from_csv.is_file():
        print(f"error: --from-csv not a file: {args.from_csv}", file=sys.stderr)
        return 2
    filters = []
    for f in args.filter:
        if "=" not in f:
            print(f"error: --filter needs COL=REGEX, got {f!r}", file=sys.stderr)
            return 2
        col, pat = f.split("=", 1)
        filters.append((col, pat))

    all_cols = []
    for title, _, specs in COLUMNS:
        series = collect(args.from_csv, specs, filters)
        print(f"  {title}: {len(series)} series, {sum(len(r) for _, r, _ in series)} arms")
        all_cols.append(series)
    if not any(all_cols):
        print("nothing to plot", file=sys.stderr)
        return 1
    draw(all_cols, args.out, args.peak_tflops, args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())

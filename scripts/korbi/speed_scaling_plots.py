#!/usr/bin/env python3
r"""Plot strong/weak scaling curves from gather_speed.py's CSV.

This script no longer parses Megatron logs. `gather_speed.py` does that once,
across every run folder, and writes `dump/all_runs_speed.csv`; everything here
reads that. Re-parsing 200+ multi-MB logs took minutes per figure, which made
iterating on a plot painful; from the CSV it is well under a second.

USAGE
-----
    # default source is dump/all_runs_speed.csv
    python scripts/korbi/scaling_plots.py \
        --series '32B=speedscale_fp8strong-n(\d+)_' \
        --filter fp8=hybrid --paper --split --peak-tflops 989.4 \
        --out dump/fig_32b.png --csv dump/fig_32b.csv

    # several series, explicit source
    python scripts/korbi/scaling_plots.py --from-csv dump/all_runs_speed.csv \
        --series '0.4B=fp8small-strong-400m-n(\d+)_' \
        --series '1.7B=fp8small-strong-1b7-n(\d+)_' \
        --series '7B=fp8small-strong-7b-n(\d+)_' --out dump/fig_small.png

Each --series is `LABEL=REGEX`, matched against the CSV's `run` column. A capture
group supplies the node count; otherwise the `nodes` column is used.

--filter COL=REGEX (repeatable) narrows rows by any CSV column, e.g.
`--filter fp8=hybrid`, `--filter tp=4`, `--filter mock_data=False`. Matching is
re.fullmatch, so `tp=4` cannot also match 40.

FILTER DELIBERATELY, OR GET A MIXED CURVE
-----------------------------------------
When several rows share a geometry (nodes, GBS, TP, PP) the one with the most
iterations wins. That silently mixes precisions if both exist: without a filter
the 32B 1024-node point flips to the bf16 run (275.3) because it logged more
iterations than the FP8 one (322.7). The script prints a loud `!` warning
naming every geometry where this happened -- do not ignore it.

WHY MEDIANS, NOT MEANS
----------------------
Megatron throughput is noisy in ways a mean hides: manual GC costs ~+45% on one
iteration in `manual_gc_interval`, a sick node produces occasional 4x dips, and
startup iterations run far below steady state. Every number here is a MEDIAN
with a p10-p90 band. `drift` (median vs last-quartile plateau) flags arms that
never converged -- their medians are biased, not merely noisy.

STRONG vs WEAK IS DETECTED, NOT ASSUMED
---------------------------------------
    GBS constant across arms        -> strong: efficiency = speedup / (N/N_ref)
    GBS/nodes constant across arms  -> weak:   efficiency = perGPU / perGPU_ref
Anything else is reported as 'mixed' and only raw curves are drawn.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Run:
    label: str
    name: str
    log: Path
    nodes: int | None = None
    world: int | None = None
    dp: int | None = None
    tp: int | None = None
    pp: int | None = None
    gbs: int | None = None
    mbs: int | None = None
    seq: int | None = None
    # When the data comes from gather_speed.py's CSV we have the STATISTICS but
    # not the per-iteration series. Storing them here lets every downstream
    # property work unchanged, so plotting is identical from either source.
    pre: dict | None = None
    cfg_sig: str = ""

    # ---- derived -----------------------------------------------------------
    @property
    def ok(self) -> bool:
        return bool(self.pre) and self.nodes is not None

    @property
    def m(self) -> int | None:
        """Micro-batches per DP rank — the quantity that actually drives the
        curve."""
        if None in (self.gbs, self.mbs, self.dp) or not self.dp:
            return None
        return self.gbs // (self.mbs * self.dp)

    @property
    def per_gpu(self) -> tuple[float, float, float]:
        return (self.pre["p10"], self.pre["median"], self.pre["p90"])

    @property
    def plateau(self) -> float:
        """Median of the LAST QUARTER of iterations — the converged rate.

        --skip-first drops a fixed prefix, which assumes the warmup is short. It
        is not always: speedscale_strong-n256 (real data, 64 samples/rank/iter)
        bounced between 178 and 376 for ~40 of its 50 iterations before settling
        at 376, so its whole-run median of 303.6 sat 19% BELOW the true rate and
        reported a physically impossible efficiency above 1.0.

        Comparing this against the whole-run median detects that case: if they
        disagree the arm never converged and the median is a biased estimate,
        not a noisy one. See `drift` in the table.
        """
        return self.pre.get("plateau") or self.pre["median"]

    @property
    def aggregate(self) -> float:
        """Median per-GPU TFLOP/s x world size, in PFLOP/s."""
        return self.per_gpu[1] * (self.world or 0) / 1000.0

    @property
    def tokens_per_s(self) -> float | None:
        return self.pre.get("tok_per_gpu_s")


def load_from_csv(
    path: Path, label: str, pattern: str, filters: list[tuple[str, str]]
) -> list[Run]:
    """Build Runs from gather_speed.py's CSV instead of re-parsing logs.

    Re-reading 200+ multi-MB logs takes minutes; the CSV takes milliseconds, so
    iterating on a figure stops being gated on I/O. The statistics are identical
    because gather_speed.py applies the same warmup trimming -- but note the
    trimming happened AT GATHER TIME, so --skip-first/--skip-frac/--tail have no
    effect here. Re-run gather_speed.py to change them.

    `pattern` is matched against the `run` column (same semantics as the
    directory-name match). `filters` are extra COLUMN=REGEX constraints, e.g.
    fp8=hybrid or tp=4, applied with re.fullmatch so tp=4 cannot match 40.
    """
    rx = re.compile(pattern)

    def num(v, cast=float):
        try:
            return cast(v)
        except (TypeError, ValueError):
            return None

    runs: list[Run] = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            name = row.get("run", "")
            m = rx.search(name)
            if not m:
                continue
            if any(not re.fullmatch(pat, str(row.get(col, ""))) for col, pat in filters):
                continue
            med = num(row.get("tflops_median"))
            if med is None:
                continue
            r = Run(label=label, name=name, log=Path(row.get("log", "")))
            r.world = num(row.get("world_size"), int)
            r.dp, r.tp, r.pp = (num(row.get(k), int) for k in ("dp", "tp", "pp"))
            r.gbs = num(row.get("global_batch_size"), int)
            r.mbs = num(row.get("micro_batch_size"), int)
            r.seq = num(row.get("seq_length"), int)
            r.nodes = int(m.group(1)) if m.groups() else num(row.get("nodes"), int)
            # short signature of the settings that must NOT be mixed in one curve
            r.cfg_sig = "/".join(
                str(row.get(c, ""))
                for c in ("fp8", "fp8_recipe", "grad_reduce_in_bf16", "micro_batch_size")
            )
            r.pre = dict(
                median=med,
                plateau=num(row.get("tflops_plateau")),
                p10=num(row.get("tflops_p10")) or med,
                p90=num(row.get("tflops_p90")) or med,
                drift=num(row.get("drift_pct")) or 0.0,
                n_iters=num(row.get("iters_used"), int) or 0,
                job_id=row.get("job_id", ""),
            )
            runs.append(r)
    # One arm can appear several times (reruns). Keep the row with the most
    # iterations, mirroring the "most iterations wins" rule used for logs.
    best: dict[tuple, Run] = {}
    variants: dict[tuple, set] = {}
    for r in runs:
        k = (r.nodes, r.gbs, r.tp, r.pp)
        variants.setdefault(k, set()).add(r.cfg_sig)
        cur = best.get(k)
        if cur is None or (r.pre["n_iters"] or 0) > (cur.pre["n_iters"] or 0):
            best[k] = r
    # LOUD WARNING: if one geometry has rows with DIFFERENT precision/parallel
    # settings, "most iterations wins" silently picks one of them, which can mix
    # an FP8 arm and a bf16 arm into the same curve. Observed for real: without a
    # filter the 1024-node point flipped to the bf16 run (275.3) because it had
    # more iterations than the FP8 one (322.7). Always filter when both exist.
    for k, sigs in sorted(variants.items()):
        if len(sigs) > 1:
            chosen = best[k].cfg_sig
            print(
                f"  ! {label}: {k[0]} nodes has {len(sigs)} configurations "
                f"{sorted(sigs)}; kept {chosen!r} (most iterations). "
                f"Add --filter to choose deliberately.",
                file=sys.stderr,
            )
    return sorted([r for r in best.values() if r.ok], key=lambda r: r.nodes or 0)


def classify(runs: list[Run]) -> str:
    """Strong (GBS fixed) / weak (GBS proportional to nodes) / mixed."""
    gbs = {r.gbs for r in runs if r.gbs}
    if len(gbs) <= 1:
        return "strong"
    ratios = {round((r.gbs or 0) / r.nodes, 6) for r in runs if r.nodes}
    return "weak" if len(ratios) == 1 else "mixed"


# --- reporting --------------------------------------------------------------


def table(runs: list[Run], kind: str) -> list[dict]:
    ref = runs[0]
    rows = []
    for r in runs:
        lo, med, hi = r.per_gpu
        if kind == "strong":
            ideal = r.nodes / ref.nodes
            speedup = r.aggregate / ref.aggregate if ref.aggregate else float("nan")
            eff = speedup / ideal if ideal else float("nan")
        elif kind == "weak":
            eff = med / ref.per_gpu[1] if ref.per_gpu[1] else float("nan")
            speedup = float("nan")
        else:
            eff = speedup = float("nan")
        plat = r.plateau
        rows.append(
            dict(
                series=r.label,
                run=r.name,
                nodes=r.nodes,
                world=r.world,
                tp=r.tp,
                pp=r.pp,
                dp=r.dp,
                gbs=r.gbs,
                mbs=r.mbs,
                M=r.m,
                n_iters=r.pre.get("n_iters", 0),
                p10=round(lo, 1),
                median=round(med, 1),
                p90=round(hi, 1),
                plateau=round(plat, 1),
                drift=round(100.0 * (plat - med) / med, 1) if med else float("nan"),
                pflops=round(r.aggregate, 2),
                speedup=round(speedup, 2),
                efficiency=round(eff, 3),
            )
        )
    return rows


def print_table(rows: list[dict], kind: str) -> None:
    print(f"\n  === {rows[0]['series']}  ({kind} scaling, {len(rows)} arms) ===")
    hdr = (
        f"  {'nodes':>6} {'TP':>3} {'PP':>3} {'DP':>5} {'GBS':>6} {'M':>5} {'n':>5} "
        f"{'p10':>7} {'median':>7} {'p90':>7} {'plateau':>8} {'drift%':>7} {'PFLOP/s':>9} "
    )
    hdr += f"{'speedup':>8} {'eff':>6}" if kind == "strong" else f"{'eff':>6}"
    print(hdr)
    unconverged = []
    for r in rows:
        flag = "  <-- NOT CONVERGED" if abs(r["drift"]) > 5.0 else ""
        if flag:
            unconverged.append(r)
        line = (
            f"  {r['nodes']:>6} {r['tp']:>3} {r['pp']:>3} {r['dp']:>5} {r['gbs']:>6} "
            f"{str(r['M']):>5} {r['n_iters']:>5} {r['p10']:>7.1f} {r['median']:>7.1f} "
            f"{r['p90']:>7.1f} {r['plateau']:>8.1f} {r['drift']:>7.1f} {r['pflops']:>9.2f} "
        )
        line += (
            f"{r['speedup']:>8.2f} {r['efficiency']:>6.3f}"
            if kind == "strong"
            else f"{r['efficiency']:>6.3f}"
        )
        print(line + flag)
    if unconverged:
        print(
            f"\n  WARNING: {len(unconverged)} arm(s) never reached steady state — their "
            "medians are BIASED, not merely noisy:"
        )
        for r in unconverged:
            if r["drift"] > 0:
                # plateau above median: a long warmup dragged the median down
                print(
                    f"    {r['nodes']:>5}n  still RISING at the end "
                    f"(median {r['median']:.1f} -> plateau {r['plateau']:.1f}, "
                    f"{r['drift']:+.1f}%). Median biased LOW; true rate is nearer the plateau."
                )
            else:
                # plateau below median: it ran clean then degraded (straggler, thermal, FS)
                print(
                    f"    {r['nodes']:>5}n  DEGRADED at the end "
                    f"(median {r['median']:.1f} -> plateau {r['plateau']:.1f}, "
                    f"{r['drift']:+.1f}%). Median biased HIGH; suspect a sick node or FS stall."
                )
        print("  Re-run these arms longer; do not compare them against converged arms.")


PAPER_RC = {
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.linewidth": 0.9,
    "lines.linewidth": 1.9,
    "lines.markersize": 5.5,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 160,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "legend.frameon": False,
}


def _node_ticks(ax, xs):
    """Label the x axis with the ACTUAL node counts.

    A log2 axis otherwise renders 2^3, 2^4, ... which nobody reads as "8 nodes".
    """
    xs = sorted(set(xs))
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(v) for v in xs])
    ax.minorticks_off()


def _panel_pergpu(ax, all_rows, colors, peak=None, annotate_m=False, marks=()):
    for label, (rows, kind) in all_rows.items():
        x = [r["nodes"] for r in rows]
        med = [r["median"] for r in rows]
        lo = [r["median"] - r["p10"] for r in rows]
        hi = [r["p90"] - r["median"] for r in rows]
        ax.errorbar(
            x,
            med,
            yerr=[lo, hi],
            marker="o",
            capsize=2.5,
            elinewidth=1,
            color=colors[label],
            label=label,
        )
        # For WEAK scaling the ideal is a flat line: constant work per GPU should
        # give constant per-GPU rate. For STRONG scaling per-GPU rate is EXPECTED
        # to fall, so a flat reference there would be misleading and is omitted.
        if kind == "weak":
            ax.axhline(med[0], ls="--", lw=1.1, alpha=0.55, color=colors[label])
        if annotate_m:
            for r in rows:
                if r.get("M"):
                    ax.annotate(
                        f"M={r['M']}",
                        (r["nodes"], r["median"]),
                        textcoords="offset points",
                        xytext=(0, 7),
                        ha="center",
                        fontsize=7.5,
                        alpha=0.75,
                    )
    # Points measured OUTSIDE this sweep -- in practice the sustained production
    # rate. Worth overlaying because a 50-iteration shakeout never pays for a
    # checkpoint, an eval or a restart, so the curve above is an upper bound on
    # what a real run holds; the gap between the two is the honest correction.
    for label, nodes, value in marks:
        ax.plot([nodes], [value], marker="*", ms=15, ls="none", color="C3", zorder=5, label=label)
        ax.annotate(
            f"{value:.0f}",
            (nodes, value),
            textcoords="offset points",
            xytext=(0, -16),
            ha="center",
            fontsize=8.5,
            color="C3",
            fontweight="bold",
        )
    _node_ticks(ax, [r["nodes"] for rows, _ in all_rows.values() for r in rows])
    ax.set_xlabel("nodes (4 GH200 per node)")
    ax.set_ylabel("TFLOP/s per GPU")
    ax.set_ylim(bottom=0)
    if peak:
        sec = ax.secondary_yaxis(
            "right", functions=(lambda v: 100.0 * v / peak, lambda m: m * peak / 100.0)
        )
        sec.set_ylabel(f"MFU (% of {peak:g} TFLOP/s peak)")
    ax.set_title("Per-GPU throughput")


def _panel_aggregate(ax, all_rows, colors):
    for label, (rows, _kind) in all_rows.items():
        x = [r["nodes"] for r in rows]
        ax.plot(x, [r["pflops"] for r in rows], marker="o", color=colors[label], label=label)
    # ONE ideal-linear reference per series, anchored at that series' smallest arm.
    # Colour-matched because on log-log such a line is fixed entirely by its
    # per-node rate, so two series with similar small-arm rates draw two almost
    # coincident dashed lines and look like a single shared reference.
    for i, (label, (rows, _k)) in enumerate(all_rows.items()):
        x0, y0 = rows[0]["nodes"], rows[0]["pflops"]
        xs = [r["nodes"] for r in rows]
        ax.plot(
            xs,
            [y0 * xi / x0 for xi in xs],
            ls="--",
            lw=1.1,
            alpha=0.55,
            color=colors[label],
            label="ideal linear scaling" if i == 0 else None,
        )
    _node_ticks(ax, [r["nodes"] for rows, _ in all_rows.values() for r in rows])
    ax.set_yscale("log", base=10)
    ax.set_xlabel("nodes (4 GH200 per node)")
    ax.set_ylabel("aggregate PFLOP/s")
    ax.set_title("Aggregate throughput")


def _panel_efficiency(ax, all_rows, colors):
    for label, (rows, _kind) in all_rows.items():
        x = [r["nodes"] for r in rows]
        ax.plot(
            x, [100.0 * r["efficiency"] for r in rows], marker="o", color=colors[label], label=label
        )
        r = rows[-1]
        ax.annotate(
            f"{100 * r['efficiency']:.0f}%",
            (r["nodes"], 100 * r["efficiency"]),
            textcoords="offset points",
            xytext=(-4, -14),
            ha="center",
            fontsize=9,
            color=colors[label],
            fontweight="bold",
        )
    ax.axhline(100.0, ls="--", lw=1.1, alpha=0.55, color="0.35", label="ideal (no scaling loss)")
    _node_ticks(ax, [r["nodes"] for rows, _ in all_rows.values() for r in rows])
    ax.set_xlabel("nodes (4 GH200 per node)")
    ax.set_ylabel("parallel efficiency (%)")
    ax.set_ylim(0, 112)
    ax.set_title("Parallel efficiency")


def plot(
    all_rows, out, paper=False, split=False, peak=None, annotate_m=False, suptitle=None, marks=()
):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "\n  matplotlib not available — table printed above, no plot written.", file=sys.stderr
        )
        return

    ctx = plt.rc_context(PAPER_RC) if paper else plt.rc_context({})
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2"])
    colors = {lab: cycle[i % len(cycle)] for i, lab in enumerate(all_rows)}
    panels = [
        ("pergpu", _panel_pergpu),
        ("aggregate", _panel_aggregate),
        ("efficiency", _panel_efficiency),
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    written = []

    def _save(fig, path):
        # Proposals go into LaTeX: emit vector PDF alongside the raster preview.
        fig.savefig(path)
        written.append(path)
        if paper and path.suffix.lower() == ".png":
            pdf = path.with_suffix(".pdf")
            fig.savefig(pdf)
            written.append(pdf)
        plt.close(fig)

    with ctx:
        if split:
            for name, fn in panels:
                fig, ax = plt.subplots(figsize=(5.2, 3.9))
                fn(
                    ax,
                    all_rows,
                    colors,
                    **(
                        {"peak": peak, "annotate_m": annotate_m, "marks": marks}
                        if name == "pergpu"
                        else {}
                    ),
                )
                ax.legend()
                _save(fig, out.with_name(f"{out.stem}_{name}{out.suffix}"))
        else:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
            for ax, (name, fn) in zip(axes, panels):
                fn(
                    ax,
                    all_rows,
                    colors,
                    **(
                        {"peak": peak, "annotate_m": annotate_m, "marks": marks}
                        if name == "pergpu"
                        else {}
                    ),
                )
                ax.legend()
            if suptitle:
                fig.suptitle(suptitle, fontsize=13, y=1.02)
            fig.tight_layout()
            _save(fig, out)

    for p in written:
        print(f"  plot written: {p}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--from-csv",
        type=Path,
        default=Path("dump/all_runs_speed.csv"),
        help="gather_speed.py output; the ONLY data source (default: "
        "dump/all_runs_speed.csv). Warmup trimming happens at gather "
        "time -- re-run gather_speed.py to change it.",
    )
    ap.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="COL=REGEX",
        help="repeatable; keep only rows whose COLUMN "
        "fully matches REGEX (e.g. fp8=hybrid, tp=4, mock_data=False)",
    )
    ap.add_argument(
        "--series",
        action="append",
        required=True,
        metavar="LABEL=REGEX",
        help="repeatable; regex matched against subdirectory names, "
        "optional capture group = node count",
    )
    ap.add_argument("--out", type=Path, default=Path("dump/scaling.png"))
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument(
        "--paper",
        action="store_true",
        help="publication styling; also emits a vector .pdf beside each .png",
    )
    ap.add_argument(
        "--split",
        action="store_true",
        help="one file per panel (_pergpu/_aggregate/_efficiency) instead "
        "of a 3-panel figure — usually what a proposal wants",
    )
    ap.add_argument(
        "--peak-tflops",
        type=float,
        default=None,
        help="add an MFU axis using this per-GPU peak. NOT set by default: "
        "MFU is only as honest as the peak you quote, so pass it "
        "explicitly (GH200 BF16 dense w/ sparsity is 989.4)",
    )
    ap.add_argument(
        "--annotate-m",
        action="store_true",
        help="label each point with M = GBS/(mbs*DP), the microbatches per "
        "DP rank — the quantity that actually drives the strong curve",
    )
    ap.add_argument("--suptitle", default=None)
    ap.add_argument(
        "--mark",
        action="append",
        default=[],
        metavar="LABEL=NODES:TFLOPS",
        help="repeatable; overlay a star on the per-GPU panel for a point "
        "measured outside this sweep, e.g. the SUSTAINED production "
        "rate. A shakeout curve is an upper bound — marking the real "
        "run beside it is what makes the figure quotable",
    )
    args = ap.parse_args()

    marks = []
    for m in args.mark:
        try:
            label, rhs = m.split("=", 1)
            nodes, value = rhs.split(":", 1)
            marks.append((label, int(nodes), float(value)))
        except ValueError:
            print(f"error: --mark needs LABEL=NODES:TFLOPS, got {m!r}", file=sys.stderr)
            return 2

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

    all_rows: dict[str, tuple[list[dict], str]] = {}
    flat: list[dict] = []
    for spec in args.series:
        if "=" not in spec:
            print(f"error: --series needs LABEL=REGEX, got {spec!r}", file=sys.stderr)
            return 2
        label, pattern = spec.split("=", 1)
        runs = load_from_csv(args.from_csv, label, pattern, filters)
        if not runs:
            print(f"  ! no runs matched {pattern!r}", file=sys.stderr)
            continue
        kind = classify(runs)
        rows = table(runs, kind)
        print_table(rows, kind)
        all_rows[label] = (rows, kind)
        flat.extend(rows)

    if not all_rows:
        print("nothing to plot", file=sys.stderr)
        return 1

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(flat[0].keys()))
            w.writeheader()
            w.writerows(flat)
        print(f"  csv written : {args.csv}")

    plot(
        all_rows,
        args.out,
        paper=args.paper,
        split=args.split,
        peak=args.peak_tflops,
        annotate_m=args.annotate_m,
        suptitle=args.suptitle,
        marks=marks,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

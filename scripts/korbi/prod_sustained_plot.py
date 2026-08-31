#!/usr/bin/env python3
r"""Plot SUSTAINED throughput of the 32B flagship, across restarts, at 512
nodes.

WHY THIS EXISTS ALONGSIDE speed_scaling_plots.py
------------------------------------------------
`speed_scaling_plots.py` answers "how does throughput scale with node count?"
from 50-iteration shakeouts. It cannot answer "can that rate be HELD?", because
a shakeout never pays for a checkpoint, an eval, a node failure, or a queue
wait. Those are exactly the costs a multi-week run is made of, and they are the
difference between a benchmark number and a schedule you can promise.

So this script reads the real production logs and reports two different things
that are both routinely (and wrongly) called "throughput":

  panel A  INSTANTANEOUS rate  — TFLOP/s/GPU while the job is running.
                                 This is the number the scaling plots produce.
  panel B  DELIVERED tokens vs WALL CLOCK — the same run measured end to end,
                                 including every restart gap, queue wait and
                                 startup. The gap between the two curves IS the
                                 cost of running at scale, and it is invisible
                                 in panel A.

Quoting panel A alone overstates a real schedule; quoting panel B alone hides
whether the configuration is fast. Both, side by side, is the honest pair.

INPUT
-----
Not the Megatron logs themselves — they are ~1.3 GB for one campaign and live on
JUPITER. Grep them there first and copy the ~1 MB of matching lines back:

    B=/e/project1/e-sta-openeurollm/production_training
    for d in $B/oellm_32b_dense_prod_gbs4096_lr3e-4 \
             $B/oellm_32b_dense_prod_dataopt5_gbs4096_lr3e-4; do
      for lg in $d/logs/slurm-*.log; do
        jid=$(basename $lg .log); jid=${jid#slurm-}
        grep -ah "elapsed time per iteration" "$lg" \
          | sed "s#^#$(basename $d)|$jid|#" >> prod_iterlines.txt
      done
    done
    gzip prod_iterlines.txt

    python scripts/korbi/prod_sustained_plot.py --lines dump/prod_iterlines.txt.gz \
        --campaign oellm_32b_dense_prod_dataopt5_gbs4096_lr3e-4 \
        --paper --peak-tflops 989.4 --out dump/final/fig_prod_sustained.png

RESTARTS REPLAY ITERATIONS — DEDUPLICATE OR DOUBLE-COUNT
-------------------------------------------------------
A job that dies at iteration 34450 resumes from the last checkpoint, not from
34450, so the next job re-logs a few hundred iterations that already exist. Here
job 1512329 and job 1516079 both cover 25880-25980. Concatenating the logs would
count those tokens twice and invent throughput that never happened. Every
iteration is therefore keyed on its NUMBER and the LATEST timestamp wins, which
is the same rule `gather_speed.py` applies within a single log.

MEDIANS, AND A ROLLING MEDIAN
-----------------------------
Same reasoning as the scaling plots: `manual_gc_interval` puts a ~+45% spike on
one iteration in every N, checkpoint saves stall one logged interval outright,
and a sick node produces occasional 4x dips. A mean folds all of that into the
headline; a median does not. The rolling curve is a median too, so a checkpoint
stall shows up as a notch rather than dragging the trend.
"""

from __future__ import annotations

import argparse
import gzip
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

# run|job|[default3]: [2026-08-29 11:53:02.898210] iteration 48501/ 894000 | consumed
# samples: 198660096 | elapsed time per iteration (ms): 47977.7 | mem usages: 0.7592 |
# throughput per GPU (TFLOP/s/GPU): 34.6 | ...
LINE_RE = re.compile(
    r"^(?P<run>[^|]+)\|(?P<job>\d+)\|"
    r".*?\[(?P<ts>\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d(?:\.\d+)?)\]"
    r"\s*iteration\s+(?P<it>\d+)\s*/\s*(?P<total>\d+)"
    r".*?consumed samples:\s*(?P<samples>\d+)"
    r".*?elapsed time per iteration \(ms\):\s*(?P<ms>[\d.]+)"
    r".*?mem usages:\s*(?P<mem>[\d.]+)"
    r".*?TFLOP/s/GPU\):\s*(?P<tf>[\d.]+)"
)


def parse(path: Path, campaign: str | None):
    """-> (records, skipped). One record per LOGGED line, not per iteration."""
    opener = gzip.open if path.suffix == ".gz" else open
    recs, skipped = [], 0
    with opener(path, "rt", errors="ignore") as fh:
        for line in fh:
            m = LINE_RE.match(line)
            if not m:
                skipped += 1
                continue
            d = m.groupdict()
            if campaign and d["run"] != campaign:
                continue
            ts = d["ts"].replace("T", " ")
            ts = datetime.strptime(ts.split(".")[0], "%Y-%m-%d %H:%M:%S")
            recs.append(
                dict(
                    run=d["run"],
                    job=d["job"],
                    ts=ts,
                    it=int(d["it"]),
                    total=int(d["total"]),
                    samples=int(d["samples"]),
                    ms=float(d["ms"]),
                    mem=float(d["mem"]),
                    tf=float(d["tf"]),
                )
            )
    return recs, skipped


def dedup(recs: list[dict]) -> list[dict]:
    """Keep one record per iteration: the one written LAST.

    See the module docstring — resumed jobs replay iterations they did not run,
    and keeping both would double-count their tokens.
    """
    best: dict[int, dict] = {}
    for r in recs:
        cur = best.get(r["it"])
        if cur is None or r["ts"] >= cur["ts"]:
            best[r["it"]] = r
    return [best[k] for k in sorted(best)]


def rolling_median(xs: list[float], win: int) -> list[float]:
    if win <= 1:
        return xs
    half = win // 2
    return [statistics.median(xs[max(0, i - half) : i + half + 1]) for i in range(len(xs))]


def steady_seconds(recs: list[dict], max_gap: int = 20) -> float:
    """Total time actually spent iterating, weighted by the LOG INTERVAL.

    Megatron reports the MEAN ms/iter since the previous log line, so with
    `log_interval: 5` each record stands for five iterations. Summing the raw
    per-record ms therefore undercounts the run 5x — here 18.1 h instead of the
    true 90.0 h, which would have turned an 87% efficiency into 17%.

    Intervals wider than `max_gap` are dropped: those are the seams where a
    restart resumed from an older checkpoint, and their "elapsed per iteration"
    describes a startup, not training.
    """
    total = 0.0
    for prev, cur in zip(recs, recs[1:]):
        d = cur["it"] - prev["it"]
        if 0 < d <= max_gap:
            total += d * cur["ms"] / 1000.0
    return total


def summarize(recs, seq, gpus, peak, alloc=None) -> dict:
    """Instantaneous rate, delivered tokens, and the two efficiencies they
    imply."""
    tf = sorted(r["tf"] for r in recs)
    med = statistics.median(tf)
    span_s = (recs[-1]["ts"] - recs[0]["ts"]).total_seconds()
    # Tokens actually delivered over the campaign. `samples` is Megatron's own
    # running total, so it already accounts for a batch-size ramp if one ran.
    tokens = (recs[-1]["samples"] - recs[0]["samples"]) * seq
    busy_s = steady_seconds(recs)
    s = dict(
        n_logged=len(recs),
        n_jobs=len({r["job"] for r in recs}),
        it_lo=recs[0]["it"],
        it_hi=recs[-1]["it"],
        it_total=recs[-1]["total"],
        t_lo=recs[0]["ts"],
        t_hi=recs[-1]["ts"],
        span_h=span_s / 3600.0,
        tf_p10=tf[int(len(tf) * 0.10)],
        tf_med=med,
        tf_p90=tf[int(len(tf) * 0.90)],
        mfu=(100.0 * med / peak) if peak else None,
        tokens=tokens,
        tokens_per_day=tokens / (span_s / 86400.0) if span_s else 0.0,
        busy_h=busy_s / 3600.0,
        busy_frac=busy_s / span_s if span_s else 0.0,
        pflops=med * gpus / 1000.0,
        mem_max=max(r["mem"] for r in recs),
        alloc_h=None,
        alloc_frac=None,
        gaps=[],
    )
    if alloc:
        # SLURM is the only source for allocation time: the logs start at the
        # FIRST iteration, so they cannot see the 8-15 min of NCCL sync and
        # kernel compilation that every 512-node allocation pays before it.
        alloc_s = sum((e - b).total_seconds() for b, e in alloc.values())
        s["alloc_h"] = alloc_s / 3600.0
        s["alloc_frac"] = busy_s / alloc_s if alloc_s else None
        win = sorted(alloc.values())
        s["gaps"] = sorted(
            ((b2 - e1).total_seconds() / 3600.0 for (_, e1), (b2, _) in zip(win, win[1:])),
            reverse=True,
        )
    return s


def print_summary(s: dict, label: str) -> None:
    print(f"\n  === {label} ===")
    print(f"  jobs (restarts)      : {s['n_jobs']}")
    print(
        f"  iterations           : {s['it_lo']} -> {s['it_hi']}  of {s['it_total']} "
        f"({100.0 * s['it_hi'] / s['it_total']:.1f}% of the 15 T schedule)"
    )
    print(
        f"  wall clock           : {s['t_lo']:%Y-%m-%d %H:%M} -> {s['t_hi']:%Y-%m-%d %H:%M} "
        f"({s['span_h']:.1f} h)"
    )
    print(
        f"  TFLOP/s/GPU          : p10 {s['tf_p10']:.1f} | median {s['tf_med']:.1f} | "
        f"p90 {s['tf_p90']:.1f}" + (f"   -> MFU {s['mfu']:.1f}%" if s["mfu"] else "")
    )
    print(f"  aggregate            : {s['pflops']:.1f} PFLOP/s while running")
    print(
        f"  tokens delivered     : {s['tokens'] / 1e12:.3f} T  "
        f"({s['tokens_per_day'] / 1e9:.1f} B tokens/day end-to-end)"
    )
    print(f"  steady-state time    : {s['busy_h']:.1f} h")
    if s["alloc_h"]:
        print(f"  SLURM allocated      : {s['alloc_h']:.1f} h")
        print(
            f"    in-allocation eff. : {100.0 * s['alloc_frac']:.1f}%  "
            "<- startup, checkpoints, eval, teardown"
        )
    print(
        f"    of calendar span   : {100.0 * s['busy_frac']:.1f}%  "
        "<- the above PLUS queue waits and pauses"
    )
    if s["gaps"]:
        # Without this line the calendar number reads as a system property. It is
        # not: two gaps here are a 21 h queue wait and a 17 h stretch with nothing
        # submitted, while the median handover between allocations is ~2 minutes.
        big = [g for g in s["gaps"] if g > 1.0]
        rest = sorted(g for g in s["gaps"] if g <= 1.0)
        print(
            f"  gaps between allocs  : {len(s['gaps'])} total; "
            f"{len(big)} over 1 h ({', '.join(f'{g:.1f} h' for g in big)})"
            + (
                f"; median of the other {len(rest)} = {rest[len(rest) // 2] * 60:.0f} min"
                if rest
                else ""
            )
        )
    print(f"  peak memory reported : {s['mem_max']:.3f}")


def load_sacct(path: Path) -> dict:
    """Parse `sacct -X -j <ids> --format=JobID,...,Start,End` output.

    Accepts either the raw table or bare `jobid start end` lines;
    anything whose 2nd and 3rd whitespace fields are not ISO timestamps
    is skipped, which drops the header and the dashed rule without
    needing to know the column widths.
    """
    out = {}
    for line in path.read_text().splitlines():
        parts = line.replace("|", " ").split()  # accepts sacct -P as well as the table
        if len(parts) < 3:
            continue
        job = parts[0].split(".")[0]
        stamps = [p for p in parts if re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d", p)]
        if len(stamps) < 2 or not job.isdigit():
            continue
        out[job] = (datetime.fromisoformat(stamps[0]), datetime.fromisoformat(stamps[1]))
    return out


def plot(recs, s, out: Path, peak, paper, window, label, seq, alloc_windows=None):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "\n  matplotlib not available — summary printed above, no plot written.",
            file=sys.stderr,
        )
        return

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from speed_scaling_plots import PAPER_RC  # one styling source for both figures
    except ImportError:
        PAPER_RC = {}

    it = [r["it"] for r in recs]
    tf = [r["tf"] for r in recs]
    roll = rolling_median(tf, window)
    # Restart boundaries: the first iteration each job contributes after dedup.
    firsts, seen = [], set()
    for r in recs:
        if r["job"] not in seen:
            seen.add(r["job"])
            firsts.append(r["it"])

    out.parent.mkdir(parents=True, exist_ok=True)
    written = []
    with plt.rc_context(PAPER_RC if paper else {}):
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.1))

        # --- A: instantaneous rate -------------------------------------------
        ax.plot(it, tf, lw=0.5, alpha=0.30, color="C0", label="per logged interval")
        ax.plot(it, roll, lw=1.8, color="C0", label=f"rolling median ({window})")
        ax.axhline(
            s["tf_med"],
            ls="--",
            lw=1.1,
            color="0.35",
            label=f"campaign median {s['tf_med']:.0f} TFLOP/s",
        )
        for i, x in enumerate(firsts[1:]):
            ax.axvline(x, color="C3", lw=0.7, alpha=0.45, label="restart" if i == 0 else None)
        ax.set_xlabel("training iteration")
        ax.set_ylabel("TFLOP/s per GPU")
        ax.set_ylim(0, max(tf) * 1.08)
        if peak:
            sec = ax.secondary_yaxis(
                "right", functions=(lambda v: 100.0 * v / peak, lambda m: m * peak / 100.0)
            )
            sec.set_ylabel(f"MFU (% of {peak:g} TFLOP/s peak)")
        ax.set_title("Instantaneous throughput")
        ax.legend(loc="lower right")

        # --- B: delivered tokens vs wall clock --------------------------------
        h = [(r["ts"] - recs[0]["ts"]).total_seconds() / 3600.0 for r in recs]
        tok = [(r["samples"] - recs[0]["samples"]) * seq / 1e12 for r in recs]
        bx.plot(h, tok, lw=1.9, color="C0", label="delivered")
        # Reference = the same configuration never interrupted, i.e. every
        # wall-clock hour spent at the achieved steady-state token rate. The gap
        # to it is startup + checkpoints + queue, which is exactly the quantity
        # panel A cannot show.
        rate = tok[-1] / s["busy_h"] if s["busy_h"] else 0.0
        bx.plot(
            [0, h[-1]],
            [0, rate * h[-1]],
            ls="--",
            lw=1.2,
            color="0.35",
            label="continuous training (no restart, no queue)",
        )
        if alloc_windows:
            t0 = recs[0]["ts"]
            for i, (b, e) in enumerate(sorted(alloc_windows.values())):
                bx.axvspan(
                    (b - t0).total_seconds() / 3600.0,
                    (e - t0).total_seconds() / 3600.0,
                    color="C0",
                    alpha=0.10,
                    label="inside a SLURM allocation" if i == 0 else None,
                )
        bx.set_xlabel("wall clock since campaign start (h)")
        bx.set_ylabel("tokens delivered (T)")
        bx.set_xlim(left=0)
        bx.set_ylim(bottom=0)
        title = f"Delivered tokens — {100.0 * s['busy_frac']:.0f}% of calendar time training"
        if s["alloc_frac"]:
            title += f", {100.0 * s['alloc_frac']:.0f}% of allocated time"
        bx.set_title(title)
        bx.legend(loc="upper left")

        if label:
            fig.suptitle(label, fontsize=13, y=1.03)
        fig.tight_layout()
        fig.savefig(out)
        written.append(out)
        if paper and out.suffix.lower() == ".png":
            fig.savefig(out.with_suffix(".pdf"))
            written.append(out.with_suffix(".pdf"))
        plt.close(fig)
    for p in written:
        print(f"  plot written: {p}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--lines",
        type=Path,
        default=Path("dump/prod_iterlines.txt.gz"),
        help="grepped log lines, prefixed 'run|jobid|' (plain or .gz); "
        "see the module docstring for how to produce it",
    )
    ap.add_argument(
        "--campaign",
        default=None,
        help="keep only this run name. Two campaigns BOTH start at "
        "iteration 1, so merging them onto one iteration axis is "
        "meaningless — always pick one.",
    )
    ap.add_argument("--list", action="store_true", help="list campaigns and exit")
    ap.add_argument(
        "--sacct",
        type=Path,
        default=None,
        help="output of `sacct -X -j <ids> --format=JobID,State,Start,End`. "
        "Without it the script can only report training time against "
        "CALENDAR time, which charges the run for queue waits it did "
        "not cause; with it you also get in-allocation efficiency.",
    )
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--gpus", type=int, default=2048, help="512 nodes x 4 GH200")
    ap.add_argument("--window", type=int, default=101, help="rolling-median width")
    ap.add_argument(
        "--peak-tflops",
        type=float,
        default=None,
        help="add an MFU axis using this per-GPU peak. NOT set by default: "
        "MFU is only as honest as the peak you quote (GH200 BF16 "
        "dense is 989.4; the FP8 peak is 2x that)",
    )
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", type=Path, default=Path("dump/fig_prod_sustained.png"))
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    if not args.lines.is_file():
        print(f"error: --lines not a file: {args.lines}", file=sys.stderr)
        return 2
    recs, skipped = parse(args.lines, None if args.list else args.campaign)
    if not recs:
        print("no records parsed", file=sys.stderr)
        return 1
    if skipped:
        print(f"  note: {skipped} line(s) did not match the iteration pattern", file=sys.stderr)

    if args.list or not args.campaign:
        by = {}
        for r in recs:
            by.setdefault(r["run"], []).append(r)
        print("  campaigns found:")
        for run, rs in sorted(by.items()):
            print(
                f"    {run:<50} {len(rs):>6} lines, "
                f"iterations {min(r['it'] for r in rs)}..{max(r['it'] for r in rs)}, "
                f"{len({r['job'] for r in rs})} jobs"
            )
        if args.list:
            return 0
        print(
            "\n  pass --campaign to pick one (they share an iteration axis otherwise)",
            file=sys.stderr,
        )
        return 2

    recs = dedup(recs)
    alloc = load_sacct(args.sacct) if args.sacct else None
    if alloc:
        # Keep only allocations that actually contributed iterations to this
        # campaign, so a stray job id in the sacct dump cannot inflate the
        # allocated-time denominator.
        jobs = {r["job"] for r in recs}
        extra = {j: w for j, w in alloc.items() if j not in jobs}
        if extra:
            lo, hi = recs[0]["ts"], recs[-1]["ts"]
            # A failed startup burns allocation without ever logging an
            # iteration; it belongs in the denominator if it fell inside the
            # campaign window. A job from a different campaign does not.
            alloc = {
                **{j: w for j, w in alloc.items() if j in jobs},
                **{j: w for j, w in extra.items() if lo <= w[0] <= hi},
            }
    s = summarize(recs, args.seq, args.gpus, args.peak_tflops, alloc)
    print_summary(s, args.campaign)

    if args.csv:
        import csv as _csv

        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(
                [
                    "iteration",
                    "job",
                    "timestamp",
                    "consumed_samples",
                    "ms_per_iter",
                    "tflops_per_gpu",
                    "mem",
                ]
            )
            for r in recs:
                w.writerow(
                    [
                        r["it"],
                        r["job"],
                        r["ts"].isoformat(),
                        r["samples"],
                        r["ms"],
                        r["tf"],
                        r["mem"],
                    ]
                )
        print(f"  csv written : {args.csv}")

    plot(recs, s, args.out, args.peak_tflops, args.paper, args.window, args.label, args.seq, alloc)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Paired readout of warm-start arms against a reference arm: lm loss / grad norm per step (slurm logs), validation
losses and Engram gate metrics (TensorBoard, when importable: run in the training container).

    python compare_runs_loss.py ctrl=<run dir> read=<run dir> ... [--from N] [--to N] [--every K]

The first arm is the reference. A run dir holds slurm-*.log (merged in time order: a restart's steps replace the
earlier attempt's) and tb/.
"""

import argparse
import glob
import math
import os
import re
import sys

ITER = re.compile(r" iteration +(\d+)/ *\d+ \|")
FIELD = {"loss": re.compile(r"lm loss: ([0-9.Ee+-]+)"), "gnorm": re.compile(r"grad norm: ([0-9.Ee+-]+)")}
TB_TAGS = ("lm loss validation", "arch/engram/gate", "engram/gate_stream_spread")


def read_log(run_dir):
    out = {}
    for log in sorted(glob.glob(os.path.join(run_dir, "slurm-*.log")), key=os.path.getmtime):
        for line in open(log, errors="replace"):
            m = ITER.search(line)
            if m:
                row = {k: float(r.search(line).group(1)) for k, r in FIELD.items() if r.search(line)}
                if "loss" in row:
                    out[int(m.group(1))] = row
    return out


def read_tb(run_dir):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return None
    out = {}
    for f in sorted(glob.glob(os.path.join(run_dir, "tb", "**", "events*"), recursive=True), key=os.path.getmtime):
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        for tag in ea.Tags()["scalars"]:
            if any(k in tag for k in TB_TAGS) and "vs samples" not in tag and "vs tokens" not in tag:
                out.setdefault(tag, {}).update({e.step: e.value for e in ea.Scalars(tag)})
    return out


def stats(xs):
    xs = [x for x in xs if not math.isnan(x)]
    if not xs:
        return "-"
    mean = sum(xs) / len(xs)
    return f"mean {mean:+.4f}, max |.| {max(map(abs, xs)):.4f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="+", help="name=run_dir; the first is the reference")
    ap.add_argument("--from", dest="lo", type=int, default=0)
    ap.add_argument("--to", dest="hi", type=int, default=10**9)
    ap.add_argument("--every", type=int, default=25, help="print every K-th step of the per-step table")
    a = ap.parse_args()
    arms = dict(x.split("=", 1) for x in a.arms)
    names = list(arms)
    logs = {n: {i: r for i, r in read_log(d).items() if a.lo <= i <= a.hi} for n, d in arms.items()}
    ref = names[0]

    print("## lm loss (grad norm) per step\n")
    print("| step | " + " | ".join(names) + " |")
    print("|---" * (len(names) + 1) + "|")
    steps = sorted(set().union(*[set(v) for v in logs.values()]))
    for i, s in enumerate(steps):
        if i % a.every and s != steps[-1]:
            continue
        cells = [f"{logs[n][s]['loss']:.4f} ({logs[n][s].get('gnorm', float('nan')):.3f})" if s in logs[n] else "-"
                 for n in names]
        print(f"| {s} | " + " | ".join(cells) + " |")

    print(f"\n## against {ref} (paired steps)\n")
    print("| arm | steps | loss diff (first step) | loss diff, first 50 | loss diff, last 100 | grad norm ratio, max |")
    print("|---|---:|---:|---:|---:|---:|")
    for n in names[1:]:
        shared = sorted(set(logs[ref]) & set(logs[n]))
        if not shared:
            print(f"| {n} | 0 | - | - | - | - |")
            continue
        d = [logs[n][s]["loss"] - logs[ref][s]["loss"] for s in shared]
        g = [logs[n][s].get("gnorm", float("nan")) / logs[ref][s].get("gnorm", float("nan")) for s in shared]
        print(f"| {n} | {len(shared)} | {d[0]:+.2e} | {stats(d[:50])} | {stats(d[-100:])} | "
              f"{max((x for x in g if not math.isnan(x)), default=float('nan')):.2f} |")

    tbs = {n: read_tb(d) for n, d in arms.items()}
    if all(v is None for v in tbs.values()):
        print("\n(tensorboard not importable: no validation / gate readout)")
        return 0
    tags = sorted(set().union(*[set(v or {}) for v in tbs.values()]))
    print("\n## validation and Engram gates (TensorBoard; last value in the window, and at the first logged step)\n")
    print("| tag | " + " | ".join(names) + " |")
    print("|---" * (len(names) + 1) + "|")
    for tag in tags:
        cells = []
        for n in names:
            pts = {s: v for s, v in (tbs[n] or {}).get(tag, {}).items() if a.lo <= s <= a.hi}
            if not pts:
                cells.append("-")
                continue
            first, last = min(pts), max(pts)
            cells.append(f"{pts[last]:.4f} @{last}" + (f" (from {pts[first]:.4f})" if first != last else ""))
        print(f"| {tag} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())

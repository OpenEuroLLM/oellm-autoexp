#!/usr/bin/env python3
"""Summarise EP speed suites: one row per arm (latest attempt).

    python3 scripts/korbi/ep_speed/collect.py $OUTPUT_DIR/hybrid_speed/e1_epslice [--from 10] [--md]

Reads every array group under the suite directory (task_<i>.sh -> arm, slurm-<job>_<i>.log) and
plain jobs (<arm>/*.sbatch + slurm-*.log). Per arm: median tok/s/GPU and TFLOP/s/GPU over
iterations >= --from, max memory fraction, lm loss at the first and last logged iteration, the
number of iterations, and the failure signature (Python exception / OOM / NCCL / watchdog).
Host python3 (3.6+), no dependencies.
"""
import argparse
import glob
import os
import re
import statistics

ITER = re.compile(r"iteration\s+(\d+)/\s*(\d+) \|.*?elapsed time per iteration \(ms\): ([\d.]+)"
                  r"(?:.*?mem usages: ([\d.]+))?(?:.*?TFLOP/s/GPU\): ([\d.]+))?"
                  r"(?:.*?Tok/s/GPU\): ([\d.]+))?(?:.*?lm loss: ([\d.E+-]+))?")
ERRORS = [
    ("OOM", re.compile(r"CUDA out of memory|OutOfMemoryError|out of memory")),
    ("WATCHDOG", re.compile(r"WATCHDOG: .*")),
    ("NCCL-timeout", re.compile(r"Watchdog caught collective operation timeout|NCCL.*[Tt]imeout")),
    ("assert", re.compile(r"AssertionError.*")),
    ("error", re.compile(r"^\S*\s*(?:\[default\d+\]:)?\s*(\w+(?:Error|Exception)): (.*)$", re.M)),
]


def arm_of_task(task_sh):
    txt = open(task_sh).read()
    m = re.search(r"--data-cache-path \S*/([^/\s]+)/data_cache", txt)
    if m:
        return m.group(1)
    m = re.search(r"--tensorboard-dir \S*/([^/\s]+)/tb", txt)
    return m.group(1) if m else os.path.basename(task_sh)


def parse_log(path, start):
    rows, err = [], None
    txt = open(path, errors="replace").read()
    for m in ITER.finditer(txt):
        it, tot, ms, mem, tf, tok, loss = m.groups()
        rows.append((int(it), float(ms), float(mem) if mem else None, float(tf) if tf else None,
                     float(tok) if tok else None, float(loss) if loss else None))
    for name, rx in ERRORS:
        m = rx.search(txt)
        if m:
            err = name + ": " + m.group(0).strip()[:160]
            break
    total = int(re.findall(r"iteration\s+\d+/\s*(\d+)", txt)[-1]) if rows else 1
    steady = [r for r in rows if r[0] >= start] or rows[-3:]
    med = lambda i: statistics.median([r[i] for r in steady if r[i] is not None]) if any(  # noqa: E731
        r[i] is not None for r in steady) else None
    out = {
        "iters": rows[-1][0] if rows else 0,
        "ms": med(1), "tflops": med(3), "tok": med(4),
        "mem": max((r[2] for r in rows if r[2] is not None), default=None),
        "loss0": next((r[5] for r in rows if r[5] is not None), None),
        "loss1": next((r[5] for r in reversed(rows) if r[5] is not None), None),
        "err": err if (not rows or rows[-1][0] < total) else None,
        "log": path,
    }
    wd = re.search(r"WATCHDOG: .*", txt)
    if wd:
        out["err"] = wd.group(0)[:200]
    return out


def collect(suite_dir, start):
    arms = {}
    for task in glob.glob(os.path.join(suite_dir, "*", "task_*.sh")):
        group = os.path.dirname(task)
        idx = re.search(r"task_(\d+)\.sh", task).group(1)
        arm = arm_of_task(task)
        logs = sorted(glob.glob(os.path.join(group, "slurm-*_%s.log" % idx)), key=os.path.getmtime)
        if logs:
            arms[arm] = parse_log(logs[-1], start)
            arms[arm]["attempts"] = len(logs)
    for sb in glob.glob(os.path.join(suite_dir, "*", "*.sbatch")):
        d = os.path.dirname(sb)
        if glob.glob(os.path.join(d, "task_*.sh")):
            continue
        logs = sorted(glob.glob(os.path.join(d, "slurm-*.log")), key=os.path.getmtime)
        if logs:
            arms[os.path.basename(d)] = parse_log(logs[-1], start)
    return arms


def fmt(v, f):
    return "-" if v is None else f % v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("suite_dirs", nargs="+")
    ap.add_argument("--from", dest="start", type=int, default=10)
    ap.add_argument("--ref", default=None, help="reference arm for the tok/s ratio")
    args = ap.parse_args()
    for sd in args.suite_dirs:
        arms = collect(sd, args.start)
        ref = arms.get(args.ref, {}).get("tok") if args.ref else None
        print("## %s\n" % sd)
        print("| arm | iters | ms/step | tok/s/GPU | vs ref | TFLOP/s | mem | loss first -> last | status |")
        print("|---|---:|---:|---:|---:|---:|---:|---|---|")
        for arm in sorted(arms):
            a = arms[arm]
            ratio = (a["tok"] / ref) if (ref and a["tok"]) else None
            print("| %s | %d | %s | %s | %s | %s | %s | %s -> %s | %s |" % (
                arm, a["iters"], fmt(a["ms"], "%.0f"), fmt(a["tok"], "%.0f"), fmt(ratio, "%.2f"),
                fmt(a["tflops"], "%.1f"), fmt(a["mem"], "%.0f%%" if False else "%.2f"),
                fmt(a["loss0"], "%.3f"), fmt(a["loss1"], "%.3f"),
                (a["err"] or "ok").replace("|", "/")))
        print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""RULER long-context table (docs/recall_surgery_ambitious.md §17): per model a task x context-length grid from
the rows "<task>__L<length>" (config/eval_tasks/ruler_long*.yaml, results root flag_evals_ruler), plus the mean
over tasks per length. lm-eval's RULER metric list holds every length; a row's other lengths are -1.0 ("not run")
and are ignored -- only the row's own length is read.

    python3 ruler_table.py [--root /gpfs/scratch/ehpc533/hybrid_nucleus/flag_evals_ruler] <export> [...]
"""
import argparse
import glob
import json
import os
import re

TASKS = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
         "niah_multivalue", "niah_multiquery", "ruler_vt"]
SHORT = {"niah_single_1": "s1", "niah_single_2": "s2", "niah_single_3": "s3", "niah_multikey_1": "mk1",
         "niah_multikey_2": "mk2", "niah_multikey_3": "mk3", "niah_multivalue": "mv", "niah_multiquery": "mq",
         "ruler_vt": "vt"}


def scores(root: str, export: str) -> dict:
    """{(task, length): score in % or None}."""
    out = {}
    for d in glob.glob(os.path.join(root, export, "results", "*__L*_0")):
        m = re.match(r"(.+)__L(\d+)_0$", os.path.basename(d))
        if not m:
            continue
        task, length = m.group(1), int(m.group(2))
        files = sorted(glob.glob(os.path.join(d, "**", "results_*.json"), recursive=True))
        if not files:
            continue
        res = json.load(open(files[-1]))["results"].get(task, {})
        v = res.get(f"{length},none", res.get(str(length)))
        out[(task, length)] = None if v is None or v < 0 else 100 * v
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/gpfs/scratch/ehpc533/hybrid_nucleus/flag_evals_ruler")
    ap.add_argument("exports", nargs="+")
    a = ap.parse_args(argv)
    for export in a.exports:
        s = scores(a.root, export)
        lengths = sorted({n for _, n in s})
        print(f"== {export}  ({sum(v is not None for v in s.values())} rows)")
        if not lengths:
            continue
        print(f"{'length':>7} | " + " ".join(f"{SHORT[t]:>5}" for t in TASKS) + " |  mean")
        for n in lengths:
            vals = [s.get((t, n)) for t in TASKS]
            got = [v for v in vals if v is not None]
            cells = " ".join(f"{v:5.1f}" if v is not None else f"{'-':>5}" for v in vals)
            mean = f"{sum(got) / len(got):5.1f}" if len(got) == len(TASKS) else f"({sum(got) / len(got):4.1f})" if got else "    -"
            print(f"{n:>7} | {cells} | {mean}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""The done markers of one mn5_oellm_run_one.sh call: which of its rows have a result.

A row is done when
  lm-eval-harness  flag_lmeval.py wrote its marker <lm-done>/<task>_<n_shot>.ok
  evalchemy        a results_*.json under <results> has the task       (key: <task>)
  lighteval        a results_*.json under <results>/results has it     (key: <task>|<n_shot>)
For each done row without one, <out-done>/<task key>_<n_shot>.ok is written (the runner's
per-row marker, as for a single row). The rows without a result are printed, space-separated
as <task>|<n_shot> (nothing when all are done).

    oellm_rows_done.py --suite lighteval --results <out>/lighteval_group_x --lm-done <dir> \
        --out-done <out>/done --limit 0 --rows $'flores200:a|4\\nflores200:b|4'
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time


def key_of(task: str) -> str:
    return task.translate(str.maketrans({c: "_" for c in "/:| "}))


def result_keys(results_dir: str, suite: str) -> set[str]:
    root = os.path.join(results_dir, "results") if suite == "lighteval" else results_dir
    keys: set[str] = set()
    for path in glob.glob(os.path.join(root, "**", "results_*.json"), recursive=True):
        try:
            with open(path) as fh:
                keys.update((json.load(fh).get("results") or {}).keys())
        except (OSError, ValueError):
            continue
    return keys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", required=True, choices=["lm-eval-harness", "evalchemy", "lighteval"])
    p.add_argument("--results", required=True, help="this call's output dir (evalchemy / lighteval)")
    p.add_argument("--lm-done", required=True, help="flag_lmeval.py's done dir (lm-eval)")
    p.add_argument("--out-done", required=True)
    p.add_argument("--limit", default="0")
    p.add_argument("--rows", required=True, help="<task>|<n_shot> per line")
    a = p.parse_args(argv)
    rows = [line.rsplit("|", 1) for line in a.rows.splitlines() if line.strip()]
    keys = result_keys(a.results, a.suite) if a.suite != "lm-eval-harness" else set()
    os.makedirs(a.out_done, exist_ok=True)
    missing = []
    for task, n_shot in rows:
        if a.suite == "lm-eval-harness":
            ok = os.path.exists(os.path.join(a.lm_done, f"{task}_{n_shot}.ok"))
        elif a.suite == "evalchemy":
            ok = task in keys
        else:
            ok = f"{task}|{n_shot}" in keys
        if not ok:
            missing.append(f"{task}|{n_shot}")
            continue
        marker = os.path.join(a.out_done, f"{key_of(task)}_{n_shot}.ok")
        if not os.path.exists(marker):
            with open(marker, "w") as fh:
                json.dump({"task": task, "n_shot": int(n_shot), "suite": a.suite,
                           "limit": int(float(a.limit)), "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, fh)
                fh.write("\n")
    print(" ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

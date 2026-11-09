#!/usr/bin/env python3
"""Which suite rows have no result in a collected CSV (the counterpart of `oellm-eval collect --check`).

A row (task, n_shot) of the suite file counts as present when the CSV of collect_raw_mn5.py has any
metric for that task at that n_shot (lm-eval groups count by their group row). Writes the missing
rows as CSV (name, task, n_shot, suite) and prints a one-line count per suite:

    python3 flag_missing_rows.py --suite config/eval_tasks/flag_evals.yaml --csv <export>.tasks.csv \
        [-o <export>.missing.csv] [--strict]

--strict exits 1 when a row is missing.
"""

from __future__ import annotations

import argparse
import collections
import csv
import sys
from pathlib import Path

import yaml

# collect_raw_mn5.py writes summarize-evals' spelling of this group (see its --lmeval-object path)
CSV_SPELLING = {"include_base_44_north_macedonian": "include_base_44_north macedonian"}


def missing_rows(suite: list[dict], rows: list[dict]) -> list[dict]:
    have = {(r["task"], str(r["n_shot"])) for r in rows}
    return [t for t in suite if (CSV_SPELLING.get(t["task"], t["task"]), str(t["n_shot"])) not in have]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", type=Path, required=True, help="suite file (config/eval_tasks/flag_evals.yaml)")
    ap.add_argument("--csv", type=Path, required=True, help="collect_raw_mn5.py output")
    ap.add_argument("-o", "--out", type=Path, help="write the missing rows here")
    ap.add_argument("--strict", action="store_true", help="exit 1 if any row is missing")
    a = ap.parse_args(argv)
    suite = yaml.safe_load(a.suite.read_text())["tasks"]
    with a.csv.open() as fh:
        miss = missing_rows(suite, list(csv.DictReader(fh)))
    if a.out:
        with a.out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["name", "task", "n_shot", "suite"])
            w.writeheader()
            w.writerows({k: t[k] for k in w.fieldnames} for t in miss)
    per = collections.Counter(t["suite"] for t in miss)
    total = collections.Counter(t["suite"] for t in suite)
    print(f"{a.csv.name}: {len(suite) - len(miss)}/{len(suite)} rows present; missing "
          + (", ".join(f"{s} {per[s]}/{total[s]}" for s in sorted(per)) or "none"))
    for t in miss[:20]:
        print(f"  missing {t['name']}")
    if len(miss) > 20:
        print(f"  ... {len(miss) - 20} more" + (f" (in {a.out})" if a.out else ""))
    return 1 if a.strict and miss else 0


if __name__ == "__main__":
    sys.exit(main())

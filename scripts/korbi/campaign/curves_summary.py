"""Per-suite summary of the learning-curve fits (curves.py --csv): the fitted parameters with
their standard errors, fit / extrapolation quality and health, one table per suite.

    python scripts/korbi/campaign/curves_summary.py dump/campaign_3b/curves_all.csv [--suites ...]
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict


def f(r, k, fmt=".3f"):
    v = r.get(k, "")
    try:
        return format(float(v), fmt)
    except ValueError:
        return "-"


def pm(r, k, se, fmt=".3f"):
    v, e = f(r, k, fmt), f(r, se, fmt)
    return v if e in ("-", "nan") else f"{v} ± {e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv")
    ap.add_argument("--suites", nargs="*")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv)))
    by = defaultdict(list)
    for r in rows:
        if not a.suites or r["suite"] in a.suites:
            by[r["suite"]].append(r)
    cols = "| arm | steps | warmup | stable loss | E | alpha | lr_power | fit rmse | backtest | flattening | spikes (max) | healthy | extrapolable |"
    for suite in sorted(by):
        print(f"\n### {suite}\n")
        print(cols)
        print("|" + "---|" * (cols.count("|") - 1))
        for r in sorted(by[suite], key=lambda r: float(r["stable_end_loss"] or 9)):
            print(f"| {r['arm']} | {r['iters']}/{r['total']} | {r['warmup']} | {f(r, 'stable_end_loss', '.4f')} "
                  f"| {pm(r, 'E', 'E_se')} | {pm(r, 'alpha', 'alpha_se')} | {f(r, 'lr_power', '.2f')} "
                  f"| {f(r, 'fit_rmse', '.4f')} | {f(r, 'backtest_err', '+.4f')} | {f(r, 'flattening', '+.4f')} "
                  f"| {r['spikes']} ({f(r, 'max_spike', '.3f')}) | {r['healthy']} | {r.get('extrapolable', '')} |")


if __name__ == "__main__":
    main()

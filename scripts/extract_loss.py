#!/usr/bin/env python3
"""Pull iteration/lm-loss/grad-norm out of the Megatron slurm logs.

Restarts replay iterations, so the same iteration appears in several
logs with different values. We keep the newest line per iteration, by
the timestamp Megatron stamps on it.

python3 scripts/extract_loss.py <run_dir>/logs --csv docs/fp8-loss-
turn/data/loss.csv
"""

import argparse
import csv
import re
from pathlib import Path

LINE = re.compile(
    r"\[(?P<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d(?:\.\d+)?)\].*?"
    r"iteration\s+(?P<it>\d+)/.*?"
    r"lm loss:\s*(?P<loss>[\d.]+E[+-]\d+).*?"
    r"grad norm:\s*(?P<gn>[\d.]+)"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logdir", type=Path)
    ap.add_argument("--csv", type=Path, default=Path("loss.csv"))
    args = ap.parse_args()

    best = {}  # iter -> (ts, loss, grad_norm)
    files = sorted(args.logdir.glob("slurm-*.log"))
    for f in files:
        with open(f, errors="replace") as fh:
            for line in fh:
                m = LINE.search(line)
                if not m:
                    continue
                it = int(m["it"])
                rec = (m["ts"], float(m["loss"]), float(m["gn"]))
                if it not in best or rec[0] > best[it][0]:
                    best[it] = rec

    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "lm_loss", "grad_norm"])
        for it in sorted(best):
            w.writerow([it, best[it][1], best[it][2]])
    print(f"{len(files)} logs -> {len(best)} unique iterations -> {args.csv}")


if __name__ == "__main__":
    main()

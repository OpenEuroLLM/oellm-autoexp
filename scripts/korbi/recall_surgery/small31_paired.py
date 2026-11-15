#!/usr/bin/env python3
"""Paired comparison of the small31 late arms with the control C (docs/recall_surgery_ambitious.md §14.8): for every
arm and readout step, the difference to C at the same step (same batches) on the recall-F1 columns (SQuAD v2 F1 when
answering, HasAns F1 under the uniform abstain rule, CoQA F1, FDA, SWDE, squad_completion), Winogrande and the
v8ml-valid loss; then per arm the mean of those paired differences over its readouts (incomplete readouts skipped per
column) and the mean of its last three readouts' values. Reads small31_table.py's output (run it first, or this
script calls it).

    small31_paired.py [--arms K W Wp D2 DK2 D2h T M L] [--ref C]
"""

import argparse
import subprocess
import sys
from pathlib import Path

COLS = ["CoQA F1", "Wino", "SQuAD F1", "HasAnsU", "F1|ans", "abst@ans", "abst@un", "discrim", "SWDE", "FDA", "SQcomp",
        "RULER4k", "valid"]
SHOW = [("F1|ans", "F1 ans"), ("HasAnsU", "HasAns"), ("CoQA F1", "CoQA"), ("FDA", "FDA"), ("SWDE", "SWDE"),
        ("SQcomp", "SQcomp"), ("Wino", "Wino"), ("valid", "valid")]


def table(arms):
    here = Path(__file__).resolve().parent
    out = subprocess.run([sys.executable, str(here / "small31_table.py"), "--arms", *arms], capture_output=True,
                         text=True, check=True).stdout
    rows = {}
    for line in out.splitlines():
        if "|" not in line or line.startswith("arm"):
            continue
        head, *cells = [c.strip() for c in line.split("|")]
        parts = head.split()
        if len(parts) < 2 or not parts[1].isdigit():
            continue
        vals = {}
        for name, cell in zip(COLS, cells):
            try:
                vals[name] = float(cell)
            except ValueError:
                pass
        rows[(parts[0], int(parts[1]))] = vals
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="*", default=["K", "W", "Wp", "D2", "DK2", "D2h", "T", "M", "L"])
    ap.add_argument("--ref", default="C")
    a = ap.parse_args(argv)
    rows = table([a.ref] + a.arms)
    fmt = lambda x, k: "-" if x is None else (f"{x:+.4f}" if k == "valid" else f"{x:+.2f}")  # noqa: E731
    print("paired differences arm - " + a.ref + " (same step); mean over the arm's readouts; last-3 mean of values")
    print(f"{'arm':5s} {'step':>6s} " + " ".join(f"{s:>8s}" for _, s in SHOW))
    for arm in a.arms:
        steps = sorted(s for (x, s) in rows if x == arm)
        diffs = {k: [] for k, _ in SHOW}
        for step in steps:
            ref = rows.get((a.ref, step))
            if ref is None:
                continue
            cur = rows[(arm, step)]
            line = []
            for k, _ in SHOW:
                d = cur[k] - ref[k] if k in cur and k in ref else None
                if d is not None:
                    diffs[k].append(d)
                line.append(fmt(d, k))
            print(f"{arm:5s} {step:6d} " + " ".join(f"{x:>8s}" for x in line))
        mean = [fmt(sum(v) / len(v), k) if v else "-" for (k, _), v in ((ks, diffs[ks[0]]) for ks in SHOW)]
        print(f"{arm:5s} {'mean':>6s} " + " ".join(f"{x:>8s}" for x in mean))
        last = [rows[(arm, s)] for s in steps[-3:]]
        lm = []
        for k, _ in SHOW:
            v = [r[k] for r in last if k in r]
            lm.append("-" if not v else (f"{sum(v) / len(v):.4f}" if k == "valid" else f"{sum(v) / len(v):.2f}"))
        print(f"{arm:5s} {'last3':>6s} " + " ".join(f"{x:>8s}" for x in lm) + f"   (steps {steps[-3:]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

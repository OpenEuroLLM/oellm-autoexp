#!/usr/bin/env python
"""Compare two Megatron training logs iteration-by-iteration on lm loss.

Written for the megatron-core 0.16 -> 0.19 port, where the question is not "are
the two runs bit-identical" -- they cannot be, the kernel mix differs -- but
"do they diverge the way two numerically equivalent runs diverge, or the way a
broken one does".

WHAT THE OUTPUT MEANS
---------------------
Two numerically equivalent runs of a chaotic system agree to near machine
precision for the first tens of steps and then separate exponentially, with the
sign of the difference flipping about as often as not. A real defect looks
different: either a step-1 mismatch (different init, different data order, a
different LR at iteration 1) or a drift whose sign never changes (a dropped loss
term, a different weight decay, a different schedule).

So the script reports both the magnitude AND the sign balance, and the sign
balance is the load-bearing half. `--max-rel` alone would flag a healthy run at
step 200 and miss a small consistent bias.

USAGE
    python scripts/korbi/compare_loss_parity.py <log_a> <log_b>
    python scripts/korbi/compare_loss_parity.py a.log b.log --max-rel 3e-3

Logs may be local paths. To pull them off JUPITER first:
    scp jupiter:/e/scratch/.../slurm-1511166.log /tmp/a.log
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Matches the standard Megatron progress line, e.g.
#   iteration       10/   11445 | consumed samples: ... | lm loss: 1.067719E+01 | ...
# `lm loss` is not always immediately after the iteration number (throughput,
# memory and grad-norm fields move around between versions and configs), hence
# the non-greedy gap rather than a fixed field order.
_ITER_LOSS = re.compile(r"iteration\s+(\d+)/\s*\d+.*?lm loss:\s*([0-9.]+E?[+-]?[0-9]*)")


def parse(path: Path) -> dict[int, float]:
    """Iteration -> lm loss, for every progress line in a Megatron log.

    A dict rather than a list because the two runs need not have logged the same
    iterations: a run that logged iteration 1 (the first step is always logged,
    independent of log_interval) would otherwise shift every subsequent
    comparison by one and report a spurious mismatch on every row.
    """
    losses: dict[int, float] = {}
    with path.open(errors="replace") as handle:
        for line in handle:
            match = _ITER_LOSS.search(line)
            if match:
                losses[int(match.group(1))] = float(match.group(2))
    return losses


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("log_a", type=Path, help="reference log (e.g. the 0.16 arm)")
    parser.add_argument("log_b", type=Path, help="candidate log (e.g. the 0.19 arm)")
    parser.add_argument(
        "--max-rel",
        type=float,
        default=5e-3,
        help="fail if any shared iteration exceeds this relative difference (default 5e-3)",
    )
    args = parser.parse_args()

    a, b = parse(args.log_a), parse(args.log_b)
    shared = sorted(set(a) & set(b))
    if not shared:
        print(f"no shared iterations: {len(a)} in A, {len(b)} in B", file=sys.stderr)
        return 2

    print(f"{'iter':>8}  {'A':>14}  {'B':>14}  {'rel(B-A)/A':>12}")
    worst_iter, worst_rel = 0, 0.0
    positive = 0
    for iteration in shared:
        rel = (b[iteration] - a[iteration]) / a[iteration]
        if abs(rel) > abs(worst_rel):
            worst_iter, worst_rel = iteration, rel
        if rel > 0:
            positive += 1
        print(f"{iteration:>8}  {a[iteration]:>14.6E}  {b[iteration]:>14.6E}  {rel:>12.2E}")

    # Sign balance. Under floating-point chaos the two curves cross repeatedly,
    # so `positive` lands near half of the samples. A run that is consistently
    # above or below has a bias, and a bias is a bug however small it is.
    fraction = positive / len(shared)
    print()
    print(f"shared iterations : {len(shared)}")
    print(f"worst rel diff    : {worst_rel:.2E} at iteration {worst_iter}")
    print(f"B above A         : {positive}/{len(shared)} ({fraction:.0%})")

    failed = False
    if abs(worst_rel) > args.max_rel:
        print(f"FAIL: worst |rel| {abs(worst_rel):.2E} exceeds --max-rel {args.max_rel:.2E}")
        failed = True
    # Only meaningful with enough points to be a trend rather than noise; below
    # ~8 samples an all-positive run is unremarkable.
    if len(shared) >= 8 and fraction in (0.0, 1.0):
        print("FAIL: B is on the same side of A at EVERY iteration -- that is a bias, not chaos")
        failed = True
    if not failed:
        print("PASS: differences are unbiased and within tolerance")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

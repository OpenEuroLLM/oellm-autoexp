"""Paired loss gaps vs the suite's baseline arm at matched steps, and their trend: the gap of
each arm (same data order as the baseline) per step window, a fit gap(t) = a + b ln t over the
windows from --fit-start on, and the gap predicted at --predict-tokens. More robust than per-run
curve fits on short runs (a shared data order cancels most of the noise; no irreducible-loss
degeneracy).

    python scripts/korbi/campaign/gap_trend.py <suite log dir> [--baseline h5_moe]
        [--ref-suite <dir with the baseline, e.g. the round-5 suite of an extension>]
"""

import argparse
import glob
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

IT = re.compile(r"iteration\s+(\d+)/\s*(\d+) \|.*?lm loss: ([0-9.Ee+-]+)")


def runs(suite: Path) -> dict:
    out = defaultdict(dict)
    pat = re.compile(rf"{re.escape(suite.name)}/([a-z0-9_.-]+)/ckpt")
    for f in sorted(glob.glob(str(suite / suite.name / "slurm-*.log"))):
        text = Path(f).read_text(errors="replace")
        m = pat.search(text)
        if not m:
            continue
        for it, _, loss in IT.findall(text):
            out[m.group(1)][int(it)] = float(loss)  # later logs (restarts) win
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("suite", type=Path)
    p.add_argument("--baseline", default="h5_moe")
    p.add_argument("--ref-suite", type=Path)
    p.add_argument("--window", type=int, default=250)
    p.add_argument("--fit-start", type=int, default=1000)
    p.add_argument("--tokens-per-step", type=float, default=64 * 8192)
    p.add_argument("--predict-tokens", type=float, nargs="+", default=[4e9, 1e10])
    a = p.parse_args()
    r = runs(a.suite)
    ref = (runs(a.ref_suite) if a.ref_suite else r)[a.baseline]
    marks = [500, 1000, 1500, 2000, 3000, 4000]
    head = " | ".join(f"{m}" for m in marks)
    preds = " | ".join(f"gap@{t:.0e}" for t in a.predict_tokens)
    print(f"| arm | steps | {head} | slope/ln t | {preds} |")
    print("|---|---:|" + "---:|" * (len(marks) + 1 + len(a.predict_tokens)))
    for arm in sorted(r):
        if arm == a.baseline and not a.ref_suite:
            continue
        d = r[arm]
        common = sorted(i for i in d if i in ref)
        if not common:
            continue
        top = common[-1]
        cells = []
        for m in marks:
            w = [d[i] - ref[i] for i in common if m - a.window < i <= m]
            cells.append(f"{np.mean(w):+.4f}" if len(w) >= 5 and m <= top else "")
        # windowed gaps from fit-start on -> a + b ln t
        centers, gaps = [], []
        for lo in range(a.fit_start, top - a.window + 1, a.window):
            w = [d[i] - ref[i] for i in common if lo < i <= lo + a.window]
            if len(w) >= 5:
                centers.append(lo + a.window / 2)
                gaps.append(np.mean(w))
        if len(centers) >= 3:
            b, c = np.polyfit(np.log(centers), gaps, 1)
            slope = f"{b:+.4f}"
            pr = []
            for tok in a.predict_tokens:
                pr.append(f"{c + b * np.log(tok / a.tokens_per_step):+.4f}")
        else:
            slope, pr = "", ["" for _ in a.predict_tokens]
        print(f"| {arm} | {top} | " + " | ".join(cells) + f" | {slope} | " + " | ".join(pr) + " |")


if __name__ == "__main__":
    main()

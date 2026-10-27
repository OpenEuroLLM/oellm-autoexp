"""Status of the GQA rematch (screen5resgqa_s{1,2}): paired loss gaps of LDM vs mHC-4 (and each vs
plain) at matched steps, and their extrapolation, as gap_trend.py does it (same data order, a fit
gap(t) = a + b ln t over 100-step windows from step 1000 on -- after the 1000-step warmup -- and
the gap predicted at the end of the run (step 4000, 2.1B tokens) and at 10B tokens). A fit needs
3 windows past 1000, i.e. step >= 1300; earlier only the measured gaps are shown.

    python scripts/korbi/campaign/rematch_report.py [<campaign_3b dir>]
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gap_trend import runs  # noqa: E402

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/gpfs/scratch/ehpc533/hybrid_nucleus/campaign_3b")
PAIRS = [("ldm", "mhc4"), ("ldm_lr2", "mhc4"), ("ldm", "ldm_lr2"), ("mhc4", "plain"),
         ("ldm", "plain"), ("mhc4_lr0125", "mhc4")]
WINDOW, FIT_START, STEP_TOKENS = 100, 1000, 64 * 8192


def pair(d, ref):
    common = sorted(i for i in d if i in ref)
    if not common:
        return None
    top = common[-1]
    last = [d[i] - ref[i] for i in common if i > top - WINDOW]
    centers, gaps = [], []
    for lo in range(FIT_START, top - WINDOW + 1, WINDOW):
        w = [d[i] - ref[i] for i in common if lo < i <= lo + WINDOW]
        if len(w) >= 5:
            centers.append(lo + WINDOW / 2)
            gaps.append(np.mean(w))
    fit = None
    if len(centers) >= 3:
        b, c = np.polyfit(np.log(centers), gaps, 1)
        fit = (b, c + b * np.log(4000), c + b * np.log(1e10 / STEP_TOKENS))
    return top, float(np.mean(last)), fit


def main():
    print("| size | pair | common steps | gap (last 100) | slope / ln t | gap @ step 4000 | gap @ 10B |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for size in ("s1", "s2"):
        r = runs(ROOT / f"screen5resgqa_{size}")
        for a, b in PAIRS:
            if a not in r or b not in r:
                continue
            res = pair(r[a], r[b])
            if res is None:
                continue
            top, last, fit = res
            tail = (f"{fit[0]:+.4f} | {fit[1]:+.4f} | {fit[2]:+.4f}" if fit
                    else "| (fit from step 1300) |")
            print(f"| {size} | {a} - {b} | {top} | {last:+.4f} | {tail} |")


if __name__ == "__main__":
    main()

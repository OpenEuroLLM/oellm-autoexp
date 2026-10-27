"""Full read-out of the GQA rematch (screen5resgqa_s{1,2}): validation loss per eval, paired
train-loss gaps by phase (stable lr / WSD cooldown 3200-4000), the gap trajectory, and the long-
context set. Reads a local copy (rsync of suite/*/slurm-*.log and ds5lcresgqa*_s2/*/ckpt/
downstream.json).

The runs: 4000 steps x 64 x 8192 tokens (2.1B), WSD with a linear 800-step cooldown, one seed per
arm (seed pairs of round 4 agree at step 4000 to <= 0.0003; extrapolations past the run do not).

    python scripts/korbi/campaign/rematch_analysis.py <local copy dir>
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1])
ARMS = ["plain", "mhc4", "mhc4_lr0125", "ldm", "ldm_lr2"]
NAMES = {"plain": "plain", "mhc4": "mHC-4 x0.25", "mhc4_lr0125": "mHC-4 x0.125",
         "ldm": "LDM x1", "ldm_lr2": "LDM x2"}
IT = re.compile(r"iteration +(\d+)/ +\d+ \|.*?lm loss: ([0-9.E+-]+)")
VAL = re.compile(r"validation loss at iteration (\d+) \| lm loss value: ([0-9.E+-]+)")
COOL = 3200


def load(size):
    train, val = defaultdict(dict), defaultdict(dict)
    pat = re.compile(rf"screen5resgqa_{size}/([a-z0-9_]+)/ckpt")
    for f in sorted((ROOT / f"screen5resgqa_{size}").glob("*/slurm-*.log")):
        text = f.read_text(errors="replace")
        m = pat.search(text)
        if not m:
            continue
        for it, loss in IT.findall(text):
            train[m.group(1)][int(it)] = float(loss)
        for it, loss in VAL.findall(text):
            val[m.group(1)][int(it)] = float(loss)
    return train, val


def gap(train, a, b, lo, hi):
    d = [train[a][i] - train[b][i] for i in train[a] if lo < i <= hi and i in train[b]]
    return np.mean(d), np.std(d) / np.sqrt(len(d)), len(d)


def main():
    for size in ("s1", "s2"):
        train, val = load(size)
        print(f"\n## {size}: validation loss (20 x 64 x 8192 tokens per eval)\n")
        evals = sorted({i for a in ARMS for i in val[a]})
        print("| arm | " + " | ".join(str(i) for i in evals) + " | vs mHC-4 @4000 |")
        print("|---|" + "---:|" * (len(evals) + 1))
        for a in ARMS:
            ref = val["mhc4"].get(4000)
            d = f"{val[a][4000] - ref:+.4f}" if 4000 in val[a] and ref else ""
            print(f"| {NAMES[a]} | " + " | ".join(f"{val[a].get(i, float('nan')):.4f}"
                                                  for i in evals) + f" | {d} |")
        print(f"\n## {size}: paired train-loss gap vs mHC-4 x0.25 (mean over the window +- SE "
              "of the per-log differences)\n")
        wins = [(500, 1000), (1000, 2000), (2000, 3000), (3000, COOL), (COOL, 3600),
                (3600, 3900), (3900, 4000)]
        print("| arm | " + " | ".join(f"{lo}-{hi}" for lo, hi in wins) + " |")
        print("|---|" + "---:|" * len(wins))
        for a in ARMS:
            if a == "mhc4":
                continue
            cells = []
            for lo, hi in wins:
                m, se, n = gap(train, a, "mhc4", lo, hi)
                cells.append(f"{m:+.4f} ({se:.4f})")
            print(f"| {NAMES[a]} | " + " | ".join(cells) + " |")
        # the change through the cooldown: gap at its end minus gap just before it
        print()
        for a in ("ldm", "ldm_lr2", "plain", "mhc4_lr0125"):
            pre = gap(train, a, "mhc4", 3000, COOL)[0]
            end = gap(train, a, "mhc4", 3900, 4000)[0]
            print(f"- {NAMES[a]} - mHC-4: {pre:+.4f} before the cooldown -> {end:+.4f} at its end "
                  f"(cooldown {end - pre:+.4f})")

    print("\n## s2 long context (plain eval; eval-time log-n in brackets)\n")
    lc = {}
    for a in ARMS:
        p = ROOT / "ds5lcresgqa_s2" / a / "ckpt" / "downstream.json"
        g = ROOT / "ds5lcresgqalogn_s2" / a / "ckpt" / "downstream.json"
        lc[a] = (json.loads(p.read_text()) if p.exists() else {},
                 json.loads(g.read_text()) if g.exists() else {})
    keys = sorted({k for a in ARMS for k in lc[a][0]})
    print("keys:", ", ".join(keys))
    # retrieval aggregates: mean accuracy over lengths 16k-64k (150 questions per arm and set)
    print("\n| arm | NLL 4-8k | 16-32k | 32-64k | kvret64 mean 16-64k | kvret64 mean 16-64k log-n |"
          " kvret16 64k | kvret16 64k log-n |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for a in ARMS:
        p, g = lc[a]
        lp = p.get("longppl_64k", {})
        kv = [p.get(f"kvret64_{n}", {}).get("acc") for n in ("16k", "32k", "64k")]
        kvg = [g.get(f"kvret64_{n}", {}).get("acc") for n in ("16k", "32k", "64k")]
        f = lambda xs: f"{np.mean(xs):.3f}" if all(x is not None for x in xs) else ""  # noqa: E731
        print(f"| {NAMES[a]} | {lp.get('nll_4096-8192', float('nan')):.3f} | "
              f"{lp.get('nll_16384-32768', float('nan')):.3f} | "
              f"{lp.get('nll_32768-65536', float('nan')):.3f} | {f(kv)} | {f(kvg)} | "
              f"{p.get('kvret_64k', {}).get('acc', float('nan')):.2f} | "
              f"{g.get('kvret_64k', {}).get('acc', float('nan')):.2f} |")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prompt-identity check: our greedy outputs (math_sample.py --check) vs the release run's AIME24 outputs.

    python3 check.py DIR
Greedy outputs only drift apart late (batch-dependent bf16 noise, median ~2800 chars), so identical
prompts give identical openings; a different prompt or tokenization diverges within the first tokens.
The release file is the one the regenerated results CSV cites for AIME24.
"""

import csv
import glob
import json
import sys

R = "/e/scratch/e-sta-openeurollm/poeppel1/flag-evals/results"


def common(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


for f in sorted(glob.glob(f"{sys.argv[1]}/*.check.jsonl")):
    rows = [json.loads(line) for line in open(f)]
    ck = rows[0]["ck"]
    src = [
        r["results_file"]
        for r in csv.DictReader(open(f"{R}/{ck}.flag-evals-438.tasks.csv"))
        if r["task"] == "AIME24" and r["metric"] == "accuracy_avg"
    ][0]
    ex = json.load(open(src))["results"]["AIME24"]["examples"]
    pre, ok = [], 0
    for r in rows:
        ours = r["samples"][0]["text"]
        best = max(ex[r["idx"]]["model_outputs"], key=lambda o: common(ours, o))
        n = common(ours, best)
        pre.append(n)
        ok += n >= min(
            300, len(ours), len(best)
        )  # short answers ("100") count when fully identical
    print(
        f"{ck:24s} common prefix with release outputs (chars): {pre}  -> {ok}/{len(pre)} match "
        f"{'OK' if ok >= len(pre) - 1 else 'PROMPT MISMATCH?'}"
    )

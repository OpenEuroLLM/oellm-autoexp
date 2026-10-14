#!/usr/bin/env python3
"""Grade math_sample.py outputs (runs inside oellm-eval-vllm.sif for
Evalchemy's extraction + is_equiv).

    python3 grade.py DIR            -> DIR/summary.csv, DIR/per_problem.csv, DIR/check.txt
Per checkpoint x task:
  acc_grader   Evalchemy's rule (last \\boxed{} only, hendrycks is_equiv) = sampled pass@1
  acc_lenient  boxed -> "final answer/answer is" phrase -> last number (lenient_rescore.py), numeric compare
  pass@4 / pass@16  unbiased estimator over each problem's n samples (lenient)
  capped       finish_reason == length (hit the 4096-token context)
  loop         capped AND distinct-20-gram ratio of the last 1000 tokens < 0.5
  acc_finished lenient accuracy on samples that stopped on their own
  ci95         bootstrap over problems for acc_lenient
"""

import csv
import glob
import json
import os
import sys
from math import comb

import numpy as np

sys.path.insert(0, "/opt/evalchemy")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # lenient_rescore.py
from lm_eval.tasks.hendrycks_math.utils import is_equiv, last_boxed_only_string, remove_boxed  # noqa: E402
from lenient_rescore import extract, to_number  # noqa: E402

try:
    from eval.utils.parsers import extract_math_answer  # noqa: E402
except Exception:
    extract_math_answer = None


def grader_answer(task, text):
    try:
        if task == "MATH500" and extract_math_answer is not None:
            return extract_math_answer(text)
        return remove_boxed(last_boxed_only_string(text))
    except Exception:
        return ""


def lenient_ok(ans, text):
    v, _ = extract(text)
    a, b = to_number(str(ans)), (to_number(v) if v is not None else None)
    if a is not None and b is not None:
        return abs(a - b) < 1e-6
    return v is not None and v.strip() == str(ans).strip()


def pass_at(n, c, k):
    return 1.0 if n - c < k else 1 - comb(n - c, k) / comb(n, k)


def main():
    d = sys.argv[1]
    probs = []
    for f in sorted(glob.glob(f"{d}/*.shard*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            S = r["samples"]
            g = [is_equiv(str(r["answer"]), grader_answer(r["task"], s["text"])) for s in S]
            ok = [bool(x) or lenient_ok(r["answer"], s["text"]) for x, s in zip(g, S)]
            cap = [s["finish"] == "length" for s in S]
            loop = [c and s["distinct20"] < 0.5 for c, s in zip(cap, S)]
            probs.append(
                dict(
                    ck=r["ck"],
                    task=r["task"],
                    idx=r["idx"],
                    n=len(S),
                    c_grader=sum(g),
                    c_lenient=sum(ok),
                    capped=sum(cap),
                    loop=sum(loop),
                    fin=sum(not c for c in cap),
                    fin_ok=sum(x and not c for x, c in zip(ok, cap)),
                    mean_tokens=float(np.mean([s["n_tokens"] for s in S])),
                )
            )
    with open(f"{d}/per_problem.csv", "w") as f:
        w = csv.DictWriter(f, fieldnames=list(probs[0]))
        w.writeheader()
        w.writerows(probs)
    rng = np.random.default_rng(0)
    rows = []
    for ck in sorted({p["ck"] for p in probs}):
        for t in ["AIME24", "AIME25", "AMC23", "MATH500"]:
            P = [p for p in probs if p["ck"] == ck and p["task"] == t]
            if not P:
                continue
            N = sum(p["n"] for p in P)
            pl = np.array([p["c_lenient"] / p["n"] for p in P])
            boot = [pl[rng.integers(0, len(pl), len(pl))].mean() for _ in range(2000)]
            fin = sum(p["fin"] for p in P)
            rows.append(
                dict(
                    checkpoint=ck,
                    task=t,
                    problems=len(P),
                    samples_per_problem=P[0]["n"],
                    acc_grader=round(sum(p["c_grader"] for p in P) / N, 4),
                    acc_lenient=round(float(pl.mean()), 4),
                    ci95_lo=round(float(np.percentile(boot, 2.5)), 4),
                    ci95_hi=round(float(np.percentile(boot, 97.5)), 4),
                    pass_at_4=round(
                        float(np.mean([pass_at(p["n"], p["c_lenient"], 4) for p in P])), 4
                    )
                    if P[0]["n"] >= 4
                    else "",
                    pass_at_16=round(
                        float(np.mean([pass_at(p["n"], p["c_lenient"], 16) for p in P])), 4
                    )
                    if P[0]["n"] >= 16
                    else "",
                    problems_solved_ever=sum(p["c_lenient"] > 0 for p in P),
                    capped=round(sum(p["capped"] for p in P) / N, 4),
                    loop=round(sum(p["loop"] for p in P) / N, 4),
                    acc_finished=round(sum(p["fin_ok"] for p in P) / fin, 4) if fin else "",
                    mean_tokens=round(float(np.mean([p["mean_tokens"] for p in P])), 1),
                )
            )
    with open(f"{d}/summary.csv", "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()

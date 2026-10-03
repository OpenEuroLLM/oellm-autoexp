#!/usr/bin/env python3
"""5. Reasoning evals (vLLM DP4, oellm-eval feat/vllm-data-parallel): GSM8K 4-shot, MBPP 3-shot, MATH500 0-shot.
Reads every reasoning_vllm/*/**/results/**/results_*.json; the newest result per (model, task) wins.
python3 table_reasoning.py [--md]"""

import glob
import json
import math
import os
import re
import sys

D = os.environ.get(
    "EVAL_WORK_ROOT", os.path.dirname(os.path.realpath(__file__))
)  # exports, caches and runs
HERE = os.path.dirname(os.path.realpath(__file__))  # the scripts themselves
# (task, metric key in results[task], label)
COLS = [
    ("gsm8k", "exact_match,strict-match", "gsm8k strict"),
    ("gsm8k", "exact_match,flexible-extract", "gsm8k flex"),
    ("mbpp", "pass_at_1,none", "mbpp pass@1"),
    ("MATH500", "accuracy", "MATH500"),
    ("HumanEval", "python_pass@1", "HumanEval py"),
    ("GPQADiamond", "accuracy_avg", "GPQA-Diamond"),
]
N_EVALCHEMY = {"MATH500": 500, "HumanEval": 164, "GPQADiamond": 198}  # for binomial se

ROOT = os.environ.get("REASONING_ROOT", f"{D}/reasoning_vllm")  # e.g. a BOS A/B run tree
res = {}  # (label, model) -> (value, se, mtime)
for f in glob.glob(f"{ROOT}/*/**/results/**/*.json", recursive=True):
    d = json.load(open(f))
    # lm-eval writes results/<hash>_<time>.json with the model in model_name;
    # evalchemy writes results/<hash>/<..identity__model>/results_<time>.json.
    m = re.search(r"identity__([^/]+)/", f)
    model = m.group(1) if m else os.path.basename(str(d.get("model_name", "")).rstrip("/"))
    if not model:
        continue
    mtime, r = os.path.getmtime(f), d.get("results", {})
    for task, key, label in COLS:
        if task not in r or key not in r[task]:
            continue
        v = r[task][key]
        if v > 1:  # some evalchemy tasks report percent
            v /= 100
        if task in N_EVALCHEMY:  # evalchemy reports no stderr: binomial se
            se = math.sqrt(v * (1 - v) / r[task].get("num_total", N_EVALCHEMY[task]))
        else:
            name, _, filt = key.partition(",")
            se = r[task].get(f"{name}_stderr,{filt}", float("nan"))
        if (label, model) not in res or res[label, model][2] < mtime:
            res[label, model] = (v, se, mtime)

# HumanEval whose Bash grading hung (evalchemy's outer deadline) writes no results json, but its
# Python score is already in humaneval_artifacts/*/scores_python.json. Recover it: the traceback in
# the task's .err names the results/<hash>/ folder, and that task's last "Model:" line is the model.
for sp in glob.glob(f"{ROOT}/*/*/results/*/humaneval_artifacts/*/scores_python.json"):
    hdir = sp.split("/humaneval_artifacts/")[0]
    h, run = os.path.basename(hdir), os.path.dirname(os.path.dirname(hdir))
    errs = [
        e
        for e in glob.glob(f"{run}/slurm_logs/*.err")
        if f"results/{h}/" in open(e, errors="ignore").read()
    ]
    if len(errs) != 1:
        continue
    models_in_task = re.findall(
        r"Model: \S*/identity/(\S+)", open(errs[0][:-4] + ".out", errors="ignore").read()
    )
    if not models_in_task:
        continue
    model, v = models_in_task[-1], json.load(open(sp))["pass@1"]
    if ("HumanEval py", model) not in res:
        res["HumanEval py", model] = (v, math.sqrt(v * (1 - v) / 164), os.path.getmtime(sp))

order = {run: i for i, run in enumerate(["control", "v1", "v2", "fork", "b1"])}  # others follow
models = sorted(
    {m for _, m in res},
    key=lambda m: (
        order.get(m.rpartition("_")[0], len(order)),
        m.rpartition("_")[0],
        int(re.sub(r"\D", "", m.rpartition("_")[2]) or 0),
    ),
)
labels = [c[2] for c in COLS if any((c[2], m) in res for m in models)]


def main():
    md = "--md" in sys.argv

    def cell(label, model):
        return f"{res[label, model][0]:.3f}" if (label, model) in res else "-"

    def se(label):  # mean standard error over the models that have this task
        have = [m for m in models if (label, m) in res]
        return sum(res[label, m][1] for m in have) / max(1, len(have))

    if md:
        print("| task | ± se | " + " | ".join(models) + " |")
        print("|---|---:|" + "---:|" * len(models))
        for label in labels:
            vals = [cell(label, m) for m in models]
            best = max((v for v in vals if v != "-"), default=None)
            print(
                f"| {label} | {se(label):.3f} | "
                + " | ".join(f"**{v}**" if v == best else v for v in vals)
                + " |"
            )
    else:
        print(f"{'task':14} {'± se':>6} " + " ".join(f"{m:>12}" for m in models))
        for label in labels:
            print(
                f"{label:14} {se(label):6.3f} " + " ".join(f"{cell(label, m):>12}" for m in models)
            )
    missing = [f"{m}:{label}" for label in labels for m in models if (label, m) not in res]
    if missing:
        print(f"\nmissing: {', '.join(missing)}")


if __name__ == "__main__":
    main()

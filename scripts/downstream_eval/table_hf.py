#!/usr/bin/env python3
"""3. Results table: one row per task, one column per model in hf/ (acc_norm where reported, else acc).
   python3 table_hf.py        # terminal
   python3 table_hf.py --md   # markdown, best per row in bold
Reads every oellm-eval run under oellm_shared_evals/$USER; the newest result per (task, model) wins."""

import glob
import json
import os
import re
import sys

D = os.environ.get(
    "EVAL_WORK_ROOT", os.path.dirname(os.path.realpath(__file__))
)  # exports, caches and runs
HERE = os.path.dirname(os.path.realpath(__file__))  # the scripts themselves
EXPORTS = os.environ.get("EVAL_EXPORT_ROOT", f"{D}/hf")
RUNS = os.environ.get(
    "EVAL_SHARED_RUNS", "/e/data1/datasets/playground/mmlaion/shared/oellm_shared_evals"
)
RUNS = f"{RUNS}/{os.environ['USER']}"
TASKS = "arc_challenge arc_easy boolq commonsense_qa copa hellaswag lambada_openai mmlu openbookqa piqa social_iqa winogrande".split()
GROUPS = [
    "control",
    "v1",
    "v2",
    "fork",
    "b1",
]  # preferred column order; others follow alphabetically


def order(name):  # <run>_<iteration>: v1_64k, v2_40k, fork_86k, b1_110k, ...
    run, _, it = name.rpartition("_")
    return (
        GROUPS.index(run) if run in GROUPS else len(GROUPS),
        run,
        int(re.sub(r"\D", "", it) or 0),
    )


res = {}  # (task, model) -> (value, stderr, metric, n-shot)
for f in sorted(glob.glob(f"{RUNS}/*/results/**/*.json", recursive=True)):  # sorted = chronological
    r = json.load(open(f))
    ma = r["config"]["model_args"]
    ma = ma["pretrained"] if isinstance(ma, dict) else re.search(r"pretrained=([^,]+)", ma).group(1)
    if os.path.dirname(ma.rstrip("/")) != EXPORTS:  # e.g. a BOS A/B tree reuses the model names
        continue
    m = os.path.basename(ma.rstrip("/"))
    for t in TASKS:
        if t in r["results"]:
            v = r["results"][t]
            k = "acc_norm" if "acc_norm,none" in v else "acc"
            n = r["n-shot"].get(
                t, next(iter(r["n-shot"].values()))
            )  # mmlu: n-shot only listed per subtask
            res[t, m] = (v[f"{k},none"], v[f"{k}_stderr,none"], k, n)

models = sorted({m for _, m in res if os.path.isdir(f"{EXPORTS}/{m}")}, key=order)
rows = [t for t in TASKS if any((t, m) in res for m in models)]
full = [t for t in rows if all((t, m) in res for m in models)]
val = {(t, m): res[t, m][0] for t in rows for m in models if (t, m) in res}
for m in models:
    if full:
        val["MEAN", m] = sum(val[t, m] for t in full) / len(full)
info = {t: next(res[t, m] for m in models if (t, m) in res) for t in rows}
se = {
    t: sum(res[t, m][1] for m in models if (t, m) in res) / sum((t, m) in res for m in models)
    for t in rows
}
labels = [(t, t, str(info[t][3]), info[t][2], f"{se[t]:.3f}") for t in rows]
labels += [("MEAN", f"mean ({len(full)} tasks)", "", "", "")] if full else []


def main():
    md = "--md" in sys.argv
    if md:
        print("| task | shots | metric | ± se | " + " | ".join(models) + " |")
        print("|---|---:|---|---:|" + "---:|" * len(models))
    else:
        print(
            f"{'task':20}{'shots':>6}{'metric':>10}{'± se':>7}"
            + "".join(f"{m:>13}" for m in models)
        )
    for key, name, n, k, s in labels:
        best = f"{max(val[key, m] for m in models if (key, m) in val):.3f}"  # ties at display precision all bold
        cells = []
        for m in models:
            c = f"{val[key, m]:.3f}" if (key, m) in val else "-"
            cells.append(f"**{c}**" if md and c == best else c)
        if md:
            print(f"| {name} | {n} | {k} | {s} | " + " | ".join(cells) + " |")
        else:
            print(f"{name:20}{n:>6}{k:>10}{s:>7}" + "".join(f"{c:>13}" for c in cells))


if __name__ == "__main__":  # 4_plot.py loads the parsed results from here
    main()

#!/usr/bin/env python3
"""Lenient re-scoring of flag-evals math generations, no GPU.
Tasks: lm-eval sample files (polymath_*, default polymath_en_low) and Evalchemy results (AIME24,
AIME25, AMC23: all repeats of every problem are scored).

Why: the flag-evals grader for these tasks credits ONLY answers written as \\boxed{...}. An answer
like "Final Answer: Janet makes $18 every day" scores 0 even when 18 is correct. Base models drift
between a long reasoning style that ends in \\boxed{} and a short "Step 1 ... Final Answer:" style,
so the grader score mixes answer FORMAT with math CORRECTNESS. This script separates the two.

Each generation is first CUT at a newly started "Question:" / "Problem:" header (base models keep
going with invented follow-up problems). Extraction order on the rest (first hit wins):
  1. boxed   the last \\boxed{...}; the last number inside it (else its raw content)
  2. phrase  the first number after "final answer" / "the answer is" / "answer:" on that line
  3. last    the last number anywhere in the generation
Numbers are compared after stripping "," "$" "%" and a trailing "."; fractions a/b are evaluated;
correct = |extracted - target| < 1e-6. The lenient extraction can also be wrong (e.g. "last number"
picks an intermediate result) -- check the per-sample CSV.

Input:  lm-eval:     <runs>/<ckpt>/vllm/*/results/samples_<task>_*.jsonl            (newest by mtime)
        Evalchemy:   <runs>/<ckpt>/vllm/*/results/*/*/results_*.json holding <task>  (newest by mtime)
                     grader-correct there = Evalchemy's own extracted model_answer == expected_answer
Output: <out>/<task>_summary.csv   one row per checkpoint: n, grader acc, lenient acc, boxed count,
                                   correct/total per extraction source
        <out>/<task>_samples.csv   one row per generation: checkpoint, doc_id, target, grader score,
                                   extracted answer, source, lenient correct, generation tail

    python3 lenient_rescore.py [--task polymath_en_low] [--runs <flag-evals>/runs] [--out .] [CKPT ...]
    (no CKPT: every checkpoint folder under --runs that has samples for the task)
"""

import argparse
import csv
import glob
import json
import os
import re

NUM = r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?"


def response(r):
    x = r.get("resps") or r.get("filtered_resps")
    while isinstance(x, list) and x:
        x = x[0]
    return x if isinstance(x, str) else ""


def to_number(s):
    s = s.replace(",", "").replace("$", "").replace("\\%", "").replace("%", "").strip().rstrip(".")
    try:
        if "/" in s:
            a, b = s.split("/")
            return float(a) / float(b)
        return float(s)
    except Exception:
        return None


EVALCHEMY = {"AIME24", "AIME25", "AMC23"}
NEXT_Q = re.compile(r"\n\s*(?:#+\s*)?(?:Question|Problem)\s*\d*\s*[:.]", re.I)


def truncate(g):
    m = NEXT_Q.search(g, 1)
    return g[: m.start()] if m else g


def items(task, runs, ck):
    """Yield (doc_id, target, grader_correct, generation) for one checkpoint,
    plus the source file."""
    if task in EVALCHEMY:
        fs = [
            f
            for f in sorted(
                glob.glob(f"{runs}/{ck}/vllm/*/results/*/*/results_*.json"), key=os.path.getmtime
            )
            if f'"{task}": {{' in open(f).read()
        ]
        if not fs:
            return None, []
        ex = json.load(open(fs[-1]))["results"][task]["examples"]
        out = []
        for e in ex:
            tgt = e.get(
                "expected_answer", e.get("answer")
            )  # AIME24: expected_answer; AIME25/AMC23: answer
            ans = e.get("model_answers") or [None] * len(e["model_outputs"])
            for k, (g, ma) in enumerate(zip(e["model_outputs"], ans)):
                t, v = to_number(str(tgt)), (to_number(str(ma)) if ma not in (None, "") else None)
                out.append(
                    (
                        f"{e['id']}#{k}",
                        tgt,
                        int(v is not None and t is not None and abs(v - t) < 1e-6),
                        g,
                    )
                )
        return fs[-1], out
    fs = sorted(
        glob.glob(f"{runs}/{ck}/vllm/*/results/samples_{task}_20*.jsonl"), key=os.path.getmtime
    )
    if not fs:
        return None, []
    rows = [json.loads(line) for line in open(fs[-1])]
    return fs[-1], [
        (r.get("doc_id"), r["target"], r.get("exact_match") or 0, response(r)) for r in rows
    ]


def extract(g):
    g = truncate(g)
    m = re.findall(r"\\boxed\{([^{}]*)\}", g)
    if m:
        n = re.findall(NUM, m[-1])
        return (n[-1] if n else m[-1]), "boxed"
    m = re.search(r"(?:final answer|the answer is|answer:)[^\n]*?(" + NUM + ")", g, re.I)
    if m:
        return m.group(1), "phrase"
    n = re.findall(NUM, g)
    return (n[-1], "last") if n else (None, "none")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("checkpoints", nargs="*")
    ap.add_argument("--task", default="polymath_en_low")
    ap.add_argument("--runs", default="/e/scratch/e-sta-openeurollm/poeppel1/flag-evals/runs")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    cks = a.checkpoints or sorted(
        c for c in os.listdir(a.runs) if os.path.isdir(f"{a.runs}/{c}/vllm")
    )
    os.makedirs(a.out, exist_ok=True)
    srcs = ["boxed", "phrase", "last", "none"]
    summ, samp = [], []
    for c in cks:
        src_file, rows = items(a.task, a.runs, c)
        if not rows:
            if a.checkpoints:
                print(f"{c}: no {a.task} outputs")
            continue
        st = {s: [0, 0] for s in srcs}
        lenient = grader = 0
        for doc, target, gr, g in rows:
            ans, src = extract(g)
            t = to_number(str(target))
            v = to_number(ans) if ans is not None else None
            ok = v is not None and t is not None and abs(v - t) < 1e-6
            lenient += ok
            grader += gr
            st[src][0] += ok
            st[src][1] += 1
            samp.append(
                dict(
                    checkpoint=c,
                    doc_id=doc,
                    target=target,
                    grader=gr,
                    extracted=ans,
                    source=src,
                    lenient_correct=int(ok),
                    generation_tail=truncate(g)[-300:].replace("\n", "\\n"),
                )
            )
        n = len(rows)
        summ.append(
            dict(
                checkpoint=c,
                n=n,
                grader_acc=round(grader / n, 4),
                lenient_acc=round(lenient / n, 4),
                boxed=st["boxed"][1],
                samples_file=src_file,
                **{f"{s}_correct": st[s][0] for s in srcs},
                **{f"{s}_total": st[s][1] for s in srcs},
            )
        )
        print(
            f"{c:24s} n={n} grader={grader / n:.3f} lenient={lenient / n:.3f} boxed={st['boxed'][1]} "
            + " ".join(f"{s}={st[s][0]}/{st[s][1]}" for s in srcs[:3])
        )
    for name, rows in ((f"{a.task}_summary.csv", summ), (f"{a.task}_samples.csv", samp)):
        if rows:
            with open(os.path.join(a.out, name), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            print("wrote", os.path.join(a.out, name))


if __name__ == "__main__":
    main()

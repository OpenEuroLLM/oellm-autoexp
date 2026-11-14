#!/usr/bin/env python3
"""Before / after table of the 1.7B fine-tuning recoverability test (docs/recall_surgery_ambitious.md §16): CoQA F1 /
EM and Winogrande acc from the ladder17b_eval result files, SQuAD v2 re-scored UNIFORMLY from the logged samples
(squad_uniform.py logic: abstain = empty output or containing "unanswerable"; HasAns F1 with abstain = 0, abstention
on answerable, F1 when answering, NoAnsAcc = abstention on unanswerable, discrimination = NoAnsAcc - abst@ans, total
F1). The ladder evaluator logs doc_id / response / one target per sample, so the gold answers come from the squad_v2
validation split (offline HF cache), joined by doc_id; the join is checked against the logged target.

    python3 ft17b_table.py <ft17b_evals root> [names ...]     (default: the six §16 readouts; needs `datasets`)
"""

from __future__ import annotations

import collections
import glob
import json
import os
import re
import string
import sys

NAMES = ["base_hyb", "hyb_gen", "hyb_rc", "base_attn", "attn_gen", "attn_rc"]


def norm(s):
    s = s.lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(p, g):
    p, g = norm(p).split(), norm(g).split()
    if not p or not g:
        return float(p == g)
    c = sum((collections.Counter(p) & collections.Counter(g)).values())
    if c == 0:
        return 0.0
    pr, rc = c / len(p), c / len(g)
    return 2 * pr * rc / (pr + rc)


def gen(x):
    r = x["filtered_resps"]
    g = r[0] if isinstance(r, list) else r
    while isinstance(g, list) and g:
        g = g[0]
    return g if isinstance(g, str) else str(g)


def ab(g):
    return g.strip() == "" or "unanswerable" in g.lower()


def metric(path, *keys):
    try:
        r = json.load(open(path))["results"]
    except (OSError, KeyError, ValueError):
        return None
    for k in keys:
        for rk, v in r.items():
            if rk.split(",")[0] == k and isinstance(v, (int, float)):
                return 100 * v
    return None


def squad(out, val):
    H, AN, NA, T, A, bad, n = [], [], [], [], 0, 0, 0
    for f in sorted(glob.glob(os.path.join(out, "squadv2*.samples.jsonl"))):
        for line in open(f):
            x = json.loads(line)
            d = val[int(x["doc_id"])]
            golds = d["answers"]["text"]
            tgt = str(x.get("target") or "").strip().lower()
            if tgt and tgt != (golds[0].strip().lower() if golds else "unanswerable"):
                bad += 1
            n += 1
            g = gen(x)
            if golds:
                v = 0.0 if ab(g) else max(f1(g, gg) for gg in golds)
                H.append(v)
                T.append(v)
                if ab(g):
                    A += 1
                else:
                    AN.append(v)
            else:
                v = 1.0 if ab(g) else 0.0
                NA.append(v)
                T.append(v)
    if not n:
        return None
    mean = lambda v: 100 * sum(v) / max(1, len(v))
    abst = 100 * A / max(1, len(H))
    return dict(n=n, join_mismatch=bad, hasans=mean(H), abst=abst, f1_ans=mean(AN), noans=mean(NA),
                disc=mean(NA) - abst, total=mean(T))


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    root = argv[0]
    names = argv[1:] or NAMES
    from datasets import load_dataset

    val = load_dataset("squad_v2", split="validation")
    print(f"{'model':10s} | {'CoQA F1':>7s} {'EM':>5s} | {'Wino':>5s} | SQuAD v2 uniform: {'HasAnsF1':>8s} {'abst@ans':>8s} "
          f"{'F1|answ':>7s} {'NoAnsAcc':>8s} {'disc':>6s} {'totalF1':>7s} (n, join mismatches)")
    for m in names:
        out = os.path.join(root, m)
        coqa_f1 = metric(os.path.join(out, "coqa.json"), "f1")
        coqa_em = metric(os.path.join(out, "coqa.json"), "em")
        wino = metric(os.path.join(out, "winogrande.json"), "acc")
        s = squad(out, val)
        fmt = lambda v, w=5: f"{v:{w}.1f}" if v is not None else f"{'-':>{w}s}"
        sq = (f"{s['hasans']:8.1f} {s['abst']:7.1f}% {s['f1_ans']:7.1f} {s['noans']:7.1f}% {s['disc']:6.1f} "
              f"{s['total']:7.1f} ({s['n']}, {s['join_mismatch']})") if s else "-"
        print(f"{m:10s} | {fmt(coqa_f1, 7)} {fmt(coqa_em)} | {fmt(wino)} | {'':17s} {sq}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

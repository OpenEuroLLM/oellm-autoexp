#!/usr/bin/env python3
"""lm-eval samples_*.jsonl -> one compact record per test item (stdlib only: runs on any login node).

    python compact_samples.py --task squadv2 --out x.jsonl.gz samples_squadv2_part0of8.jsonl ...

Per item: key (stable doc id), score (the row's metric: acc_norm for belebele / hellaswag / piqa / openbookqa, acc
for boolq / winogrande / wsc273 / social_iqa, exact (+ f1) for squadv2 recomputed from the prediction with the
SQuAD-v2 normalization (lm-eval's per-sample entries hold the prediction, not the score; abstention never fires at
its default threshold), em / f1 for coqa), and align: the normalized texts used to find the item among the
decontamination references (passage / question fields)."""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
import string
import sys

_PUNCT = set(string.punctuation)


def squad_norm(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def squad_em_f1(pred: str, golds: list[str]) -> tuple[float, float]:
    golds = [g for g in golds if squad_norm(g)] or [""]
    p = squad_norm(pred)
    em = max(float(p == squad_norm(g)) for g in golds)
    f1 = 0.0
    for g in golds:
        gt, pt = squad_norm(g).split(), p.split()
        if not gt or not pt:
            f1 = max(f1, float(gt == pt))
            continue
        common = collections.Counter(gt) & collections.Counter(pt)
        same = sum(common.values())
        if same:
            pr, rc = same / len(pt), same / len(gt)
            f1 = max(f1, 2 * pr * rc / (pr + rc))
    return em, f1


def tnorm(s) -> str:
    """Alignment normalization: lowercase alphanumerics, single spaces."""
    return " ".join(re.sub(r"[^0-9a-z]+", " ", str(s).lower()).split())


def _lit(v):
    if isinstance(v, str) and v[:1] in "[{":
        import ast
        try:
            return ast.literal_eval(v)
        except Exception:
            return v
    return v


def record(task: str, r: dict) -> dict:
    d = {k: _lit(v) for k, v in r["doc"].items()}
    f = lambda k: float(r[k])  # noqa: E731
    if task == "squadv2":
        pred = r["exact"][0]["prediction_text"] if isinstance(r["exact"], list) else _lit(r["exact"])[0]["prediction_text"]
        golds = list(d["answers"]["text"])
        em, f1 = squad_em_f1(pred, golds)
        return {"key": d["id"], "score": em, "f1": f1, "has_ans": bool(golds),
                "align": {"passage": tnorm(d["context"]), "question": tnorm(d["question"])}}
    if task == "coqa":
        return {"key": d["id"], "score": f("em"), "f1": f("f1"), "align": {"passage": tnorm(d["story"])}}
    if task == "boolq":
        return {"key": tnorm(d["question"]) + "|" + tnorm(d["passage"])[:80], "score": f("acc"),
                "align": {"passage": tnorm(d["passage"]), "question": tnorm(d["question"])}}
    if task == "winogrande":
        return {"key": tnorm(d["sentence"]), "score": f("acc"), "align": {"question": tnorm(d["sentence"])}}
    if task == "wsc273":
        return {"key": tnorm(d["text"]), "score": f("acc"), "align": {"question": tnorm(d["text"])}}
    if task == "belebele":
        return {"key": f'{d["link"]}|{d["question_number"]}', "score": f("acc_norm"),
                "align": {"passage": tnorm(d["flores_passage"]), "question": tnorm(d["question"])}}
    if task == "hellaswag":
        return {"key": f'{d["ind"]}|{d["source_id"]}', "score": f("acc_norm"),
                "align": {"question": tnorm(d["ctx"])}}
    if task == "piqa":
        return {"key": tnorm(d["goal"]) + "|" + tnorm(d["sol1"])[:60], "score": f("acc_norm"),
                "align": {"question": tnorm(d["goal"])}}
    if task == "social_iqa":
        return {"key": tnorm(d["context"]) + "|" + tnorm(d["question"]), "score": f("acc"),
                "align": {"passage": tnorm(d["context"]), "question": tnorm(d["question"])}}
    if task == "openbookqa":
        return {"key": d["id"], "score": f("acc_norm"), "align": {"question": tnorm(d["question_stem"])}}
    raise SystemExit(f"unknown task {task}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    n = 0
    seen = set()
    with gzip.open(a.out, "wt", encoding="utf-8") as out:
        for fn in a.files:
            for line in open(fn, encoding="utf-8"):
                r = json.loads(line)
                rec = record(a.task, r)
                if rec["key"] in seen:  # a doc split across parts is one item
                    continue
                seen.add(rec["key"])
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    print(f"{a.task}: {n} items -> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

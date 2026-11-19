#!/usr/bin/env python3
"""Multi-document QA at long context: an answerable SQuAD v2 question whose passage sits at a controlled depth among
other SQuAD passages (other articles) filling the context to a target length -- "a question on a document far back".
Greedy generation with the HF export as it is (its sparse attention, config.sparse_attention, in prefill and decode),
SQuAD-normalized F1 / exact match per (length, depth). Batch size 1 (the HF attention takes no padding mask).

    multidoc_qa.py --model <hf export> --squad <dev-v2.0.json> --out <dir> [--lengths 8192 32768 65536]
                   [--depths far=0.05 mid=0.5 near=0.95 only] [--n 30] [--shots 3] [--seed 0]

``--shots K``: K worked examples (question + short answer) before the question, their passages among the documents
at random positions -- a base model then answers with a span instead of a sentence. Depth ``only``: the question's
passage (and the examples') alone, no distractors -- the ceiling without long context.

Writes <out>/samples.jsonl (prompt length, gold position, prediction, F1 / EM) and <out>/summary.json; cells already
in samples.jsonl are skipped (resumable).
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import string
import time
from pathlib import Path

INSTR = "Answer the question based on the given documents. Only give me the answer and do not output any other words."


def norm(s: str) -> str:
    s = s.lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred: str, gold: str) -> float:
    p, g = norm(pred).split(), norm(gold).split()
    if not p or not g:
        return float(p == g)
    common = sum((collections.Counter(p) & collections.Counter(g)).values())
    if common == 0:
        return 0.0
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc)


def load_squad(path: Path):
    """(questions [(article, paragraph id, question, answers)], paragraphs {id: (article, text)}): answerable only."""
    data = json.loads(path.read_text())["data"]
    qs, paras = [], {}
    for ai, art in enumerate(data):
        for pi, p in enumerate(art["paragraphs"]):
            pid = f"{ai}:{pi}"
            paras[pid] = (ai, p["context"])
            for qa in p["qas"]:
                answers = [a["text"] for a in qa.get("answers", [])]
                if answers and not qa.get("is_impossible", False):
                    qs.append((ai, pid, qa["question"], answers))
    return qs, paras


def build(tok, q, paras, length: int, depth, rng: random.Random, shots=()):
    """Prompt ids of about ``length`` tokens: distractor passages of other articles, the gold one at ``depth`` (fraction
    of the passages before it; None = the gold passage alone), the examples' passages at random positions and their
    question / answer pairs before the question. Returns (ids, gold index, number of passages)."""
    art, pid, question, _ = q
    head = f"{INSTR}\n\nThe following are given documents.\n\n"
    examples = "".join(f"Question: {sq} Answer: {sa[0]}\n\n" for _, _, sq, sa in shots)
    tail = f"\n\n{INSTR}\n\n{examples}Question: {question} Answer:"
    budget = length - len(tok(head + tail, add_special_tokens=False).input_ids) - 8
    taken = {art} | {sa for sa, _, _, _ in shots}
    others = [k for k, (a, _) in paras.items() if a not in taken]
    rng.shuffle(others)
    gold_text = paras[pid][1]
    shot_texts = [paras[spid][1] for _, spid, _, _ in shots]
    used = sum(len(tok(t, add_special_tokens=False).input_ids) + 8 for t in [gold_text] + shot_texts)
    docs = []
    for k in others if depth is not None else []:
        n = len(tok(paras[k][1], add_special_tokens=False).input_ids) + 8
        if used + n > budget:
            break
        docs.append(paras[k][1])
        used += n
    for t in shot_texts:
        docs.insert(rng.randint(0, len(docs)), t)
    gi = 0 if depth is None else min(len(docs), int(round(depth * len(docs))))
    docs.insert(gi, gold_text)
    body = "".join(f"Document {i + 1}:\n{d}\n\n" for i, d in enumerate(docs)).rstrip()
    ids = tok(head + body + tail, add_special_tokens=True).input_ids
    return ids, gi, len(docs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--squad", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=[8192, 32768, 65536])
    ap.add_argument("--depths", nargs="+", default=["far=0.05", "mid=0.5", "near=0.95"],
                    help="name=fraction, or 'only' (the gold passage alone)")
    ap.add_argument("--shots", type=int, default=0, help="worked examples before the question")
    ap.add_argument("--n", type=int, default=30, help="questions per (length, depth) cell")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    a.out.mkdir(parents=True, exist_ok=True)
    samples = a.out / "samples.jsonl"
    done = collections.Counter()
    rows = []
    if samples.exists():
        for line in samples.read_text().splitlines():
            r = json.loads(line)
            rows.append(r)
            done[(r["length"], r["depth"])] += 1
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, dtype=torch.bfloat16,
                                                 device_map="auto").eval()
    sa = getattr(model.config, "sparse_attention", None)
    print(f"[multidoc_qa] {a.model}: sparse_attention {json.dumps(sa)}", flush=True)
    qs, paras = load_squad(a.squad)
    depths = [(d, None) if d == "only" else (d.split("=")[0], float(d.split("=")[1])) for d in a.depths]
    eos = tok.eos_token_id
    with samples.open("a") as fh:
        for length in a.lengths:
            for dname, dval in depths:
                rng = random.Random(f"{a.seed}:{length}:{dname}")  # the same questions / contexts for every model
                cell = rng.sample(qs, a.n)
                for i, q in enumerate(cell):
                    if i < done[(length, dname)]:
                        continue
                    srng = random.Random(f"{a.seed}:{length}:{dname}:{i}:shots")
                    shots = []
                    while len(shots) < a.shots:  # examples from other articles than the question's and each other's
                        e = srng.choice(qs)
                        if e[0] != q[0] and all(e[0] != x[0] for x in shots):
                            shots.append(e)
                    ids, gi, ndocs = build(tok, q, paras, length, dval, random.Random(f"{a.seed}:{length}:{dname}:{i}"),
                                           shots)
                    t0 = time.time()
                    x = torch.tensor([ids], device=model.device)
                    with torch.no_grad():
                        out = model.generate(x, max_new_tokens=a.max_new_tokens, do_sample=False, eos_token_id=eos,
                                             pad_token_id=eos)
                    pred = tok.decode(out[0, len(ids):], skip_special_tokens=True).strip().split("\n")[0].strip()
                    golds = q[3]
                    r = {"length": length, "depth": dname, "i": i, "prompt_tokens": len(ids), "gold_doc": gi,
                         "docs": ndocs, "question": q[2], "golds": golds, "pred": pred,
                         "f1": max(f1(pred, g) for g in golds), "em": float(max(norm(pred) == norm(g) for g in golds)),
                         "seconds": round(time.time() - t0, 2)}
                    fh.write(json.dumps(r) + "\n")
                    fh.flush()
                    rows.append(r)
                cellrows = [r for r in rows if r["length"] == length and r["depth"] == dname]
                print(f"[multidoc_qa] length {length} depth {dname}: F1 "
                      f"{sum(r['f1'] for r in cellrows) / max(len(cellrows), 1):.3f} over {len(cellrows)}", flush=True)
    summary = {}
    for (length, dname), _ in sorted({(r["length"], r["depth"]): 1 for r in rows}.items()):
        cellrows = [r for r in rows if r["length"] == length and r["depth"] == dname]
        summary[f"{length}/{dname}"] = {"f1": sum(r["f1"] for r in cellrows) / len(cellrows),
                                        "em": sum(r["em"] for r in cellrows) / len(cellrows), "n": len(cellrows)}
    (a.out / "summary.json").write_text(json.dumps({"model": a.model, "sparse_attention": sa, "cells": summary},
                                                   indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

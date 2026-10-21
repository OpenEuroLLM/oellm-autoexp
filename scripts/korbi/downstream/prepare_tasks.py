"""Downstream tasks for the in-Megatron evaluator (megatron/training/downstream_eval.py).

Two steps (MN5 compute / login nodes have no internet; the container has the tokenizer libraries):

    # where datasets can be downloaded: raw text, lm-eval-harness prompt conventions
    python prepare_tasks.py raw  <out_dir>
    # inside the training container: token ids with the run's tokenizer + the passkey task
    python prepare_tasks.py tokenize <out_dir> <tokenizer dir>

The result <out_dir>/requests.jsonl has one line per question:
    {"task", "qid", "gold", "kind": "mc" | "greedy", "items": [{"ctx": [ids], "cont": [ids], "nchars": int}]}
Scoring: sum log p(cont | ctx) per item; "mc" -> acc (argmax) and acc_norm (per character);
"greedy" (LAMBADA) -> correct when every continuation token is the argmax.

Tasks (lm-eval-harness's prompts): HellaSwag (first 2000 of validation), ARC-Easy / -Challenge
(test), PIQA (validation), Winogrande (xl validation, partial scoring), LAMBADA (OpenAI, first 2000),
and a passkey retrieval (Mohtashami & Jaggi 2023) at 2k-32k tokens, keys scored as a 10-way choice.
"""

import json
import random
import re
import sys
from pathlib import Path


def _hellaswag_pre(text):
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def raw(out):
    from datasets import load_dataset

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    hs = load_dataset("Rowan/hellaswag", split="validation").select(range(2000))
    for i, d in enumerate(hs):
        ctx = _hellaswag_pre(d["activity_label"] + ": " + d["ctx_a"] + " " + d["ctx_b"].capitalize())
        rows.append({"task": "hellaswag", "qid": i, "gold": int(d["label"]), "kind": "mc",
                     "items": [{"ctx": ctx, "cont": " " + _hellaswag_pre(e)} for e in d["endings"]]})
    for sub, task in (("ARC-Easy", "arc_easy"), ("ARC-Challenge", "arc_challenge")):
        ds = load_dataset("allenai/ai2_arc", sub, split="test")
        for i, d in enumerate(ds):
            labels = d["choices"]["label"]
            if d["answerKey"] not in labels:
                continue
            ctx = f"Question: {d['question']}\nAnswer:"
            rows.append({"task": task, "qid": i, "gold": labels.index(d["answerKey"]), "kind": "mc",
                         "items": [{"ctx": ctx, "cont": " " + t} for t in d["choices"]["text"]]})
    pq = load_dataset("baber/piqa", split="validation")
    for i, d in enumerate(pq):
        ctx = f"Question: {d['goal']}\nAnswer:"
        rows.append({"task": "piqa", "qid": i, "gold": int(d["label"]), "kind": "mc",
                     "items": [{"ctx": ctx, "cont": " " + d["sol1"]}, {"ctx": ctx, "cont": " " + d["sol2"]}]})
    wg = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
    for i, d in enumerate(wg):
        idx = d["sentence"].index("_")
        cont = " " + d["sentence"][idx + 1:].strip()
        items = [{"ctx": d["sentence"][:idx] + opt, "cont": cont} for opt in (d["option1"], d["option2"])]
        rows.append({"task": "winogrande", "qid": i, "gold": int(d["answer"]) - 1, "kind": "mc", "items": items})
    lb = load_dataset("EleutherAI/lambada_openai", "default", split="test").select(range(2000))
    for i, d in enumerate(lb):
        text = d["text"]
        ctx, last = text.rsplit(" ", 1)
        rows.append({"task": "lambada", "qid": i, "gold": 0, "kind": "greedy",
                     "items": [{"ctx": ctx, "cont": " " + last}]})
    with open(out / "raw.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} questions -> {out / 'raw.jsonl'}")


FILLER = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "


def passkey_rows(tok, lengths=(2048, 4096, 8192, 16384, 32768), depths=(0.1, 0.5, 0.9), n=20, seed=0):
    rng = random.Random(seed)
    filler_ids = tok.encode(FILLER, add_special_tokens=False)
    rows = []
    for L in lengths:
        for depth in depths:
            for j in range(n):
                key = rng.randint(10000, 99999)
                keys = [key] + rng.sample([k for k in range(10000, 100000) if k != key], 9)
                needle = tok.encode(f" The pass key is {key}. Remember it. {key} is the pass key. ",
                                    add_special_tokens=False)
                question = tok.encode(" What is the pass key? The pass key is", add_special_tokens=False)
                body = L - len(needle) - len(question) - 8
                hay = (filler_ids * (body // len(filler_ids) + 1))[:body]
                cut = int(len(hay) * depth)
                ctx = [tok.bos_token_id] + hay[:cut] + needle + hay[cut:] + question
                items = [{"ctx": ctx, "cont": tok.encode(f" {k}", add_special_tokens=False), "nchars": 6}
                         for k in keys]
                rows.append({"task": f"passkey_{L // 1024}k", "qid": len(rows), "gold": 0, "kind": "mc",
                             "depth": depth, "items": items})
    return rows


def tokenize(out, tokdir):
    from transformers import AutoTokenizer

    out = Path(out)
    tok = AutoTokenizer.from_pretrained(tokdir)
    n = 0
    with open(out / "raw.jsonl") as f, open(out / "requests.jsonl", "w") as g:
        for line in f:
            r = json.loads(line)
            items = []
            for it in r["items"]:
                ctx_ids = tok.encode(it["ctx"])  # with BOS, as the training documents start
                whole = tok.encode(it["ctx"] + it["cont"])
                cont_ids = whole[len(ctx_ids):] if whole[:len(ctx_ids)] == ctx_ids else \
                    tok.encode(it["cont"], add_special_tokens=False)
                items.append({"ctx": ctx_ids, "cont": cont_ids, "nchars": len(it["cont"])})
            r["items"] = items
            g.write(json.dumps(r) + "\n")
            n += 1
        for r in passkey_rows(tok):
            g.write(json.dumps(r) + "\n")
            n += 1
    print(f"{n} questions -> {out / 'requests.jsonl'}")


if __name__ == "__main__":
    {"raw": raw, "tokenize": tokenize}[sys.argv[1]](*sys.argv[2:])

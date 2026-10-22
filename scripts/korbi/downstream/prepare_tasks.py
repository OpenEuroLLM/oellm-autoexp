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
a passkey retrieval (Mohtashami & Jaggi 2023) at 2k-32k tokens, keys scored as a 10-way choice, and
a multi-key retrieval in natural text (RULER-style: 16 facts, one asked; 10-way vs the other facts).
"""

import json
import random
import string
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


WORDS = ("apple river candle mirror garden pencil violin tunnel harbor lantern meadow rocket "
         "saddle thunder velvet window anchor basket castle desert falcon glacier helmet island "
         "jacket kettle ladder marble needle orchard pepper quarry ribbon silver timber umbrella").split()


def kv_rows(tok, raw_path, lengths=(2048, 4096, 8192, 16384, 32768), n=50, pairs=16, seed=1):
    """Multi-key retrieval in natural text (RULER's multi-key NIAH): `pairs` facts "The special
    number for <word> is <NNNNN>." spread evenly through natural filler (LAMBADA / HellaSwag passages),
    then asked for one; scored as a 10-way choice against the other inserted numbers."""
    rng = random.Random(seed)
    texts = []
    with open(raw_path) as f:
        for line in f:
            q = json.loads(line)
            if q["task"] in ("lambada", "hellaswag"):
                texts.append(q["items"][0]["ctx"] + q["items"][0]["cont"])
    filler = tok.encode(" ".join(texts), add_special_tokens=False)
    rows = []
    for L in lengths:
        for j in range(n):
            # more facts than words: two-word keys ("apple river")
            pool = WORDS if pairs <= len(WORDS) else [f"{a} {b}" for a in WORDS for b in WORDS if a != b]
            words = rng.sample(pool, pairs)
            vals = rng.sample(range(10000, 100000), pairs)
            facts = [tok.encode(f" The special number for {w} is {v}. ", add_special_tokens=False)
                     for w, v in zip(words, vals)]
            ask = rng.randrange(pairs)
            question = tok.encode(f" What is the special number for {words[ask]}? The special number for "
                                  f"{words[ask]} is", add_special_tokens=False)
            body = L - sum(map(len, facts)) - len(question) - 8
            start = rng.randrange(0, len(filler) - body)
            hay = filler[start:start + body]
            seg = len(hay) // pairs
            ctx = [tok.bos_token_id]
            for i, fact in enumerate(facts):
                ctx += hay[i * seg:(i + 1) * seg] + fact
            ctx += hay[pairs * seg:] + question
            others = [v for i, v in enumerate(vals) if i != ask]
            keys = [vals[ask]] + rng.sample(others, 9)
            items = [{"ctx": ctx, "cont": tok.encode(f" {k}", add_special_tokens=False), "nchars": 6} for k in keys]
            rows.append({"task": f"kvret_{L // 1024}k", "qid": len(rows), "gold": 0, "kind": "mc",
                         "asked_pos": ask / (pairs - 1), "items": items})
    return rows


def vt_rows(tok, raw_path, lengths=(4096, 8192, 16384, 32768, 65536), n=50, chains=8, hops=4, seed=2):
    """Variable tracking in natural text (RULER's VT): `chains` assignment chains "VAR A = 12345."
    "VAR B = VAR A." ... (`hops` re-assignments each), their statements interleaved (each chain in
    order) and spread evenly through filler; asked for the value of one chain's last variable,
    scored as a 10-way choice among the chains' values (+ 2 unused numbers)."""
    rng = random.Random(seed)
    filler = _filler(tok, raw_path)
    rows = []
    for L in lengths:
        for j in range(n):
            names = rng.sample(["".join(rng.choices(string.ascii_uppercase, k=5)) for _ in range(4 * chains * (hops + 1))],
                               chains * (hops + 1))
            vals = rng.sample(range(10000, 100000), chains + 2)
            chain_names = [names[c * (hops + 1):(c + 1) * (hops + 1)] for c in range(chains)]
            stmts = {c: [f" VAR {chain_names[c][0]} = {vals[c]}. "] +
                     [f" VAR {chain_names[c][h]} = VAR {chain_names[c][h - 1]}. " for h in range(1, hops + 1)]
                     for c in range(chains)}
            order = [c for c in range(chains) for _ in range(hops + 1)]
            rng.shuffle(order)
            pos = {c: 0 for c in range(chains)}
            seq = []
            for c in order:
                seq.append(tok.encode(stmts[c][pos[c]], add_special_tokens=False))
                pos[c] += 1
            ask = rng.randrange(chains)
            question = tok.encode(f" Question: what is the value of VAR {chain_names[ask][-1]}? Answer: the "
                                  f"value of VAR {chain_names[ask][-1]} is", add_special_tokens=False)
            body = L - sum(map(len, seq)) - len(question) - 8
            if body <= 0:
                continue
            start = rng.randrange(0, len(filler) - body)
            hay = filler[start:start + body]
            seg = len(hay) // len(seq)
            ctx = [tok.bos_token_id]
            for i, st in enumerate(seq):
                ctx += hay[i * seg:(i + 1) * seg] + st
            ctx += hay[len(seq) * seg:] + question
            keys = [vals[ask]] + [v for i, v in enumerate(vals) if i != ask]
            items = [{"ctx": ctx, "cont": tok.encode(f" {k}", add_special_tokens=False), "nchars": 6} for k in keys]
            rows.append({"task": f"vt_{L // 1024}k", "qid": len(rows), "gold": 0, "kind": "mc", "items": items})
    return rows


def _filler(tok, raw_path):
    texts = []
    with open(raw_path) as f:
        for line in f:
            q = json.loads(line)
            if q["task"] in ("lambada", "hellaswag"):
                texts.append(q["items"][0]["ctx"] + q["items"][0]["cont"])
    return tok.encode(" ".join(texts), add_special_tokens=False)


def longdoc_rows(tok, prefixes, max_len=65536, n=24, seed=3):
    """Long documents for the per-position loss ("ppl" items): `n` documents of at least `max_len`
    tokens from Megatron indexed datasets (`prefixes`), cut to max_len (BOS first)."""
    from megatron.core.datasets.indexed_dataset import IndexedDataset

    rng = random.Random(seed)
    docs = []
    for prefix in prefixes:
        ds = IndexedDataset(prefix)
        lens = ds.sequence_lengths
        cand = [i for i in range(len(lens)) if lens[i] >= max_len]
        rng.shuffle(cand)
        for i in cand[: n - len(docs)]:
            docs.append([tok.bos_token_id] + ds[i][: max_len - 1].tolist())
        if len(docs) >= n:
            break
    return [{"task": f"longppl_{max_len // 1024}k", "qid": q, "gold": 0, "kind": "ppl",
             "items": [{"ctx": d, "cont": [], "nchars": 0}]} for q, d in enumerate(docs)]


def longctx(out, tokdir, *prefixes):
    """The harder long-context set (CKLSTM experiment C, user 2026-10-22) -> requests_longctx.jsonl:
    long-document loss per position bin to 64k, multi-key retrieval with 64 facts (4k-64k), the
    16-fact retrieval and the passkey at 64k, variable tracking (4k-64k)."""
    from transformers import AutoTokenizer

    out = Path(out)
    tok = AutoTokenizer.from_pretrained(tokdir)
    rows = longdoc_rows(tok, prefixes) if prefixes else []
    rows += kv_rows(tok, out / "raw.jsonl", lengths=(4096, 8192, 16384, 32768, 65536), pairs=64, seed=4)
    for r in rows:
        if r["task"].startswith("kvret_"):
            r["task"] = "kvret64_" + r["task"].split("_", 1)[1]
    rows += kv_rows(tok, out / "raw.jsonl", lengths=(65536,), seed=5)
    rows += passkey_rows(tok, lengths=(65536,))
    rows += vt_rows(tok, out / "raw.jsonl")
    with open(out / "requests_longctx.jsonl", "w") as g:
        for q, r in enumerate(rows):
            r["qid"] = q
            g.write(json.dumps(r) + "\n")
    print(f"{len(rows)} questions -> {out / 'requests_longctx.jsonl'}")


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
        for r in kv_rows(tok, out / "raw.jsonl"):
            g.write(json.dumps(r) + "\n")
            n += 1
    print(f"{n} questions -> {out / 'requests.jsonl'}")


if __name__ == "__main__":
    {"raw": raw, "tokenize": tokenize, "longctx": longctx}[sys.argv[1]](*sys.argv[2:])

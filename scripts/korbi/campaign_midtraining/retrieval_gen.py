#!/usr/bin/env python3
"""Synthetic long-context retrieval, generative (RULER-style, offline): the downstream generators of
scripts/korbi/downstream/prepare_tasks.py -- passkey in repeated filler (depths 0.1 / 0.5 / 0.9), multi-key NIAH in
natural filler (16 "special number" facts, one asked), variable tracking (8 interleaved assignment chains, 4 hops)
-- with greedy generation by the HF export as it is (its sparse attention), scored by exact match of the first
5-digit number generated against the inserted value. Batch size 1.

    retrieval_gen.py --model <hf export> --raw <downstream raw.jsonl> --out <dir> [--lengths 4096 ... 65536] [--n 30]

Writes <out>/samples.jsonl and <out>/summary.json (accuracy per task and length; passkey also per depth).
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "downstream"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--raw", required=True, help="the downstream raw.jsonl (natural filler for kv / vt)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192, 16384, 32768, 65536])
    ap.add_argument("--n", type=int, default=30, help="questions per (task, length); passkey: n / 3 per depth")
    ap.add_argument("--tasks", nargs="+", default=["passkey", "kv", "vt"])
    a = ap.parse_args()

    import prepare_tasks as pt
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, dtype=torch.bfloat16,
                                                 device_map="auto").eval()
    sa = getattr(model.config, "sparse_attention", None)
    print(f"[retrieval_gen] {a.model}: sparse_attention {json.dumps(sa)}", flush=True)
    lengths = tuple(a.lengths)
    rows = []
    if "passkey" in a.tasks:
        rows += pt.passkey_rows(tok, lengths=lengths, n=max(1, a.n // 3))
    if "kv" in a.tasks:
        rows += pt.kv_rows(tok, a.raw, lengths=lengths, n=a.n)
    if "vt" in a.tasks:
        rows += pt.vt_rows(tok, a.raw, lengths=lengths, n=a.n)
    a.out.mkdir(parents=True, exist_ok=True)
    samples = a.out / "samples.jsonl"
    done = set()
    if samples.exists():
        done = {(json.loads(line)["task"], json.loads(line)["qid"]) for line in samples.read_text().splitlines()}
    eos = tok.eos_token_id
    with samples.open("a") as fh:
        for r in rows:
            if (r["task"], r["qid"]) in done:
                continue
            ctx = r["items"][0]["ctx"]
            gold = tok.decode(r["items"][0]["cont"]).strip()
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(torch.tensor([ctx], device=model.device), max_new_tokens=10, do_sample=False,
                                     eos_token_id=eos, pad_token_id=eos)
            text = tok.decode(out[0, len(ctx):], skip_special_tokens=True)
            m = re.search(r"\d{5}", text)
            rec = {"task": r["task"], "qid": r["qid"], "tokens": len(ctx), "gold": gold, "pred": text.strip()[:40],
                   "correct": bool(m and m.group(0) == gold), "depth": r.get("depth"), "asked_pos": r.get("asked_pos"),
                   "seconds": round(time.time() - t0, 2)}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
    recs = [json.loads(line) for line in samples.read_text().splitlines()]
    acc = collections.defaultdict(list)
    for r in recs:
        acc[r["task"]].append(r["correct"])
        if r["depth"] is not None:
            acc[f"{r['task']}@{r['depth']}"].append(r["correct"])
    summary = {k: {"acc": sum(v) / len(v), "n": len(v)} for k, v in sorted(acc.items())}
    (a.out / "summary.json").write_text(json.dumps({"model": a.model, "sparse_attention": sa, "cells": summary},
                                                   indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

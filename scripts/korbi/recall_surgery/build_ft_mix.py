#!/usr/bin/env python3
"""Fine-tuning data of the 1.7B recoverability test (docs/recall_surgery_ambitious.md §16): render reading-comprehension
and general instruction data as plain-text documents, one Megatron indexed dataset per source (NeoX tokenizer, EOD
appended), and a manifest with document / token counts. The two mixes of ladder17b_ft_jupiter differ only in blend
weights over these sources (+ the ladder's own Nemotron pretraining data).

Sources (train splits only; the evaluations use validation splits):
  squadv2  rajpurkar/squad_v2       one document per question, lm-eval's squadv2 prompt format, "unanswerable" for the
                                    unanswerable ones (so the answer/abstain behaviour itself is trained)
  coqa     stanfordnlp/coqa         one document per story, lm-eval's coqa format (story, then "Q: ...\\n\\nA: ..." turns)
  race     ehovy/race (all)         article + multiple-choice question + options + answer
  drop     ucinlp/drop              passage + question + answer
  tulu     allenai/tulu-3-sft-mixture, without the FLAN subset (FLAN contains SQuAD / CoQA): "User: ...\\n\\nAssistant:"
           turns -- the general instruction mix, also of the RC-excluded mix

    python build_ft_mix.py render --out DIR [--sources squadv2,coqa,race,drop,tulu]     (needs internet: login node)
    python build_ft_mix.py tokenize --out DIR --megatron MEGATRON_TREE --tokenizer TOKENIZER_DIR [--workers 32]
    python build_ft_mix.py manifest --out DIR
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

SOURCES = ["squadv2", "coqa", "race", "drop", "tulu"]


def render_squadv2():
    from datasets import load_dataset

    for d in load_dataset("rajpurkar/squad_v2", split="train"):
        ans = d["answers"]["text"][0] if d["answers"]["text"] else "unanswerable"
        yield (f"Title: {d['title']}\n\nBackground: {d['context']}\n\nQuestion: {d['question']}\n\nAnswer: {ans}")


def render_coqa():
    from datasets import load_dataset

    for d in load_dataset("stanfordnlp/coqa", split="train"):
        turns = "".join(f"Q: {q}\n\nA: {a}\n\n" for q, a in zip(d["questions"], d["answers"]["input_text"]))
        yield f"{d['story']}\n\n{turns}".rstrip()


def render_race():
    from datasets import load_dataset

    for d in load_dataset("ehovy/race", "all", split="train"):
        opts = "\n".join(f"{chr(65 + i)}) {o}" for i, o in enumerate(d["options"]))
        k = ord(d["answer"]) - 65
        yield (f"{d['article']}\n\nQuestion: {d['question']}\n{opts}\nAnswer: {d['answer']}) {d['options'][k]}")


def render_drop():
    from datasets import load_dataset

    for d in load_dataset("ucinlp/drop", split="train"):
        spans = d["answers_spans"]["spans"]
        if not spans:
            continue
        yield f"{d['passage']}\n\nQuestion: {d['question']}\n\nAnswer: {', '.join(spans)}"


def render_tulu():
    from datasets import load_dataset

    for d in load_dataset("allenai/tulu-3-sft-mixture", split="train"):
        if "flan" in (d.get("source") or "").lower():
            continue
        parts = []
        for m in d["messages"]:
            role = {"user": "User", "assistant": "Assistant", "system": "System"}.get(m["role"], m["role"])
            parts.append(f"{role}: {m['content']}")
        yield "\n\n".join(parts)


def cmd_render(a) -> int:
    os.makedirs(a.out, exist_ok=True)
    for s in a.sources.split(","):
        path = os.path.join(a.out, f"{s}.jsonl")
        if os.path.exists(path) and not a.force:
            print(f"{s}: {path} exists (--force to redo)")
            continue
        n = 0
        with open(path + ".tmp", "w") as f:
            for text in globals()[f"render_{s}"]():
                f.write(json.dumps({"text": text}) + "\n")
                n += 1
        os.replace(path + ".tmp", path)
        print(f"{s}: {n} documents -> {path}", flush=True)
    return 0


def cmd_tokenize(a) -> int:
    for s in a.sources.split(","):
        src = os.path.join(a.out, f"{s}.jsonl")
        prefix = os.path.join(a.out, "neox", s)
        if os.path.exists(f"{prefix}_text_document.idx") and not a.force:
            print(f"{s}: tokenized already")
            continue
        os.makedirs(os.path.dirname(prefix), exist_ok=True)
        cmd = [sys.executable, os.path.join(a.megatron, "tools/preprocess_data.py"), "--input", src,
               "--output-prefix", prefix, "--json-keys", "text", "--tokenizer-type", "HuggingFaceTokenizer",
               "--tokenizer-model", a.tokenizer, "--append-eod", "--workers", str(a.workers)]
        print(" ".join(cmd), flush=True)
        env = dict(os.environ, PYTHONPATH=f"{a.megatron}:{os.environ.get('PYTHONPATH', '')}")
        subprocess.run(cmd, check=True, env=env)
    return 0


def cmd_manifest(a) -> int:
    import numpy as np

    sys.path.insert(0, a.megatron) if a.megatron else None
    man = {}
    for s in SOURCES:
        jl = os.path.join(a.out, f"{s}.jsonl")
        idx = os.path.join(a.out, "neox", f"{s}_text_document")
        if not os.path.exists(idx + ".bin"):
            continue
        docs = sum(1 for _ in open(jl)) if os.path.exists(jl) else None
        tokens = os.path.getsize(idx + ".bin") // np.dtype(np.uint16).itemsize
        man[s] = {"documents": docs, "tokens": int(tokens), "prefix": idx}
        print(f"{s:8s} docs {docs} tokens {tokens / 1e6:.1f}M")
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("render", "tokenize", "manifest"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--sources", default=",".join(SOURCES))
        p.add_argument("--force", action="store_true")
        p.add_argument("--megatron", default="")
        p.add_argument("--tokenizer", default="")
        p.add_argument("--workers", type=int, default=32)
    a = ap.parse_args(argv)
    return {"render": cmd_render, "tokenize": cmd_tokenize, "manifest": cmd_manifest}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())

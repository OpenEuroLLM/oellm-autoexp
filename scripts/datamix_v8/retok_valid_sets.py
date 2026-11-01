"""The fixed validation sets (build_valid_sets.py, tokenizer-256k-agent) in the ORIGINAL tokenizer-256k, for models
trained on it (user 2026-11-01: the conservative and the first ambitious run). The agent tokenizer renames only
the reserved ids 13..71 to chat / agent delimiters; every other piece and merge is the same, so text without
delimiters encodes identically. A sequence that holds one of those ids is decoded with the agent model and
re-encoded with the base model (the delimiter becomes the multi-token sequence the old models trained on);
leading / trailing <bos> / <eos> are kept; every other sequence is copied unchanged. Document boundaries are
unchanged; a re-encoded sequence is longer, so a set grows by a few tokens (counted in the info file).
Checks: no delimiter id left; decode_base(new) == decode_agent(old) counted per re-encoded sequence.

    python scripts/datamix_v8/retok_valid_sets.py --src <data>/v8ml-valid --out <data>/v8ml-valid-basetok
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "flag_decontam"))
from common import Index, write_index  # noqa: E402

BASE_TOK = "/gpfs/projects/ehpc533/tokenizers/tokenizer-256k/tokenizer.model"
AGENT_TOK = "/gpfs/projects/ehpc533/users/korbip/data/tokenizers/tokenizer-256k-agent/tokenizer.model"
AGENT_IDS = (13, 71)                   # ids the agent tokenizer renamed (scripts/flag_retok/make_tokenizer.py)
CONTROL = (1, 2)                       # <bos>, <eos>
SETS = ["all", "web_en", "multilingual", "code", "math", "reasoning", "long_64k"]


def split_control(t):
    """(head control ids, body, tail control ids)."""
    i, k = 0, len(t)
    while i < k and t[i] in CONTROL:
        i += 1
    while k > i and t[k - 1] in CONTROL:
        k -= 1
    return t[:i], t[i:k], t[k:]


def retok_set(name, src, out, base, agent):
    sp, op = os.path.join(src, f"{name}_text_document"), os.path.join(out, f"{name}_text_document")
    idx = Index(sp + ".idx")
    assert idx.dtype_code == 4, f"{name}: int32 tokens expected"
    data = np.fromfile(sp + ".bin", np.int32)
    lengths = idx.lengths.astype(np.int64).copy()
    lo, hi = AGENT_IDS
    changed = not_preserved = 0
    with open(op + ".bin.tmp", "wb") as fo:
        for q in range(len(lengths)):
            p = int(idx.pointers[q]) // 4
            t = data[p:p + int(idx.lengths[q])]
            if not ((t >= lo) & (t <= hi)).any():
                fo.write(t.tobytes())
                continue
            head, body, tail = split_control(t)
            text = agent.decode(body.tolist())
            new_body = base.encode(text)
            not_preserved += base.decode(new_body) != text   # SentencePiece normalization (counted)
            new = np.concatenate([head, np.asarray(new_body, np.int32), tail]).astype(np.int32)
            assert not ((new >= lo) & (new <= hi)).any(), f"{name} seq {q}: delimiter id left"
            fo.write(new.tobytes())
            lengths[q] = len(new)
            changed += 1
    os.replace(op + ".bin.tmp", op + ".bin")
    pointers = np.concatenate([[0], np.cumsum(lengths[:-1] * 4)])
    write_index(op + ".idx", 4, lengths.astype(np.int32), pointers, np.asarray(idx.doc_idx, np.int64))
    info = json.load(open(os.path.join(src, f"{name}.info.json")))
    info["tokenizer"] = BASE_TOK
    info["retokenized_from"] = {"src": sp, "tokenizer": AGENT_TOK, "sequences_reencoded": changed,
                                "text_not_preserved": int(not_preserved),
                                "tokens_before": int(idx.lengths.astype(np.int64).sum()),
                                "tokens_after": int(lengths.sum())}
    info["tokens"] = int(lengths.sum())
    json.dump(info, open(os.path.join(out, f"{name}.info.json"), "w"), indent=1)
    print(f"{name}: {changed} of {len(lengths)} sequences re-encoded ({not_preserved} not text-identical), tokens "
          f"{info['retokenized_from']['tokens_before']} -> {info['tokens']}", flush=True)
    return f"{name} {info['sequences_at']['seq_length']} {op}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", default=",".join(SETS))
    a = ap.parse_args()
    import sentencepiece as spm
    base, agent = spm.SentencePieceProcessor(model_file=BASE_TOK), spm.SentencePieceProcessor(model_file=AGENT_TOK)
    assert base.vocab_size() == agent.vocab_size()
    os.makedirs(a.out, exist_ok=True)
    lines = [retok_set(n, a.src, a.out, base, agent) for n in a.sets.split(",")]
    vs = os.path.join(a.out, "valid_sets.txt")             # "name seq_length prefix"; merged across runs
    old = {l.split()[0]: l.strip() for l in open(vs)} if os.path.exists(vs) else {}
    old.update({l.split()[0]: l for l in lines})
    with open(vs, "w") as f:
        f.write("\n".join(old[n] for n in SETS if n in old) + "\n")
    shutil.copy(__file__, os.path.join(a.out, os.path.basename(__file__)))


if __name__ == "__main__":
    main()

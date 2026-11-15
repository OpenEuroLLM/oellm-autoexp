#!/usr/bin/env python3
"""Which tokenizer produced each Megatron prefix? Decode -> re-encode round trip per candidate tokenizer (§19.1).

For every prefix (relative to ROOT) it samples 6 evenly spaced sequences, takes their first 1024 ids, and for each
tokenizer decodes them (special tokens kept) and re-encodes the text (no added specials). The share of the first
1000 ids reproduced exactly is ~1.0 for the producing tokenizer and clearly lower for a different vocabulary
(tokenizer-256k vs -v2). Also counts the delimiter id ranges: 13-36 (tokenizer-256k-agent's chat / agent / FIM
slots) and 69-92 (tokenizer-256k-v2(-agent)'s), which separate "re-tokenized with the agent tokenizer" from
"left as text".

    jsc_tok_fingerprint.py ROOT LIST OUT.tsv SNIPPETS.txt TOKDIR [TOKDIR ...]

FP_DOCS / FP_IDS (env) set the sample (default 6 sequences x 1024 ids). The literal_pieces column counts
delimiter strings left after decoding with special tokens skipped, i.e. delimiters tokenized as ordinary pieces.

Needs the `tokenizers` package (any Megatron image). Reads .idx / .bin like jsc_data_audit.py.
"""

from __future__ import annotations

import os
import sys
from array import array

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jsc_data_audit import DTYPES, read_idx, seq_entry  # noqa: E402

from tokenizers import Tokenizer  # noqa: E402

N_DOCS = int(os.environ.get("FP_DOCS", 6))
N_IDS = int(os.environ.get("FP_IDS", 1024))
N_CMP = N_IDS - 24
LITERALS = ("<think>", "</think>", "<|im_start|>", "<|im_end|>", "<tool_call>", "<tools>")


def docs(root, rel):
    b, i = os.path.join(root, rel + ".bin"), os.path.join(root, rel + ".idx")
    code, n_seq, _, off = read_idx(i)
    tc, width = DTYPES[code]
    out = []
    with open(b, "rb") as fb:
        for k in range(N_DOCS):
            j = k * (n_seq - 1) // (N_DOCS - 1)
            length, ptr = seq_entry(i, off, n_seq, j)
            fb.seek(ptr)
            a = array(tc)
            a.frombytes(fb.read(min(length, N_IDS) * width))
            out.append(list(a))
    return out


def agreement(tok, ids):
    text = tok.decode(ids, skip_special_tokens=False)
    re_ids = tok.encode(text, add_special_tokens=False).ids
    n = min(N_CMP, len(ids) - 8, len(re_ids))
    if n <= 0:
        return 1.0
    same = 0
    for a, b in zip(ids[:n], re_ids[:n]):
        if a != b:
            break
        same += 1
    return same / n


def main():
    root, lst, out, snip = sys.argv[1:5]
    tok_dirs = sys.argv[5:]
    toks = [(os.path.basename(d.rstrip("/")), Tokenizer.from_file(os.path.join(d, "tokenizer.json")))
            for d in tok_dirs]
    rels = [ln.strip() for ln in open(lst) if ln.strip()]
    with open(out, "w") as fo, open(snip, "w") as fs:
        fo.write("\t".join(["rel", "bos_start", "eos_end", "ids_13_36", "ids_69_92", "literal_pieces"]
                           + [f"rt_{n}" for n, _ in toks]) + "\n")
        for rel in rels:
            ds = docs(root, rel)
            flat = [t for d in ds for t in d]
            row = [rel, f"{sum(d[0] == 1 for d in ds if d) / len(ds):.2f}",
                   "-",  # sequences are truncated to 1024 ids here; jsc_data_audit.py has the <eos> share
                   str(sum(13 <= t <= 36 for t in flat)), str(sum(69 <= t <= 92 for t in flat))]
            # delimiter strings that survive decoding WITHOUT special tokens were tokenized as ordinary pieces
            # (text left on tokenizer-256k, or literal mentions)
            plain = "".join(toks[0][1].decode(d, skip_special_tokens=True) for d in ds)
            row.append(str(sum(plain.count(x) for x in LITERALS)))
            for name, tok in toks:
                row.append(f"{sum(agreement(tok, d) for d in ds) / len(ds):.3f}")
            fo.write("\t".join(row) + "\n")
            fo.flush()
            fs.write(f"=== {rel}\n{toks[0][1].decode(ds[0][:120], skip_special_tokens=False)!r}\n")


if __name__ == "__main__":
    main()

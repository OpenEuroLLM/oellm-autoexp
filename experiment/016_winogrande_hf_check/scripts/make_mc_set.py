"""A Megatron indexed dataset of lm-eval multiple-choice requests, for the Megatron-vs-export check.

Every request (context + continuation, encoded as lm-eval does: score_mc_hf.encode_pair) becomes one
document without its BOS (Megatron's dataloader_add_bos puts it back), so Megatron's fixed-validation
evaluation scores exactly the token sequences the HF / vLLM ports scored. One filler document (sentence
text of the same task) makes the epoch exactly ``--samples`` x ``--seq-len`` (+1) tokens; ``--samples``
must be a multiple of the evaluation's data-parallel size. ``<prefix>.requests.json`` maps documents back
to requests (task, doc_id, choice, context / continuation lengths).

    python experiment/016_winogrande_hf_check/scripts/make_mc_set.py --tokenizer <export> \
        --samples-files <samples_winogrande_*.jsonl> <samples_wsc273_*.jsonl> --seq-len 512 --dp 32 \
        --out <dir>/mc_text_document
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_mc_hf import encode_pair  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--samples-files", type=Path, nargs="+", required=True)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--dp", type=int, default=32, help="the sample count is rounded up to a multiple of this")
    p.add_argument("--out", required=True, help="indexed dataset prefix")
    a = p.parse_args()
    from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    bos = tok.bos_token_id
    docs, meta, filler_text = [], [], []
    for f in a.samples_files:
        task = f.name.split("_")[1]
        for d in sorted(map(json.loads, open(f)), key=lambda d: d["doc_id"]):
            args = [d["arguments"][k] for k in sorted(d["arguments"], key=lambda k: int(k.split("_")[-1]))]
            for j, g in enumerate(args):
                ctx, cont = encode_pair(tok, g["arg_0"], g["arg_1"])
                ids = ctx + cont
                assert ids[0] == bos, "expects the tokenizer's BOS first"
                docs.append(ids[1:])
                meta.append({"task": task, "doc_id": d["doc_id"], "choice": j, "n_ctx": len(ctx),
                             "n_cont": len(cont), "ids": ids})
            filler_text.append(d["doc"].get("sentence") or d["doc"].get("text") or "")
    total = sum(len(x) + 1 for x in docs)  # + one BOS each
    samples = math.ceil((total + 2) / a.seq_len / a.dp) * a.dp
    target = samples * a.seq_len + 1
    need = target - total - 1  # the filler document's tokens (+ its BOS)
    filler = []
    while len(filler) < need:
        filler += tok(" ".join(filler_text), add_special_tokens=False).input_ids
    filler = filler[:need]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    b = IndexedDatasetBuilder(a.out + ".bin", dtype=np.int32)
    for x in docs + [filler]:
        b.add_document(np.asarray(x, dtype=np.int32), [len(x)])
    b.finalize(a.out + ".idx")
    info = {"samples": samples, "seq_len": a.seq_len, "requests": len(docs), "request_tokens": total,
            "filler_tokens": len(filler), "sources": [str(f) for f in a.samples_files], "map": meta}
    Path(a.out + ".requests.json").write_text(json.dumps(info) + "\n")
    print(f"{a.out}: {len(docs)} requests ({total} tokens incl. BOS) + filler {len(filler)} = "
          f"{samples} samples of {a.seq_len}")


if __name__ == "__main__":
    main()

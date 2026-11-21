#!/usr/bin/env python3
"""Check an extracted stream chunk against Megatron's own GPTDataset at production scale: for random slots,
the dataset's sample (tokens, cu_seqlens) equals the extracted pieces read from the .bin.

    verify_stream.py --out <pipeline dir> --tokenizer <training tokenizer dir> [--slots 64]

The GPTDatasets are built from the run's cache entries through a temporary directory of symlinks
(nothing is ever written into the shared cache); a cache miss there (a regular file appears: the
configuration does not reproduce the run's description) is a failure.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cds_data import BlendStream, find_cache_entry, load_stream, read_blend  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-cache-path", required=True)
    ap.add_argument("--data-args-path", required=True)
    ap.add_argument("--tokenizer", default=None, help="training tokenizer (for BOS / EOD ids)")
    ap.add_argument("--bos-id", type=int, default=None)
    ap.add_argument("--eod-id", type=int, default=None)
    ap.add_argument("--vocab-size", type=int, default=262144)
    ap.add_argument("--slots", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import json

    from cds_megatron import MegatronBlend

    blend = json.load(open(os.path.join(a.out, "blend.json")))
    _, prefixes = read_blend(a.data_args_path)
    entry = find_cache_entry(a.data_cache_path, prefixes, blend["seq_length"], blend_md5=blend["blend_md5"])
    stream = BlendStream(entry)
    bos, eod, vocab = a.bos_id, a.eod_id, a.vocab_size
    if a.tokenizer:
        from transformers import AutoTokenizer

        hf = AutoTokenizer.from_pretrained(a.tokenizer)
        bos, eod, vocab = hf.bos_token_id, hf.eos_token_id, len(hf)
    mb = MegatronBlend(entry, bos, eod, vocab)

    chunk = sorted(p for p in glob.glob(os.path.join(a.out, "stream", "ck*_*")) if not p.endswith(".tmp"))[0]
    arr, _ = load_stream(chunk)
    rng = np.random.default_rng(a.seed)
    slots = rng.choice(len(arr["slot_bidx"]), size=min(a.slots, len(arr["slot_bidx"])), replace=False)
    gpt: dict = {}
    bad = 0
    for s in slots:
        b = int(arr["slot_bidx"][s])
        ds = int(stream.dataset_index[b])
        dsi = int(stream.dataset_sample_index[b])
        gpt[ds] = mb.gpt(ds)
        ref = gpt[ds][dsi]
        a_, b_ = arr["seg_ptr"][s], arr["seg_ptr"][s + 1]
        toks = np.concatenate([stream.read(ds, int(arr["seg_doc"][k]), int(arr["seg_off"][k]),
                                           int(arr["seg_len"][k]), bos) for k in range(a_, b_)])
        ok = np.array_equal(toks, ref["tokens"].numpy())
        cu = ref["cu_seqlens"].numpy()
        cu = cu[: int(np.argmax(cu == blend["seq_length"])) + 1]
        ok &= cu.tolist() == np.r_[0, np.cumsum(arr["seg_len"][a_:b_])].tolist()
        bad += not ok
        if not ok:
            print(f"MISMATCH slot {s} (iteration {arr['slot_iter'][s]}, position {arr['slot_pos'][s]}, sample {b})")
    print(f"verify_stream {os.path.basename(chunk)}: {len(slots) - bad}/{len(slots)} slots equal to Megatron's "
          f"GPTDataset ({len(gpt)} datasets)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

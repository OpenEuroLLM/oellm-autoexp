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
import tempfile

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

    from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
    from megatron.core.datasets.indexed_dataset import IndexedDataset
    from megatron.core.datasets.utils import Split

    blend = json.load(open(os.path.join(a.out, "blend.json")))
    _, prefixes = read_blend(a.data_args_path)
    entry = find_cache_entry(a.data_cache_path, prefixes, blend["seq_length"], blend_md5=blend["blend_md5"])
    stream = BlendStream(entry)
    bos, eod, vocab = a.bos_id, a.eod_id, a.vocab_size
    if a.tokenizer:
        from transformers import AutoTokenizer

        hf = AutoTokenizer.from_pretrained(a.tokenizer)
        bos, eod, vocab = hf.bos_token_id, hf.eos_token_id, len(hf)

    class _Tok:  # the run's tokenizer identity, as its cache descriptions record it
        def __init__(self, uid):
            self.unique_identifiers = uid
            self.bos, self.eod, self.pad, self.vocab_size = bos, eod, -1, vocab

    chunk = sorted(p for p in glob.glob(os.path.join(a.out, "stream", "ck*_*")) if not p.endswith(".tmp"))[0]
    arr, _ = load_stream(chunk)
    rng = np.random.default_rng(a.seed)
    slots = rng.choice(len(arr["slot_bidx"]), size=min(a.slots, len(arr["slot_bidx"])), replace=False)
    tmp = tempfile.mkdtemp(prefix="cds_verify_")
    gpt: dict = {}
    bad = 0
    for s in slots:
        b = int(arr["slot_bidx"][s])
        ds = int(stream.dataset_index[b])
        dsi = int(stream.dataset_sample_index[b])
        if ds not in gpt:
            d = entry.desc["datasets"][ds]
            md5 = entry.gpt_md5(ds)
            for f in glob.glob(os.path.join(entry.cache_dir, f"{md5}-GPTDataset-train-*")):
                dst = os.path.join(tmp, os.path.basename(f))
                if not os.path.exists(dst):
                    os.symlink(f, dst)
            sm = d["split_matrix"][0]
            cfg = GPTDatasetConfig(
                random_seed=d["random_seed"], sequence_length=d["sequence_length"], split=d["split"],
                blend=([d["dataset_path"]], None),  # without a blend the config replaces the split
                tokenizer=_Tok(d["tokenizer"]), reset_position_ids=False, reset_attention_mask=False,
                eod_mask_loss=False, create_attention_mask=False, inter_document_masking=True,
                add_bos=bool(d.get("add_bos", False)), path_to_cache=tmp)
            idx = IndexedDataset(d["dataset_path"], mmap=True)
            n = len(idx.sequence_lengths)
            ids = np.arange(int(round(sm[0] * n)), int(round(sm[1] * n)), dtype=np.int32)
            g = GPTDataset(idx, d["dataset_path"], ids, d["num_samples"], Split.train, cfg)
            assert g.unique_description_hash == md5, f"dataset {ds}: {g.unique_description} != {json.dumps(d, indent=4)}"
            built = [f for f in os.listdir(tmp) if not os.path.islink(os.path.join(tmp, f))]
            assert not built, f"dataset {ds}: cache miss, Megatron built {built} (configuration differs)"
            gpt[ds] = g
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

"""A small indexed dataset of seeded-random documents of a validation set, for the paired Megatron vs HF
check of an export (``paired_valid_check``): Megatron evaluates the subset completely as a fixed validation
set (one epoch), the HF side scores the very same packed samples. The same document ids are taken from each
given tokenization (v8ml-valid and v8ml-valid-basetok hold the same documents in the same order), so both
models see the same text.

    python -m oellm_autoexp.hf_export.make_valid_subset --samples 64 --seq-len 4096 --seed 0 \\
        $D/v8ml-valid/all_text_document=$OUT/v8ml-valid-sub/all_text_document \\
        $D/v8ml-valid-basetok/all_text_document=$OUT/v8ml-valid-basetok-sub/all_text_document
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from megatron.core.datasets.indexed_dataset import IndexedDataset, IndexedDatasetBuilder


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pairs", nargs="+", help="SRC_PREFIX=DST_PREFIX; the first one sets the document choice")
    p.add_argument("--samples", type=int, default=64, help="samples per epoch (a multiple of the eval's DP size)")
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    pairs = [x.split("=", 1) for x in a.pairs]
    srcs = [IndexedDataset(s) for s, _ in pairs]
    n = len(srcs[0])
    for d in srcs:  # one sequence per document (the v8ml sets)
        if len(d.document_indices) != len(d) + 1:
            raise ValueError("expects one sequence per document")
    if any(len(d) != n for d in srcs):
        raise ValueError(f"document counts differ: {[len(d) for d in srcs]}")
    order = np.random.default_rng(a.seed).permutation(n)
    # one BOS per document (dataloader_add_bos): an epoch of exactly S samples is S * seq_len + 1 tokens
    target = a.samples * a.seq_len + 1
    lens0 = srcs[0].sequence_lengths
    chosen, total = [], 0
    for i in order:  # a margin, so every tokenization reaches the target before the trim
        chosen.append(int(i))
        total += int(lens0[i]) + 1
        if total >= target * 1.05:
            break
    chosen.sort()
    for (src, dst), ds in zip(pairs, srcs):
        lens = {i: int(ds.sequence_lengths[i]) for i in chosen}
        excess = sum(lens.values()) + len(chosen) - target
        if excess < 0:
            raise ValueError(f"{src}: {-excess} tokens short of {a.samples} samples")
        while excess > 0:  # cut the longest documents' tails
            i = max(lens, key=lens.get)
            cut = min(excess, lens[i] - 1)
            lens[i] -= cut
            excess -= cut
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        b = IndexedDatasetBuilder(dst + ".bin", dtype=ds.index.dtype)
        for i in chosen:
            b.add_document(np.asarray(ds[i])[: lens[i]], [lens[i]])
        b.finalize(dst + ".idx")
        info = {"source": src, "seed": a.seed, "samples": a.samples, "seq_len": a.seq_len, "documents": chosen,
                "lengths": [lens[i] for i in chosen], "tokens": sum(lens.values())}
        Path(dst + ".subset.json").write_text(json.dumps(info) + "\n")
        print(f"{dst}: {len(chosen)} documents, {info['tokens']} tokens, {a.samples} samples of {a.seq_len}")

if __name__ == "__main__":
    main()

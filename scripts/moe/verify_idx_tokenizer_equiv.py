#!/usr/bin/env python3
"""Prove that a gpt-neox-20b-tokenized Megatron .idx is reusable as-is for a
pythia-12b model, by round-tripping sample documents:

    ids --decode(neox)--> text --encode(pythia)--> ids'      expect ids' == ids

We already know the two tokenizers are byte-identical (same vocab + merges), so
this is expected to be an exact identity. This script confirms it on the *actual*
eval data rather than trusting the global claim, and reports any mismatching docs.

Run inside the Megatron container (needs megatron.core for IndexedDataset and
`transformers` for the tokenizers). Example:

    python scripts/moe/verify_idx_tokenizer_equiv.py \
        --idx-path /leonardo_work/OELLM_prod2026/datasets/Nemotron-cc-2024-HQ-LUMI-sample-valid/high-all \
        --neox-tokenizer /leonardo_scratch/large/userexternal/ajha0001/tokenizers/gpt-neox-20b \
        --pythia-tokenizer /leonardo_scratch/large/userexternal/ajha0001/tokenizers/pythia-12b \
        --num-docs 200

Exit code 0 = all sampled docs round-trip identically; 1 = mismatches found.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MEGATRON_ROOT = REPO_ROOT / "submodules" / "Megatron-LM"
sys.path.append(str(MEGATRON_ROOT))


def _load_tokenizer(path_or_name: str):
    from transformers import AutoTokenizer

    # Local dirs load offline; hub ids need HF_HUB_OFFLINE unset / cache present.
    return AutoTokenizer.from_pretrained(path_or_name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idx-path", required=True,
                        help="Megatron IndexedDataset prefix (dir + basename, no .bin/.idx).")
    parser.add_argument("--neox-tokenizer", required=True,
                        help="Local dir / hub id of the tokenizer the .idx was built with.")
    parser.add_argument("--pythia-tokenizer", required=True,
                        help="Local dir / hub id of the target (FLAME) tokenizer.")
    parser.add_argument("--num-docs", type=int, default=200,
                        help="Number of documents to sample and round-trip.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-show", type=int, default=5,
                        help="How many mismatching docs to print in detail.")
    args = parser.parse_args()

    from megatron.core.datasets.indexed_dataset import IndexedDataset

    base = Path(args.idx_path).expanduser()
    if not (Path(str(base) + ".bin").is_file() and Path(str(base) + ".idx").is_file()):
        print(f"ERROR: no IndexedDataset at {base}.bin / {base}.idx", file=sys.stderr)
        return 2

    ds = IndexedDataset(str(base), multimodal=False, mmap=True)
    n_docs = len(ds)
    print(f"IndexedDataset: {n_docs:,} documents at {base}")

    neox = _load_tokenizer(args.neox_tokenizer)
    pythia = _load_tokenizer(args.pythia_tokenizer)
    print(f"neox vocab_size={neox.vocab_size}  pythia vocab_size={pythia.vocab_size}")

    rng = np.random.default_rng(args.seed)
    sample = rng.permutation(n_docs)[: min(args.num_docs, n_docs)]

    mismatches = 0
    shown = 0
    total = 0
    for doc_id in sample:
        ids = np.asarray(ds.get(int(doc_id)), dtype=np.int64).tolist()
        if not ids:
            continue
        total += 1
        # Decode with the source tokenizer, re-encode with the target tokenizer.
        text = neox.decode(ids, skip_special_tokens=False)
        ids2 = pythia.encode(text, add_special_tokens=False)
        if ids2 != ids:
            mismatches += 1
            if shown < args.max_show:
                shown += 1
                # Find first differing position for a compact diagnostic.
                first = next(
                    (i for i in range(min(len(ids), len(ids2))) if ids[i] != ids2[i]),
                    min(len(ids), len(ids2)),
                )
                print(
                    f"  MISMATCH doc={int(doc_id)} len={len(ids)}->{len(ids2)} "
                    f"first_diff@{first}: {ids[first:first+6]} != {ids2[first:first+6]}"
                )

    print(
        f"\nChecked {total} non-empty docs: "
        f"{total - mismatches} identical, {mismatches} mismatched."
    )
    if mismatches == 0:
        print("OK: the .idx token IDs are identical under pythia-12b — reuse as-is.")
        return 0
    print(
        "WARNING: mismatches found. The tokenizers may not be identical for this data; "
        "consider fresh pythia-12b tokenization (hf_dataset mode) instead.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

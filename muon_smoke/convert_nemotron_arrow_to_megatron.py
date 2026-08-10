"""Convert the pre-tokenized Nemotron-sample arrow shards into Megatron's
native .bin/.idx indexed-dataset format.

Source (read-only, never written to):
    /data/horse/ws/jasi149i-fastlm/data/nemotron-cc-sample-mtsynth/tokenized_gpt2/ctx_2048/{train,valid}/*.arrow
Each row already has input_ids (2049 int32 = 2048 ctx + 1) and docs_lengths
(document boundaries within the packed row) — already tokenized with
GPT2Tokenizer, no re-tokenization here, just a format conversion.

Output goes only under our own workspace, e.g.:
    /data/horse/ws/sala597i-slaing/data/nemotron_gpt2_megatron/{train,valid}.{bin,idx}

Usage:
    python3 muon_smoke/convert_nemotron_arrow_to_megatron.py --split train --limit-shards 1  # quick test
    python3 muon_smoke/convert_nemotron_arrow_to_megatron.py --split train                   # full run
"""

import argparse
import glob
import os
import time

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch

from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder, get_bin_path, get_idx_path

SOURCE_DIR = "/data/horse/ws/jasi149i-fastlm/data/nemotron-cc-sample-mtsynth/tokenized_gpt2/ctx_2048"
OUTPUT_DIR = "/data/horse/ws/sala597i-slaing/data/nemotron_gpt2_megatron"


def iter_arrow_rows(shard_path):
    with pa.memory_map(shard_path, "r") as source:
        reader = ipc.open_stream(source)
        while True:
            try:
                batch = reader.read_next_batch()
            except StopIteration:
                break
            df = batch.to_pandas()
            for _, row in df.iterrows():
                yield np.asarray(row["input_ids"], dtype=np.int32), list(row["docs_lengths"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "valid"], required=True)
    parser.add_argument("--limit-shards", type=int, default=None, help="Only process the first N shards (for a quick benchmark/test run)")
    args = parser.parse_args()

    shard_dir = os.path.join(SOURCE_DIR, args.split)
    shards = sorted(glob.glob(os.path.join(shard_dir, "data-*.arrow")))
    if args.limit_shards:
        shards = shards[: args.limit_shards]
    print(f"Found {len(shards)} shard(s) in {shard_dir}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_prefix = os.path.join(OUTPUT_DIR, args.split)
    bin_path = get_bin_path(out_prefix)
    idx_path = get_idx_path(out_prefix)
    print(f"Writing {bin_path} / {idx_path}")

    builder = IndexedDatasetBuilder(bin_path, dtype=np.int32, multimodal=False)

    t0 = time.time()
    n_rows = 0
    n_tokens = 0
    for shard_idx, shard_path in enumerate(shards):
        shard_t0 = time.time()
        shard_rows = 0
        for input_ids, docs_lengths in iter_arrow_rows(shard_path):
            builder.add_document(torch.from_numpy(input_ids), docs_lengths)
            shard_rows += 1
            n_tokens += len(input_ids)
        n_rows += shard_rows
        elapsed = time.time() - shard_t0
        print(
            f"[{shard_idx + 1}/{len(shards)}] {os.path.basename(shard_path)}: "
            f"{shard_rows} rows in {elapsed:.1f}s ({os.path.getsize(shard_path) / 1e6 / elapsed:.1f} MB/s)"
        )

    builder.finalize(idx_path)
    total_elapsed = time.time() - t0
    print(
        f"DONE: {n_rows} rows, {n_tokens} tokens, {len(shards)} shards in "
        f"{total_elapsed:.1f}s ({total_elapsed / max(len(shards), 1):.1f}s/shard avg)"
    )


if __name__ == "__main__":
    main()

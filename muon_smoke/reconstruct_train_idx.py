"""Reconstruct train.idx after the --split train conversion run was OOM-killed
just before finalize() (train.bin fully written, .idx never flushed).

Every row is a fixed 2049 int32 tokens (2048 ctx + 1), written back-to-back with no
padding by IndexedDatasetBuilder.add_document -- confirmed train.bin's size divides
evenly by 8196 bytes (2049 * 4), so it died cleanly on a row boundary. The only thing
lost was the in-memory sequence_lengths/document_indices bookkeeping, which is
trivially reconstructible since every sequence length is the fixed 2049.

This treats each packed row as a single document, dropping the original intra-row
docs_lengths sub-boundaries -- fine here since reset_attention_mask/reset_position_ids
are both false in muon_50M_50BT.yaml (base_defaults.yaml:552-553), so those boundaries
were never going to be used during training anyway.
"""

import os

import numpy as np

from megatron.core.datasets.indexed_dataset import IndexedDataset, _IndexWriter, get_bin_path, get_idx_path

OUT_PREFIX = "/data/horse/ws/sala597i-slaing/data/nemotron_gpt2_megatron/train"
ROW_TOKENS = 2049

bin_path = get_bin_path(OUT_PREFIX)
idx_path = get_idx_path(OUT_PREFIX)

bin_size = os.path.getsize(bin_path)
row_bytes = ROW_TOKENS * 4
assert bin_size % row_bytes == 0, f"bin size {bin_size} is not a multiple of {row_bytes}"
n_rows = bin_size // row_bytes
print(f"{n_rows} complete rows found in {bin_path} ({bin_size} bytes)")

sequence_lengths = np.full(n_rows, ROW_TOKENS, dtype=np.int32)
document_indices = np.arange(n_rows + 1, dtype=np.int64)

with _IndexWriter(idx_path, np.int32) as writer:
    writer.write(sequence_lengths, None, document_indices)
print(f"Wrote {idx_path}")

ds = IndexedDataset(OUT_PREFIX)
print(f"Verification: len(ds) = {len(ds)}, ds[0] shape = {ds[0].shape}, ds[-1] shape = {ds[-1].shape}")

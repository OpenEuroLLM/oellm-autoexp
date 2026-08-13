"""Locality-preserving (K-lane transposed) shuffle to replace Megatron's global
shuffle.

Motivation: under the default global document shuffle, a sub-one-epoch run reads documents
scattered uniformly across the whole .bin -> cold random GPFS reads (~1 MiB/s), which a
prefetcher cannot accelerate (it's the same random work). We instead make the consumption
order *locality-preserving* so the prefetcher reads (near-)sequentially, while staying well
mixed across sources for source-CLUSTERED data:

  * document_index = file order (identity) -> each sample is a contiguous byte range.
  * shuffle_index  = K-lane transposed interleave with a local block-shuffle inside each lane:
      - the file is split into K contiguous lanes (sample ranges); K >> #sources so every
        round of K samples spans all sources (good mixing);
      - lanes are visited round-robin (transpose), each swept in increasing offset order, so
        the prefetcher reads K sequential streams;
      - a local block-shuffle within each lane breaks any within-source ordering while keeping
        reads inside a small window (still near-sequential).

Installed via megatron_patch when OELLM_SHUFFLE_LANES is set. Tunables (env):
  OELLM_SHUFFLE_LANES  (K, default 256)
  OELLM_SHUFFLE_BLOCK  (local block shuffle size in samples, default 8192)
  OELLM_SHUFFLE_SEED   (default 1234)
"""

from __future__ import annotations

import numpy as np


def file_order_document_index(documents, num_epochs: int = 1) -> np.ndarray:
    """The split's document ids in file order (sorted; NO global shuffle),
    TILED num_epochs times.

    ``documents`` is Megatron's per-split index subset (e.g. the train portion [0, 0.989N)); we
    must return exactly those ids (sorted), not a fresh arange, or build_sample_idx mismatches the
    token total. Sorted order makes consecutive samples contiguous on disk. For num_epochs>1 each
    epoch is the same file-order pass (re-randomization happens per-epoch in the shuffle index), so
    every epoch reads the SAME lane=file-region structure -> the prefetcher stages it once and it is
    reused across epochs.
    """
    base = np.sort(np.asarray(documents)).astype(np.int32)
    return base if num_epochs <= 1 else np.tile(base, num_epochs)


def _lane_interleave_one(total_size: int, num_lanes: int, block: int, seed: int) -> np.ndarray:
    """One epoch: a permutation of [0, total_size) -- K-lane transpose with local block-shuffle."""
    if total_size <= 1:
        return np.arange(total_size, dtype=np.int64)
    K = max(1, min(num_lanes, total_size))
    M = total_size // K
    if M == 0:
        return np.arange(total_size, dtype=np.int64)
    rng = np.random.RandomState(seed)
    lanes = np.arange(K * M, dtype=np.int64).reshape(
        K, M
    )  # row j = lane j = samples [j*M, (j+1)*M)
    if block and block < M:
        for j in range(K):
            row = lanes[j]
            for s in range(0, M, block):
                rng.shuffle(row[s : min(s + block, M)])
    else:
        for j in range(K):
            rng.shuffle(lanes[j])
    order = lanes.T.reshape(-1)  # transpose (column-major) -> round-robin across lanes
    if K * M < total_size:
        order = np.concatenate([order, np.arange(K * M, total_size, dtype=np.int64)])
    return order


def lane_interleave_shuffle_index(
    total_size: int, num_lanes: int, block: int, seed: int, epoch_size: int = 0, dtype=np.uint32
) -> np.ndarray:
    """A permutation of [0, total_size). With epoch_size<=0 (or >= total), a.

    single K-lane transpose (consecutive entries from K different file regions
    -> mixed; within a lane block-shuffled).

    With epoch_size>0, the range is split into consecutive epochs of `epoch_size` samples (the final
    one possibly partial); each epoch is independently lane-interleaved with seed+e ('chunk-aware
    second reshuffle') and visited in epoch order. Every epoch sweeps the same lanes (file regions)
    in a DIFFERENT within-lane order -> locality preserved each epoch, order re-randomized, and the
    prefetcher's staged lanes are reused across epochs. epoch_size MUST equal Megatron's actual
    samples-per-epoch so the lane boundaries match the per-epoch file structure.
    """
    if dtype is None:
        dtype = np.uint32 if total_size < (np.iinfo(np.uint32).max - 1) else np.int64
    if epoch_size <= 0 or epoch_size >= total_size:
        return _lane_interleave_one(total_size, num_lanes, block, seed).astype(dtype)
    parts = []
    start = epoch = 0
    while start < total_size:
        m = min(epoch_size, total_size - start)
        parts.append(_lane_interleave_one(m, num_lanes, block, seed + epoch) + start)
        start += m
        epoch += 1
    return np.concatenate(parts).astype(dtype)

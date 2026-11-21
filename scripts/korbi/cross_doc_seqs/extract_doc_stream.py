#!/usr/bin/env python3
"""Step 1 of the cross-document packing pipeline (docs/cross_doc_seqs/README.md): the document pieces
every training step of a phase reads, from a start checkpoint on.

    extract_doc_stream.py --data-args-path <datamix.txt> --data-cache-path <cache> \
        --seq-length 65536 --global-batch-size 1024 --train-iters 236733 \
        --phase-start 23691 --ref-dp 512 --ref-mbs 1 \
        --from-ckpt 49000 --steps 4000 --chunk 250 --out <dir>

writes <out>/stream/ck<a>_<b>/ (one per --chunk steps; a chunk already present is skipped, so a
rerun resumes and a later --from-ckpt extends): per slot (step x position in the global batch) the
logged iteration, the position, the blended sample index the sampler serves there, and the slot's
pieces (dataset, document, virtual offset, owned length). The pieces of a slot sum to seq_length.

Nothing here is specific to 49000 / 50000: --from-ckpt is any checkpoint iteration inside the phase.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cds_data import (BlendStream, SamplerGeometry, StepSampler, find_cache_entry, read_blend,  # noqa: E402
                      save_stream, stream_dir)


def phase_train_samples(train_iters: int, phase_start: int, data_batch: int) -> int:
    """Megatron's train-sample target of the phase that starts at phase_start and runs to train_iters
    (training.py get_train_valid_test_num_samples with --phase-transition-iterations)."""
    return (train_iters - phase_start) * data_batch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-args-path", required=True)
    ap.add_argument("--data-cache-path", required=True)
    ap.add_argument("--blend-md5", default=None, help="pick the cache entry directly (else matched by paths/size)")
    ap.add_argument("--seq-length", type=int, required=True)
    ap.add_argument("--global-batch-size", type=int, required=True)
    ap.add_argument("--train-iters", type=int, required=True)
    ap.add_argument("--phase-start", type=int, required=True, help="the phase's first iteration (0 without phases)")
    ap.add_argument("--next-phase", type=int, default=None, help="next --phase-transition-iterations entry, if any")
    ap.add_argument("--ref-dp", type=int, required=True, help="--data-sampler-reference-dp-size")
    ap.add_argument("--ref-mbs", type=int, required=True, help="--data-sampler-reference-micro-batch-size")
    ap.add_argument("--ref-gbs", type=int, default=None, help="--data-sampler-reference-global-batch-size")
    ap.add_argument("--from-ckpt", type=int, required=True, help="first step = the one taken from this checkpoint")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--chunk", type=int, default=250, help="steps per output directory")
    ap.add_argument("--check-megatron", action="store_true",
                    help="compare the sampler replay with Megatron's sampler at every chunk start")
    ap.add_argument("--workers", type=int, default=1, help="parallel chunks (processes)")
    ap.add_argument("--no-preload", action="store_true",
                    help="memory-map the index files instead of reading them (~120 GB for the main run)")
    ap.add_argument("--preload-threads", type=int, default=32)
    ap.add_argument("--task-rank", type=int, default=0, help="this task extracts chunks rank, rank + world, ...")
    ap.add_argument("--task-world", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    weights, prefixes = read_blend(a.data_args_path)
    end_phase = a.next_phase or a.train_iters
    data_batch = a.ref_gbs or a.global_batch_size
    n_train = phase_train_samples(end_phase, a.phase_start, data_batch)
    # BlendedDataset size = sum_j ceil(N w_j) >= N; the exact value is in the description
    entry = find_cache_entry(a.data_cache_path, prefixes, a.seq_length, blend_md5=a.blend_md5)
    assert entry.size >= n_train and entry.size - n_train <= len(prefixes), (
        f"cache entry {entry.blend_md5} has {entry.size} samples, the phase needs {n_train}: "
        f"wrong --train-iters / --phase-start / --next-phase?")
    w = np.asarray(weights, dtype=np.float64)
    w /= w.sum()
    w2 = np.asarray(entry.desc["weights"], dtype=np.float64)
    assert np.allclose(w, w2, rtol=0, atol=1e-12), "datamix weights differ from the cache entry's"
    geo = SamplerGeometry(total_samples=entry.size, global_batch_size=a.global_batch_size,
                          phase_start=a.phase_start, ref_dp=a.ref_dp, ref_mbs=a.ref_mbs, ref_gbs=a.ref_gbs)
    assert a.from_ckpt >= a.phase_start and a.from_ckpt + a.steps <= end_phase, "steps outside the phase"
    sampler = StepSampler(geo)
    stream = BlendStream(entry)
    assert entry.sequence_length == a.seq_length
    print(f"blend {entry.blend_md5}: {len(prefixes)} datasets, {entry.size} samples, seq {a.seq_length}, "
          f"add_bos {entry.add_bos}; sampler B={geo.mb_per_epoch}", flush=True)
    os.makedirs(os.path.join(a.out, "stream"), exist_ok=True)
    tmp = os.path.join(a.out, f"blend.json.tmp{a.task_rank}")
    with open(tmp, "w") as f:
        json.dump({"blend_md5": entry.blend_md5, "prefixes": prefixes, "size": entry.size,
                   "seq_length": a.seq_length, "add_bos": entry.add_bos, "data_args_path": a.data_args_path,
                   "geometry": geo.__dict__}, f, indent=1)
    os.replace(tmp, os.path.join(a.out, "blend.json"))

    chunks = [(c0, min(c0 + a.chunk, a.from_ckpt + a.steps)) for c0 in range(a.from_ckpt, a.from_ckpt + a.steps, a.chunk)]
    chunks = [c for c in chunks[a.task_rank::a.task_world] if not os.path.isdir(stream_dir(a.out, *c))]
    if a.check_megatron:
        for c0, _ in chunks:
            sampler.check_against_megatron(c0)
        print(f"sampler replay == Megatron's sampler at {len(chunks)} chunk starts", flush=True)
    if chunks and not a.no_preload:
        # every chunk touches nearly every dataset: load the index arrays once, before forking
        stream.preload(threads=a.preload_threads)
    global _CTX
    _CTX = (sampler, stream, entry, a)
    if a.workers > 1 and len(chunks) > 1:
        # random page reads of the GPFS index files dominate: overlap them (fork shares the state)
        import multiprocessing as mp

        with mp.get_context("fork").Pool(min(a.workers, len(chunks))) as pool:
            for msg in pool.imap_unordered(_extract_chunk, chunks):
                print(msg, flush=True)
    else:
        for c in chunks:
            print(_extract_chunk(c), flush=True)


_CTX = None


def _extract_chunk(c: tuple[int, int]) -> str:
    sampler, stream, entry, a = _CTX
    c0, c1 = c
    path = stream_dir(a.out, c0, c1)
    t0 = time.time()
    slot_iter, slot_pos, slot_bidx, seg_ptr = [], [], [], [0]
    seg_ds, seg_doc, seg_off, seg_len = [], [], [], []
    for ck in range(c0, c1):
        bidx = sampler.step_after(ck)
        for pos, b in enumerate(bidx.tolist()):
            ds, pieces = stream.pieces(b)
            slot_iter.append(ck + 1)
            slot_pos.append(pos)
            slot_bidx.append(b)
            for doc, off, n in pieces:
                seg_ds.append(ds)
                seg_doc.append(doc)
                seg_off.append(off)
                seg_len.append(n)
            seg_ptr.append(len(seg_ds))
    arrays = {
        "slot_iter": np.asarray(slot_iter, np.int32), "slot_pos": np.asarray(slot_pos, np.int32),
        "slot_bidx": np.asarray(slot_bidx, np.int64), "seg_ptr": np.asarray(seg_ptr, np.int64),
        "seg_ds": np.asarray(seg_ds, np.int16), "seg_doc": np.asarray(seg_doc, np.int64),
        "seg_off": np.asarray(seg_off, np.int32), "seg_len": np.asarray(seg_len, np.int32),
    }
    assert len(np.unique(arrays["slot_bidx"])) == len(slot_bidx), "a sample is served twice in the range"
    save_stream(path, arrays, {"from_ckpt": c0, "to_ckpt": c1, "blend_md5": entry.blend_md5,
                               "seq_length": a.seq_length, "global_batch_size": a.global_batch_size})
    n = len(seg_ds)
    return (f"{path}: {len(slot_bidx)} slots, {n} pieces ({n / len(slot_bidx):.1f}/sequence), "
            f"{time.time() - t0:.0f} s")

if __name__ == "__main__":
    main()

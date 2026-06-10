#!/usr/bin/env python3
"""Parallel JSONL tokenizer producing Megatron .idx/.bin files.

Fully standalone — no Megatron or torch imports.
Dependencies: transformers, numpy, Python stdlib.

Each worker is an independent OS process that reads its own file region,
batch-tokenizes with AutoTokenizer, and writes its own .idx/.bin shard.
Zero IPC after launch. Optionally merges shards via merge_indexed_datasets_streaming.py.
"""

import argparse
import glob
import gzip
import json
import multiprocessing
import os
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np

_INDEX_HEADER = b"MMIDIDX\x00\x00"
_DTYPE_CODES = {np.dtype(np.uint16): 8, np.dtype(np.int32): 4}
_CODE_TO_DTYPE = {8: np.uint16, 4: np.int32}


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


class ShardWriter:
    """Writes Megatron .idx/.bin format. No Megatron/torch imports."""

    def __init__(self, prefix, dtype):
        self._dtype = np.dtype(dtype)
        self._dtype_code = _DTYPE_CODES[self._dtype]
        self._itemsize = self._dtype.itemsize
        self._bin_path = prefix + ".bin"
        self._idx_path = prefix + ".idx"
        self._bin_file = open(self._bin_path, "wb", buffering=4 * 1024 * 1024)
        self._sequence_lengths = []
        self._sequence_pointers = []
        self._doc_indices = [0]
        self._byte_offset = 0

    def add_document(self, token_ids):
        if not token_ids:
            return
        arr = np.array(token_ids, dtype=self._dtype)
        self._bin_file.write(arr.tobytes(order="C"))
        n = len(arr)
        self._sequence_pointers.append(self._byte_offset)
        self._sequence_lengths.append(n)
        self._byte_offset += n * self._itemsize
        self._doc_indices.append(len(self._sequence_lengths))

    def finalize(self):
        self._bin_file.flush()
        self._bin_file.close()
        seq_count = len(self._sequence_lengths)
        doc_count = len(self._doc_indices)
        with open(self._idx_path, "wb") as f:
            f.write(_INDEX_HEADER)
            f.write(struct.pack("<Q", 1))
            f.write(struct.pack("<B", self._dtype_code))
            f.write(struct.pack("<Q", seq_count))
            f.write(struct.pack("<Q", doc_count))
            f.write(np.array(self._sequence_lengths, dtype=np.int32).tobytes())
            f.write(np.array(self._sequence_pointers, dtype=np.int64).tobytes())
            f.write(np.array(self._doc_indices, dtype=np.int64).tobytes())

    @property
    def doc_count(self):
        return len(self._sequence_lengths)


def select_dtype(vocab_size):
    if vocab_size is not None and vocab_size < 65500:
        return np.uint16
    return np.int32


def compute_byte_boundaries(path, num_workers):
    file_size = os.path.getsize(path)
    if file_size == 0:
        return []
    chunk_size = file_size // num_workers
    if chunk_size == 0:
        return [(0, file_size)]
    boundaries = []
    with open(path, "rb") as f:
        for i in range(num_workers):
            if i == 0:
                start = 0
            else:
                f.seek(i * chunk_size)
                f.readline()
                start = f.tell()
            if i < num_workers - 1:
                end = (i + 1) * chunk_size
            else:
                end = file_size
            if start >= file_size:
                break
            if start < end:
                boundaries.append((start, end))
    return boundaries


def assign_files_to_workers(paths, num_workers):
    sizes = [(os.path.getsize(p), p) for p in paths]
    sizes.sort(reverse=True)
    actual_workers = min(num_workers, len(paths))
    worker_files = [[] for _ in range(actual_workers)]
    worker_sizes = [0] * actual_workers
    for size, path in sizes:
        min_worker = min(range(actual_workers), key=lambda i: worker_sizes[i])
        worker_files[min_worker].append(path)
        worker_sizes[min_worker] += size
    return worker_files


def _read_lines_from_region(path, byte_start, byte_end):
    with open(path, "r", encoding="utf-8") as f:
        f.seek(byte_start)
        while f.tell() < byte_end:
            line = f.readline()
            if not line:
                break
            yield line


def _read_lines_from_file(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            yield from f
    else:
        with open(path, "r", encoding="utf-8") as f:
            yield from f


def _worker_single_file(
    worker_id,
    input_path,
    byte_start,
    byte_end,
    output_prefix,
    tokenizer_name,
    text_key,
    append_eod,
    eod_id,
    batch_size,
    log_interval,
    dtype_code,
):
    return _worker_impl(
        worker_id=worker_id,
        line_iter_fn=lambda: _read_lines_from_region(input_path, byte_start, byte_end),
        output_prefix=output_prefix,
        tokenizer_name=tokenizer_name,
        text_key=text_key,
        append_eod=append_eod,
        eod_id=eod_id,
        batch_size=batch_size,
        log_interval=log_interval,
        dtype_code=dtype_code,
    )


def _worker_multi_file(
    worker_id,
    file_list,
    output_prefix,
    tokenizer_name,
    text_key,
    append_eod,
    eod_id,
    batch_size,
    log_interval,
    dtype_code,
):
    def line_iter():
        for path in file_list:
            yield from _read_lines_from_file(path)

    return _worker_impl(
        worker_id=worker_id,
        line_iter_fn=line_iter,
        output_prefix=output_prefix,
        tokenizer_name=tokenizer_name,
        text_key=text_key,
        append_eod=append_eod,
        eod_id=eod_id,
        batch_size=batch_size,
        log_interval=log_interval,
        dtype_code=dtype_code,
    )


def _worker_impl(
    worker_id,
    line_iter_fn,
    output_prefix,
    tokenizer_name,
    text_key,
    append_eod,
    eod_id,
    batch_size,
    log_interval,
    dtype_code,
):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    if append_eod and eod_id is None:
        eod_id = tokenizer.eos_token_id
        if eod_id is None:
            _log(f"worker {worker_id}: ERROR: --append-eod but tokenizer has no eos_token_id")
            sys.exit(1)

    dtype = _CODE_TO_DTYPE[dtype_code]
    shard_prefix = f"{output_prefix}_worker{worker_id:04d}"
    writer = ShardWriter(shard_prefix, dtype)

    total_docs = 0
    total_tokens = 0
    skipped = 0
    t0 = time.monotonic()

    batch_texts = []
    for line in line_iter_fn():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        text = data.get(text_key)
        if not text:
            skipped += 1
            continue
        batch_texts.append(text)

        if len(batch_texts) >= batch_size:
            _process_batch(batch_texts, tokenizer, writer, append_eod, eod_id)
            total_docs += len(batch_texts)
            total_tokens = sum(writer._sequence_lengths)
            batch_texts = []

            if total_docs % log_interval < batch_size:
                elapsed = time.monotonic() - t0
                _log(
                    f"worker {worker_id}: docs={total_docs:,} "
                    f"tokens={total_tokens:,} "
                    f"elapsed={elapsed:.1f}s "
                    f"({total_docs / elapsed:.0f} docs/s)"
                )

    if batch_texts:
        _process_batch(batch_texts, tokenizer, writer, append_eod, eod_id)
        total_docs += len(batch_texts)

    writer.finalize()
    total_tokens = sum(writer._sequence_lengths) if writer._sequence_lengths else 0
    elapsed = time.monotonic() - t0

    stats = {
        "worker_id": worker_id,
        "docs": writer.doc_count,
        "tokens": int(total_tokens),
        "skipped": skipped,
        "elapsed": round(elapsed, 2),
    }
    stats_path = shard_prefix + ".stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f)

    _log(
        f"worker {worker_id} DONE: docs={stats['docs']:,} "
        f"tokens={stats['tokens']:,} skipped={skipped} "
        f"elapsed={elapsed:.1f}s"
    )


def _process_batch(texts, tokenizer, writer, append_eod, eod_id):
    encoded = tokenizer(
        texts,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    for ids in encoded.input_ids:
        if not ids:
            continue
        if append_eod:
            ids.append(eod_id)
        writer.add_document(ids)


def run_merge(shard_prefixes, output_prefix, verify=False):
    merge_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "merge_indexed_datasets_streaming.py",
    )
    if not os.path.isfile(merge_script):
        _log(f"ERROR: merge script not found at {merge_script}")
        sys.exit(1)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as flist:
        for p in shard_prefixes:
            flist.write(p + "\n")
        flist_path = flist.name

    cmd = [
        sys.executable,
        merge_script,
        "--input",
        flist_path,
        "--output-prefix",
        output_prefix,
    ]
    if verify:
        cmd.append("--verify")

    _log(f"Merging {len(shard_prefixes)} shards -> {output_prefix}")
    subprocess.run(cmd, check=True)
    os.unlink(flist_path)


def main():
    parser = argparse.ArgumentParser(
        description="Parallel JSONL tokenizer producing Megatron .idx/.bin files"
    )
    parser.add_argument(
        "--input", required=True, help="JSONL file path or glob pattern"
    )
    parser.add_argument(
        "--output-prefix", required=True, help="Output prefix (no extension)"
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="HuggingFace model name/path for AutoTokenizer",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="Number of worker processes (default: cpu_count)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Lines per tokenization batch per worker (default: 512)",
    )
    parser.add_argument(
        "--text-key", default="text", help="JSON key for text field (default: text)"
    )
    parser.add_argument(
        "--append-eod",
        action="store_true",
        help="Append EOS token after each document",
    )
    parser.add_argument(
        "--eod-id",
        type=int,
        default=None,
        help="Override EOS token ID (default: tokenizer.eos_token_id)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Auto-merge shards after tokenization",
    )
    parser.add_argument(
        "--keep-shards",
        action="store_true",
        help="Keep shard files after merge",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run verification on merged output",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10000,
        help="Worker progress interval in docs (default: 10000)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Override vocab size for dtype selection (default: auto-detect)",
    )
    args = parser.parse_args()

    out_dir = os.path.dirname(args.output_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    t0 = time.monotonic()

    # Detect vocab size for dtype selection
    if args.vocab_size is not None:
        vocab_size = args.vocab_size
    else:
        _log("Loading tokenizer to detect vocab size...")
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
        vocab_size = tok.vocab_size
        del tok

    dtype = select_dtype(vocab_size)
    dtype_code = _DTYPE_CODES[np.dtype(dtype)]
    _log(f"vocab_size={vocab_size}, dtype={'uint16' if dtype == np.uint16 else 'int32'}")

    # Resolve input files
    input_files = sorted(glob.glob(args.input))
    if not input_files:
        if os.path.isfile(args.input):
            input_files = [args.input]
        else:
            _log(f"ERROR: no files matched '{args.input}'")
            sys.exit(1)

    total_size = sum(os.path.getsize(f) for f in input_files)
    _log(
        f"Input: {len(input_files)} file(s), {total_size / (1024**3):.2f} GiB total"
    )

    # Decide mode and launch workers
    processes = []

    if len(input_files) == 1 and not input_files[0].endswith(".gz"):
        # Single plain file: byte-offset chunking
        input_path = input_files[0]
        boundaries = compute_byte_boundaries(input_path, args.workers)
        actual_workers = len(boundaries)
        _log(f"Single-file mode: {actual_workers} workers via byte-offset chunking")

        for wid, (start, end) in enumerate(boundaries):
            p = multiprocessing.Process(
                target=_worker_single_file,
                args=(
                    wid, input_path, start, end,
                    args.output_prefix, args.tokenizer, args.text_key,
                    args.append_eod, args.eod_id, args.batch_size,
                    args.log_interval, dtype_code,
                ),
            )
            p.start()
            processes.append(p)
    else:
        # Multi-file or gzipped: assign whole files to workers
        file_assignments = assign_files_to_workers(input_files, args.workers)
        file_assignments = [fa for fa in file_assignments if fa]
        actual_workers = len(file_assignments)
        _log(f"Multi-file mode: {actual_workers} workers, {len(input_files)} files")

        for wid, file_list in enumerate(file_assignments):
            p = multiprocessing.Process(
                target=_worker_multi_file,
                args=(
                    wid, file_list,
                    args.output_prefix, args.tokenizer, args.text_key,
                    args.append_eod, args.eod_id, args.batch_size,
                    args.log_interval, dtype_code,
                ),
            )
            p.start()
            processes.append(p)

    _log(f"Launched {actual_workers} workers. Waiting...")

    for p in processes:
        p.join()

    failed = [i for i, p in enumerate(processes) if p.exitcode != 0]
    if failed:
        _log(f"ERROR: workers {failed} failed. Aborting.")
        sys.exit(1)

    # Collect stats
    shard_prefixes = []
    total_docs = 0
    total_tokens = 0
    total_skipped = 0
    for wid in range(actual_workers):
        prefix = f"{args.output_prefix}_worker{wid:04d}"
        stats_path = prefix + ".stats.json"
        if os.path.isfile(stats_path):
            with open(stats_path) as f:
                stats = json.load(f)
            total_docs += stats["docs"]
            total_tokens += stats["tokens"]
            total_skipped += stats["skipped"]
        if os.path.isfile(prefix + ".idx"):
            shard_prefixes.append(prefix)

    elapsed = time.monotonic() - t0
    _log(
        f"All workers done: docs={total_docs:,} tokens={total_tokens:,} "
        f"skipped={total_skipped:,} elapsed={elapsed:.1f}s "
        f"({total_docs / elapsed:.0f} docs/s, {total_tokens / elapsed / 1e6:.1f}M tok/s)"
    )

    if args.merge and shard_prefixes:
        run_merge(shard_prefixes, args.output_prefix, verify=args.verify)

        if not args.keep_shards:
            for prefix in shard_prefixes:
                for ext in (".idx", ".bin", ".stats.json"):
                    path = prefix + ext
                    if os.path.isfile(path):
                        os.remove(path)
            _log("Shard files cleaned up")

    _log(f"Total elapsed: {time.monotonic() - t0:.1f}s")


if __name__ == "__main__":
    main()

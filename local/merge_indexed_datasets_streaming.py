#!/usr/bin/env python3
"""Streaming merge of Megatron indexed datasets (.idx + .bin).

Merges terabytes of preprocessed data with ~84 MB peak RAM (at default
chunk size), compared to the original merge_datasets.py which loads all
index metadata into memory.

Only depends on numpy and the Python stdlib — no torch, no GPU.
"""

import argparse
import os
import shutil
import struct
import sys
import time
from dataclasses import dataclass
from typing import List

import numpy as np

_INDEX_HEADER = b"MMIDIDX\x00\x00"
_HEADER_SIZE = 34  # 9 (magic) + 8 (version) + 1 (dtype) + 8 (seq_count) + 8 (doc_count)

_DTYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 8, 7: 4, 8: 2}


@dataclass
class SourceInfo:
    prefix: str
    sequence_count: int
    document_count: int
    dtype_code: int
    bin_size: int


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def read_idx_header(idx_path: str) -> tuple:
    with open(idx_path, "rb") as f:
        header = f.read(9)
        if header != _INDEX_HEADER:
            raise ValueError(f"Bad header in {idx_path}")
        version = struct.unpack("<Q", f.read(8))[0]
        if version != 1:
            raise ValueError(f"Unsupported version {version} in {idx_path}")
        dtype_code = struct.unpack("<B", f.read(1))[0]
        sequence_count = struct.unpack("<Q", f.read(8))[0]
        document_count = struct.unpack("<Q", f.read(8))[0]
    return dtype_code, sequence_count, document_count


def discover_sources(input_path: str) -> List[str]:
    if os.path.isdir(input_path):
        prefixes = set()
        for name in os.listdir(input_path):
            base, ext = os.path.splitext(name)
            if ext in (".idx", ".bin"):
                full_prefix = os.path.join(input_path, base)
                if (
                    os.path.isfile(full_prefix + ".idx")
                    and os.path.isfile(full_prefix + ".bin")
                ):
                    prefixes.add(full_prefix)
        return sorted(prefixes)
    else:
        base_dir = os.path.dirname(os.path.abspath(input_path))
        prefixes = []
        with open(input_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if not os.path.isabs(line):
                    line = os.path.join(base_dir, line)
                prefixes.append(line)
        return prefixes


def phase1_scan_headers(prefixes: List[str]) -> List[SourceInfo]:
    sources = []
    ref_dtype = None
    for prefix in prefixes:
        idx_path = prefix + ".idx"
        bin_path = prefix + ".bin"
        dtype_code, seq_count, doc_count = read_idx_header(idx_path)
        if ref_dtype is None:
            ref_dtype = dtype_code
        elif dtype_code != ref_dtype:
            raise ValueError(
                f"dtype mismatch: {idx_path} has code {dtype_code}, "
                f"expected {ref_dtype}"
            )
        bin_size = os.path.getsize(bin_path)
        sources.append(SourceInfo(prefix, seq_count, doc_count, dtype_code, bin_size))
    return sources


def phase2_write_header(out_fd, dtype_code: int, total_seq: int, total_doc: int):
    out_fd.write(_INDEX_HEADER)
    out_fd.write(struct.pack("<Q", 1))
    out_fd.write(struct.pack("<B", dtype_code))
    out_fd.write(struct.pack("<Q", total_seq))
    out_fd.write(struct.pack("<Q", total_doc))


def phase3_stream_lengths(out_fd, sources: List[SourceInfo], chunk_size: int):
    for src in sources:
        with open(src.prefix + ".idx", "rb") as f:
            f.seek(_HEADER_SIZE)
            remaining = src.sequence_count
            while remaining > 0:
                n = min(chunk_size, remaining)
                data = f.read(n * 4)
                out_fd.write(data)
                remaining -= n


def phase4_write_pointers(
    out_fd,
    out_path: str,
    sources: List[SourceInfo],
    total_seq: int,
    dtype_itemsize: int,
    chunk_size: int,
):
    pointers_offset = _HEADER_SIZE + total_seq * 4
    out_fd.seek(pointers_offset)

    lengths_fd = open(out_path, "rb")
    lengths_fd.seek(_HEADER_SIZE)

    source_bin_start = np.int64(0)
    intra_ptr = np.int64(0)

    for src in sources:
        remaining = src.sequence_count
        while remaining > 0:
            n = min(chunk_size, remaining)
            raw = lengths_fd.read(n * 4)
            lengths_chunk = np.frombuffer(raw, dtype=np.dtype("<i4"))

            pointers = np.empty(n, dtype=np.dtype("<i8"))
            byte_lengths = lengths_chunk.astype(np.int64) * np.int64(dtype_itemsize)

            pointers[0] = source_bin_start + intra_ptr
            if n > 1:
                pointers[1:] = source_bin_start + intra_ptr + np.cumsum(byte_lengths[:-1])

            out_fd.write(pointers.tobytes())
            intra_ptr += byte_lengths.sum()
            remaining -= n

        source_bin_start += np.int64(src.bin_size)
        intra_ptr = np.int64(0)

    lengths_fd.close()


def phase5_stream_doc_indices(
    out_fd,
    sources: List[SourceInfo],
    total_seq: int,
    total_doc: int,
    chunk_size: int,
):
    doc_offset = _HEADER_SIZE + total_seq * 4 + total_seq * 8
    out_fd.seek(doc_offset)

    running_seq_offset = np.int64(0)

    for i, src in enumerate(sources):
        src_doc_start = (
            _HEADER_SIZE
            + src.sequence_count * 4
            + src.sequence_count * 8
        )

        with open(src.prefix + ".idx", "rb") as f:
            f.seek(src_doc_start)
            remaining = src.document_count
            skip_first = i > 0

            if skip_first:
                f.read(8)
                remaining -= 1

            while remaining > 0:
                n = min(chunk_size, remaining)
                raw = f.read(n * 8)
                chunk = np.frombuffer(raw, dtype=np.dtype("<i8")).copy()
                chunk += running_seq_offset
                out_fd.write(chunk.tobytes())
                remaining -= n

        running_seq_offset += np.int64(src.sequence_count)


def phase5b_stream_modes(
    out_fd,
    sources: List[SourceInfo],
    total_seq: int,
    total_doc: int,
    chunk_size: int,
):
    modes_offset = _HEADER_SIZE + total_seq * 12 + total_doc * 8
    out_fd.seek(modes_offset)

    for src in sources:
        src_modes_start = (
            _HEADER_SIZE
            + src.sequence_count * 12
            + src.document_count * 8
        )
        with open(src.prefix + ".idx", "rb") as f:
            f.seek(src_modes_start)
            remaining = src.sequence_count
            while remaining > 0:
                n = min(chunk_size, remaining)
                out_fd.write(f.read(n))
                remaining -= n


def phase6_stream_bins(out_path: str, sources: List[SourceInfo], buffer_mb: int):
    buf_size = buffer_mb * 1024 * 1024
    with open(out_path, "wb") as out_fd:
        for src in sources:
            with open(src.prefix + ".bin", "rb") as f:
                shutil.copyfileobj(f, out_fd, length=buf_size)


def verify(out_prefix: str, sources: List[SourceInfo], dtype_itemsize: int):
    _log("Verification: checking header...")
    idx_path = out_prefix + ".idx"
    dtype_code, seq_count, doc_count = read_idx_header(idx_path)

    expected_seq = sum(s.sequence_count for s in sources)
    expected_doc = sum(s.document_count for s in sources) - len(sources) + 1

    assert seq_count == expected_seq, f"seq count {seq_count} != {expected_seq}"
    assert doc_count == expected_doc, f"doc count {doc_count} != {expected_doc}"

    _log("Verification: checking .bin size...")
    expected_bin = sum(s.bin_size for s in sources)
    actual_bin = os.path.getsize(out_prefix + ".bin")
    assert actual_bin == expected_bin, f"bin size {actual_bin} != {expected_bin}"

    _log("Verification: spot-checking sequence_lengths...")
    merged_seq_offset = 0
    with open(idx_path, "rb") as merged_fd:
        for src in sources:
            if src.sequence_count == 0:
                continue
            check_positions = [0, src.sequence_count // 2, src.sequence_count - 1]
            with open(src.prefix + ".idx", "rb") as src_fd:
                for pos in check_positions:
                    src_fd.seek(_HEADER_SIZE + pos * 4)
                    src_val = struct.unpack("<i", src_fd.read(4))[0]

                    merged_pos = merged_seq_offset + pos
                    merged_fd.seek(_HEADER_SIZE + merged_pos * 4)
                    merged_val = struct.unpack("<i", merged_fd.read(4))[0]

                    assert src_val == merged_val, (
                        f"length mismatch at source {src.prefix} pos {pos}: "
                        f"{src_val} != {merged_val}"
                    )
            merged_seq_offset += src.sequence_count

    _log("Verification: spot-checking sequence_pointers...")
    pointers_start = _HEADER_SIZE + expected_seq * 4
    merged_seq_offset = 0
    source_bin_start = 0
    with open(idx_path, "rb") as merged_fd:
        for src in sources:
            if src.sequence_count == 0:
                source_bin_start += src.bin_size
                continue
            check_positions = [0, src.sequence_count // 2, src.sequence_count - 1]
            src_ptrs_start = _HEADER_SIZE + src.sequence_count * 4
            with open(src.prefix + ".idx", "rb") as src_fd:
                for pos in check_positions:
                    src_fd.seek(src_ptrs_start + pos * 8)
                    src_ptr = struct.unpack("<q", src_fd.read(8))[0]

                    merged_pos = merged_seq_offset + pos
                    merged_fd.seek(pointers_start + merged_pos * 8)
                    merged_ptr = struct.unpack("<q", merged_fd.read(8))[0]

                    expected_ptr = source_bin_start + src_ptr
                    assert merged_ptr == expected_ptr, (
                        f"pointer mismatch at source {src.prefix} pos {pos}: "
                        f"merged={merged_ptr} expected={expected_ptr}"
                    )
            merged_seq_offset += src.sequence_count
            source_bin_start += src.bin_size

    _log("Verification: checking last document index...")
    doc_start = _HEADER_SIZE + expected_seq * 12
    with open(idx_path, "rb") as merged_fd:
        merged_fd.seek(doc_start + (expected_doc - 1) * 8)
        last_doc = struct.unpack("<q", merged_fd.read(8))[0]
        assert last_doc == expected_seq, (
            f"last doc index {last_doc} != total sequences {expected_seq}"
        )

    _log("Verification passed.")


def main():
    parser = argparse.ArgumentParser(
        description="Streaming merge of Megatron indexed datasets"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Directory of .idx/.bin pairs, or text file with one prefix per line",
    )
    parser.add_argument(
        "--output-prefix",
        required=True,
        help="Output file prefix (no extension)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="Elements per chunk (default: 1M, ~4-8 MB RAM per chunk)",
    )
    parser.add_argument(
        "--bin-copy-buffer-mb",
        type=int,
        default=64,
        help="Buffer size in MB for .bin concatenation (default: 64)",
    )
    parser.add_argument(
        "--multimodal",
        action="store_true",
        help="Enable multimodal sequence_modes handling",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run spot-check verification after merge",
    )
    args = parser.parse_args()

    out_dir = os.path.dirname(args.output_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    t0 = time.monotonic()

    _log("Discovering sources...")
    prefixes = discover_sources(args.input)
    if not prefixes:
        _log("No .idx/.bin pairs found. Exiting.")
        sys.exit(1)
    _log(f"Found {len(prefixes)} source datasets")

    _log("Phase 1: scanning headers...")
    sources = phase1_scan_headers(prefixes)

    dtype_code = sources[0].dtype_code
    dtype_itemsize = _DTYPE_SIZES[dtype_code]
    total_seq = sum(s.sequence_count for s in sources)
    total_doc = sum(s.document_count for s in sources) - len(sources) + 1
    total_bin = sum(s.bin_size for s in sources)

    _log(f"  total sequences: {total_seq:,}")
    _log(f"  total documents: {total_doc:,}")
    _log(f"  total .bin size: {total_bin / (1024**3):.2f} GiB")
    _log(f"  dtype code: {dtype_code} (itemsize={dtype_itemsize})")

    idx_tmp = args.output_prefix + ".idx.tmp"
    bin_tmp = args.output_prefix + ".bin.tmp"

    with open(idx_tmp, "w+b") as out_fd:
        _log("Phase 2: writing header...")
        phase2_write_header(out_fd, dtype_code, total_seq, total_doc)

        _log("Phase 3: streaming sequence_lengths...")
        phase3_stream_lengths(out_fd, sources, args.chunk_size)
        out_fd.flush()

        _log("Phase 4: computing and writing sequence_pointers...")
        phase4_write_pointers(
            out_fd, idx_tmp, sources, total_seq, dtype_itemsize, args.chunk_size
        )

        _log("Phase 5: streaming document_indices...")
        phase5_stream_doc_indices(
            out_fd, sources, total_seq, total_doc, args.chunk_size
        )

        if args.multimodal:
            _log("Phase 5b: streaming sequence_modes...")
            phase5b_stream_modes(
                out_fd, sources, total_seq, total_doc, args.chunk_size
            )

    _log("Phase 6: streaming .bin files...")
    phase6_stream_bins(bin_tmp, sources, args.bin_copy_buffer_mb)

    os.rename(idx_tmp, args.output_prefix + ".idx")
    os.rename(bin_tmp, args.output_prefix + ".bin")

    elapsed = time.monotonic() - t0
    _log(f"Done in {elapsed:.1f}s")

    if args.verify:
        verify(args.output_prefix, sources, dtype_itemsize)


if __name__ == "__main__":
    main()

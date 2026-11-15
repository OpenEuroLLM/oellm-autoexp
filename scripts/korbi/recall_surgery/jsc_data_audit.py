#!/usr/bin/env python3
"""Byte-identity + tokenizer fingerprint of Megatron .bin/.idx prefixes, run identically on MN5 and JSC (§19.1).

Stdlib only (MN5 / JUPITER login nodes). One process per prefix keeps each process far below the MN5 login
nodes' 300 s CPU rlimit; drive it with xargs -P.

    jsc_data_audit.py ROOT REL [REL ...]          one TSV line per REL (prefix relative to ROOT, no suffix)

Columns: rel, realpath(.bin), bin_size, idx_size, sha256 of 16 x 4 MiB ranges of the .bin (evenly spaced, first
and last included), sha256 of the full .idx, dtype code, n_seq, n_doc, and a token fingerprint over 64 evenly
spaced sequences: share starting with <bos>=1, share ending with <eos>=2, count of ids 13-71 (the agent delimiter
slots of tokenizer-256k-agent: 0 in text left on tokenizer-256k), count of id 0, max id.
"""

from __future__ import annotations

import hashlib
import os
import struct
import sys
from array import array

RANGES, RANGE = 16, 4 << 20
DTYPES = {1: ("B", 1), 2: ("b", 1), 3: ("h", 2), 4: ("i", 4), 5: ("q", 8), 8: ("H", 2)}
SAMPLES = 64


def range_hash(path: str, size: int) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        if size <= RANGES * RANGE:
            h.update(f.read())
        else:
            step = (size - RANGE) // (RANGES - 1)
            for k in range(RANGES):
                f.seek(k * step)
                h.update(f.read(RANGE))
    return h.hexdigest()


def full_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_idx(path: str):
    with open(path, "rb") as f:
        magic = f.read(9)
        if magic != b"MMIDIDX\x00\x00":
            raise ValueError(f"{path}: bad magic {magic!r}")
        (version,) = struct.unpack("<Q", f.read(8))
        (code,) = struct.unpack("<B", f.read(1))
        n_seq, n_doc = struct.unpack("<QQ", f.read(16))
        off = f.tell()
    return code, n_seq, n_doc, off


def seq_entry(idx_path: str, off: int, n_seq: int, i: int):
    with open(idx_path, "rb") as f:
        f.seek(off + 4 * i)
        (length,) = struct.unpack("<i", f.read(4))
        f.seek(off + 4 * n_seq + 8 * i)
        (ptr,) = struct.unpack("<q", f.read(8))
    return length, ptr


def fingerprint(bin_path: str, idx_path: str, code: int, n_seq: int, off: int):
    tc, width = DTYPES[code]
    bos = eos = agent = zero = 0
    mx = 0
    n = min(SAMPLES, n_seq)
    with open(bin_path, "rb") as fb:
        for k in range(n):
            i = k * (n_seq - 1) // max(n - 1, 1)
            length, ptr = seq_entry(idx_path, off, n_seq, i)
            fb.seek(ptr)
            a = array(tc)
            a.frombytes(fb.read(length * width))
            if not a:
                continue
            bos += a[0] == 1
            eos += a[-1] == 2
            agent += sum(1 for t in a if 13 <= t <= 71)
            zero += sum(1 for t in a if t == 0)
            mx = max(mx, max(a))
    return f"{bos / n:.3f}", f"{eos / n:.3f}", str(agent), str(zero), str(mx)


def audit(root: str, rel: str) -> str:
    b, i = os.path.join(root, rel + ".bin"), os.path.join(root, rel + ".idx")
    if not (os.path.exists(b) and os.path.exists(i)):
        return "\t".join([rel, "MISSING"] + [""] * 12)
    bs, is_ = os.path.getsize(b), os.path.getsize(i)
    code, n_seq, n_doc, off = read_idx(i)
    fp = fingerprint(b, i, code, n_seq, off)
    return "\t".join([rel, os.path.realpath(b), str(bs), str(is_), range_hash(b, bs), full_hash(i), str(code),
                      str(n_seq), str(n_doc), *fp])


if __name__ == "__main__":
    root = sys.argv[1]
    for rel in sys.argv[2:]:
        print(audit(root, rel), flush=True)

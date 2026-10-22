"""Shared pieces of the flag decontamination (docs/flag_post_decontamination.md).

Matching happens on token ids, never on detokenised text. The OELLM 256k tokenizer is a
SentencePiece BPE with split_by_whitespace=True, an identity normaliser and no dummy prefix, so
no piece spans a space and a word tokenises the same wherever it stands -- except for its leading
'▁', which depends on whether a space, a newline or nothing precedes it. The canonical stream
removes that: every id maps through a lookup table to the id of its piece with surrounding
whitespace stripped, and pieces that are only whitespace (and control tokens: <bos>, <eos>, ...)
are dropped. Matching canonical n-grams is then matching word sequences up to whitespace.

N-gram hashes: a polynomial over canonical ids in int64 arithmetic (wrapping), seeded with n and
finalised with a xor-shift-multiply mix, computed by the same torch code for the reference table
(CPU) and the scan (GPU), so the two agree bit for bit.
"""

from __future__ import annotations

import json
import os
import struct

import numpy as np
import torch

NGRAM_SIZES = (8, 13)          # n used per reference segment: 13, or 8 for short segments
SHORT_BELOW = 20               # segments with fewer canonical tokens use n = 8
DROP = -1

_M = -7046029254386353131      # 0x9E3779B97F4A7C15 as int64
_MIX1 = -4658895280553007687   # 0xBF58476D1CE4E5B9
_MIX2 = -7723592293110705685   # 0x94D049BB133111EB


def _mix(h: torch.Tensor) -> torch.Tensor:
    # splitmix64 finaliser; >> on int64 is arithmetic, identical on CPU and GPU
    h = h ^ ((h >> 30) & 0x3FFFFFFFF)
    h = h * _MIX1
    h = h ^ ((h >> 27) & 0x1FFFFFFFFF)
    h = h * _MIX2
    h = h ^ ((h >> 31) & 0x1FFFFFFFF)
    return h


def ngram_hashes(canon: torch.Tensor, n: int) -> torch.Tensor:
    """int64 hashes of all len(canon)-n+1 windows of an int64 canonical-id tensor."""
    L = canon.shape[0] - n + 1
    if L <= 0:
        return torch.empty(0, dtype=torch.int64, device=canon.device)
    h = torch.full((L,), n * 1_000_003, dtype=torch.int64, device=canon.device)
    for k in range(n):
        h = h * _M + canon[k:k + L] + 1
    return _mix(h)


def build_lut(sp) -> tuple[np.ndarray, list[str]]:
    """Canonical id per token id (DROP for whitespace / control) from a SentencePieceProcessor."""
    V = sp.get_piece_size()
    lut = np.full(V, DROP, dtype=np.int64)
    keys: dict[str, int] = {}
    for i in range(V):
        if sp.is_control(i) or sp.is_unknown(i):
            continue
        if sp.is_byte(i):
            b = int(sp.id_to_piece(i)[1:-1], 16)            # <0x0A>
            if chr(b).isspace():
                continue
            key = f"<byte {b:02X}>"
        else:
            key = sp.id_to_piece(i).replace("▁", " ").strip()
            if not key:
                continue
        lut[i] = keys.setdefault(key, len(keys))
    return lut, list(keys)


def save_lut(path: str, lut: np.ndarray) -> None:
    np.save(path, lut)


def load_lut(path: str) -> np.ndarray:
    return np.load(path)


# ---- Megatron indexed dataset (.idx) I/O, the format of megatron/core/datasets/indexed_dataset.py
_HEADER = b"MMIDIDX\x00\x00"
_DTYPES = {1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32, 5: np.int64, 6: np.float64,
           7: np.float32, 8: np.uint16}


class Index:
    """Read-only view of a Megatron .idx file (sequence lengths / pointers, document indices)."""

    def __init__(self, path: str):
        with open(path, "rb") as f:
            assert f.read(9) == _HEADER, f"bad header {path}"
            assert struct.unpack("<Q", f.read(8))[0] == 1
            self.dtype_code = struct.unpack("<B", f.read(1))[0]
            self.seq_count = struct.unpack("<Q", f.read(8))[0]
            self.doc_count = struct.unpack("<Q", f.read(8))[0]
            off = f.tell()
        self.dtype = _DTYPES[self.dtype_code]
        mm = np.memmap(path, mode="r", dtype=np.uint8)
        self.lengths = np.frombuffer(mm, np.int32, self.seq_count, off)
        off += self.lengths.nbytes
        self.pointers = np.frombuffer(mm, np.int64, self.seq_count, off)
        off += self.pointers.nbytes
        self.doc_idx = np.frombuffer(mm, np.int64, self.doc_count, off)
        off += self.doc_idx.nbytes
        self.multimodal_tail = mm.shape[0] - off   # bytes of sequence modes, if any


def write_index(path: str, dtype_code: int, lengths: np.ndarray, pointers: np.ndarray,
                doc_idx: np.ndarray) -> None:
    """Write an .idx with explicit pointers (into an unchanged .bin), atomically."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(_HEADER)
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<B", dtype_code))
        f.write(struct.pack("<Q", len(lengths)))
        f.write(struct.pack("<Q", len(doc_idx)))
        f.write(np.ascontiguousarray(lengths, dtype=np.int32).tobytes())
        f.write(np.ascontiguousarray(pointers, dtype=np.int64).tobytes())
        f.write(np.ascontiguousarray(doc_idx, dtype=np.int64).tobytes())
    os.replace(tmp, path)


def filter_index(idx: Index, keep_doc: np.ndarray):
    """(lengths, pointers, doc_idx) keeping the documents where keep_doc is True."""
    starts, ends = idx.doc_idx[:-1], idx.doc_idx[1:]
    n_seq = ends - starts
    keep_seq = np.repeat(keep_doc, n_seq)
    lengths = idx.lengths[keep_seq]
    pointers = idx.pointers[keep_seq]
    doc_idx = np.concatenate([[0], np.cumsum(n_seq[keep_doc])]).astype(np.int64)
    return lengths, pointers, doc_idx


def read_json(path: str):
    with open(path) as f:
        return json.load(f)

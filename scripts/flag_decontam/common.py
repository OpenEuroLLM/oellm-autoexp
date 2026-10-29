"""Shared pieces of the flag decontamination (docs/flag_post_decontamination.md).

Matching happens on token ids, never on detokenised text. The OELLM 256k tokenizer is a
SentencePiece BPE with split_by_whitespace=True, an identity normaliser and no dummy prefix, so
no piece spans a space and a word tokenises the same wherever it stands -- except for its leading
'▁', which depends on whether a space, a newline or nothing precedes it. The canonical stream
removes that: every id maps through a lookup table to the id of its piece with surrounding
whitespace stripped, and pieces that are only whitespace (and control tokens: <bos>, <eos>, ...)
are dropped. Matching canonical n-grams is then matching word sequences up to whitespace.
v3 (build_lut(words=True)): the key is the piece lower-cased with every non-alphanumeric
character removed, and pieces without letters or digits are dropped -- matching is then on word
fragments, insensitive to case, punctuation, LaTeX delimiters and spacing around brackets.

Matching unit (v2): the longest *run* of consecutive matching windows of one segment in one
sequence (gaps of up to n // 2 windows bridged: pruned common n-grams inside a copied passage),
in canonical tokens. A verbatim copy is one run as long as the segment; a changed word removes n
consecutive windows and splits it; n-grams scattered over a long document are many short runs.

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

NGRAM_SIZES = (13, 24, 40)     # n per reference segment: the smallest whose n-grams span MIN_GRAM_CHARS
MIN_GRAM_CHARS = 30            # median characters of a segment's n-grams (byte-fallback scripts need larger n)
MIN_SEG_TOKENS = 20            # shorter segments (answer options, short premises) are not matched alone
MIN_SEG_CHARS = 80
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


def word_key(piece: str) -> str:
    """v3 canonical key of a piece: lower-cased, every non-alphanumeric character removed."""
    return "".join(ch for ch in piece.lower() if ch.isalnum())


def build_lut(sp, words: bool = True) -> tuple[np.ndarray, list[str]]:
    """Canonical id per token id (DROP for whitespace / control) from a SentencePieceProcessor.

    words=True (v3): the key is word_key(piece) -- case, punctuation, LaTeX delimiters, spacing
    around brackets no longer change the stream; pieces without letters or digits are dropped.
    words=False (v1 / v2): the piece with surrounding whitespace stripped.
    """
    V = sp.get_piece_size()
    lut = np.full(V, DROP, dtype=np.int64)
    keys: dict[str, int] = {}
    for i in range(V):
        if sp.is_control(i) or sp.is_unknown(i):
            continue
        if sp.is_byte(i):
            b = int(sp.id_to_piece(i)[1:-1], 16)            # <0x0A>
            if chr(b).isspace() or (words and b < 128 and not chr(b).isalnum()):
                continue
            key = f"<byte {b:02X}>"
        else:
            piece = sp.id_to_piece(i).replace("\u2581", " ")
            key = word_key(piece) if words else piece.strip()
            if not key:
                continue
        lut[i] = keys.setdefault(key, len(keys))
    return lut, list(keys)


def key_chars(keys: list[str]) -> np.ndarray:
    """Characters per canonical id; a byte-fallback piece counts 1/3 (a UTF-8 character is 2-4 bytes)."""
    return np.array([1 / 3 if k.startswith("<byte ") else len(k) for k in keys], dtype=np.float32)


def runs(pos: torch.Tensor, seg: torch.Tensor, seq: torch.Tensor, n_of_seg: torch.Tensor):
    """Runs of matching windows: (seq, seg, start, end, windows) per run, end exclusive in tokens.

    pos: position of each matching window in the canonical stream (a window covers pos .. pos+n-1),
    seg / seq: its segment and sequence. Windows of one segment whose positions differ by at most
    n // 2 + 1 belong to one run.
    """
    if pos.numel() == 0:
        e = pos.new_zeros(0)
        return e, e, e, e, e
    key = seg * (int(pos.max()) + 2) + pos
    order = torch.argsort(key)
    pos, seg, seq = pos[order], seg[order], seq[order]
    n = n_of_seg[seg]
    new = torch.ones_like(pos, dtype=torch.bool)
    new[1:] = (seg[1:] != seg[:-1]) | (seq[1:] != seq[:-1]) | (pos[1:] - pos[:-1] > n[1:] // 2 + 1)
    first = torch.nonzero(new).squeeze(1)
    last = torch.cat([first[1:] - 1, first.new_tensor([pos.numel() - 1])])
    return seq[first], seg[first], pos[first], pos[last] + n[last], last - first + 1


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


def excluded_segments(refs_path, primary_src, exclude):
    """Boolean mask of the segments whose every source is in `exclude` (source names): from the
    attribution file next to the refs (refs.attrib.jsonl.gz: all benchmarks holding the exact text),
    else from each segment's primary source. A text shared with a kept benchmark stays."""
    import gzip
    exclude = set(exclude)
    mask = np.array([s in exclude for s in primary_src], bool)
    p = refs_path.replace(".jsonl.gz", ".attrib.jsonl.gz")
    if not exclude or not os.path.exists(p):
        return mask
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if mask[i]:
                srcs = {a[0] for a in json.loads(line)["also_in"] if a[3]}
                mask[i] = srcs <= exclude
    return mask


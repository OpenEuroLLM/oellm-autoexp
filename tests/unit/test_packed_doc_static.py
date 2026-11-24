"""PACKED_DOC_STATIC_SEGMENTS: the padded cu_seqlens path.

The mode exists so that a CUDA graph can be replayed with inter-document masking on: the merged
cu_seqlens keeps ``micro_batch_size * N + 1`` entries instead of the real document count, and the
extra entries are zero-length segments. These tests pin the two things that make that safe:

  * the segment lengths the attention sees are unchanged (the padded tail only adds zeros), and
  * the merged length depends only on the micro-batch size, never on the documents.

Needs torch + the Megatron submodule (CPU only).
"""

from __future__ import annotations

import os
import sys

import pytest

torch = pytest.importorskip("torch")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "submodules", "Megatron-LM"))

packed_doc_static = pytest.importorskip("megatron.core.packed_doc_static")
from megatron.core.utils import _merge_cu_seqlens_across_micro_batch  # noqa: E402

SEQ = 64
N = 256


def _row(doc_lengths, pad_to, seq_length=SEQ):
    """A padded cu_seqlens row exactly as gpt_dataset builds it."""
    cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(doc_lengths), 0).tolist()), dtype=torch.int32)
    row = torch.full((pad_to,), seq_length, dtype=torch.int32)
    row[: cu.numel()] = cu
    return row


def _segment_lengths(merged):
    """The non-zero segment lengths, i.e. what the attention kernel actually attends over."""
    diffs = (merged[1:] - merged[:-1]).tolist()
    return [int(d) for d in diffs if d]


@pytest.fixture()
def static_on(monkeypatch):
    monkeypatch.setattr(packed_doc_static, "STATIC_SEGMENTS", N)


@pytest.fixture()
def static_off(monkeypatch):
    monkeypatch.setattr(packed_doc_static, "STATIC_SEGMENTS", 0)


def test_padded_length_follows_the_mode(static_on):
    assert packed_doc_static.padded_length(SEQ) == N + 1
    assert packed_doc_static.enabled()


def test_padded_length_is_unchanged_without_the_mode(static_off):
    assert packed_doc_static.padded_length(65536) == 65537
    assert not packed_doc_static.enabled()


def test_single_micro_batch_keeps_the_padded_row(static_on):
    docs = [[20, 30, 14]]
    rows = torch.stack([_row(d, N + 1) for d in docs])
    merged = _merge_cu_seqlens_across_micro_batch(rows, SEQ)

    assert merged.numel() == N + 1, "the merged length must not depend on the document count"
    assert _segment_lengths(merged) == [20, 30, 14]
    assert torch.equal(merged, rows[0])


def test_two_micro_batches_are_static_and_keep_their_segments(static_on):
    docs = [[20, 30, 14], [64]]
    rows = torch.stack([_row(d, N + 1) for d in docs])
    merged = _merge_cu_seqlens_across_micro_batch(rows, SEQ)

    assert merged.numel() == 2 * N + 1
    # same documents in the same order, so the same non-zero segment lengths as the stripped merge
    assert _segment_lengths(merged) == [20, 30, 14, 64]
    assert int(merged[-1]) == 2 * SEQ


def test_the_merged_length_does_not_depend_on_the_documents(static_on):
    # same micro-batch size (2), very different document layouts: one row of 64 documents, one
    # row of 4, one row of a single document
    layouts = [
        [[64], [64]],
        [[1] * 64, [64]],
        [[20, 30, 14], [64]],
        [[8] * 8, [2] * 32],
    ]
    lengths = set()
    for docs in layouts:
        rows = torch.stack([_row(d, N + 1) for d in docs])
        lengths.add(int(_merge_cu_seqlens_across_micro_batch(rows, SEQ).numel()))
    assert lengths == {2 * N + 1}


def test_the_cu_seqlens_never_decrease(static_on):
    docs = [[20, 30, 14], [64], [5, 5, 5, 49]]
    rows = torch.stack([_row(d, N + 1) for d in docs])
    merged = _merge_cu_seqlens_across_micro_batch(rows, SEQ)
    assert torch.all(merged[1:] >= merged[:-1]), "the padded tail must be equal neighbours"


def test_without_the_mode_the_strip_is_unchanged(static_off):
    docs = [[20, 30, 14], [64]]
    rows = torch.stack([_row(d, SEQ + 1) for d in docs])
    merged = _merge_cu_seqlens_across_micro_batch(rows, SEQ)
    assert merged.tolist() == [0, 20, 50, 64, 128]

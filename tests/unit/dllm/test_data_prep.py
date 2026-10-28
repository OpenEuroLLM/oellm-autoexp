"""Tests for oellm_autoexp.dllm.data_prep (shard assignment and the .idx
reader)."""

from __future__ import annotations

import struct

import pytest

from oellm_autoexp.dllm.data_prep import assign_row_groups, read_idx_header


def test_assign_row_groups_covers_every_group_once_in_order() -> None:
    plan = assign_row_groups([3, 0, 5, 2], num_shards=4)
    flat = [pair for shard in plan for pair in shard]
    expected = [(0, 0), (0, 1), (0, 2), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (3, 0), (3, 1)]
    assert flat == expected
    sizes = [len(shard) for shard in plan]
    assert max(sizes) - min(sizes) <= 1


def test_assign_row_groups_is_deterministic_and_last_shard_is_the_tail() -> None:
    counts = [726, 729, 727, 183]
    plan = assign_row_groups(counts, num_shards=16)
    assert plan == assign_row_groups(counts, num_shards=16)
    assert plan[-1][-1] == (3, 182)


@pytest.mark.parametrize(("counts", "shards"), [([1, 1], 3), ([4], 0)])
def test_assign_row_groups_rejects_bad_requests(counts: list[int], shards: int) -> None:
    with pytest.raises(ValueError):
        assign_row_groups(counts, shards)


def test_read_idx_header_counts_tokens(tmp_path) -> None:
    np = pytest.importorskip("numpy")
    lengths = np.array([3, 5, 2], dtype=np.int32)
    path = tmp_path / "x_text_document.idx"
    with open(path, "wb") as handle:
        handle.write(b"MMIDIDX\x00\x00")
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<B", 4))  # int32 tokens
        handle.write(struct.pack("<Q", len(lengths)))
        handle.write(struct.pack("<Q", len(lengths) + 1))
        handle.write(lengths.tobytes(order="C"))
        pointers = np.array([0, 12, 32], dtype=np.int64)
        handle.write(pointers.tobytes(order="C"))
        handle.write(np.arange(len(lengths) + 1, dtype=np.int64).tobytes(order="C"))
    header = read_idx_header(path)
    assert header["dtype"] == "int32"
    assert header["num_sequences"] == 3
    assert header["num_documents"] == 3
    assert header["num_tokens"] == 10


def test_read_idx_header_rejects_other_files(tmp_path) -> None:
    pytest.importorskip("numpy")
    path = tmp_path / "bad.idx"
    path.write_bytes(b"NOTANIDX" + b"\x00" * 40)
    with pytest.raises(ValueError):
        read_idx_header(path)

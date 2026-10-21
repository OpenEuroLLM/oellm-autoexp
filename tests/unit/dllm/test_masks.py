"""Invariant I1 and unit tests for the block-diffusion and AR attention masks
(spec section 5).

Each mask_mod is compared with a dense Python-loop reference, once through FlexAttention's
``create_mask`` and once through ``materialize_mask``.

Run with:
    pytest tests/unit/dllm/test_masks.py -v
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch
from torch.nn.attention.flex_attention import BlockMask, create_mask

from oellm_autoexp.dllm.masks import (
    build_block_mask,
    compute_doc_ids,
    dense_ar_reference,
    dense_bd3_reference,
    make_ar_mask_mod,
    make_bd3_mask_mod,
    materialize_mask,
)

EOD_ID = 7
_BATCH = 2
_LENGTHS = (8, 16, 24, 32)
_BLOCK_SIZES = (1, 2, 4, 8)
_BD3_CASES = [
    (length, block) for length in _LENGTHS for block in _BLOCK_SIZES if length % block == 0
]

_EXPECTED_L4_B2 = torch.tensor(
    [
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1],
    ],
    dtype=torch.bool,
)


def _random_x0(length: int, batch: int, seed: int) -> torch.Tensor:
    """Return random clean tokens in [0, 6] with 0 to 3 EOD tokens (id 7) per
    row.

    Args:
        length: Sequence length L.
        batch: Number of rows.
        seed: Seed of the generator that draws the tokens and the EOD positions.

    Returns:
        LongTensor with shape [batch, length].
    """
    gen = torch.Generator().manual_seed(seed)
    x0 = torch.randint(0, EOD_ID, (batch, length), generator=gen, dtype=torch.long)
    for row in range(batch):
        n_eod = int(torch.randint(0, 4, (1,), generator=gen))
        positions = torch.randperm(length, generator=gen)[:n_eod]
        x0[row, positions] = EOD_ID
    return x0


def _doc_rows(doc_ids: torch.Tensor | None, batch: int) -> list[list[int] | None]:
    """Return one document row per batch index, or None rows when there are no
    document ids."""
    if doc_ids is None:
        return [None] * batch
    return doc_ids.tolist()


@pytest.mark.parametrize("use_docs", [False, True], ids=["no_docs", "docs"])
@pytest.mark.parametrize(("length", "block"), _BD3_CASES)
def test_bd3_mask_matches_dense_reference(length: int, block: int, use_docs: bool) -> None:
    """I1: create_mask and materialize_mask equal the Python-loop reference in every row."""
    x0 = _random_x0(length, _BATCH, seed=1000 * length + block + 1)
    doc_ids = compute_doc_ids(x0, EOD_ID) if use_docs else None
    if doc_ids is not None:
        assert any(len(set(row)) > 1 for row in doc_ids.tolist()), "documents must split a row"
    expected = torch.stack(
        [dense_bd3_reference(length, block, row) for row in _doc_rows(doc_ids, _BATCH)]
    )
    mask_mod = make_bd3_mask_mod(length, block, doc_ids)

    model_batch = _BATCH if use_docs else 1
    flex = create_mask(
        mask_mod, B=model_batch, H=1, Q_LEN=2 * length, KV_LEN=2 * length, device="cpu"
    )
    assert flex.dtype == torch.bool
    assert flex.shape == (model_batch, 1, 2 * length, 2 * length)
    for row in range(model_batch):
        assert torch.equal(flex[row, 0], expected[row])

    materialized = materialize_mask(mask_mod, _BATCH, 2 * length, 2 * length)
    assert torch.equal(materialized, expected)


@pytest.mark.parametrize("use_docs", [False, True], ids=["no_docs", "docs"])
@pytest.mark.parametrize("length", _LENGTHS)
def test_ar_mask_matches_dense_reference(length: int, use_docs: bool) -> None:
    """I1 for AR: create_mask and materialize_mask equal the causal reference in every row."""
    x0 = _random_x0(length, _BATCH, seed=2000 + length)
    doc_ids = compute_doc_ids(x0, EOD_ID) if use_docs else None
    if doc_ids is not None:
        assert any(len(set(row)) > 1 for row in doc_ids.tolist()), "documents must split a row"
    expected = torch.stack([dense_ar_reference(length, row) for row in _doc_rows(doc_ids, _BATCH)])
    mask_mod = make_ar_mask_mod(length, doc_ids)

    model_batch = _BATCH if use_docs else 1
    flex = create_mask(mask_mod, B=model_batch, H=1, Q_LEN=length, KV_LEN=length, device="cpu")
    assert flex.dtype == torch.bool
    assert flex.shape == (model_batch, 1, length, length)
    for row in range(model_batch):
        assert torch.equal(flex[row, 0], expected[row])

    assert torch.equal(materialize_mask(mask_mod, _BATCH, length, length), expected)


def test_bd3_literal_matrix_for_l4_b2_without_docs() -> None:
    """The hand-checked 8x8 mask for L = 4 and B = 2 matches the reference and
    create_mask."""
    assert torch.equal(dense_bd3_reference(4, 2, None), _EXPECTED_L4_B2)

    mask_mod = make_bd3_mask_mod(4, 2, None)
    flex = create_mask(mask_mod, B=1, H=1, Q_LEN=8, KV_LEN=8, device="cpu")
    assert torch.equal(flex[0, 0], _EXPECTED_L4_B2)


def test_compute_doc_ids_counts_eod_tokens_before_each_position() -> None:
    """The EOD token ends its own document, so the count excludes the current
    position."""
    x0 = torch.tensor([[5, 7, 5, 5, 7, 5]])
    doc_ids = compute_doc_ids(x0, EOD_ID)
    assert doc_ids.dtype == torch.long
    assert torch.equal(doc_ids, torch.tensor([[0, 0, 1, 1, 1, 2]]))


def test_compute_doc_ids_rejects_one_dimensional_input() -> None:
    """compute_doc_ids requires the [b, L] layout."""
    with pytest.raises(ValueError, match="must have shape"):
        compute_doc_ids(torch.tensor([5, 7, 5]), EOD_ID)


def test_bd3_mask_mod_rejects_block_size_that_does_not_divide_length() -> None:
    """make_bd3_mask_mod(10, 4, None) raises because 10 is not a multiple of
    4."""
    with pytest.raises(ValueError, match="multiple of B"):
        make_bd3_mask_mod(10, 4, None)


@pytest.mark.parametrize("block", [0, -2])
def test_bd3_mask_mod_rejects_block_size_below_one(block: int) -> None:
    """A block size below 1 is rejected before any modulo is taken."""
    with pytest.raises(ValueError, match="B must be >= 1"):
        make_bd3_mask_mod(8, block, None)


def test_bd3_mask_mod_rejects_doc_ids_with_wrong_length() -> None:
    """Document ids must have one column per position."""
    doc_ids = torch.zeros((2, 4), dtype=torch.long)
    with pytest.raises(ValueError, match="doc_ids must have shape"):
        make_bd3_mask_mod(8, 2, doc_ids)


def test_build_block_mask_on_cpu_with_and_without_batch() -> None:
    """build_block_mask builds for L = 128, B = 8 on CPU, with and without
    document ids.

    The BlockMask covers Q_LEN = KV_LEN = 256. Its block-level dense
    form (FlexAttention's default 128-wide blocks) marks a block when
    the element mask has at least one allowed entry in it.
    """
    length, block = 128, 8
    q_len = kv_len = 2 * length
    flex_block = 128
    doc_ids = compute_doc_ids(_random_x0(length, _BATCH, seed=7), EOD_ID)

    for mask_mod, batch in (
        (make_bd3_mask_mod(length, block, None), None),
        (make_bd3_mask_mod(length, block, doc_ids), _BATCH),
    ):
        rows = 1 if batch is None else batch
        block_mask = build_block_mask(mask_mod, batch, q_len, kv_len, "cpu")
        assert isinstance(block_mask, BlockMask)
        assert block_mask.shape == (rows, 1, q_len, kv_len)

        n_q_blocks, n_kv_blocks = q_len // flex_block, kv_len // flex_block
        element_mask = materialize_mask(mask_mod, rows, q_len, kv_len)
        expected_blocks = (
            element_mask.reshape(rows, n_q_blocks, flex_block, n_kv_blocks, flex_block)
            .any(dim=2)
            .any(dim=3)
        )
        dense_blocks = block_mask.to_dense().bool().reshape(rows, n_q_blocks, n_kv_blocks)
        assert torch.equal(dense_blocks, expected_blocks)


@pytest.mark.parametrize(("length", "block"), [(16, 4), (24, 8), (32, 8)])
def test_noisy_rows_do_not_depend_on_tokens_of_their_own_block(length: int, block: int) -> None:
    """D-039: whether a token of block k is EOD never changes the mask rows of block k's noisy
    queries, so the doc mask cannot reveal masked content."""
    gen = torch.Generator().manual_seed(length + block)
    x0 = torch.randint(0, 6, (1, length), generator=gen)
    x0[0, block - 1] = 7  # one document boundary before block 1
    for k in range(length // block):
        for offset in range(block):
            pos = k * block + offset
            with_eod = x0.clone()
            with_eod[0, pos] = 7
            without_eod = x0.clone()
            without_eod[0, pos] = 3
            rows = []
            for variant in (with_eod, without_eod):
                docs = compute_doc_ids(variant, 7)
                mask = materialize_mask(
                    make_bd3_mask_mod(length, block, docs), 1, 2 * length, 2 * length
                )
                rows.append(mask[0, k * block : (k + 1) * block])
            assert torch.equal(rows[0], rows[1]), f"block {k}, position {pos}"

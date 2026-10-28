"""Invariant I5: prediction sources, document starts, and loss reduction (spec sections 6 and 7).

Run with:
    pytest tests/unit/dllm/test_targets.py -v
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch

from oellm_autoexp.dllm.targets import (
    apply_doc_source_rule,
    ar_loss_terms,
    ar_sources,
    bd3_loss_terms,
    prediction_sources,
    scorable_positions,
    token_weights,
)

_LOSS_KEYS = ("elbo_sum", "masked_ce_sum", "num_targets", "scored", "loss_sum", "num_tokens")


def _reference_shift1(seq_len: int, block_size: int) -> tuple[list[int], list[bool]]:
    """Return the token-shift-1 sources with a Python loop over the spec
    formulas."""
    src: list[int] = []
    has_src: list[bool] = []
    for i in range(seq_len):
        if i == 0:
            src.append(0)
            has_src.append(False)
        elif i % block_size == 0:
            src.append(seq_len + i - 1)
            has_src.append(True)
        else:
            src.append(i - 1)
            has_src.append(True)
    return src, has_src


def test_token_shift_zero_reads_the_noisy_copy() -> None:
    """Shift 0 reads the N stream at position i, and every position has a
    source."""
    src, has_src = prediction_sources(8, 4, 0)
    assert src.dtype == torch.long
    assert torch.equal(src, torch.arange(8))
    assert has_src.dtype == torch.bool
    assert bool(has_src.all())


def test_token_shift_one_block_starts_read_the_clean_copy() -> None:
    """Shift 1 reads i - 1 inside a block and L + i - 1 at a block start. Position 0 is empty."""
    src, has_src = prediction_sources(8, 4, 1)
    assert src.tolist() == [0, 0, 1, 2, 11, 4, 5, 6]
    assert has_src.tolist() == [False, True, True, True, True, True, True, True]


def test_token_shift_one_with_block_size_one() -> None:
    """With B = 1 every position after 0 is a block start, so every source is
    in the C stream."""
    src, has_src = prediction_sources(6, 1, 1)
    assert src.tolist() == [0, 6, 7, 8, 9, 10]
    assert has_src.tolist() == [False, True, True, True, True, True]


def test_token_shift_one_single_block() -> None:
    """With B = L the only block start is position 0, so shift 1 reads the N
    stream."""
    src, has_src = prediction_sources(8, 8, 1)
    assert src.tolist() == [0, 0, 1, 2, 3, 4, 5, 6]
    assert has_src.tolist() == [False] + [True] * 7


@pytest.mark.parametrize(
    ("seq_len", "block_size"),
    [(4, 4), (6, 1), (8, 2), (8, 4), (12, 3), (16, 8)],
)
def test_token_shift_one_matches_loop_reference(seq_len: int, block_size: int) -> None:
    """The vectorized source table equals a Python loop over the spec
    formulas."""
    src, has_src = prediction_sources(seq_len, block_size, 1)
    ref_src, ref_has = _reference_shift1(seq_len, block_size)
    assert src.tolist() == ref_src
    assert has_src.tolist() == ref_has


def test_prediction_sources_rejects_unsupported_settings() -> None:
    """Unsupported token shifts, block sizes, and lengths raise instead of
    returning a table."""
    with pytest.raises(ValueError, match="token_shift"):
        prediction_sources(8, 4, 2)
    with pytest.raises(ValueError, match="block_size"):
        prediction_sources(8, 0, 1)
    with pytest.raises(ValueError, match="multiple"):
        prediction_sources(6, 4, 1)


def test_document_start_removes_the_source_under_shift_one() -> None:
    """Position 3 starts document 1, so it has no source.

    No other flag changes.
    """
    doc_ids = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]])
    _, has_src = prediction_sources(8, 4, 1)
    without_docs = apply_doc_source_rule(has_src, None, 1, batch=1)
    with_docs = apply_doc_source_rule(has_src, doc_ids, 1, batch=1)
    assert without_docs.tolist() == [[False, True, True, True, True, True, True, True]]
    assert with_docs.tolist() == [[False, True, True, False, True, True, True, True]]
    assert (with_docs != without_docs).nonzero().tolist() == [[0, 3]]


def test_document_rule_is_skipped_under_shift_zero() -> None:
    """Shift 0 has no document rule, so the same document ids keep every
    source."""
    doc_ids = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]])
    _, has_src = prediction_sources(8, 4, 0)
    out = apply_doc_source_rule(has_src, doc_ids, 0, batch=1)
    assert out.tolist() == [[True] * 8]


def test_document_rule_matches_loop_reference() -> None:
    """The vectorized document rule equals a per-position comparison for random
    document ids."""
    gen = torch.Generator().manual_seed(0)
    doc_ids = torch.randint(0, 3, (4, 12), generator=gen)
    _, has_src = prediction_sources(12, 4, 1)
    out = apply_doc_source_rule(has_src, doc_ids, 1, batch=4)
    expected = [
        [bool(has_src[i]) and (i == 0 or doc[i] == doc[i - 1]) for i in range(12)]
        for doc in doc_ids.tolist()
    ]
    assert out.tolist() == expected


def test_document_rule_returns_a_new_tensor() -> None:
    """The output never shares storage with has_src, so a cached source table
    stays unchanged."""
    _, has_src = prediction_sources(8, 4, 1)
    out = apply_doc_source_rule(has_src, None, 1, batch=2)
    assert out.shape == (2, 8)
    out.fill_(False)
    assert has_src.tolist() == [False, True, True, True, True, True, True, True]


def test_document_rule_rejects_unsupported_inputs() -> None:
    """Document ids must have shape [batch, L] under shift 1, and the token
    shift must be 0 or 1."""
    _, has_src = prediction_sources(8, 4, 1)
    with pytest.raises(ValueError, match="doc_ids"):
        apply_doc_source_rule(has_src, torch.zeros(1, 7, dtype=torch.long), 1, batch=1)
    with pytest.raises(ValueError, match="token_shift"):
        apply_doc_source_rule(has_src, None, 2, batch=1)


def test_ar_sources_point_one_position_back() -> None:
    """The AR output at index i - 1 predicts x0[i], and index 0 has no source."""
    src, has_src = ar_sources(5)
    assert src.tolist() == [0, 0, 1, 2, 3]
    assert has_src.tolist() == [False, True, True, True, True]


def test_scorable_positions_exclude_position_zero_and_unlabeled_positions() -> None:
    """A scorable position has a source, a valid label, and an index of at
    least 1."""
    has_src = torch.tensor([[True, True, True, True, False]])
    target_ok = torch.tensor([[True, False, True, True, True]])
    out = scorable_positions(has_src, target_ok)
    assert out.dtype == torch.bool
    assert out.tolist() == [[False, False, True, True, False]]


def test_token_weights_repeat_each_block_weight() -> None:
    """Each block weight 1 / p_blk repeats for the B positions of its block."""
    weights = token_weights(torch.tensor([[1.0, 0.5]]), 2)
    assert weights.dtype == torch.float32
    assert weights.tolist() == [[1.0, 1.0, 2.0, 2.0]]


def test_token_weights_raises_on_non_positive_fraction() -> None:
    """A block with p_blk = 0 has no finite weight, so the call raises."""
    with pytest.raises(ValueError, match="p_block"):
        token_weights(torch.tensor([[1.0, 0.0]]), 2)
    with pytest.raises(ValueError, match="p_block"):
        token_weights(torch.tensor([[-0.5, 1.0]]), 2)


def test_bd3_loss_terms_hand_case_elbo() -> None:
    """Hand case of I5 with B = 2: target = [F, T, F, T], so elbo_sum = 1 * 1 +
    4 * 2 = 9."""
    ce = torch.tensor([[0.5, 1.0, 2.0, 4.0]], requires_grad=True)
    masked = torch.tensor([[True, True, False, True]])
    scorable = torch.tensor([[False, True, True, True]])
    p_block = torch.tensor([[1.0, 0.5]])
    terms = bd3_loss_terms(ce, masked, scorable, p_block, 2, "elbo")
    assert set(terms) == set(_LOSS_KEYS)
    for key in _LOSS_KEYS:
        assert terms[key].dtype == torch.float32, key
        assert terms[key].dim() == 0, key
    assert terms["elbo_sum"].item() == pytest.approx(9.0)
    assert terms["masked_ce_sum"].item() == pytest.approx(5.0)
    assert terms["num_targets"].item() == 2.0
    assert terms["scored"].item() == 3.0
    assert terms["loss_sum"].item() == pytest.approx(9.0)
    assert terms["num_tokens"].item() == 3.0
    terms["loss_sum"].backward()
    assert ce.grad.tolist() == [[0.0, 1.0, 0.0, 2.0]]


def test_bd3_loss_terms_hand_case_uniform() -> None:
    """Uniform weighting sums the plain ce over the targets and counts the
    targets."""
    ce = torch.tensor([[0.5, 1.0, 2.0, 4.0]])
    masked = torch.tensor([[True, True, False, True]])
    scorable = torch.tensor([[False, True, True, True]])
    p_block = torch.tensor([[1.0, 0.5]])
    terms = bd3_loss_terms(ce, masked, scorable, p_block, 2, "uniform")
    assert terms["loss_sum"].item() == pytest.approx(5.0)
    assert terms["num_tokens"].item() == 2.0
    assert terms["masked_ce_sum"].item() == pytest.approx(5.0)
    assert terms["elbo_sum"].item() == pytest.approx(9.0)


def test_bd3_loss_terms_rejects_unsupported_inputs() -> None:
    """An unknown weighting, a non-boolean mask, or a shape mismatch raises
    ValueError."""
    ce = torch.zeros(1, 4)
    masked = torch.ones(1, 4, dtype=torch.bool)
    scorable = torch.ones(1, 4, dtype=torch.bool)
    p_block = torch.ones(1, 2)
    with pytest.raises(ValueError, match="weighting"):
        bd3_loss_terms(ce, masked, scorable, p_block, 2, "focal")
    with pytest.raises(ValueError, match="BoolTensor"):
        bd3_loss_terms(ce, masked.long(), scorable, p_block, 2, "elbo")
    with pytest.raises(ValueError, match="shape"):
        bd3_loss_terms(ce, masked, scorable, torch.ones(1, 3), 2, "elbo")


def test_non_finite_ce_outside_targets_stays_out_of_the_sums() -> None:
    """Inf and NaN at non-target positions reach no sum, and the gradient stays
    finite."""
    ce = torch.tensor([[0.5, 1.0, float("inf"), float("nan")]], requires_grad=True)
    masked = torch.tensor([[True, True, False, False]])
    scorable = torch.tensor([[False, True, True, True]])
    p_block = torch.tensor([[1.0, 0.5]])
    for weighting in ("elbo", "uniform"):
        terms = bd3_loss_terms(ce, masked, scorable, p_block, 2, weighting)
        for key in _LOSS_KEYS:
            assert bool(torch.isfinite(terms[key])), (weighting, key)
    terms = bd3_loss_terms(ce, masked, scorable, p_block, 2, "elbo")
    terms["loss_sum"].backward()
    assert bool(torch.isfinite(ce.grad).all())


def test_ar_loss_terms_sum_scorable_positions() -> None:
    """The AR pair sums ce over the scorable positions and counts them."""
    ce = torch.tensor([[float("inf"), 1.0, 2.0, 4.0]], requires_grad=True)
    scorable = torch.tensor([[False, True, True, False]])
    loss_sum, num_tokens = ar_loss_terms(ce, scorable)
    assert loss_sum.dtype == torch.float32
    assert loss_sum.dim() == 0
    assert loss_sum.item() == pytest.approx(3.0)
    assert num_tokens.item() == 2.0
    loss_sum.backward()
    assert ce.grad.tolist() == [[0.0, 1.0, 1.0, 0.0]]

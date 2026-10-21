"""Invariants I2 and I3 (spec section 11) on the tiny reference model.

I2: with xt == x0 (no masking), the N-stream outputs equal the C-stream outputs at every layer.
    A negative control with positions arange(2L) shows that the check can fail.
I3: with B = 1, token shift 1, and every target masked, the bd3 per-token cross entropy equals the
    AR next-token cross entropy at every position that has a source.

The file also has shape and tied-embedding smoke tests, and checks that unsupported settings raise.

Run with:
    pytest tests/unit/dllm/test_reference_invariants.py -v
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch
from torch.nn import functional as F

from oellm_autoexp.dllm.masks import compute_doc_ids
from oellm_autoexp.dllm.reference import (
    RefConfig,
    RefModel,
    ar_dense_mask,
    ar_logits,
    bd3_dense_mask,
    bd3_logits,
)
from oellm_autoexp.dllm.targets import apply_doc_source_rule, prediction_sources

_CONFIG = RefConfig()
_VOCAB = _CONFIG.vocab_size
_MASK_ID = _VOCAB - 1
_EOD_ID = 7
_BATCH = 2
_I2_CASES = [(16, 4), (16, 1), (16, 16), (24, 8)]


def _make_model(config: RefConfig | None = None) -> RefModel:
    """Return a float64 reference model initialized under manual_seed(0).

    Args:
        config: Model settings, or None for the default RefConfig.

    Returns:
        The model, converted to float64.
    """
    torch.manual_seed(0)
    return RefModel(config or _CONFIG).double()


def _random_tokens(batch: int, length: int, seed: int) -> torch.Tensor:
    """Return random ids in [0, V - 2], which never include the mask id.

    Args:
        batch: Number of rows.
        length: Sequence length L.
        seed: Seed of the generator that draws the ids.

    Returns:
        LongTensor [batch, length].
    """
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, _VOCAB - 1, (batch, length), generator=gen)


def _with_documents(x0: torch.Tensor, seed: int) -> torch.Tensor:
    """Return x0 with EOD ids at about 20 percent of the positions.

    Args:
        x0: Clean token ids, LongTensor [b, L].
        seed: Seed of the generator that picks the EOD positions.

    Returns:
        LongTensor [b, L] with the same shape as x0.
    """
    gen = torch.Generator().manual_seed(seed)
    is_eod = torch.rand(x0.shape, generator=gen) < 0.2
    return torch.where(is_eod, torch.full_like(x0, _EOD_ID), x0)


def _with_block_aligned_documents(x0: torch.Tensor, block: int, seed: int) -> torch.Tensor:
    """Return x0 with EOD ids at about half of the block-final positions.

    Under the D-039 doc rule, I2 holds when every document boundary falls on a block boundary.

    Args:
        x0: Clean token ids, LongTensor [b, L].
        block: Block size B.
        seed: Seed of the generator that picks the EOD positions.

    Returns:
        LongTensor [b, L] with the same shape as x0.
    """
    gen = torch.Generator().manual_seed(seed)
    x0 = torch.where(x0 == _EOD_ID, x0 + 1, x0)  # random tokens may already hold the EOD id
    block_final = (torch.arange(x0.shape[1]) % block) == block - 1
    is_eod = (torch.rand(x0.shape, generator=gen) < 0.5) & block_final
    return torch.where(is_eod, torch.full_like(x0, _EOD_ID), x0)


def _token_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return the cross entropy of each target from logits.

    Args:
        logits: Tensor [b, n, V].
        targets: LongTensor [b, n].

    Returns:
        Tensor [b, n] with the cross entropy of every target.
    """
    flat = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
    )
    return flat.reshape(targets.shape)


@pytest.mark.parametrize(("length", "block"), _I2_CASES)
@pytest.mark.parametrize("with_docs", [False, True], ids=["no-docs", "docs"])
def test_i2_noisy_and_clean_streams_agree_at_every_layer(
    length: int, block: int, with_docs: bool
) -> None:
    """I2: with xt == x0, hidden[:, :L] equals hidden[:, L:] at every layer.

    With documents, the boundaries fall on block boundaries (spec section 11, D-039).
    """
    model = _make_model()
    x0 = _random_tokens(_BATCH, length, seed=length + block)
    doc_ids = None
    if with_docs:
        x0 = _with_block_aligned_documents(x0, block, seed=block)
        doc_ids = compute_doc_ids(x0, _EOD_ID)
        if block < length:
            assert int(doc_ids.max()) > 0, "the document split must not be trivial"
    ids = torch.cat([x0, x0], dim=1)
    positions = torch.cat([torch.arange(length), torch.arange(length)]).expand(_BATCH, -1)
    mask = bd3_dense_mask(length, block, doc_ids, _BATCH)
    with torch.no_grad():
        _, hidden = model(ids, positions, mask, output_hidden_states=True)
    assert len(hidden) == _CONFIG.num_layers
    for layer, states in enumerate(hidden):
        torch.testing.assert_close(
            states[:, :length],
            states[:, length:],
            atol=1e-10,
            rtol=0,
            msg=f"layer {layer}: the noisy stream differs from the clean stream",
        )


def test_i2_negative_control_fails_without_duplicated_positions() -> None:
    """Negative control: arange(2L) positions make the two streams differ.

    The I2 comparison can therefore fail, so passing it shows that the duplication works.
    """
    length = 16
    model = _make_model()
    x0 = _random_tokens(_BATCH, length, seed=1)
    ids = torch.cat([x0, x0], dim=1)
    positions = torch.arange(2 * length).expand(_BATCH, -1)
    mask = bd3_dense_mask(length, 4, None, _BATCH)
    with torch.no_grad():
        _, hidden = model(ids, positions, mask, output_hidden_states=True)
    for layer, states in enumerate(hidden):
        assert not torch.allclose(states[:, :length], states[:, length:]), f"layer {layer}"


@pytest.mark.parametrize("with_docs", [False, True], ids=["no-docs", "docs"])
def test_i3_bd3_ce_equals_ar_next_token_ce(with_docs: bool) -> None:
    """I3: with B = 1 and shift 1, bd3 ce at i equals AR ce at i - 1."""
    length = 16
    model = _make_model()
    x0 = _random_tokens(_BATCH, length, seed=2)
    doc_ids = None
    if with_docs:
        x0 = _with_documents(x0, seed=3)
        doc_ids = compute_doc_ids(x0, _EOD_ID)
    xt = torch.full_like(x0, _MASK_ID)
    with torch.no_grad():
        bd = bd3_logits(model, x0, xt, block_size=1, token_shift=1, doc_ids=doc_ids)
        ar = ar_logits(model, x0, doc_ids=doc_ids)
    bd_ce = _token_ce(bd, x0)  # bd_ce[:, i] is the ce of x0[:, i] under the bd3 prediction
    ar_ce = _token_ce(ar[:, :-1], x0[:, 1:])  # ar_ce[:, i - 1] is the ce of x0[:, i] under AR
    _, has_src = prediction_sources(length, 1, 1)
    scored = apply_doc_source_rule(has_src, doc_ids, 1, _BATCH)[:, 1:]  # positions 1 .. L-1
    assert bool(scored.any())
    if with_docs:
        assert int(scored.sum()) < _BATCH * (length - 1), "the document rule must drop a source"
    torch.testing.assert_close(bd_ce[:, 1:][scored], ar_ce[scored], atol=1e-10, rtol=0)


def test_forward_returns_logits_and_one_hidden_state_per_layer() -> None:
    """Return [b, s, V] logits and one [b, s, h] hidden state per layer."""
    model = _make_model()
    length = 8
    ids = _random_tokens(_BATCH, length, seed=4)
    positions = torch.arange(length).expand(_BATCH, -1)
    causal = torch.tril(torch.ones(length, length)).bool().expand(_BATCH, -1, -1)
    with torch.no_grad():
        logits = model(ids, positions, causal)
        logits_again, hidden = model(ids, positions, causal, output_hidden_states=True)
    assert logits.shape == (_BATCH, length, _VOCAB)
    assert logits.dtype == torch.float64
    assert torch.equal(logits, logits_again)
    assert len(hidden) == _CONFIG.num_layers
    assert all(states.shape == (_BATCH, length, _CONFIG.hidden_size) for states in hidden)


def test_dense_masks_and_logits_have_the_documented_shapes() -> None:
    """Masks are [b, 2L, 2L] (bd3) or [b, L, L] (AR); logits are [b, L, V]."""
    length, block = 8, 2
    model = _make_model()
    x0 = _random_tokens(_BATCH, length, seed=5)
    xt = torch.full_like(x0, _MASK_ID)
    docs = compute_doc_ids(_with_documents(x0, seed=6), _EOD_ID)
    for doc_ids in (None, docs):
        bd_mask = bd3_dense_mask(length, block, doc_ids, _BATCH)
        assert bd_mask.shape == (_BATCH, 2 * length, 2 * length)
        assert ar_dense_mask(length, doc_ids, _BATCH).shape == (_BATCH, length, length)
    with torch.no_grad():
        for doc_ids in (None, docs):
            bd = bd3_logits(model, x0, xt, block_size=block, token_shift=0, doc_ids=doc_ids)
            ar = ar_logits(model, x0, doc_ids=doc_ids)
            assert bd.shape == (_BATCH, length, _VOCAB)
            assert ar.shape == (_BATCH, length, _VOCAB)


def test_tied_embeddings_share_storage() -> None:
    """Tied embeddings share storage, also after model.double()."""
    tied = _make_model(RefConfig(tie_embeddings=True))
    assert tied.lm_head.weight is tied.embed_tokens.weight
    assert tied.lm_head.weight.data_ptr() == tied.embed_tokens.weight.data_ptr()
    untied = _make_model(RefConfig(tie_embeddings=False))
    assert untied.lm_head.weight.data_ptr() != untied.embed_tokens.weight.data_ptr()


def test_config_rejects_unsupported_sizes() -> None:
    """Sizes below 1, a bad head count, or an odd head_dim raise."""
    with pytest.raises(ValueError, match="vocab_size"):
        RefConfig(vocab_size=0)
    with pytest.raises(ValueError, match="num_heads"):
        RefConfig(num_heads=3, num_kv_heads=2)
    with pytest.raises(ValueError, match="head_dim"):
        RefConfig(head_dim=7)


def test_forward_rejects_a_query_row_without_keys() -> None:
    """A query row without any allowed key raises instead of giving NaN."""
    model = _make_model()
    ids = torch.zeros(1, 4, dtype=torch.long)
    positions = torch.arange(4).expand(1, -1)
    mask = torch.ones(1, 4, 4, dtype=torch.bool)
    mask[0, 2] = False
    with pytest.raises(ValueError, match="at least one key"):
        model(ids, positions, mask)


def test_dense_masks_reject_doc_ids_with_another_batch_size() -> None:
    """doc_ids must have one row per batch row, or the call raises."""
    doc_ids = torch.zeros(3, 8, dtype=torch.long)
    with pytest.raises(ValueError, match="doc_ids"):
        bd3_dense_mask(8, 2, doc_ids, _BATCH)
    with pytest.raises(ValueError, match="doc_ids"):
        ar_dense_mask(8, doc_ids, _BATCH)

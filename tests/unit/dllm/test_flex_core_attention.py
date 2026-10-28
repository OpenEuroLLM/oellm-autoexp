"""Tests for the dllm BlockMask context and the block-diffusion FlexAttention
core attention.

The FlexAttention path runs on CPU in eager mode. The reference is torch
scaled dot-product attention with dense boolean masks.
"""

from __future__ import annotations

import re
import types
from collections.abc import Iterator
from typing import Any

import pytest

pytest.importorskip("torch")

import torch
from torch.nn.attention.flex_attention import create_block_mask

from oellm_autoexp.dllm import context
from oellm_autoexp.dllm.flex_core_attention import BlockDiffusionFlexAttention

_SEQ = 32
_BATCH = 2
_HEADS = 4
_KV_GROUPS = 2
_HEAD_DIM = 16
_UNSET_MESSAGE = "no BlockMask set; call oellm_autoexp.dllm.context.set_block_mask first"


@pytest.fixture(autouse=True)
def _clear_block_mask() -> Iterator[None]:
    """Clear the BlockMask context before and after every test."""
    context.clear_block_mask()
    yield
    context.clear_block_mask()


def _config(
    attention_dropout: float = 0.0, context_parallel_size: int = 1
) -> types.SimpleNamespace:
    """Return a minimal transformer config for the core attention."""
    return types.SimpleNamespace(
        attention_dropout=attention_dropout, context_parallel_size=context_parallel_size
    )


def _causal_mask_mod(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
    """Allow each query to attend to keys at or before its position."""
    return q_idx >= kv_idx


def _causal_block_mask() -> Any:
    """Return a causal BlockMask for one sequence of _SEQ tokens."""
    return create_block_mask(
        _causal_mask_mod, B=None, H=None, Q_LEN=_SEQ, KV_LEN=_SEQ, device="cpu"
    )


def _random_qkv() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return random queries, keys, and values in the Megatron layout."""
    query = torch.randn(_SEQ, _BATCH, _HEADS, _HEAD_DIM)
    key = torch.randn(_SEQ, _BATCH, _KV_GROUPS, _HEAD_DIM)
    value = torch.randn(_SEQ, _BATCH, _KV_GROUPS, _HEAD_DIM)
    return query, key, value


def _dense_reference(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Compute attention with scaled dot-product attention and a dense boolean
    mask.

    Args:
        query: Queries of shape [sq, b, np, hn].
        key: Keys of shape [sk, b, ng, hn].
        value: Values of shape [sk, b, ng, hn].
        mask: Boolean mask of shape [b, sq, sk], where True means attend.

    Returns:
        Attention output of shape [sq, b, np * hn].
    """
    sq, b, num_heads, head_dim = query.shape
    repeats = num_heads // key.shape[2]
    q = query.permute(1, 2, 0, 3)
    k = key.permute(1, 2, 0, 3).repeat_interleave(repeats, dim=1)
    v = value.permute(1, 2, 0, 3).repeat_interleave(repeats, dim=1)
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None])
    return out.permute(2, 0, 1, 3).reshape(sq, b, num_heads * head_dim)


def test_get_block_mask_raises_when_unset() -> None:
    """get_block_mask raises RuntimeError with the documented message when
    nothing is set."""
    context.clear_block_mask()
    with pytest.raises(RuntimeError, match=re.escape(_UNSET_MESSAGE)):
        context.get_block_mask()


def test_set_get_clear_roundtrip() -> None:
    """get_block_mask returns the stored object until clear_block_mask removes
    it."""
    sentinel = object()
    context.set_block_mask(sentinel)
    assert context.get_block_mask() is sentinel
    context.clear_block_mask()
    with pytest.raises(RuntimeError, match=re.escape(_UNSET_MESSAGE)):
        context.get_block_mask()


@pytest.mark.filterwarnings("ignore:flex_attention called without torch.compile")
def test_forward_matches_sdpa_causal_gqa() -> None:
    """Causal FlexAttention with grouped-query heads matches SDPA on the dense
    causal mask."""
    torch.manual_seed(0)
    query, key, value = _random_qkv()
    context.set_block_mask(_causal_block_mask())
    module = BlockDiffusionFlexAttention(_config(), 1, None, "self")
    out = module(query, key, value, None)

    assert out.shape == (_SEQ, _BATCH, _HEADS * _HEAD_DIM)
    assert out.is_contiguous()
    causal = torch.ones(_SEQ, _SEQ, dtype=torch.bool).tril()
    expected = _dense_reference(query, key, value, causal.expand(_BATCH, -1, -1))
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.filterwarnings("ignore:flex_attention called without torch.compile")
def test_forward_matches_sdpa_with_batch_mask() -> None:
    """Document masks that differ between batch rows each match their own dense
    reference."""
    torch.manual_seed(0)
    query, key, value = _random_qkv()
    doc_ids = torch.zeros(_BATCH, _SEQ, dtype=torch.long)
    doc_ids[0, 12:] = 1
    doc_ids[1, 20:] = 1

    def document_mask_mod(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        return (q_idx >= kv_idx) & (doc_ids[b, q_idx] == doc_ids[b, kv_idx])

    block_mask = create_block_mask(
        document_mask_mod, B=_BATCH, H=None, Q_LEN=_SEQ, KV_LEN=_SEQ, device="cpu"
    )
    context.set_block_mask(block_mask)
    module = BlockDiffusionFlexAttention(_config(), 1, None, "self")
    out = module(query, key, value, None)

    positions = torch.arange(_SEQ)
    causal = positions[:, None] >= positions[None, :]
    same_doc = doc_ids[:, :, None] == doc_ids[:, None, :]
    dense = causal[None] & same_doc
    assert not torch.equal(dense[0], dense[1])
    expected = _dense_reference(query, key, value, dense)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_forward_raises_without_mask() -> None:
    """Forward raises RuntimeError when no BlockMask is stored in the
    context."""
    query, key, value = _random_qkv()
    module = BlockDiffusionFlexAttention(_config(), 1, None, "self")
    with pytest.raises(RuntimeError, match=re.escape(_UNSET_MESSAGE)):
        module(query, key, value, None)


def test_rejects_dropout() -> None:
    """Non-zero dropout raises, whether it comes from the config or from the
    argument."""
    with pytest.raises(ValueError, match="dropout"):
        BlockDiffusionFlexAttention(_config(attention_dropout=0.1), 1, None, "self")
    with pytest.raises(ValueError, match="dropout"):
        BlockDiffusionFlexAttention(_config(), 1, None, "self", attention_dropout=0.1)


def test_rejects_context_parallel() -> None:
    """Context parallelism above 1 raises, because the mask covers the whole
    sequence."""
    with pytest.raises(ValueError, match="context_parallel_size"):
        BlockDiffusionFlexAttention(_config(context_parallel_size=2), 1, None, "self")


def test_rejects_cross_attention_and_separate_channels() -> None:
    """Only self-attention with shared key and value head sizes is
    supported."""
    with pytest.raises(ValueError, match="self-attention"):
        BlockDiffusionFlexAttention(_config(), 1, None, "cross")
    with pytest.raises(ValueError, match="k_channels or v_channels"):
        BlockDiffusionFlexAttention(_config(), 1, None, "self", k_channels=_HEAD_DIM)
    with pytest.raises(ValueError, match="k_channels or v_channels"):
        BlockDiffusionFlexAttention(_config(), 1, None, "self", v_channels=_HEAD_DIM)


def test_rejects_packed_seq() -> None:
    """Packed sequences raise NotImplementedError, even when a BlockMask is
    set."""
    context.set_block_mask(_causal_block_mask())
    query, key, value = _random_qkv()
    module = BlockDiffusionFlexAttention(_config(), 1, None, "self")
    with pytest.raises(NotImplementedError, match="packed sequences"):
        module(query, key, value, None, packed_seq_params=object())


def test_rejects_attention_bias() -> None:
    """An attention bias raises NotImplementedError, even when a BlockMask is
    set."""
    context.set_block_mask(_causal_block_mask())
    query, key, value = _random_qkv()
    module = BlockDiffusionFlexAttention(_config(), 1, None, "self")
    with pytest.raises(NotImplementedError, match="attention_bias"):
        module(query, key, value, None, attention_bias=torch.zeros(_SEQ, _SEQ))


def test_extra_state_matches_torch_flex_attention() -> None:
    """The checkpoint keys match TorchFlexAttention: one _extra_state entry that is None."""
    module = BlockDiffusionFlexAttention(_config(), 1, None, "self")
    assert module.get_extra_state() is None
    module.set_extra_state(None)
    assert module.state_dict() == {"_extra_state": None}

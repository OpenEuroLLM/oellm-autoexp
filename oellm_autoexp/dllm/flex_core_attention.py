"""FlexAttention core attention for block-diffusion training.

``BlockDiffusionFlexAttention`` replaces the MegaDLMs ``TorchFlexAttention`` in the layer spec. It
reads the BlockMask of the current micro-batch from ``oellm_autoexp.dllm.context`` and ignores
``attention_mask``. It supports self-attention with grouped-query heads. It does not support
dropout, context parallelism, packed sequences, or attention bias.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import flex_attention

from oellm_autoexp.dllm import context

if TYPE_CHECKING:
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.transformer_config import TransformerConfig

_COMPILED_FLEX_ATTENTION: Callable[..., Tensor] | None = None


def _flex_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    block_mask: Any,
    scale: float | None,
    enable_gqa: bool,
) -> Tensor:
    """Run FlexAttention, compiled on CUDA and eager on any other device.

    The compiled function is created on the first CUDA call and cached in a module global.

    Args:
        q: Queries of shape [b, np, sq, hn].
        k: Keys of shape [b, ng, sk, hn].
        v: Values of shape [b, ng, sk, hn].
        block_mask: FlexAttention BlockMask that defines the attention pattern.
        scale: Softmax scale, or None for 1 / sqrt(hn).
        enable_gqa: True when the number of query heads differs from the number of kv heads.

    Returns:
        Attention output of shape [b, np, sq, hn].
    """
    global _COMPILED_FLEX_ATTENTION
    if q.is_cuda:
        if _COMPILED_FLEX_ATTENTION is None:
            _COMPILED_FLEX_ATTENTION = torch.compile(flex_attention, dynamic=False)
        fn = _COMPILED_FLEX_ATTENTION
    else:
        fn = flex_attention
    return fn(q, k, v, block_mask=block_mask, scale=scale, enable_gqa=enable_gqa)


class BlockDiffusionFlexAttention(torch.nn.Module):
    """FlexAttention core attention that reads its BlockMask from the dllm
    context.

    The constructor has the signature of MegaDLMs ``TorchFlexAttention``, so Megatron builds this
    module the same way. The module has no parameters and no checkpoint state.

    Args:
        config: Transformer config. Its ``attention_dropout`` must be 0.0, and its
            ``context_parallel_size`` must be 1.
        layer_number: 1-based index of the transformer layer.
        attn_mask_type: Megatron attention mask type. ``forward`` ignores it, because the
            BlockMask defines the pattern.
        attention_type: Must be ``"self"``.
        attention_dropout: Optional override of ``config.attention_dropout``. Must be 0.0.
        softmax_scale: Softmax scale. None means 1 / sqrt(head_dim).
        k_channels: Must be None. Separate key head sizes are not supported.
        v_channels: Must be None. Separate value head sizes are not supported.
        cp_comm_type: Unused, because context parallelism is 1.

    Raises:
        ValueError: If dropout is not 0.0, context parallelism is not 1, the attention type is not
            ``"self"``, or ``k_channels`` or ``v_channels`` is set.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType | None,
        attention_type: str,
        attention_dropout: float | None = None,
        softmax_scale: float | None = None,
        k_channels: int | None = None,
        v_channels: int | None = None,
        cp_comm_type: str = "p2p",
    ) -> None:
        super().__init__()
        dropout = config.attention_dropout if attention_dropout is None else attention_dropout
        if dropout != 0.0:
            raise ValueError(
                f"BlockDiffusionFlexAttention needs attention dropout 0.0, got {dropout}"
            )
        cp_size = getattr(config, "context_parallel_size", 1)
        if cp_size != 1:
            raise ValueError(
                f"BlockDiffusionFlexAttention needs context_parallel_size 1, got {cp_size}"
            )
        if attention_type != "self":
            raise ValueError(
                f"BlockDiffusionFlexAttention supports only self-attention, got {attention_type!r}"
            )
        if k_channels is not None or v_channels is not None:
            raise ValueError(
                "BlockDiffusionFlexAttention does not support k_channels or v_channels"
            )
        self.config = config
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type
        self.softmax_scale = softmax_scale

    def get_extra_state(self) -> None:
        """Return None as the extra state.

        torch still writes a ``core_attention._extra_state`` key (value None), as for MegaDLMs
        ``TorchFlexAttention``, so checkpoints of both modules have the same keys.

        Returns:
            None.
        """

    def set_extra_state(self, extra_state: Any) -> None:
        """Ignore extra state from a checkpoint, because the module has no
        state.

        Args:
            extra_state: Extra state from a checkpoint. Ignored.
        """

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None,
        attn_mask_type: AttnMaskType | None = None,
        attention_bias: Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
    ) -> Tensor:
        """Compute attention over the BlockMask stored in the dllm context.

        Args:
            query: Queries of shape [sq, b, np, hn].
            key: Keys of shape [sk, b, ng, hn], where np is a multiple of ng.
            value: Values of shape [sk, b, ng, hn].
            attention_mask: Ignored. The BlockMask defines the pattern.
            attn_mask_type: Ignored. The BlockMask defines the pattern.
            attention_bias: Must be None.
            packed_seq_params: Must be None.

        Returns:
            Attention output of shape [sq, b, np * hn], contiguous.

        Raises:
            NotImplementedError: If packed_seq_params or attention_bias is set.
            RuntimeError: If no BlockMask is stored in the dllm context.
        """
        if packed_seq_params is not None:
            raise NotImplementedError(
                "BlockDiffusionFlexAttention does not support packed sequences"
            )
        if attention_bias is not None:
            raise NotImplementedError("BlockDiffusionFlexAttention does not support attention_bias")
        block_mask = context.get_block_mask()
        sq, b, num_heads, head_dim = query.shape
        q = query.permute(1, 2, 0, 3)
        k = key.permute(1, 2, 0, 3)
        v = value.permute(1, 2, 0, 3)
        out = _flex_attention(
            q,
            k,
            v,
            block_mask=block_mask,
            scale=self.softmax_scale,
            enable_gqa=q.shape[1] != k.shape[1],
        )
        return out.permute(2, 0, 1, 3).reshape(sq, b, num_heads * head_dim).contiguous()

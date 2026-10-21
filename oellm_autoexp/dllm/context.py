"""Module-level holder for the BlockMask that the block-diffusion core
attention reads.

The model stores the BlockMask of the current micro-batch with ``set_block_mask`` before it runs
the decoder. ``BlockDiffusionFlexAttention`` reads it with ``get_block_mask``. The mask never
travels through ``attention_mask``, because Megatron recompute saves positional arguments and a
BlockMask is not a tensor. This is safe because pipeline parallel size 1 runs the forward and
backward passes of one micro-batch in order.

This module does not import torch, so core autoexp code can import it without torch.
"""

from typing import Any

_BLOCK_MASK: Any = None


def set_block_mask(block_mask: Any) -> None:
    """Store the BlockMask for the decoder pass that follows.

    Args:
        block_mask: FlexAttention BlockMask for the current micro-batch.
    """
    global _BLOCK_MASK
    _BLOCK_MASK = block_mask


def get_block_mask() -> Any:
    """Return the BlockMask stored by the last call to ``set_block_mask``.

    Returns:
        The stored BlockMask.

    Raises:
        RuntimeError: If no BlockMask is stored.
    """
    if _BLOCK_MASK is None:
        raise RuntimeError("no BlockMask set; call oellm_autoexp.dllm.context.set_block_mask first")
    return _BLOCK_MASK


def clear_block_mask() -> None:
    """Remove the stored BlockMask, so that a later ``get_block_mask``
    raises."""
    global _BLOCK_MASK
    _BLOCK_MASK = None

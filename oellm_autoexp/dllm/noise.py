"""Training noise and the deterministic validation estimator for block
diffusion.

Implements spec section 3 (bd3 training noise and complementary masking) and spec section 9
(validation estimator). Every random draw uses the caller's ``torch.Generator`` on ``x0.device``.
The global RNG is never used. With ``K = L // block_size``, ``x0`` has shape ``[b, L]`` and every
per-block tensor has shape ``[b, K]``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

_INTEGER_DTYPES: tuple[torch.dtype, ...] = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)


def _check_block_size(block_size: int) -> None:
    """Check that ``block_size`` is an integer of at least 1.

    Args:
        block_size: Diffusion block size B.

    Raises:
        ValueError: If ``block_size`` is not an integer or is smaller than 1.
    """
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise ValueError(f"block_size must be an integer >= 1, got {block_size!r}")


def _check_sampler_inputs(
    x0: Tensor, block_size: int, mask_id: int, generator: torch.Generator
) -> None:
    """Validate the inputs that every sampler shares.

    Args:
        x0: Clean tokens, expected shape ``[b, L]`` with an integer dtype.
        block_size: Diffusion block size B.
        mask_id: Mask token id M.
        generator: Generator that draws the noise.

    Raises:
        ValueError: If an input is unsupported.
    """
    if x0.dim() != 2:
        raise ValueError(f"x0 must be 2-D [b, L], got shape {tuple(x0.shape)}")
    if x0.dtype not in _INTEGER_DTYPES:
        raise ValueError(f"x0 must have an integer dtype, got {x0.dtype}")
    _check_block_size(block_size)
    length = x0.shape[1]
    if length == 0 or length % block_size != 0:
        raise ValueError(f"L={length} must be a positive multiple of block_size={block_size}")
    if isinstance(mask_id, bool) or not isinstance(mask_id, int) or mask_id < 0:
        raise ValueError(f"mask_id must be a non-negative integer, got {mask_id!r}")
    if generator.device.type != x0.device.type:
        raise ValueError(
            f"generator is on device type {generator.device.type!r} but x0 is on {x0.device.type!r}"
        )


def _check_mask_range(t_min: float, t_max: float) -> None:
    """Check that ``0 <= t_min <= t_max <= 1``.

    Args:
        t_min: Lower bound of the masking rate.
        t_max: Upper bound of the masking rate.

    Raises:
        ValueError: If the bounds are out of order or outside [0, 1]. NaN is rejected too.
    """
    if not 0.0 <= t_min <= t_max <= 1.0:
        raise ValueError(f"need 0 <= t_min <= t_max <= 1, got t_min={t_min}, t_max={t_max}")


def _draw_block_masks(
    x0: Tensor, block_size: int, t_min: float, t_max: float, generator: torch.Generator
) -> Tensor:
    """Draw one copy of the block masks in the order of spec section 3, steps 1
    to 3.

    The draws are t (``[b, K]``), then u (``[b, K, B]``), then the fallback position
    (``[b, K]``, always drawn). A block without a masked position gets its fallback position
    masked.

    Args:
        x0: Clean tokens, shape ``[b, L]``. Only the shape and the device are used.
        block_size: Diffusion block size B.
        t_min: Lower bound of the masking rate.
        t_max: Upper bound of the masking rate.
        generator: Generator on the device of ``x0``.

    Returns:
        A bool tensor ``[b, K, B]`` in which every block has at least one True entry.
    """
    batch, length = x0.shape
    num_blocks = length // block_size
    device = x0.device
    t = t_min + (t_max - t_min) * torch.rand(batch, num_blocks, generator=generator, device=device)
    u = torch.rand(batch, num_blocks, block_size, generator=generator, device=device)
    masked = u < t[..., None]
    fallback = torch.randint(0, block_size, (batch, num_blocks), generator=generator, device=device)
    empty = ~masked.any(dim=-1)
    fallback_onehot = F.one_hot(fallback, block_size).bool()
    return masked | (empty[..., None] & fallback_onehot)


def sample_block_noise(
    x0: Tensor,
    block_size: int,
    mask_id: int,
    t_min: float,
    t_max: float,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor]:
    """Draw the bd3 training noise for a batch (spec section 3).

    For each block, draw t_k uniformly from [t_min, t_max], mask each position with probability
    t_k, and mask one uniformly chosen position when the block has no mask. The draws come from
    ``generator`` in the order t, u, fallback.

    Args:
        x0: Clean tokens of shape ``[b, L]`` with an integer dtype. ``L`` must be a positive
            multiple of ``block_size``.
        block_size: Diffusion block size B.
        mask_id: Mask token id M.
        t_min: Lower bound of the masking rate, with ``0 <= t_min <= t_max <= 1``.
        t_max: Upper bound of the masking rate.
        generator: Generator on the same device type as ``x0``.

    Returns:
        A tuple ``(xt, masked, p_block)``. ``xt`` is a ``torch.long`` tensor ``[b, L]`` that
        holds ``mask_id`` at masked positions and ``x0`` elsewhere. ``masked`` is a bool tensor
        ``[b, L]``. ``p_block`` is a float32 tensor ``[b, K]`` with the masked fraction of each
        block.

    Raises:
        ValueError: If an input is unsupported.
    """
    _check_sampler_inputs(x0, block_size, mask_id, generator)
    _check_mask_range(t_min, t_max)
    batch, length = x0.shape
    masked_blocks = _draw_block_masks(x0, block_size, t_min, t_max, generator)
    p_block = masked_blocks.sum(dim=-1).float() / block_size
    masked = masked_blocks.reshape(batch, length)
    xt = torch.where(masked, mask_id, x0.long())
    return xt, masked, p_block


def sample_complementary_noise(
    x0: Tensor,
    block_size: int,
    mask_id: int,
    t_min: float,
    t_max: float,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor]:
    """Draw the complementary training noise for a batch (spec section 3).

    Copy 1 is the bd3 draw. In every block, one uniformly chosen position is unmasked when the
    whole block is masked, so copy 1 keeps between 1 and B - 1 masks per block. Copy 2 masks
    exactly the positions that copy 1 left unmasked. The caller repeats ``x0``; this function
    does not return it.

    Args:
        x0: Clean tokens of shape ``[b, L]`` with an integer dtype. ``L`` must be a positive
            multiple of ``block_size``.
        block_size: Diffusion block size B, at least 2.
        mask_id: Mask token id M.
        t_min: Lower bound of the masking rate, with ``0 <= t_min <= t_max <= 1``.
        t_max: Upper bound of the masking rate.
        generator: Generator on the same device type as ``x0``.

    Returns:
        A tuple ``(xt, masked, p_block)`` with a batch of ``2b`` rows. Rows ``0..b-1`` are copy 1
        and rows ``b..2b-1`` are copy 2. ``xt`` is a ``torch.long`` tensor ``[2b, L]``, ``masked``
        is a bool tensor ``[2b, L]``, and ``p_block`` is a float32 tensor ``[2b, K]``.

    Raises:
        ValueError: If an input is unsupported, including ``block_size < 2``.
    """
    _check_sampler_inputs(x0, block_size, mask_id, generator)
    if block_size < 2:
        raise ValueError(f"complementary masking needs block_size >= 2, got {block_size}")
    _check_mask_range(t_min, t_max)
    batch, length = x0.shape
    num_blocks = length // block_size
    device = x0.device

    copy1 = _draw_block_masks(x0, block_size, t_min, t_max, generator)
    keep = torch.randint(0, block_size, (batch, num_blocks), generator=generator, device=device)
    fully_masked = copy1.all(dim=-1)
    copy1 = copy1 & ~(fully_masked[..., None] & F.one_hot(keep, block_size).bool())
    masked_blocks = torch.cat([copy1, ~copy1], dim=0)
    masked = masked_blocks.reshape(2 * batch, length)
    p_block = masked_blocks.sum(dim=-1).float() / block_size
    x0_stacked = torch.cat([x0, x0], dim=0).long()
    xt = torch.where(masked, mask_id, x0_stacked)
    return xt, masked, p_block


def sample_validation_noise(
    x0: Tensor,
    block_size: int,
    mask_id: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor]:
    """Draw the validation noise of spec section 9 for a batch.

    For each block, draw l_k uniformly from {1, ..., B} and mask exactly l_k positions chosen
    uniformly without replacement. The block weight is B / l_k.

    Args:
        x0: Clean tokens of shape ``[b, L]`` with an integer dtype. ``L`` must be a positive
            multiple of ``block_size``.
        block_size: Diffusion block size B.
        mask_id: Mask token id M.
        generator: Generator on the same device type as ``x0``.

    Returns:
        A tuple ``(xt, masked, block_weight)``. ``xt`` is a ``torch.long`` tensor ``[b, L]``,
        ``masked`` is a bool tensor ``[b, L]``, and ``block_weight`` is a float32 tensor
        ``[b, K]`` with the value ``B / l_k``.

    Raises:
        ValueError: If an input is unsupported.
    """
    _check_sampler_inputs(x0, block_size, mask_id, generator)
    batch, length = x0.shape
    num_blocks = length // block_size
    device = x0.device
    n_mask = torch.randint(
        1, block_size + 1, (batch, num_blocks), generator=generator, device=device
    )
    keys = torch.rand(batch, num_blocks, block_size, generator=generator, device=device)
    rank = keys.argsort(dim=-1).argsort(dim=-1)
    masked_blocks = rank < n_mask[..., None]
    block_weight = block_size / n_mask.float()
    masked = masked_blocks.reshape(batch, length)
    xt = torch.where(masked, mask_id, x0.long())
    return xt, masked, block_weight


def expand_block_values(values: Tensor, block_size: int) -> Tensor:
    """Repeat each per-block value across the positions of its block.

    Args:
        values: Per-block values of shape ``[b, K]``.
        block_size: Diffusion block size B.

    Returns:
        A tensor of shape ``[b, K * block_size]`` in which position ``i`` holds
        ``values[:, i // block_size]``.

    Raises:
        ValueError: If ``values`` is not 2-D or ``block_size`` is not an integer of at least 1.
    """
    if values.dim() != 2:
        raise ValueError(f"values must be 2-D [b, K], got shape {tuple(values.shape)}")
    _check_block_size(block_size)
    return values.repeat_interleave(block_size, dim=1)


def validation_nelbo(
    ce: Tensor,
    masked: Tensor,
    block_weight: Tensor,
    scorable: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute the NELBO sum and the scored count of the validation estimator
    (spec section 9).

    The NELBO sum adds ``block_weight * ce`` over the positions that are masked and scorable.
    The scored count is the number of scorable positions. ``torch.where`` selects the terms, so a
    non-finite ``ce`` outside the targets cannot leak into the sum.

    Args:
        ce: Per-token cross entropy, a floating-point tensor of shape ``[b, L]``.
        masked: Bool tensor ``[b, L]``, True where the position is masked.
        block_weight: Per-block weights ``B / l_k`` of shape ``[b, K]``.
        scorable: Bool tensor ``[b, L]``, True where the position is scored.

    Returns:
        A tuple ``(nelbo_sum, scored)`` of 0-dim float32 tensors.

    Raises:
        ValueError: If the shapes, dtypes, or block layout are inconsistent.
    """
    if ce.dim() != 2 or ce.shape != masked.shape or ce.shape != scorable.shape:
        raise ValueError(
            "ce, masked, and scorable must share the 2-D shape [b, L], got "
            f"{tuple(ce.shape)}, {tuple(masked.shape)}, {tuple(scorable.shape)}"
        )
    if not ce.is_floating_point():
        raise ValueError(f"ce must be a floating-point tensor, got {ce.dtype}")
    if masked.dtype != torch.bool or scorable.dtype != torch.bool:
        raise ValueError("masked and scorable must be bool tensors")
    batch, length = ce.shape
    if block_weight.dim() != 2 or block_weight.shape[0] != batch:
        raise ValueError(f"block_weight must have shape [b, K], got {tuple(block_weight.shape)}")
    num_blocks = block_weight.shape[1]
    if num_blocks < 1 or length % num_blocks != 0:
        raise ValueError(f"L={length} must be a positive multiple of K={num_blocks}")
    block_size = length // num_blocks
    weights = expand_block_values(block_weight.to(torch.float32), block_size)
    target = masked & scorable
    terms = torch.where(
        target,
        weights * ce.to(torch.float32),
        torch.zeros((), dtype=torch.float32, device=ce.device),
    )
    nelbo_sum = terms.sum().to(torch.float32)
    scored = scorable.sum().to(torch.float32)
    return nelbo_sum, scored

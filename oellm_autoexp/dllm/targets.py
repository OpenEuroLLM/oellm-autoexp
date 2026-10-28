"""Prediction sources, loss weights, and per-micro-batch loss reduction (spec
sections 6 and 7).

``src[i]`` is the input index whose output predicts ``x0[i]``. Input indices ``0..L-1`` form the
N stream (noisy copy) and ``L..2L-1`` the C stream (clean copy). The functions here build the
source tables, apply the document-start rule, select the scored positions, compute the ELBO
weights, and reduce the per-token cross entropy to the (sum, count) terms that Megatron logs.

Boolean masks have shape [b, L]. Values that are used only at target positions are selected with
``torch.where``, so NaN or inf at other positions cannot reach a sum.
"""

from __future__ import annotations

import torch
from torch import Tensor

_TOKEN_SHIFTS: tuple[int, ...] = (0, 1)
_WEIGHTINGS: tuple[str, ...] = ("elbo", "uniform")


def _check_seq_len(seq_len: int) -> None:
    """Raise ValueError unless the sequence length is at least 1."""
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")


def _check_token_shift(token_shift: int) -> None:
    """Raise ValueError unless the token shift is 0 or 1."""
    if token_shift not in _TOKEN_SHIFTS:
        raise ValueError(f"token_shift must be 0 or 1, got {token_shift}")


def _check_bool(name: str, value: Tensor) -> None:
    """Raise ValueError unless ``value`` is a boolean tensor."""
    if value.dtype != torch.bool:
        raise ValueError(f"{name} must be a BoolTensor, got {value.dtype}")


def _check_shape(name: str, value: Tensor, shape: torch.Size) -> None:
    """Raise ValueError unless ``value`` has the expected shape."""
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {tuple(value.shape)}")


def prediction_sources(
    seq_len: int,
    block_size: int,
    token_shift: int,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    """Return the input index that predicts each clean position (spec section
    6).

    Args:
        seq_len: Effective length L. Must be a multiple of ``block_size``.
        block_size: Diffusion block size B.
        token_shift: 0 reads the noisy token at position ``i``. 1 reads position ``i - 1``, or
            the last clean token of the previous block (C stream) at a block start.
        device: Device of the returned tensors. None uses the default device.

    Returns:
        A tuple ``(src, has_src)``. ``src`` is a LongTensor [L] of input indices. ``has_src`` is a
        BoolTensor [L]. Under token shift 1, position 0 has no source and ``src[0]`` is 0.

    Raises:
        ValueError: If ``token_shift`` is not 0 or 1, ``block_size`` is below 1, ``seq_len`` is
            below 1, or ``seq_len`` is not a multiple of ``block_size``.
    """
    _check_seq_len(seq_len)
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    if seq_len % block_size != 0:
        raise ValueError(f"seq_len {seq_len} is not a multiple of block_size {block_size}")
    _check_token_shift(token_shift)
    idx = torch.arange(seq_len, device=device, dtype=torch.long)
    if token_shift == 0:
        return idx, torch.ones(seq_len, dtype=torch.bool, device=device)
    block_start = idx % block_size == 0
    src = torch.where(block_start, seq_len + idx - 1, idx - 1)
    src = torch.where(idx == 0, torch.zeros_like(idx), src)
    return src, idx >= 1


def apply_doc_source_rule(
    has_src: Tensor,
    doc_ids: Tensor | None,
    token_shift: int,
    batch: int,
) -> Tensor:
    """Return the per-sample source flags with the document-start rule (spec
    section 6).

    With token shift 1 and document ids, position ``i >= 1`` has no source when it starts a new
    document, that is when ``doc_ids[:, i] != doc_ids[:, i - 1]``. Other settings leave the flags
    unchanged.

    Args:
        has_src: BoolTensor [L] from :func:`prediction_sources`, or any tensor that broadcasts to
            [batch, L].
        doc_ids: LongTensor [batch, L] of document ids, or None when the document mask is off.
        token_shift: 0 or 1, as in :func:`prediction_sources`.
        batch: Batch size b.

    Returns:
        A new BoolTensor [batch, L]. It never shares storage with ``has_src``.

    Raises:
        ValueError: If ``token_shift`` is not 0 or 1, ``has_src`` is not boolean or has more than
            two dimensions, or ``doc_ids`` does not have shape [batch, L] under token shift 1.
    """
    _check_token_shift(token_shift)
    _check_bool("has_src", has_src)
    if has_src.dim() not in (1, 2):
        raise ValueError(f"has_src must have 1 or 2 dimensions, got {has_src.dim()}")
    seq_len = has_src.shape[-1]
    out = torch.broadcast_to(has_src, (batch, seq_len)).clone()
    if token_shift == 1 and doc_ids is not None:
        _check_shape("doc_ids", doc_ids, torch.Size([batch, seq_len]))
        out[:, 1:] &= doc_ids[:, 1:] == doc_ids[:, :-1]
    return out


def ar_sources(
    seq_len: int,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    """Return the sources of the autoregressive control (spec section 6,
    objective ar).

    The output at index ``i - 1`` predicts ``x0[i]``. Index 0 has no source and points to 0.

    Args:
        seq_len: Length L of the clean sequence.
        device: Device of the returned tensors. None uses the default device.

    Returns:
        A tuple ``(src, has_src)`` with a LongTensor [L] and a BoolTensor [L].

    Raises:
        ValueError: If ``seq_len`` is below 1.
    """
    _check_seq_len(seq_len)
    idx = torch.arange(seq_len, device=device, dtype=torch.long)
    src = torch.clamp(idx - 1, min=0)
    return src, idx >= 1


def scorable_positions(has_src: Tensor, target_ok: Tensor) -> Tensor:
    """Return the positions that the loss scores (spec sections 2 and 7).

    A position is scorable when it has a source, its label is valid (``target_ok``), and it is
    not position 0, which has no left context.

    Args:
        has_src: BoolTensor [b, L], for example from :func:`apply_doc_source_rule`.
        target_ok: BoolTensor [b, L] with the loss mask of ``x0``.

    Returns:
        A BoolTensor [b, L].

    Raises:
        ValueError: If an input is not boolean or the two shapes differ.
    """
    _check_bool("has_src", has_src)
    _check_bool("target_ok", target_ok)
    _check_shape("target_ok", target_ok, has_src.shape)
    not_first = torch.arange(has_src.shape[-1], device=has_src.device) >= 1
    return has_src & target_ok & not_first


def token_weights(p_block: Tensor, block_size: int) -> Tensor:
    """Return the ELBO weight 1 / p_blk of every position (spec section 7).

    Args:
        p_block: Tensor [b, K] with the exact masked fraction of each block. Every value must be
            positive.
        block_size: Diffusion block size B. The weight of each block is repeated B times.

    Returns:
        A float32 tensor [b, K * B]. With L = K * B, this is [b, L].

    Raises:
        ValueError: If ``block_size`` is below 1, ``p_block`` is not two-dimensional, or any
            value of ``p_block`` is not positive. NaN counts as not positive.
    """
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    if p_block.dim() != 2:
        raise ValueError(f"p_block must have shape [b, K], got {tuple(p_block.shape)}")
    p = p_block.to(torch.float32)
    if not bool(torch.all(p > 0)):
        raise ValueError("p_block must be > 0 in every block")
    return (1.0 / p).repeat_interleave(block_size, dim=1)


def bd3_loss_terms(
    ce: Tensor,
    masked: Tensor,
    scorable: Tensor,
    p_block: Tensor,
    block_size: int,
    weighting: str,
) -> dict[str, Tensor]:
    """Reduce the bd3 cross entropy to the per-micro-batch (sum, count) terms
    (spec section 7).

    A target is a masked position that is scorable. Under ``elbo`` each target is weighted by
    1 / p_blk, and the count is the number of scorable positions. Under ``uniform`` the targets are
    summed without weights and counted.

    Args:
        ce: Tensor [b, L] with the cross entropy of ``x0`` at every position.
        masked: BoolTensor [b, L]. True where the noisy input holds the mask token.
        scorable: BoolTensor [b, L] from :func:`scorable_positions`.
        p_block: Tensor [b, K] with the masked fraction of each block.
        block_size: Diffusion block size B.
        weighting: ``"elbo"`` or ``"uniform"``.

    Returns:
        A dict of 0-dim float32 tensors with the keys ``elbo_sum``, ``masked_ce_sum``,
        ``num_targets``, ``scored``, ``loss_sum``, and ``num_tokens``. ``loss_sum`` and
        ``num_tokens`` hold the pair that ``weighting`` selects. ``loss_sum`` keeps the autograd
        graph of ``ce``.

    Raises:
        ValueError: If ``weighting`` is not ``"elbo"`` or ``"uniform"``, or if the dtypes or
            shapes of the inputs do not match.
    """
    if weighting not in _WEIGHTINGS:
        raise ValueError(f"weighting must be 'elbo' or 'uniform', got {weighting!r}")
    _check_bool("masked", masked)
    _check_bool("scorable", scorable)
    _check_shape("masked", masked, ce.shape)
    _check_shape("scorable", scorable, ce.shape)
    weights = token_weights(p_block, block_size)
    _check_shape("token weights from p_block", weights, ce.shape)
    target = masked & scorable
    ce32 = ce.to(torch.float32)
    elbo_sum = torch.where(target, weights * ce32, 0.0).sum()
    masked_ce_sum = torch.where(target, ce32, 0.0).sum()
    num_targets = target.sum().to(torch.float32)
    scored = scorable.sum().to(torch.float32)
    if weighting == "elbo":
        loss_sum, num_tokens = elbo_sum, scored
    else:
        loss_sum, num_tokens = masked_ce_sum, num_targets
    return {
        "elbo_sum": elbo_sum,
        "masked_ce_sum": masked_ce_sum,
        "num_targets": num_targets,
        "scored": scored,
        "loss_sum": loss_sum,
        "num_tokens": num_tokens,
    }


def ar_loss_terms(ce: Tensor, scorable: Tensor) -> tuple[Tensor, Tensor]:
    """Return the autoregressive (sum, count) pair over the scorable positions
    (spec section 7).

    Args:
        ce: Tensor [b, L] with the next-token cross entropy of ``x0``.
        scorable: BoolTensor [b, L] from :func:`scorable_positions`.

    Returns:
        A tuple ``(loss_sum, num_tokens)`` of 0-dim float32 tensors. ``loss_sum`` keeps the
        autograd graph of ``ce``.

    Raises:
        ValueError: If ``scorable`` is not boolean or its shape differs from ``ce``.
    """
    _check_bool("scorable", scorable)
    _check_shape("scorable", scorable, ce.shape)
    ce32 = ce.to(torch.float32)
    loss_sum = torch.where(scorable, ce32, 0.0).sum()
    num_tokens = scorable.sum().to(torch.float32)
    return loss_sum, num_tokens

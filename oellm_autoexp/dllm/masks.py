"""Attention masks for block-diffusion (BD3) and autoregressive (AR) training.

The BD3 model input is ``[noisy | clean]`` with 2L indices. Index ``j < L`` is the noisy copy of
position ``j``, and index ``j >= L`` is the clean copy of position ``j - L``. The mask rules follow
spec section 5 of the dllm-bd3-spec skill. The dense references in this module use Python loops
and do not reuse the mask_mod code, so tests can compare the two (invariant I1).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

_MaskMod = Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]


def _check_length(length: int) -> None:
    """Raise ValueError unless the effective length L is at least 1.

    Args:
        length: Effective sequence length L.

    Raises:
        ValueError: If length < 1.
    """
    if length < 1:
        raise ValueError(f"L must be >= 1, got {length}")


def _check_bd3_args(length: int, block_size: int) -> None:
    """Raise ValueError unless L >= 1, B >= 1, and L is a multiple of B.

    Args:
        length: Effective sequence length L.
        block_size: Diffusion block size B.

    Raises:
        ValueError: If a rule above does not hold.
    """
    _check_length(length)
    if block_size < 1:
        raise ValueError(f"B must be >= 1, got {block_size}")
    if length % block_size != 0:
        raise ValueError(f"L must be a multiple of B, got L={length} and B={block_size}")


def _check_doc_ids(doc_ids: Tensor | None, length: int) -> None:
    """Raise ValueError unless doc_ids is None or has shape [b, L].

    Args:
        doc_ids: Document ids, or None.
        length: Effective sequence length L.

    Raises:
        ValueError: If doc_ids is not 2-D with L columns.
    """
    if doc_ids is None:
        return
    if doc_ids.dim() != 2 or doc_ids.shape[1] != length:
        raise ValueError(f"doc_ids must have shape [b, {length}], got {tuple(doc_ids.shape)}")


def _checked_doc_row(doc_ids_row: Sequence[int] | None, length: int) -> list[int] | None:
    """Return one document-id row as a list of ints, or None, after checking
    its length.

    Args:
        doc_ids_row: Document id of each position, or None.
        length: Effective sequence length L.

    Returns:
        The row as a list of Python ints, or None.

    Raises:
        ValueError: If the row does not have length L.
    """
    if doc_ids_row is None:
        return None
    row = [int(value) for value in doc_ids_row]
    if len(row) != length:
        raise ValueError(f"doc_ids_row must have length {length}, got {len(row)}")
    return row


def compute_doc_ids(x0: Tensor, eod_id: int) -> Tensor:
    """Number the documents of each row by counting the EOD tokens before each
    position.

    The EOD token belongs to the document that it ends. So ``doc_ids[b, i]`` is the number of EOD
    tokens in ``x0[b, 0:i]``, and position 0 always has document id 0.

    Args:
        x0: Clean token ids with shape [b, L].
        eod_id: Token id of the end-of-document token (``tokenizer.eod``).

    Returns:
        LongTensor with shape [b, L].

    Raises:
        ValueError: If x0 is not 2-D.
    """
    if x0.dim() != 2:
        raise ValueError(f"x0 must have shape [b, L], got {tuple(x0.shape)}")
    is_eod = (x0 == eod_id).long()
    c = is_eod.cumsum(1)
    return c - is_eod


def make_bd3_mask_mod(L: int, B: int, doc_ids: Tensor | None) -> _MaskMod:
    """Build the BD3 mask_mod for the ``[noisy | clean]`` input of length 2L.

    Noisy queries attend to noisy keys in their own block and to clean keys of earlier blocks.
    Clean queries attend to clean keys of their own block and earlier blocks, so no clean query
    attends to a noisy key. With doc_ids, a pair is allowed only when both positions have the same
    document id.

    Args:
        L: Effective sequence length. Must be a multiple of B.
        B: Diffusion block size. Must be at least 1.
        doc_ids: Document ids with shape [b, L], or None to disable the document mask.

    Returns:
        A function ``mask_mod(b, h, q_idx, kv_idx)`` that returns a boolean tensor. True means
        that the query may attend to the key.

    Raises:
        ValueError: If B < 1, L % B != 0, or doc_ids does not have shape [b, L].
    """
    _check_bd3_args(L, B)
    _check_doc_ids(doc_ids, L)

    def mask_mod(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        q_noisy = q_idx < L
        kv_noisy = kv_idx < L
        q_pos = torch.where(q_noisy, q_idx, q_idx - L)
        kv_pos = torch.where(kv_noisy, kv_idx, kv_idx - L)
        q_blk = q_pos // B
        kv_blk = kv_pos // B
        n2n = q_noisy & kv_noisy & (q_blk == kv_blk)  # own noisy block, both directions
        n2c = q_noisy & ~kv_noisy & (kv_blk < q_blk)  # earlier clean blocks only
        c2c = ~q_noisy & ~kv_noisy & (kv_blk <= q_blk)  # block-causal, full inside own block
        allowed = n2n | n2c | c2c  # clean never attends to noisy
        if doc_ids is not None:
            # D-039: a noisy query takes the document of its block start, which the earlier clean
            # blocks determine, so masked tokens of its own block never shape the mask. Noisy
            # keys of the own block stay visible across a document boundary.
            q_doc = torch.where(q_noisy, doc_ids[b, q_blk * B], doc_ids[b, q_pos])
            allowed = allowed & (n2n | (q_doc == doc_ids[b, kv_pos]))
        return allowed

    return _with_grid(mask_mod, 2 * L, doc_ids)


def make_ar_mask_mod(L: int, doc_ids: Tensor | None) -> _MaskMod:
    """Build the causal mask_mod for AR training on L tokens.

    Args:
        L: Sequence length.
        doc_ids: Document ids with shape [b, L], or None to disable the document mask.

    Returns:
        A function ``mask_mod(b, h, q_idx, kv_idx)`` that returns a boolean tensor. True means
        that the query may attend to the key: the key is not after the query, and with doc_ids
        both positions have the same document id.

    Raises:
        ValueError: If L < 1 or doc_ids does not have shape [b, L].
    """
    _check_length(L)
    _check_doc_ids(doc_ids, L)

    def mask_mod(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        allowed = kv_idx <= q_idx
        if doc_ids is not None:
            allowed = allowed & (doc_ids[b, q_idx] == doc_ids[b, kv_idx])
        return allowed

    return _with_grid(mask_mod, L, doc_ids)


def _with_grid(mask_mod: _MaskMod, length: int, doc_ids: Tensor | None) -> _MaskMod:
    """Record the grid a mask_mod is built for, so that the builders below can
    check it.

    Args:
        mask_mod: The mask_mod closure.
        length: Number of query and of key positions (2L for BD3, L for AR).
        doc_ids: The document ids the closure reads, or None.

    Returns:
        mask_mod, with the attributes ``grid_length`` and ``doc_rows`` (None without doc_ids).
    """
    mask_mod.grid_length = length
    mask_mod.doc_rows = None if doc_ids is None else int(doc_ids.shape[0])
    return mask_mod


def _check_grid(mask_mod: _MaskMod, batch: int | None, q_len: int, kv_len: int) -> None:
    """Check a grid against the one a mask_mod of this module is built for.

    A mask_mod that reads document ids must get its number of document rows as the batch size.
    With batch None, FlexAttention builds the block mask from row 0 only, and the compiled kernel
    applies row 0's skipped and full blocks to every row.

    Args:
        mask_mod: A mask_mod built by this module. Other callables are not checked.
        batch: Batch size, or None for a mask that does not depend on the batch index.
        q_len: Number of query positions.
        kv_len: Number of key and value positions.

    Raises:
        ValueError: If batch differs from the number of document rows, or if q_len or kv_len
            differs from the grid length of the mask_mod.
    """
    rows = getattr(mask_mod, "doc_rows", None)
    if rows is not None and batch != rows:
        raise ValueError(f"the mask reads document ids of {rows} rows, so batch must be {rows}")
    length = getattr(mask_mod, "grid_length", None)
    if length is not None and (q_len, kv_len) != (length, length):
        raise ValueError(f"the mask is built for q_len = kv_len = {length}, got {q_len}, {kv_len}")


def dense_bd3_reference(L: int, B: int, doc_ids_row: Sequence[int] | None) -> Tensor:
    """Build the BD3 mask for one row with plain Python loops.

    The rules follow spec section 5, written per index pair. They do not reuse
    ``make_bd3_mask_mod``, so the two can be compared (invariant I1).

    Args:
        L: Effective sequence length. Must be a multiple of B.
        B: Diffusion block size. Must be at least 1.
        doc_ids_row: Document id of each position (length L), or None.

    Returns:
        Boolean tensor with shape [2L, 2L]. Entry [q, kv] is True when q may attend to kv.

    Raises:
        ValueError: If B < 1, L % B != 0, or doc_ids_row does not have length L.
    """
    _check_bd3_args(L, B)
    docs = _checked_doc_row(doc_ids_row, L)
    rows: list[list[bool]] = []
    for q in range(2 * L):
        q_noisy = q < L  # stream: indices below L form the noisy copy
        q_pos = q % L  # position of the index in the sequence
        q_blk = q_pos // B
        row: list[bool] = []
        for kv in range(2 * L):
            kv_noisy = kv < L
            kv_pos = kv % L
            kv_blk = kv_pos // B
            both_noisy = q_noisy and kv_noisy
            noisy_to_clean = q_noisy and not kv_noisy
            both_clean = not q_noisy and not kv_noisy
            own_noisy_block = both_noisy and q_blk == kv_blk
            allowed = (
                own_noisy_block
                or (noisy_to_clean and kv_blk < q_blk)
                or (both_clean and kv_blk <= q_blk)
            )
            if allowed and docs is not None and not own_noisy_block:
                # D-039: a noisy query uses the document of its block start.
                q_doc = docs[q_blk * B] if q_noisy else docs[q_pos]
                allowed = q_doc == docs[kv_pos]
            row.append(bool(allowed))
        rows.append(row)
    return torch.tensor(rows, dtype=torch.bool).reshape(2 * L, 2 * L)


def dense_ar_reference(L: int, doc_ids_row: Sequence[int] | None) -> Tensor:
    """Build the causal AR mask for one row with plain Python loops.

    Args:
        L: Sequence length.
        doc_ids_row: Document id of each position (length L), or None.

    Returns:
        Boolean tensor with shape [L, L]. Entry [q, kv] is True when kv <= q and, with
        doc_ids_row, both positions have the same document id.

    Raises:
        ValueError: If L < 1 or doc_ids_row does not have length L.
    """
    _check_length(L)
    docs = _checked_doc_row(doc_ids_row, L)
    rows: list[list[bool]] = []
    for q in range(L):
        row: list[bool] = []
        for kv in range(L):
            allowed = kv <= q
            if allowed and docs is not None:
                allowed = docs[q] == docs[kv]
            row.append(bool(allowed))
        rows.append(row)
    return torch.tensor(rows, dtype=torch.bool).reshape(L, L)


def materialize_mask(
    mask_mod: _MaskMod,
    batch: int,
    q_len: int,
    kv_len: int,
    device: str | torch.device | None = None,
) -> Tensor:
    """Evaluate a mask_mod once on the full index grid, broadcast to [batch,
    q_len, kv_len].

    The index tensors broadcast against each other. The result is then broadcast to the full
    shape, so a mask that does not depend on the batch index is repeated for every row.

    Args:
        mask_mod: A mask_mod built by this module.
        batch: Batch size (number of rows).
        q_len: Number of query positions.
        kv_len: Number of key and value positions.
        device: Device for the index tensors, or None for the default device.

    Returns:
        Boolean tensor with shape [batch, q_len, kv_len].

    Raises:
        ValueError: If the sizes do not match the mask_mod (see ``_check_grid``).
    """
    _check_grid(mask_mod, batch, q_len, kv_len)
    b = torch.arange(batch, device=device).view(-1, 1, 1)
    h = torch.zeros((1, 1, 1), dtype=torch.long, device=device)
    q = torch.arange(q_len, device=device).view(1, -1, 1)
    kv = torch.arange(kv_len, device=device).view(1, 1, -1)
    allowed = mask_mod(b, h, q, kv).to(torch.bool)
    return torch.broadcast_to(allowed, (batch, q_len, kv_len)).contiguous()


def build_block_mask(
    mask_mod: _MaskMod,
    batch: int | None,
    q_len: int,
    kv_len: int,
    device: str | torch.device,
    compile: bool = False,
) -> BlockMask:
    """Build a FlexAttention BlockMask that is shared by all heads (H = None).

    Pass the batch size when the mask reads document ids, because the mask then depends on the
    batch index. Pass None for a mask that does not depend on the batch index. Rebuild the
    BlockMask whenever B or the batch size changes.

    Args:
        mask_mod: A mask_mod built by this module.
        batch: Batch size for a batch-dependent mask, or None for a shared mask.
        q_len: Number of query positions.
        kv_len: Number of key and value positions.
        device: Device for the BlockMask, for example "cuda" or "cpu".
        compile: Whether create_block_mask compiles the mask construction.

    Returns:
        The FlexAttention BlockMask.

    Raises:
        ValueError: If the sizes do not match the mask_mod (see ``_check_grid``).
    """
    _check_grid(mask_mod, batch, q_len, kv_len)
    return create_block_mask(
        mask_mod,
        B=batch,
        H=None,
        Q_LEN=q_len,
        KV_LEN=kv_len,
        device=device,
        _compile=compile,
    )

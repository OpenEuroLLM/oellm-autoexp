"""Tensor layouts for Qwen3 Hugging Face <-> MegaDLMs (Megatron) weights.

The functions in this module move tensors between the two frameworks. They do not read or write
files. They follow spec section 13 of the dllm-bd3-spec skill:

- The Megatron ``linear_qkv`` weight holds G query groups. Each group has H // G query heads, then
  one key head, then one value head. Every head is D rows wide.
- The Megatron ``linear_fc1`` weight is split per tensor-parallel rank. Rank r holds the gate rows
  and then the up rows of its slice of the FFN. SwiGLU applies SiLU to the first half, the gate.
- Column-parallel weights and the vocabulary are split along dim 0. The row-parallel
  ``linear_proj`` and ``linear_fc2`` are split along dim 1.

The round trip HF -> Megatron -> HF is exact for every vocabulary row below the padded size
(invariant I7). The exception is the mask token row, which becomes the mean of the real-token rows
when a mask token is given. The checkpoint holds only the padded vocabulary, so the rows at or above
the padded size are not in it. :func:`megatron_to_hf_state` returns zeros for them. The exporter
restores them from the base model with :func:`restore_vocab_tail`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

_LAYOUTS = ("te", "local")
_ROTARY_INV_FREQ = "rotary_emb.inv_freq"
_EXTRA_STATE = "_extra_state"
_MISSING = object()


def _hf_head_dim(cfg: dict) -> int:
    """Return the head width D of a parsed Hugging Face Qwen3 config.

    Args:
        cfg: Parsed config.json. ``head_dim`` is used when it is present and not null. Otherwise
            the width is ``hidden_size // num_attention_heads``.

    Returns:
        The head width D.

    Raises:
        ValueError: If ``num_attention_heads`` is below 1.
    """
    head_dim = cfg.get("head_dim")
    if head_dim is not None:
        return int(head_dim)
    heads = int(cfg["num_attention_heads"])
    if heads < 1:
        raise ValueError(f"num_attention_heads must be >= 1, got {heads}")
    return int(cfg["hidden_size"]) // heads


def _hf_rope_theta(cfg: dict) -> float:
    """Return the RoPE base of a parsed Hugging Face Qwen3 config.

    Transformers 4.x stores ``rope_theta`` at the top level of config.json. Transformers 5.x stores
    it in ``rope_parameters``. The top-level value is used when both are present.

    Args:
        cfg: Parsed config.json.

    Returns:
        The RoPE base.

    Raises:
        KeyError: If neither ``rope_theta`` nor ``rope_parameters.rope_theta`` is present.
    """
    theta = cfg.get("rope_theta")
    if theta is None:
        params = cfg.get("rope_parameters")
        if isinstance(params, dict):
            theta = params.get("rope_theta")
    if theta is None:
        raise KeyError("rope_theta is missing from the Hugging Face config")
    return float(theta)


def _check_supported_features(cfg: dict) -> None:
    """Raise ValueError unless cfg uses only the features that the converter
    supports.

    The converter maps the dense Qwen3 architecture: a SiLU gated FFN, the default RoPE, full
    attention, and no attention bias. Any other setting would change the computation, so it is
    refused instead of ignored.

    Args:
        cfg: Parsed config.json of a Qwen3 model.

    Raises:
        TypeError: If rope_scaling or rope_parameters is neither an object nor null.
        ValueError: If hidden_act is not "silu", if RoPE scaling is requested in rope_scaling or in
            rope_parameters, if use_sliding_window is true, or if attention_bias is true.
    """
    act = cfg.get("hidden_act", "silu")
    if act != "silu":
        raise ValueError(f"hidden_act must be 'silu', got {act!r}")
    for field in ("rope_scaling", "rope_parameters"):
        section = cfg.get(field)
        if section is None:
            continue
        if not isinstance(section, dict):
            raise TypeError(f"{field} must be an object or null, got {section!r}")
        kind = section.get("rope_type") or section.get("type") or "default"
        if kind != "default":
            raise ValueError(f"RoPE scaling {kind!r} in {field} is not supported")
    if cfg.get("use_sliding_window", False):
        raise ValueError(
            "use_sliding_window must be false; sliding-window attention is not supported"
        )
    if cfg.get("attention_bias", False):
        raise ValueError(
            "attention_bias must be false; attention projection biases are not supported"
        )


@dataclass(frozen=True)
class Qwen3Dims:
    """Architecture sizes of a Qwen3 model, named as in spec section 13.

    Attributes:
        num_layers: Number of decoder layers.
        hidden_size: Hidden width h.
        ffn_hidden_size: FFN width F (Hugging Face ``intermediate_size``).
        num_heads: Number of query heads H.
        num_kv_heads: Number of key and value heads G. G is also the number of query groups.
        head_dim: Width D of one head.
        vocab_size: Vocabulary size of the Hugging Face model.
        tie_embeddings: True when ``lm_head`` shares its weight with the input embedding.
    """

    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    tie_embeddings: bool

    def __post_init__(self) -> None:
        """Check that every size is positive and that the heads form whole
        query groups.

        Raises:
            ValueError: If a size is not positive, or if num_heads is not a multiple of
                num_kv_heads.
        """
        sizes = {
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "ffn_hidden_size": self.ffn_hidden_size,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "vocab_size": self.vocab_size,
        }
        for name, value in sizes.items():
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be a multiple of "
                f"num_kv_heads ({self.num_kv_heads})"
            )

    @classmethod
    def from_hf_config(cls, cfg: dict) -> Qwen3Dims:
        """Read the sizes from a parsed Hugging Face ``config.json``.

        Args:
            cfg: Parsed config.json of a Qwen3 model. ``head_dim`` defaults to
                ``hidden_size // num_attention_heads`` when it is absent or null.
                ``tie_word_embeddings`` defaults to False. The model must use SiLU, the default
                RoPE, full attention, and no attention bias.

        Returns:
            The sizes of the model.

        Raises:
            KeyError: If a required key is missing from cfg.
            TypeError: If rope_scaling or rope_parameters is neither an object nor null.
            ValueError: If the sizes are invalid (see ``__post_init__``), or if cfg uses a feature
                that the converter does not support (see ``_check_supported_features``).
        """
        _check_supported_features(cfg)
        hidden = int(cfg["hidden_size"])
        heads = int(cfg["num_attention_heads"])
        return cls(
            num_layers=int(cfg["num_hidden_layers"]),
            hidden_size=hidden,
            ffn_hidden_size=int(cfg["intermediate_size"]),
            num_heads=heads,
            num_kv_heads=int(cfg["num_key_value_heads"]),
            head_dim=_hf_head_dim(cfg),
            vocab_size=int(cfg["vocab_size"]),
            tie_embeddings=bool(cfg.get("tie_word_embeddings", False)),
        )


def _check_layout(layout: str) -> None:
    """Raise ValueError unless layout names a supported layer spec.

    Args:
        layout: Layer spec name, ``"te"`` or ``"local"``.

    Raises:
        ValueError: If layout is not supported.
    """
    if layout not in _LAYOUTS:
        raise ValueError(f"layout must be one of {_LAYOUTS}, got {layout!r}")


def _check_tp(size: int, tp: int, what: str) -> None:
    """Raise ValueError unless tp >= 1 and size is divisible by tp.

    Args:
        size: Length of the dimension that is split.
        tp: Tensor-parallel size.
        what: Name of the split dimension, used in the message.

    Raises:
        ValueError: If tp < 1 or size is not divisible by tp.
    """
    if tp < 1:
        raise ValueError(f"tensor-parallel size must be >= 1, got {tp}")
    if size % tp != 0:
        raise ValueError(f"{what} = {size} is not divisible by tensor-parallel size {tp}")


def _check_2d(tensor: Tensor, name: str) -> None:
    """Raise ValueError unless tensor is 2-D.

    Args:
        tensor: Tensor to check.
        name: Name of the tensor, used in the message.

    Raises:
        ValueError: If tensor is not 2-D.
    """
    if tensor.dim() != 2:
        raise ValueError(f"{name} must be 2-D, got shape {tuple(tensor.shape)}")


def _check_uniform(shards: Sequence[Tensor]) -> None:
    """Raise ValueError unless shards is a non-empty list of 2-D tensors with
    one shape.

    Args:
        shards: Tensor shards of one split weight.

    Raises:
        ValueError: If the list is empty, a shard is not 2-D, or the shapes differ.
    """
    if not shards:
        raise ValueError("at least one shard is required")
    for shard in shards:
        _check_2d(shard, "shard")
    shapes = {tuple(shard.shape) for shard in shards}
    if len(shapes) != 1:
        raise ValueError(f"shards must share one shape, got {sorted(shapes)}")


def _expect_shape(name: str, tensor: Tensor, shape: tuple[int, ...]) -> None:
    """Raise ValueError unless tensor has the expected shape.

    Args:
        name: Name of the tensor, used in the message.
        tensor: Tensor to check.
        shape: Expected shape.

    Raises:
        ValueError: If the shape differs.
    """
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")


def _hf_shape(key: str, dims: Qwen3Dims) -> tuple[int, ...]:
    """Return the shape that a Qwen3 Hugging Face weight must have.

    Args:
        key: Hugging Face parameter name, as in Qwen3ForCausalLM.
        dims: Qwen3 sizes.

    Returns:
        The expected shape.

    Raises:
        ValueError: If the converter does not map the key.
    """
    h, d, f = dims.hidden_size, dims.head_dim, dims.ffn_hidden_size
    if key in ("model.embed_tokens.weight", "lm_head.weight"):
        return (dims.vocab_size, h)
    if key == "model.norm.weight":
        return (h,)
    if key.endswith(("input_layernorm.weight", "post_attention_layernorm.weight")):
        return (h,)
    if key.endswith(("q_norm.weight", "k_norm.weight")):
        return (d,)
    if key.endswith("q_proj.weight"):
        return (dims.num_heads * d, h)
    if key.endswith(("k_proj.weight", "v_proj.weight")):
        return (dims.num_kv_heads * d, h)
    if key.endswith("o_proj.weight"):
        return (h, dims.num_heads * d)
    if key.endswith(("gate_proj.weight", "up_proj.weight")):
        return (f, h)
    if key.endswith("down_proj.weight"):
        return (h, f)
    raise ValueError(f"no expected shape for Hugging Face key {key}")


def _qkv_groups(qkv: Tensor, dims: Qwen3Dims) -> Tensor:
    """View a Megatron linear_qkv weight as [G, H // G + 2, D, h].

    Args:
        qkv: Tensor of shape [(H + 2G) * D, h] in the Megatron layout.
        dims: Qwen3 sizes.

    Returns:
        A reshaped view of qkv. Entry [g, j] is query head j of group g for j < H // G, then the
        key head (j = H // G) and the value head (j = H // G + 1) of group g.

    Raises:
        ValueError: If qkv has the wrong shape.
    """
    heads, groups, dim = dims.num_heads, dims.num_kv_heads, dims.head_dim
    width = dims.hidden_size
    _expect_shape("qkv", qkv, ((heads + 2 * groups) * dim, width))
    return qkv.reshape(groups, heads // groups + 2, dim, width)


def qkv_hf_to_megatron(q: Tensor, k: Tensor, v: Tensor, dims: Qwen3Dims) -> Tensor:
    """Group the Hugging Face q, k, and v projections into the Megatron
    linear_qkv layout.

    Args:
        q: ``q_proj.weight`` of shape [H * D, h].
        k: ``k_proj.weight`` of shape [G * D, h].
        v: ``v_proj.weight`` of shape [G * D, h].
        dims: Qwen3 sizes.

    Returns:
        Tensor of shape [(H + 2G) * D, h]. Group g holds the query heads g * (H // G) up to
        (g + 1) * (H // G) - 1, then key head g, then value head g.

    Raises:
        ValueError: If a projection has the wrong shape.
    """
    heads, groups, dim = dims.num_heads, dims.num_kv_heads, dims.head_dim
    width = dims.hidden_size
    _expect_shape("q", q, (heads * dim, width))
    _expect_shape("k", k, (groups * dim, width))
    _expect_shape("v", v, (groups * dim, width))
    q_grouped = q.reshape(groups, heads // groups, dim, width)
    k_grouped = k.reshape(groups, 1, dim, width)
    v_grouped = v.reshape(groups, 1, dim, width)
    grouped = torch.cat([q_grouped, k_grouped, v_grouped], dim=1)
    return grouped.reshape((heads + 2 * groups) * dim, width)


def qkv_megatron_to_hf(qkv: Tensor, dims: Qwen3Dims) -> tuple[Tensor, Tensor, Tensor]:
    """Split a Megatron linear_qkv weight into Hugging Face q, k, and v
    projections.

    This is the exact inverse of :func:`qkv_hf_to_megatron`.

    Args:
        qkv: Tensor of shape [(H + 2G) * D, h] in the Megatron layout.
        dims: Qwen3 sizes.

    Returns:
        Tuple (q, k, v) with shapes [H * D, h], [G * D, h], and [G * D, h].

    Raises:
        ValueError: If qkv has the wrong shape.
    """
    heads, groups, dim = dims.num_heads, dims.num_kv_heads, dims.head_dim
    width = dims.hidden_size
    per_group = heads // groups
    grouped = _qkv_groups(qkv, dims)
    q = grouped[:, :per_group].reshape(heads * dim, width)
    k = grouped[:, per_group].reshape(groups * dim, width)
    v = grouped[:, per_group + 1].reshape(groups * dim, width)
    return q, k, v


def shard_qkv(qkv_full: Tensor, dims: Qwen3Dims, tp: int) -> list[Tensor]:
    """Split a full Megatron linear_qkv weight into tensor-parallel shards by
    query group.

    Rank r receives the query groups [r * G // tp, (r + 1) * G // tp), with their query, key, and
    value rows in Megatron order.

    Args:
        qkv_full: Full weight of shape [(H + 2G) * D, h].
        dims: Qwen3 sizes.
        tp: Tensor-parallel size.

    Returns:
        tp independent tensors. Shard r has shape [(H + 2G) * D / tp, h].

    Raises:
        ValueError: If tp < 1, G % tp != 0, or qkv_full has the wrong shape.
    """
    _check_tp(dims.num_kv_heads, tp, "num_kv_heads")
    grouped = _qkv_groups(qkv_full, dims)
    per_rank = dims.num_kv_heads // tp
    width = dims.hidden_size
    shards = []
    for rank in range(tp):
        groups_of_rank = grouped[rank * per_rank : (rank + 1) * per_rank]
        shards.append(groups_of_rank.reshape(-1, width).clone())
    return shards


def unshard_qkv(shards: list[Tensor], dims: Qwen3Dims) -> Tensor:
    """Concatenate tensor-parallel linear_qkv shards into the full Megatron
    weight.

    This is the inverse of :func:`shard_qkv`. The number of shards is the tensor-parallel size.

    Args:
        shards: One tensor per rank, each of shape [(H + 2G) * D / tp, h].
        dims: Qwen3 sizes.

    Returns:
        Full weight of shape [(H + 2G) * D, h].

    Raises:
        ValueError: If there are no shards, G % len(shards) != 0, or a shard has the wrong shape.
    """
    _check_uniform(shards)
    tp = len(shards)
    _check_tp(dims.num_kv_heads, tp, "num_kv_heads")
    heads, groups, dim = dims.num_heads, dims.num_kv_heads, dims.head_dim
    width = dims.hidden_size
    per_rank = groups // tp
    group_rows = heads // groups + 2
    parts = []
    for shard in shards:
        _expect_shape("qkv shard", shard, (per_rank * group_rows * dim, width))
        parts.append(shard.reshape(per_rank, group_rows, dim, width))
    return torch.cat(parts, dim=0).reshape((heads + 2 * groups) * dim, width)


def shard_fc1(gate: Tensor, up: Tensor, tp: int) -> list[Tensor]:
    """Split a SwiGLU fc1 weight into tensor-parallel shards.

    Rank r receives the gate rows of its FFN slice, then the up rows of the same slice.

    Args:
        gate: ``gate_proj.weight`` of shape [F, h].
        up: ``up_proj.weight`` of shape [F, h].
        tp: Tensor-parallel size.

    Returns:
        tp tensors. Shard r has shape [2F / tp, h].

    Raises:
        ValueError: If tp < 1, F % tp != 0, or gate and up have different shapes.
    """
    _check_2d(gate, "gate")
    _expect_shape("up", up, tuple(gate.shape))
    ffn = gate.shape[0]
    _check_tp(ffn, tp, "ffn_hidden_size")
    per_rank = ffn // tp
    shards = []
    for rank in range(tp):
        rows = slice(rank * per_rank, (rank + 1) * per_rank)
        shards.append(torch.cat([gate[rows], up[rows]], dim=0))
    return shards


def unshard_fc1(shards: list[Tensor]) -> tuple[Tensor, Tensor]:
    """Recover the gate and up weights from tensor-parallel fc1 shards.

    This is the inverse of :func:`shard_fc1`.

    Args:
        shards: One tensor per rank, each of shape [2F / tp, h].

    Returns:
        Tuple (gate, up), each of shape [F, h].

    Raises:
        ValueError: If there are no shards, the shapes differ, or the shard row count is odd.
    """
    _check_uniform(shards)
    rows = shards[0].shape[0]
    if rows % 2 != 0:
        raise ValueError(f"fc1 shards must have an even row count, got {rows}")
    half = rows // 2
    gate = torch.cat([shard[:half] for shard in shards], dim=0)
    up = torch.cat([shard[half:] for shard in shards], dim=0)
    return gate, up


def shard_rows(weight: Tensor, tp: int) -> list[Tensor]:
    """Split a 2-D weight into tensor-parallel shards along dim 0.

    Used for column-parallel weights and for vocabulary rows.

    Args:
        weight: Tensor of shape [R, C].
        tp: Tensor-parallel size.

    Returns:
        tp independent tensors, each of shape [R / tp, C].

    Raises:
        ValueError: If weight is not 2-D, tp < 1, or R % tp != 0.
    """
    _check_2d(weight, "weight")
    _check_tp(weight.shape[0], tp, "rows")
    return [part.clone() for part in torch.chunk(weight, tp, dim=0)]


def unshard_rows(shards: list[Tensor]) -> Tensor:
    """Concatenate tensor-parallel shards along dim 0. This inverts
    :func:`shard_rows`.

    Args:
        shards: One tensor per rank, all with the same shape.

    Returns:
        Tensor of shape [tp * R / tp, C].

    Raises:
        ValueError: If there are no shards, or the shards have different shapes.
    """
    _check_uniform(shards)
    return torch.cat(shards, dim=0)


def shard_cols(weight: Tensor, tp: int) -> list[Tensor]:
    """Split a 2-D weight into tensor-parallel shards along dim 1.

    Used for the row-parallel ``linear_proj`` and ``linear_fc2`` weights.

    Args:
        weight: Tensor of shape [R, C].
        tp: Tensor-parallel size.

    Returns:
        tp independent tensors, each of shape [R, C / tp].

    Raises:
        ValueError: If weight is not 2-D, tp < 1, or C % tp != 0.
    """
    _check_2d(weight, "weight")
    _check_tp(weight.shape[1], tp, "columns")
    return [part.clone() for part in torch.chunk(weight, tp, dim=1)]


def unshard_cols(shards: list[Tensor]) -> Tensor:
    """Concatenate tensor-parallel shards along dim 1. This inverts
    :func:`shard_cols`.

    Args:
        shards: One tensor per rank, all with the same shape.

    Returns:
        Tensor of shape [R, tp * C / tp].

    Raises:
        ValueError: If there are no shards, or the shards have different shapes.
    """
    _check_uniform(shards)
    return torch.cat(shards, dim=1)


def vocab_to_megatron(
    weight_hf: Tensor,
    padded_rows: int,
    real_vocab: int,
    mask_token_id: int | None,
) -> Tensor:
    """Pad a Hugging Face vocabulary matrix to the Megatron padded size.

    The first min(V_hf, padded_rows) rows are copied, and the remaining rows are zero. If
    mask_token_id is set, that row becomes the mean of the first real_vocab rows of weight_hf,
    cast to the dtype of weight_hf. This is the initialization of spec section 12.

    Args:
        weight_hf: Hugging Face embedding or lm_head weight of shape [V_hf, h].
        padded_rows: Total padded vocabulary size of the Megatron model.
        real_vocab: Number of real tokens, len(tokenizer). Must be at most min(V_hf, padded_rows).
        mask_token_id: Mask token id M, or None when the model has no mask token. Spec section 12
            sets M = len(tokenizer), which is real_vocab. M must satisfy
            real_vocab <= M < padded_rows, so that the mask row never overwrites a real token. A
            smaller M, such as a row of the real vocabulary, is rejected.

    Returns:
        Tensor of shape [padded_rows, h].

    Raises:
        ValueError: If padded_rows < 1, real_vocab is outside [1, min(V_hf, padded_rows)], or
            mask_token_id is outside [real_vocab, padded_rows).
    """
    _check_2d(weight_hf, "weight_hf")
    vocab_hf, width = weight_hf.shape
    if padded_rows < 1:
        raise ValueError(f"padded_rows must be >= 1, got {padded_rows}")
    if not 1 <= real_vocab <= min(vocab_hf, padded_rows):
        raise ValueError(
            f"real_vocab must be in [1, min(V_hf={vocab_hf}, padded_rows={padded_rows})], "
            f"got {real_vocab}"
        )
    out = torch.zeros(padded_rows, width, dtype=weight_hf.dtype, device=weight_hf.device)
    copied = min(vocab_hf, padded_rows)
    out[:copied] = weight_hf[:copied]
    if mask_token_id is not None:
        if not real_vocab <= mask_token_id < padded_rows:
            raise ValueError(
                f"mask_token_id must be in [real_vocab={real_vocab}, "
                f"padded_rows={padded_rows}), got {mask_token_id}"
            )
        out[mask_token_id] = weight_hf[:real_vocab].float().mean(dim=0).to(weight_hf.dtype)
    return out


def vocab_to_hf(weight_mega: Tensor, vocab_size_hf: int) -> Tensor:
    """Cut a Megatron vocabulary matrix back to the Hugging Face vocabulary
    size.

    The first min(rows, vocab_size_hf) rows are copied, and any remaining rows are zero.

    Args:
        weight_mega: Megatron vocabulary matrix of shape [padded_rows, h].
        vocab_size_hf: Hugging Face ``vocab_size``.

    Returns:
        Tensor of shape [vocab_size_hf, h].

    Raises:
        ValueError: If weight_mega is not 2-D or vocab_size_hf < 1.
    """
    _check_2d(weight_mega, "weight_mega")
    if vocab_size_hf < 1:
        raise ValueError(f"vocab_size_hf must be >= 1, got {vocab_size_hf}")
    rows, width = weight_mega.shape
    out = torch.zeros(vocab_size_hf, width, dtype=weight_mega.dtype, device=weight_mega.device)
    copied = min(rows, vocab_size_hf)
    out[:copied] = weight_mega[:copied]
    return out


def restore_vocab_tail(weight_hf: Tensor, base_hf: Tensor, padded_rows: int) -> Tensor:
    """Restore the vocabulary rows at and above the padded size from the base
    model.

    The Megatron checkpoint holds only the padded rows. When the Hugging Face vocabulary is larger,
    :func:`vocab_to_hf` fills the rows from padded_rows up to V_hf with zeros. This function
    replaces those rows with the original rows of the base model. Rows below padded_rows are not
    changed.

    Args:
        weight_hf: Hugging Face vocabulary matrix of shape [V_hf, h], as exported.
        base_hf: The same matrix in the base model, of shape [V_hf, h].
        padded_rows: Total padded vocabulary size of the Megatron model. Rows [padded_rows, V_hf)
            are replaced. A value of at least V_hf replaces nothing.

    Returns:
        A new tensor with the dtype and device of weight_hf.

    Raises:
        ValueError: If a matrix is not 2-D, the shapes differ, or padded_rows is negative.
    """
    _check_2d(weight_hf, "weight_hf")
    _check_2d(base_hf, "base_hf")
    _expect_shape("base_hf", base_hf, tuple(weight_hf.shape))
    if padded_rows < 0:
        raise ValueError(f"padded_rows must be >= 0, got {padded_rows}")
    out = weight_hf.clone()
    if padded_rows < out.shape[0]:
        out[padded_rows:] = base_hf[padded_rows:].to(dtype=out.dtype, device=out.device)
    return out


def layer_key_table(layer: int, layout: str = "te") -> dict[str, tuple[str, str]]:
    """Map the Megatron parameter names of one decoder layer to their Hugging
    Face sources.

    Args:
        layer: Decoder layer index l.
        layout: ``"te"`` for the Transformer Engine layer spec. It fuses the input RMSNorm into
            ``linear_qkv`` and the post-attention RMSNorm into ``linear_fc1``. ``"local"`` for the
            Megatron-core layer spec. It has separate ``input_layernorm`` and ``pre_mlp_layernorm``
            modules.

    Returns:
        Dict from Megatron key to ``(kind, source)``. The kinds are:

        - ``"none"``: a replicated tensor. The source is one Hugging Face key.
        - ``"qkv"``: grouped by query group. The source is the prefix ``model.layers.l.self_attn``,
          and the keys ``q_proj``, ``k_proj``, and ``v_proj`` are read from it.
        - ``"cols"``: split along dim 1. The source is one Hugging Face key.
        - ``"fc1"``: split per rank. The source is the prefix ``model.layers.l.mlp``, and the
          keys ``gate_proj`` and ``up_proj`` are read from it.

    Raises:
        ValueError: If layout is not ``"te"`` or ``"local"``.
    """
    _check_layout(layout)
    mega_prefix = f"decoder.layers.{layer}"
    hf_prefix = f"model.layers.{layer}"
    if layout == "te":
        input_norm_key = f"{mega_prefix}.self_attention.linear_qkv.layer_norm_weight"
        post_norm_key = f"{mega_prefix}.mlp.linear_fc1.layer_norm_weight"
    else:
        input_norm_key = f"{mega_prefix}.input_layernorm.weight"
        post_norm_key = f"{mega_prefix}.pre_mlp_layernorm.weight"
    return {
        input_norm_key: ("none", f"{hf_prefix}.input_layernorm.weight"),
        f"{mega_prefix}.self_attention.linear_qkv.weight": ("qkv", f"{hf_prefix}.self_attn"),
        f"{mega_prefix}.self_attention.q_layernorm.weight": (
            "none",
            f"{hf_prefix}.self_attn.q_norm.weight",
        ),
        f"{mega_prefix}.self_attention.k_layernorm.weight": (
            "none",
            f"{hf_prefix}.self_attn.k_norm.weight",
        ),
        f"{mega_prefix}.self_attention.linear_proj.weight": (
            "cols",
            f"{hf_prefix}.self_attn.o_proj.weight",
        ),
        post_norm_key: ("none", f"{hf_prefix}.post_attention_layernorm.weight"),
        f"{mega_prefix}.mlp.linear_fc1.weight": ("fc1", f"{hf_prefix}.mlp"),
        f"{mega_prefix}.mlp.linear_fc2.weight": ("cols", f"{hf_prefix}.mlp.down_proj.weight"),
    }


def hf_to_megatron_state(
    hf: dict[str, Tensor],
    dims: Qwen3Dims,
    tp: int,
    padded_vocab: int,
    real_vocab: int,
    mask_token_id: int | None,
    layout: str = "te",
) -> list[dict[str, Tensor]]:
    """Convert Hugging Face Qwen3 weights into one Megatron state dict per
    tensor-parallel rank.

    The result holds every model key of the Megatron layout in spec section 13. Every Hugging Face
    key must be used, except rotary ``inv_freq`` buffers. In a tied model, ``lm_head.weight`` may
    be absent. If it is present, it must equal ``model.embed_tokens.weight``.

    Args:
        hf: Hugging Face state dict with the keys of Qwen3ForCausalLM. Shapes are checked against
            dims.
        dims: Qwen3 sizes.
        tp: Tensor-parallel size.
        padded_vocab: Total padded vocabulary size of the Megatron model. Each rank holds
            padded_vocab // tp vocabulary rows.
        real_vocab: Number of real tokens, len(tokenizer).
        mask_token_id: Mask token id, or None. It must satisfy real_vocab <= M < padded_vocab (see
            :func:`vocab_to_megatron`). Its embedding row, and its output row when the model is
            untied, is set to the mean of the real-token rows.
        layout: ``"te"`` or ``"local"``. See :func:`layer_key_table`.

    Returns:
        List of tp dicts, one per rank, that map Megatron keys to tensors.

    Raises:
        KeyError: If a required Hugging Face key is missing.
        ValueError: If the layout or tp is unsupported, a shape differs, a Hugging Face key is
            unused, or a tied lm_head differs from the embedding.
    """
    _check_layout(layout)
    _check_tp(padded_vocab, tp, "padded_vocab")
    used: set[str] = set()

    def take(key: str) -> Tensor:
        if key not in hf:
            raise KeyError(f"missing Hugging Face key {key}")
        used.add(key)
        _expect_shape(key, hf[key], _hf_shape(key, dims))
        return hf[key]

    ranks: list[dict[str, Tensor]] = [{} for _ in range(tp)]

    embed_hf = take("model.embed_tokens.weight")
    embedding = vocab_to_megatron(embed_hf, padded_vocab, real_vocab, mask_token_id)
    for rank, part in zip(ranks, shard_rows(embedding, tp), strict=True):
        rank["embedding.word_embeddings.weight"] = part

    for layer in range(dims.num_layers):
        for mega_key, (kind, source) in layer_key_table(layer, layout).items():
            if kind == "none":
                value = take(source)
                parts = [value.clone() for _ in range(tp)]
            elif kind == "qkv":
                full = qkv_hf_to_megatron(
                    take(f"{source}.q_proj.weight"),
                    take(f"{source}.k_proj.weight"),
                    take(f"{source}.v_proj.weight"),
                    dims,
                )
                parts = shard_qkv(full, dims, tp)
            elif kind == "cols":
                parts = shard_cols(take(source), tp)
            elif kind == "fc1":
                gate = take(f"{source}.gate_proj.weight")
                up = take(f"{source}.up_proj.weight")
                parts = shard_fc1(gate, up, tp)
            else:
                raise ValueError(f"unknown layer key kind {kind!r}")
            for rank, part in zip(ranks, parts, strict=True):
                rank[mega_key] = part

    final_norm = take("model.norm.weight")
    for rank in ranks:
        rank["decoder.final_layernorm.weight"] = final_norm.clone()

    if dims.tie_embeddings:
        if "lm_head.weight" in hf:
            head_hf = take("lm_head.weight")
            if not torch.equal(head_hf, embed_hf):
                raise ValueError(
                    "tied model: lm_head.weight differs from model.embed_tokens.weight"
                )
    else:
        head_hf = take("lm_head.weight")
        head = vocab_to_megatron(head_hf, padded_vocab, real_vocab, mask_token_id)
        for rank, part in zip(ranks, shard_rows(head, tp), strict=True):
            rank["output_layer.weight"] = part

    unused = sorted(k for k in hf if k not in used and not k.endswith(_ROTARY_INV_FREQ))
    if unused:
        raise ValueError(f"unused Hugging Face keys: {unused}")
    return ranks


def megatron_to_hf_state(
    shards: list[dict[str, Tensor]],
    dims: Qwen3Dims,
    layout: str = "te",
) -> dict[str, Tensor]:
    """Convert per-rank Megatron state dicts back into Hugging Face Qwen3
    weights.

    This is the inverse of :func:`hf_to_megatron_state`. Keys that end in ``_extra_state`` are
    ignored. Any other key that the converter does not map is an error. ``lm_head.weight`` is
    produced only when the model is untied. When the Hugging Face vocabulary is larger than the
    padded size, the rows from the padded size up are not in the checkpoint, so they come back as
    zeros. :func:`restore_vocab_tail` restores them from the base model.

    Args:
        shards: One Megatron state dict per tensor-parallel rank, in rank order.
        dims: Qwen3 sizes.
        layout: ``"te"`` or ``"local"``. See :func:`layer_key_table`.

    Returns:
        Hugging Face state dict with the keys of Qwen3ForCausalLM.

    Raises:
        KeyError: If a mapped Megatron key is missing from a rank.
        ValueError: If there are no shards, the layout is unsupported, a replicated tensor
            differs between ranks, a Megatron key is unused, or a shape is wrong.
    """
    _check_layout(layout)
    if not shards:
        raise ValueError("at least one tensor-parallel shard is required")
    used: set[str] = set()

    def gather(key: str) -> list[Tensor]:
        missing = [rank for rank, shard in enumerate(shards) if key not in shard]
        if missing:
            raise KeyError(f"Megatron key {key} is missing on ranks {missing}")
        used.add(key)
        return [shard[key] for shard in shards]

    def replicated(key: str) -> Tensor:
        values = gather(key)
        if any(not torch.equal(value, values[0]) for value in values[1:]):
            raise ValueError(f"replicated Megatron key {key} differs between ranks")
        return values[0]

    out: dict[str, Tensor] = {
        "model.embed_tokens.weight": vocab_to_hf(
            unshard_rows(gather("embedding.word_embeddings.weight")), dims.vocab_size
        ),
    }
    for layer in range(dims.num_layers):
        for mega_key, (kind, source) in layer_key_table(layer, layout).items():
            if kind == "none":
                out[source] = replicated(mega_key).clone()
            elif kind == "qkv":
                q, k, v = qkv_megatron_to_hf(unshard_qkv(gather(mega_key), dims), dims)
                out[f"{source}.q_proj.weight"] = q
                out[f"{source}.k_proj.weight"] = k
                out[f"{source}.v_proj.weight"] = v
            elif kind == "cols":
                out[source] = unshard_cols(gather(mega_key))
            elif kind == "fc1":
                gate, up = unshard_fc1(gather(mega_key))
                out[f"{source}.gate_proj.weight"] = gate
                out[f"{source}.up_proj.weight"] = up
            else:
                raise ValueError(f"unknown layer key kind {kind!r}")
    out["model.norm.weight"] = replicated("decoder.final_layernorm.weight").clone()
    if not dims.tie_embeddings:
        head = unshard_rows(gather("output_layer.weight"))
        out["lm_head.weight"] = vocab_to_hf(head, dims.vocab_size)

    for rank, shard in enumerate(shards):
        unused = sorted(k for k in shard if k not in used and not k.endswith(_EXTRA_STATE))
        if unused:
            raise ValueError(f"unused Megatron keys on rank {rank}: {unused}")
    for key, value in out.items():
        _expect_shape(key, value, _hf_shape(key, dims))
    return out


def _same(actual: Any, expected: Any) -> bool:
    """Return True when actual equals expected.

    Args:
        actual: Value from the checkpoint args.
        expected: Value from the Hugging Face config.

    Returns:
        The result of the equality test.
    """
    return actual == expected


def _flag(actual: Any, expected: bool) -> bool:
    """Return True when the truth value of actual equals expected.

    Args:
        actual: Flag from the checkpoint args.
        expected: Required truth value.

    Returns:
        True when bool(actual) equals expected.
    """
    return bool(actual) == expected


def _close(actual: Any, expected: float) -> bool:
    """Return True when actual is within a relative tolerance of 1e-9 of
    expected.

    Args:
        actual: Value from the checkpoint args. It must convert to float.
        expected: Value from the Hugging Face config.

    Returns:
        False when actual cannot be converted to float.
    """
    try:
        return math.isclose(float(actual), expected, rel_tol=1e-9)
    except (TypeError, ValueError):
        return False


def _as_float(actual: Any, expected: float) -> bool:
    """Return True when actual equals expected after both are converted to
    float.

    Args:
        actual: Value from the checkpoint args. It must convert to float.
        expected: Value from the Hugging Face config.

    Returns:
        False when actual cannot be converted to float.
    """
    try:
        return float(actual) == expected
    except (TypeError, ValueError):
        return False


def check_args_against_hf(args: Any, hf_config: dict, layout: str) -> None:
    """Check that the MegaDLMs arguments of a checkpoint match a Hugging Face
    Qwen3 config.

    A converted checkpoint is read with the architecture of its Hugging Face config, so the
    training run must have used the same architecture. Each pair below is compared only when args
    has the attribute. A missing attribute is skipped. The layout is validated and named in the
    error message. No pair depends on it, because the key set of the layout is checked when the
    template is filled.

    Args:
        args: Arguments of a MegaDLMs checkpoint, such as ``checkpoint["args"]``.
        hf_config: Parsed config.json of the Hugging Face model.
        layout: Layer spec, ``"te"`` or ``"local"``.

    Raises:
        ValueError: If the layout is unsupported, or if one or more pairs differ. The message lists
            every mismatch.
        KeyError: If hf_config has no ``rms_norm_eps``, or no RoPE base in ``rope_theta`` or
            ``rope_parameters.rope_theta``.
    """
    _check_layout(layout)
    heads = int(hf_config["num_attention_heads"])
    groups = int(hf_config["num_key_value_heads"])
    tied = bool(hf_config.get("tie_word_embeddings", False))
    checks: list[tuple[str, Any, Callable[[Any, Any], bool]]] = [
        ("num_layers", int(hf_config["num_hidden_layers"]), _same),
        ("hidden_size", int(hf_config["hidden_size"]), _same),
        ("ffn_hidden_size", int(hf_config["intermediate_size"]), _same),
        ("num_attention_heads", heads, _same),
        ("num_query_groups", groups, _same),
        ("kv_channels", _hf_head_dim(hf_config), _same),
        ("rotary_base", _hf_rope_theta(hf_config), _as_float),
        ("norm_epsilon", float(hf_config["rms_norm_eps"]), _close),
        ("rotary_interleaved", False, _flag),
        ("untie_embeddings_and_output_weights", not tied, _flag),
        ("swiglu", True, _flag),
        ("qk_layernorm", True, _flag),
        ("add_bias_linear", False, _flag),
        ("normalization", "RMSNorm", _same),
        ("position_embedding_type", "rope", _same),
        # Megatron options that change the forward function, which Qwen3 does not use.
        ("apply_layernorm_1p", False, _flag),
        ("apply_residual_connection_post_layernorm", False, _flag),
        ("use_rope_scaling", False, _flag),
        ("rotary_percent", 1.0, _as_float),
    ]
    if groups < heads:
        checks.append(("group_query_attention", True, _flag))
    problems: list[str] = []
    for name, expected, matches in checks:
        actual = getattr(args, name, _MISSING)
        if actual is _MISSING or matches(actual, expected):
            continue
        problems.append(f"{name} is {actual!r} but the Hugging Face config needs {expected!r}")
    if problems:
        raise ValueError(
            f"checkpoint args do not match the Hugging Face config (layout {layout!r}): "
            + "; ".join(problems)
        )

"""Tiny pure-PyTorch Qwen3-like reference model for dense boolean masks.

The model has RMSNorm, grouped-query attention with QK-norm, RoPE with explicit position ids, and
a SwiGLU MLP. It takes a dense ``[b, s, s]`` boolean mask instead of a FlexAttention BlockMask, so
CPU tests can run the block-diffusion invariants without a GPU. The CPU tests use it for I2 and I3
of spec section 11, and the planned sampler test uses it for I8. It is a test and diagnostic model
only. Training uses the MegaDLMs model.

The mask helpers build their masks with :mod:`oellm_autoexp.dllm.masks`, and the source table of
the logits helpers comes from :mod:`oellm_autoexp.dllm.targets`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from oellm_autoexp.dllm.masks import make_ar_mask_mod, make_bd3_mask_mod, materialize_mask
from oellm_autoexp.dllm.targets import prediction_sources


@dataclass
class RefConfig:
    """Sizes and settings of the reference model.

    Attributes:
        vocab_size: Number of token ids, including the mask id.
        hidden_size: Width h of the residual stream.
        num_layers: Number of decoder blocks.
        num_heads: Number of query heads H.
        num_kv_heads: Number of key and value heads G. H must be a multiple of G.
        head_dim: Width D of one head. Must be even, because RoPE rotates pairs of channels.
        ffn_hidden_size: Width F of the SwiGLU hidden layer.
        rope_theta: RoPE base.
        norm_eps: Epsilon of every RMSNorm.
        tie_embeddings: If True, ``lm_head.weight`` is the same parameter as
            ``embed_tokens.weight``.

    Raises:
        ValueError: From ``__post_init__``, if a size is below 1, H is not a multiple of G, D is
            odd, or rope_theta or norm_eps is not positive.
    """

    vocab_size: int = 64
    hidden_size: int = 32
    num_layers: int = 2
    num_heads: int = 4
    num_kv_heads: int = 2
    head_dim: int = 8
    ffn_hidden_size: int = 48
    rope_theta: float = 1e6
    norm_eps: float = 1e-6
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        """Raise ValueError if the settings cannot form a valid model."""
        sizes = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "ffn_hidden_size": self.ffn_hidden_size,
        }
        for name, value in sizes.items():
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads must be a multiple of num_kv_heads, got {self.num_heads} and "
                f"{self.num_kv_heads}"
            )
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {self.head_dim}")
        if not self.rope_theta > 0:
            raise ValueError(f"rope_theta must be > 0, got {self.rope_theta}")
        if not self.norm_eps > 0:
            raise ValueError(f"norm_eps must be > 0, got {self.norm_eps}")


class RMSNorm(nn.Module):
    """Root-mean-square norm over the last axis, with a learned scale.

    The mean square is computed in float64 when the input is float64, and in float32 otherwise.
    The output has the dtype of the input.

    Args:
        dim: Size of the normalized axis.
        eps: Value added to the mean square before the reciprocal square root.
    """

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """Normalize the last axis of ``x``.

        Args:
            x: Tensor whose last axis has the size ``dim``.

        Returns:
            A tensor with the shape and dtype of ``x``.
        """
        compute_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        xc = x.to(compute_dtype)
        normed = xc * torch.rsqrt(xc.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normed * self.weight.to(compute_dtype)).to(x.dtype)


def rope_cos_sin(position_ids: Tensor, head_dim: int, theta: float) -> tuple[Tensor, Tensor]:
    """Return the RoPE cosine and sine tables for explicit position ids.

    The tables are computed in float64. :func:`apply_rope` casts them to the dtype of its input.

    Args:
        position_ids: Integer tensor [b, s] with the rotary position of each token.
        head_dim: Head width D. Must be even.
        theta: RoPE base.

    Returns:
        A tuple ``(cos, sin)`` of float64 tensors with shape [b, s, D]. The two halves of the last
        axis are equal.

    Raises:
        ValueError: If position_ids is not 2-D, D is odd or below 2, or theta is not positive.
    """
    if position_ids.dim() != 2:
        raise ValueError(f"position_ids must have shape [b, s], got {tuple(position_ids.shape)}")
    if head_dim < 2 or head_dim % 2 != 0:
        raise ValueError(f"head_dim must be an even number >= 2, got {head_dim}")
    if not theta > 0:
        raise ValueError(f"theta must be > 0, got {theta}")
    pairs = torch.arange(0, head_dim, 2, dtype=torch.float64, device=position_ids.device)
    inv_freq = 1 / theta ** (pairs / head_dim)
    freqs = position_ids.to(torch.float64)[..., None] * inv_freq
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: Tensor) -> Tensor:
    """Rotate the last axis by halves, ``cat([-x2, x1], -1)``.

    Here x1 and x2 are the first and the second half of the last axis.

    Args:
        x: Tensor whose last axis has an even size.

    Returns:
        A tensor with the shape of ``x``.

    Raises:
        ValueError: If the last axis of ``x`` has an odd size.
    """
    size = x.shape[-1]
    if size % 2 != 0:
        raise ValueError(f"the last axis must have an even size, got {size}")
    half = size // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary embeddings to a query or key tensor.

    Args:
        x: Tensor with shape [b, heads, s, D].
        cos: Cosine table with shape [b, s, D], from :func:`rope_cos_sin`. A head axis is added.
        sin: Sine table with shape [b, s, D], from :func:`rope_cos_sin`. A head axis is added.

    Returns:
        ``x * cos + rotate_half(x) * sin``, with the shape and dtype of ``x``.

    Raises:
        ValueError: If x is not 4-D, or cos or sin does not have the shape [b, s, D] of x.
    """
    if x.dim() != 4:
        raise ValueError(f"x must have shape [b, heads, s, D], got {tuple(x.shape)}")
    batch, _, seq_len, head_dim = x.shape
    expected = (batch, seq_len, head_dim)
    if tuple(cos.shape) != expected or tuple(sin.shape) != expected:
        raise ValueError(
            f"cos and sin must have shape {expected}, got {tuple(cos.shape)} and {tuple(sin.shape)}"
        )
    cos_h = cos.to(x.dtype).unsqueeze(1)
    sin_h = sin.to(x.dtype).unsqueeze(1)
    return x * cos_h + rotate_half(x) * sin_h


class RefMLP(nn.Module):
    """SwiGLU block, ``down(silu(gate(x)) * up(x))``, without biases.

    Args:
        hidden_size: Width h of the input and output.
        ffn_hidden_size: Width F of the gate and up projections.
    """

    def __init__(self, hidden_size: int, ffn_hidden_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.down_proj = nn.Linear(ffn_hidden_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the SwiGLU block to ``x`` with shape [..., h]."""
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class RefAttention(nn.Module):
    """Grouped-query attention with QK-norm and RoPE over a dense mask.

    The query and key heads are normalized before RoPE, as in Qwen3. Each key and value head is
    shared by H / G query heads. Every query row of the mask must allow at least one key, because
    a row without keys gives NaN after the softmax.

    Args:
        config: Model settings.
    """

    def __init__(self, config: RefConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        q_width = config.num_heads * config.head_dim
        kv_width = config.num_kv_heads * config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, q_width, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, kv_width, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_width, bias=False)
        self.o_proj = nn.Linear(q_width, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(config.head_dim, config.norm_eps)
        self.k_norm = RMSNorm(config.head_dim, config.norm_eps)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, mask: Tensor) -> Tensor:
        """Attend over ``x`` under ``mask``.

        Args:
            x: Residual stream after the input norm, shape [b, s, h].
            cos: RoPE cosine table, shape [b, s, D].
            sin: RoPE sine table, shape [b, s, D].
            mask: BoolTensor [b, s, s]. True means that the query may attend to the key.

        Returns:
            Attention output with shape [b, s, h].
        """
        batch, seq_len, _ = x.shape
        d = self.head_dim
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, d).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, d).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, d).transpose(1, 2)
        q = apply_rope(self.q_norm(q), cos, sin)
        k = apply_rope(self.k_norm(k), cos, sin)
        group = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
        scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v).transpose(1, 2).reshape(batch, seq_len, self.num_heads * d)
        return self.o_proj(out)


class RefBlock(nn.Module):
    """Pre-norm decoder block: attention, then SwiGLU, each with a residual connection.

    Args:
        config: Model settings.
    """

    def __init__(self, config: RefConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.norm_eps)
        self.self_attn = RefAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.norm_eps)
        self.mlp = RefMLP(config.hidden_size, config.ffn_hidden_size)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, mask: Tensor) -> Tensor:
        """Return the residual stream after the block, shape [b, s, h]."""
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, mask)
        return x + self.mlp(self.post_attention_layernorm(x))


def _check_forward_inputs(input_ids: Tensor, position_ids: Tensor, mask: Tensor) -> None:
    """Raise ValueError unless the inputs of RefModel.forward are valid.

    Args:
        input_ids: Token ids. Must be a LongTensor [b, s].
        position_ids: Positions. Must have the shape of ``input_ids``.
        mask: Attention mask. Must be a BoolTensor [b, s, s] that allows a key for every query.

    Raises:
        ValueError: If a check above fails.
    """
    if input_ids.dim() != 2 or input_ids.dtype != torch.long:
        raise ValueError(
            f"input_ids must be a LongTensor with shape [b, s], got {input_ids.dtype} "
            f"{tuple(input_ids.shape)}"
        )
    batch, seq_len = input_ids.shape
    if tuple(position_ids.shape) != (batch, seq_len):
        raise ValueError(
            f"position_ids must have shape {(batch, seq_len)}, got {tuple(position_ids.shape)}"
        )
    if mask.dtype != torch.bool or tuple(mask.shape) != (batch, seq_len, seq_len):
        raise ValueError(
            f"mask must be a BoolTensor with shape {(batch, seq_len, seq_len)}, got "
            f"{mask.dtype} {tuple(mask.shape)}"
        )
    if not bool(mask.any(dim=-1).all()):
        raise ValueError("every query row of mask must allow at least one key")


class RefModel(nn.Module):
    """Tiny Qwen3-like decoder over explicit position ids and a dense mask.

    Args:
        config: Model settings. With ``tie_embeddings``, ``lm_head`` shares the embedding weight.
    """

    def __init__(self, config: RefConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([RefBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        mask: Tensor,
        output_hidden_states: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        """Return the logits, and optionally the per-layer residual stream.

        Args:
            input_ids: LongTensor [b, s] of token ids.
            position_ids: Tensor [b, s] of RoPE positions, used as given.
            mask: BoolTensor [b, s, s]. True means that query ``q`` may attend to key ``k``.
                Every query row must allow at least one key.
            output_hidden_states: If True, also return the residual stream after each layer.

        Returns:
            The logits, with shape [b, s, vocab_size]. With ``output_hidden_states``, a tuple
            ``(logits, hidden_states)``. ``hidden_states`` is a list of ``num_layers`` tensors with
            shape [b, s, h]. Each one is the residual stream after a layer and before the final
            norm.

        Raises:
            ValueError: If the shapes or dtypes of the inputs are not as documented, or a query
                row of ``mask`` allows no key.
        """
        _check_forward_inputs(input_ids, position_ids, mask)
        cos, sin = rope_cos_sin(position_ids, self.config.head_dim, self.config.rope_theta)
        hidden = self.embed_tokens(input_ids)
        layer_states: list[Tensor] = []
        for layer in self.layers:
            hidden = layer(hidden, cos, sin, mask)
            if output_hidden_states:
                layer_states.append(hidden)
        logits = self.lm_head(self.norm(hidden))
        if output_hidden_states:
            return logits, layer_states
        return logits


def _mask_device(doc_ids: Tensor | None, batch: int) -> torch.device | None:
    """Check the batch size of a dense mask and return the index device.

    Args:
        doc_ids: Document ids with shape [batch, L], or None.
        batch: Number of rows of the mask.

    Returns:
        The device of ``doc_ids``, or None (the default device) when ``doc_ids`` is None.

    Raises:
        ValueError: If batch is below 1, or doc_ids is not 2-D with ``batch`` rows.
    """
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}")
    if doc_ids is None:
        return None
    if doc_ids.dim() != 2 or doc_ids.shape[0] != batch:
        raise ValueError(f"doc_ids must have shape [{batch}, L], got {tuple(doc_ids.shape)}")
    return doc_ids.device


def bd3_dense_mask(L: int, B: int, doc_ids: Tensor | None, batch: int) -> Tensor:
    """Return the dense BD3 attention mask of the ``[noisy | clean]`` input.

    The mask is ``materialize_mask(make_bd3_mask_mod(L, B, doc_ids), batch, 2L, 2L)``, following
    spec section 5. It lives on the device of ``doc_ids``, or on the default device when
    ``doc_ids`` is None.

    Args:
        L: Effective length L. Must be a multiple of B.
        B: Diffusion block size. Must be at least 1.
        doc_ids: Document ids with shape [batch, L], or None to disable the document mask.
        batch: Number of rows.

    Returns:
        A BoolTensor [batch, 2L, 2L]. True means that the query may attend to the key.

    Raises:
        ValueError: If batch is below 1, B is below 1, L % B != 0, or doc_ids does not have shape
            [batch, L].
    """
    device = _mask_device(doc_ids, batch)
    return materialize_mask(make_bd3_mask_mod(L, B, doc_ids), batch, 2 * L, 2 * L, device)


def ar_dense_mask(L: int, doc_ids: Tensor | None, batch: int) -> Tensor:
    """Return the dense causal AR mask on L tokens (spec section 5).

    Args:
        L: Sequence length. Must be at least 1.
        doc_ids: Document ids with shape [batch, L], or None to disable the document mask.
        batch: Number of rows.

    Returns:
        A BoolTensor [batch, L, L]. True means that the query may attend to the key.

    Raises:
        ValueError: If batch is below 1, L is below 1, or doc_ids does not have shape [batch, L].
    """
    device = _mask_device(doc_ids, batch)
    return materialize_mask(make_ar_mask_mod(L, doc_ids), batch, L, L, device)


def _duplicated_positions(batch: int, seq_len: int, device: torch.device) -> Tensor:
    """Return ``cat([arange(L), arange(L)])`` as a [batch, 2L] tensor."""
    idx = torch.arange(seq_len, device=device)
    return torch.cat([idx, idx]).expand(batch, -1)


def bd3_logits(
    model: RefModel,
    x0: Tensor,
    xt: Tensor,
    block_size: int,
    token_shift: int,
    doc_ids: Tensor | None = None,
) -> Tensor:
    """Return the block-diffusion logits of each clean position.

    The model input is ``cat([xt, x0], 1)`` with positions ``cat([arange(L), arange(L)])``, and
    the mask is :func:`bd3_dense_mask`. Output ``i`` is the model output at the source index
    ``src[i]`` of :func:`oellm_autoexp.dllm.targets.prediction_sources`, so it predicts ``x0[i]``.

    Args:
        model: The reference model.
        x0: Clean token ids, LongTensor [b, L].
        xt: Noisy token ids, LongTensor [b, L].
        block_size: Diffusion block size B. L must be a multiple of B.
        token_shift: 0 or 1, as in ``prediction_sources``.
        doc_ids: Document ids with shape [b, L], or None to disable the document mask.

    Returns:
        Logits with shape [b, L, vocab_size].

    Raises:
        ValueError: If x0 is not 2-D, xt does not have the shape of x0, or the block size, the
            token shift, or doc_ids is not supported.
    """
    if x0.dim() != 2 or xt.shape != x0.shape:
        raise ValueError(
            f"x0 and xt must have the same shape [b, L], got {tuple(x0.shape)} and "
            f"{tuple(xt.shape)}"
        )
    batch, seq_len = x0.shape
    src, _ = prediction_sources(seq_len, block_size, token_shift, device=x0.device)
    ids = torch.cat([xt, x0], dim=1)
    positions = _duplicated_positions(batch, seq_len, x0.device)
    mask = bd3_dense_mask(seq_len, block_size, doc_ids, batch).to(x0.device)
    logits = model(ids, positions, mask)
    return logits.index_select(1, src)


def ar_logits(model: RefModel, x0: Tensor, doc_ids: Tensor | None = None) -> Tensor:
    """Return the autoregressive logits of every position of ``x0``.

    The model input is ``x0`` with positions ``arange(L)``, and the mask is :func:`ar_dense_mask`.
    Output ``i`` predicts ``x0[i + 1]``.

    Args:
        model: The reference model.
        x0: Clean token ids, LongTensor [b, L].
        doc_ids: Document ids with shape [b, L], or None to disable the document mask.

    Returns:
        Logits with shape [b, L, vocab_size].

    Raises:
        ValueError: If x0 is not 2-D, or doc_ids does not have shape [b, L].
    """
    if x0.dim() != 2:
        raise ValueError(f"x0 must have shape [b, L], got {tuple(x0.shape)}")
    batch, seq_len = x0.shape
    positions = torch.arange(seq_len, device=x0.device).expand(batch, -1)
    mask = ar_dense_mask(seq_len, doc_ids, batch).to(x0.device)
    return model(x0, positions, mask)

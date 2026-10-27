"""HuggingFace port of the OpenEuroLLM hybrid architecture-scaling models.

This is a *transcription* of the Megatron modules the checkpoints were trained
with, not a reimplementation:

* ``OellmAttention``     <- ``megatron.core.transformer.attention.SelfAttention``
* ``OellmGatedDeltaNet`` <- ``megatron.core.ssm.gated_delta_net.GatedDeltaNet``
* ``OellmMLSTM``         <- ``megatron.core.ssm.mlstm.MLSTM``
* ``OellmMamba2``        <- ``megatron.core.ssm.mamba_mixer.MambaMixer``
* ``OellmShortConv``     <- ``megatron.core.ssm.short_conv.ShortConv``
* ``OellmEngram``        <- ``megatron.core.ssm.engram.EngramLayer``
* ``OellmFLALayer``      <- ``megatron.core.ssm.fla_hybrid`` / ``complex_kda`` (the FLA
  layer itself: KimiDeltaAttention, ComplexKimiDeltaAttention, gated NoPE Attention)

The recurrences call the very same kernels Megatron calls
(``fla.ops.gated_delta_rule.chunk_gated_delta_rule``,
``mlstm_kernels.torch.get_mlstm_kernel``,
``mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined``), so the only
sources of divergence are the linear layers themselves. The fused input
projections are kept fused — exactly as in the checkpoint — so no tensor is
re-laid-out during conversion.

Layer structure (Megatron ``TransformerLayer`` with fused input layernorms;
the mixer specs set ``fuse_input_layernorm=True``, so what the checkpoint calls
``in_proj.layer_norm_weight`` / ``linear_qkv.layer_norm_weight`` is the layer's
pre-mixer norm and ``mlp.linear_fc1.layer_norm_weight`` is its pre-MLP norm)::

    h = h + mixer(input_layernorm(h))
    h = h + mlp(post_attention_layernorm(h))

Decoding state is cached per layer (``OellmHybridCache``): KV for the attention
layers, and the genuine recurrent state for the others — conv + delta state for
GDN, ``(C, n, m)`` for the mLSTM, conv + SSM state for Mamba2. Prefill uses the
chunkwise kernels with ``output_final_state``; each decode step uses the
matching single-step kernel. ``scripts/korbi/check_hf_cache_parity.py`` asserts
cached and uncached generation agree.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F
from torch import nn
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .configuration_oellm_hybrid import OellmHybridConfig

# --------------------------------------------------------------------------
# torch.cpu.device shim
# --------------------------------------------------------------------------
# `fla.utils.custom_device_ctx` calls `torch.cpu.device(...)`. That exists in
# stock torch wheels but NOT in NVIDIA's NGC builds (e.g. 2.11.0a0+...nv26.02),
# which is what the eval container inherits from the training image — so every
# GDN/mLSTM/Mamba2 forward died with
# `AttributeError: module 'torch.cpu' has no attribute 'device'`
# while the attention-only variants were unaffected. Installing a no-op context
# manager here keeps the fix with the exported checkpoint rather than depending
# on which torch build happens to be present.
if not hasattr(torch.cpu, "device"):  # pragma: no cover - environment shim

    class _CpuDeviceCtx(contextlib.nullcontext):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

    torch.cpu.device = _CpuDeviceCtx


# --------------------------------------------------------------------------
# lazily-imported third-party kernels (identical to the Megatron training path)
# --------------------------------------------------------------------------


def _chunk_gated_delta_rule():
    try:  # inside try: transformers' remote-code import check skips it
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    except ImportError as exc:
        raise ImportError(
            "GDN layers and the mLSTM conv need flash-linear-attention (fla)"
        ) from exc

    return chunk_gated_delta_rule


def _fused_recurrent_gated_delta_rule():
    try:  # inside try: transformers' remote-code import check skips it
        from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
    except ImportError as exc:
        raise ImportError(
            "GDN layers and the mLSTM conv need flash-linear-attention (fla)"
        ) from exc

    return fused_recurrent_gated_delta_rule


def _fla_causal_conv1d():
    try:  # inside try: transformers' remote-code import check skips it
        from fla.modules.convolution import causal_conv1d
    except ImportError as exc:
        raise ImportError(
            "GDN layers and the mLSTM conv need flash-linear-attention (fla)"
        ) from exc

    return causal_conv1d


def _fla_l2norm():
    try:  # inside try: transformers' remote-code import check skips it
        from fla.modules.l2norm import l2norm
    except ImportError as exc:
        raise ImportError(
            "GDN layers and the mLSTM conv need flash-linear-attention (fla)"
        ) from exc

    return l2norm


def _mlstm_kernel(name: str):
    try:  # inside try: transformers' remote-code import check skips it
        from mlstm_kernels.torch import get_mlstm_kernel
    except ImportError as exc:
        raise ImportError("mLSTM layers need mlstm_kernels") from exc

    return get_mlstm_kernel(name)


def _mlstm_step_kernel(name: str = "triton"):
    try:  # inside try: transformers' remote-code import check skips it
        from mlstm_kernels.torch import get_mlstm_step_kernel
    except ImportError as exc:
        raise ImportError("mLSTM layers need mlstm_kernels") from exc

    return get_mlstm_step_kernel(name)


def _mamba_chunk_scan_combined():
    try:  # inside try: transformers' remote-code import check skips it
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    except ImportError as exc:
        raise ImportError("Mamba2 layers need mamba_ssm") from exc

    return mamba_chunk_scan_combined


def _rms_norm_gated_cls():
    try:  # inside try: transformers' remote-code import check skips it
        from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
    except ImportError as exc:
        raise ImportError("Mamba2 layers need mamba_ssm") from exc

    return RMSNormGated


# --------------------------------------------------------------------------
# shared building blocks
# --------------------------------------------------------------------------


class OellmRMSNorm(nn.Module):
    """RMSNorm with fp32 reductions, matching Megatron/TE.

    ``zero_centered_gamma`` mirrors Megatron's ``layernorm_zero_centered_gamma``
    (TE's ``zero_centered_gamma``, CLI ``--apply-layernorm-1p``): the checkpoint
    stores ``gamma - 1`` and the norm applies ``(1 + w)``. Getting this wrong
    does not raise — every weight is simply off by one — so it is threaded from
    config at every construction site rather than defaulted per call.
    """

    def __init__(
        self, hidden_size: int, eps: float = 1e-5, zero_centered_gamma: bool = False
    ) -> None:
        super().__init__()
        # Init matches the stored convention: zero-centred checkpoints hold
        # gamma-1, so an untrained module must start at 0, not 1.
        self.weight = nn.Parameter(
            torch.zeros(hidden_size) if zero_centered_gamma else torch.ones(hidden_size)
        )
        self.variance_epsilon = eps
        self.zero_centered_gamma = zero_centered_gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        w = self.weight.float()
        if self.zero_centered_gamma:
            w = w + 1.0
        return (w * x).to(dtype)


class OellmMLP(nn.Module):
    """SwiGLU MLP.

    Megatron's fused ``linear_fc1`` is split into gate/up by the
    converter (``torch.chunk(x, 2, dim=-1)`` in ``bias_swiglu`` ⇒ first half is
    the gated branch).
    """

    def __init__(self, config: OellmHybridConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class OellmRotaryEmbedding(nn.Module):
    """Megatron ``RotaryEmbedding`` with ``rotary_interleaved=False``.

    ``inv_freq`` is deliberately **not** a registered buffer. transformers 5.x
    builds models on the meta device and only materialises tensors that exist in
    the checkpoint; a non-persistent buffer is absent from the checkpoint, so it
    came back as zeros — which silently turns RoPE into the identity
    (``cos≡1, sin≡0``) instead of raising. That cost the RoPE variants ~2.8 nats
    and was invisible to a strict-load check. Recomputing it from the two scalars
    at first use sidesteps the whole class of problem; it is 64 elements.
    """

    def __init__(self, dim: int, theta: float) -> None:
        super().__init__()
        self.dim = int(dim)
        self.theta = float(theta)
        self._inv_freq: torch.Tensor | None = None

    def _get_inv_freq(self, device: torch.device) -> torch.Tensor:
        if self._inv_freq is None or self._inv_freq.device != device:
            exponent = torch.arange(0, self.dim, 2, dtype=torch.float32, device=device)
            self._inv_freq = 1.0 / (self.theta ** (exponent / self.dim))
        return self._inv_freq

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: [b, s] -> freqs [b, s, dim/2] -> emb [b, s, dim]
        inv_freq = self._get_inv_freq(position_ids.device)
        freqs = position_ids.float().unsqueeze(-1) * inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def _apply_rope(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # t: [b, h, s, d]; cos/sin: [b, s, rot_dim] with rot_dim <= d.
    # Partial RoPE (Megatron rotary_percent < 1): rotate the leading rot_dim
    # channels and pass the tail through untouched, exactly as
    # rope_utils._apply_rotary_pos_emb_bshd does.
    cos = cos.unsqueeze(1).to(t.dtype)
    sin = sin.unsqueeze(1).to(t.dtype)
    rot_dim = cos.shape[-1]
    if rot_dim < t.shape[-1]:
        t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
        t = t * cos + _rotate_half(t) * sin
        return torch.cat((t, t_pass), dim=-1)
    return t * cos + _rotate_half(t) * sin


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


class OellmHybridCache:
    """Per-layer decoding state for a hybrid stack.

    One slot per layer, holding whatever that layer's mixer needs:

    * attention  — ``(k, v)`` at ``num_key_value_heads`` (pre-GQA-expansion)
    * gdn        — ``(conv_state, recurrent_state)``
    * mlstm      — ``(c, n, m)`` (plus ``conv_state`` when ``mlstm_conv1d``)
    * mamba2     — ``(conv_state, ssm_state)``
    * short_conv — ``conv_state``, the last ``kernel_size - 1`` conv inputs
    * fla        — the layer's own ``fla.models.utils.Cache`` (one per layer)
    * engram     — ``(id_state, conv_state)``: the last ``max_order - 1`` (shifted)
      token ids and the last ``(kernel_size - 1) * dilation`` gated values
    * none       — nothing (an MLP-only layer)

    Deliberately not a ``transformers.Cache`` subclass: that API is built around
    key/value tensors and carries assumptions (layer indexing, cropping,
    reordering) that do not apply to three of the four mixers here.
    """

    def __init__(self, num_layers: int) -> None:
        self.states: list = [None] * num_layers
        self.seen_tokens = 0

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.seen_tokens

    # HF's generate calls this when it reorders beams.
    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        def _sel(x):
            if isinstance(x, torch.Tensor):
                return x.index_select(0, beam_idx.to(x.device))
            if isinstance(x, (tuple, list)):
                return type(x)(_sel(i) for i in x)
            if isinstance(x, dict):
                return {k: _sel(v) for k, v in x.items()}
            if hasattr(x, "states"):  # an FLA Cache: per-layer dicts of tensors
                x.states = [_sel(s) for s in x.states]
            return x

        self.states = [_sel(s) for s in self.states]

    def __len__(self) -> int:
        return len(self.states)


# --------------------------------------------------------------------------
# mixers
# --------------------------------------------------------------------------


class OellmAttention(nn.Module):
    """Softmax attention with QK-layernorm, optional RoPE and optional sliding
    window.

    Megatron's ``window_size=(left, 0)`` lets a query at position ``i`` attend to
    ``j`` in ``[i - left, i]`` (``left + 1`` keys including itself); the mask
    below reproduces that inclusive bound exactly.
    """

    def __init__(self, config: OellmHybridConfig, layer_type: str) -> None:
        super().__init__()
        self.config = config
        self.layer_type = layer_type
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.n_rep = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5
        self.sliding_window = config.sliding_window if layer_type == "sliding_attention" else None

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        # Qwen3.5 output gate: a fourth projection the same shape as q, carved
        # out of Megatron's widened linear_qkv by the converter.
        self.gate_proj = (
            nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
            if getattr(config, "attention_output_gate", False)
            else None
        )

        zcg = getattr(config, "layernorm_zero_centered_gamma", False)
        if config.qk_layernorm:
            self.q_norm = OellmRMSNorm(
                self.head_dim, eps=config.rms_norm_eps, zero_centered_gamma=zcg
            )
            self.k_norm = OellmRMSNorm(
                self.head_dim, eps=config.rms_norm_eps, zero_centered_gamma=zcg
            )
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        attn_mask: torch.Tensor | None,
        state=None,
        past_len: int = 0,
    ) -> tuple[torch.Tensor, tuple | None]:
        b, s, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(b, s, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(b, s, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(b, s, self.num_kv_heads, self.head_dim)

        # Megatron normalises over the head dim before RoPE.
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)

        # Cache the pre-GQA-expansion k/v (num_key_value_heads), which is what
        # Megatron would store too, then expand for the SDPA call.
        if state is not None:
            past_k, past_v = state
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_state = (k, v)

        kv_len = k.shape[2]
        k = _repeat_kv(k, self.n_rep)
        v = _repeat_kv(v, self.n_rep)

        if attn_mask is None and self.sliding_window is None and s == kv_len:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scaling)
        else:
            mask = _build_attention_mask(
                attn_mask,
                s,
                kv_len,
                past_len,
                self.sliding_window,
                hidden_states.device,
                q.dtype,
            )
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scaling)

        out = out.transpose(1, 2).reshape(b, s, self.num_heads * self.head_dim)

        # Output gate, before o_proj. Megatron does the sigmoid in fp32
        # (attention.py::_apply_output_gate), which matters at bf16 — evaluate
        # it the same way or the parity check drifts.
        if self.gate_proj is not None:
            gate = self.gate_proj(hidden_states)
            out = (out.float() * torch.sigmoid(gate.float())).to(out.dtype)

        return self.o_proj(out), new_state


def _build_attention_mask(
    padding_mask: torch.Tensor | None,
    q_len: int,
    kv_len: int,
    past_len: int,
    sliding_window: int | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Additive float mask ``[b, 1, q_len, kv_len]`` combining causality, the
    sliding window and (optionally) a ``[b, kv_len]`` padding mask.

    Queries occupy absolute positions ``past_len … past_len + q_len - 1`` while
    keys start at 0, so the causal comparison is done in absolute coordinates —
    that keeps prefill and single-token decode on one code path.
    """
    q_pos = torch.arange(past_len, past_len + q_len, device=device).unsqueeze(1)
    k_pos = torch.arange(kv_len, device=device).unsqueeze(0)
    allowed = k_pos <= q_pos
    if sliding_window is not None:
        allowed &= (q_pos - k_pos) <= sliding_window
    allowed = allowed[None, None]  # [1, 1, q, k]
    if padding_mask is not None:
        allowed = allowed & padding_mask[:, None, None, :].bool()
    mask = torch.zeros(allowed.shape, dtype=dtype, device=device)
    return mask.masked_fill(~allowed, torch.finfo(dtype).min)


class OellmGatedDeltaNet(nn.Module):
    """GatedDeltaNet; ``in_proj`` stays fused as ``[q | k | v | z | beta |
    alpha]``."""

    def __init__(self, config: OellmHybridConfig) -> None:
        super().__init__()
        self.config = config
        self.key_head_dim = config.linear_key_head_dim
        self.value_head_dim = config.linear_value_head_dim
        self.num_key_heads = config.linear_num_key_heads
        self.num_value_heads = config.linear_num_value_heads
        self.qk_dim = self.key_head_dim * self.num_key_heads
        self.v_dim = self.value_head_dim * self.num_value_heads
        self.conv_dim = self.qk_dim * 2 + self.v_dim
        self.conv_kernel_dim = config.linear_conv_kernel_dim

        in_proj_dim = self.qk_dim * 2 + self.v_dim * 2 + self.num_value_heads * 2
        self.in_proj = nn.Linear(config.hidden_size, in_proj_dim, bias=False)
        # Depthwise causal conv over [q | k | v]; Megatron builds it without bias.
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            kernel_size=self.conv_kernel_dim,
            groups=self.conv_dim,
            padding=self.conv_kernel_dim - 1,
            bias=False,
        )
        self.dt_bias = nn.Parameter(torch.zeros(self.num_value_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_value_heads))
        self.out_norm = OellmRMSNorm(
            self.value_head_dim,
            eps=config.rms_norm_eps,
            zero_centered_gamma=getattr(config, "layernorm_zero_centered_gamma", False),
        )
        self.out_proj = nn.Linear(self.v_dim, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, state=None) -> tuple[torch.Tensor, tuple | None]:
        b, s, _ = hidden_states.shape
        conv_state, recurrent_state = state if state is not None else (None, None)
        qkvzba = self.in_proj(hidden_states)
        qkv, gate, beta, alpha = torch.split(
            qkvzba,
            [self.conv_dim, self.v_dim, self.num_value_heads, self.num_value_heads],
            dim=-1,
        )
        gate = gate.reshape(b, s, -1, self.value_head_dim)

        qkv, conv_state = _fla_causal_conv1d()(
            x=qkv,
            weight=self.conv1d.weight.squeeze(1),
            bias=None,
            activation="silu",
            initial_state=conv_state,
            output_final_state=True,
            cu_seqlens=None,
        )

        query_key, value = torch.split(qkv, [2 * self.qk_dim, self.v_dim], dim=-1)
        query_key = query_key.reshape(b, s, -1, self.key_head_dim)
        # Megatron constructs GatedDeltaNet with use_qk_l2norm=True.
        query_key = _fla_l2norm()(query_key.contiguous())
        n_kh = self.qk_dim // self.key_head_dim
        query, key = torch.split(query_key, [n_kh, n_kh], dim=2)
        value = value.reshape(b, s, -1, self.value_head_dim)

        if self.num_value_heads // self.num_key_heads > 1:
            rep = self.num_value_heads // self.num_key_heads
            query = query.repeat_interleave(rep, dim=2)
            key = key.repeat_interleave(rep, dim=2)

        g = -self.A_log.float().exp() * F.softplus(alpha.float() + self.dt_bias.float())
        beta = beta.sigmoid()

        # Single-token steps go through the recurrent kernel; the chunkwise one
        # is for T > 1. Both accept/emit the same state layout.
        gdn = _fused_recurrent_gated_delta_rule() if s == 1 else _chunk_gated_delta_rule()
        core_out, recurrent_state = gdn(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            g=g,
            beta=beta.contiguous(),
            initial_state=recurrent_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
            cu_seqlens=None,
        )

        dtype = core_out.dtype
        y = self.out_norm(core_out.reshape(-1, core_out.shape[-1]))
        y = (y * F.silu(gate.reshape(-1, gate.shape[-1]).float())).to(dtype)
        return self.out_proj(y.reshape(b, s, -1)), (conv_state, recurrent_state)


class OellmMLSTM(nn.Module):
    """MLSTM; ``in_proj`` stays fused as ``[q | k | v | ogate | igate |
    fgate]``."""

    def __init__(self, config: OellmHybridConfig) -> None:
        super().__init__()
        self.config = config
        self.key_head_dim = config.linear_key_head_dim
        self.value_head_dim = config.linear_value_head_dim
        self.num_heads = config.linear_num_value_heads
        num_key_heads = config.linear_num_key_heads
        if self.num_heads % num_key_heads:
            raise ValueError(
                "mLSTM requires linear_num_value_heads to be a multiple of linear_num_key_heads"
            )
        # Grouped-query mode (Megatron MLSTM): q/k projected at num_key_heads,
        # repeat_interleave'd to the value heads before the cell.
        self.qk_repeat = self.num_heads // num_key_heads
        self.qk_dim = self.key_head_dim * num_key_heads
        self.v_dim = self.value_head_dim * self.num_heads
        self.gate_soft_cap = config.mlstm_gate_soft_cap
        self.chunk_size = config.mlstm_chunk_size
        self.backend = config.mlstm_backend
        self.norm_eps = config.rms_norm_eps

        in_proj_dim = self.qk_dim * 2 + self.v_dim * 2 + self.num_heads * 2
        self.in_proj = nn.Linear(config.hidden_size, in_proj_dim, bias=False)
        self.igate_bias = nn.Parameter(torch.zeros(self.num_heads))
        self.fgate_bias = nn.Parameter(torch.zeros(self.num_heads))

        self.use_conv1d = bool(config.mlstm_conv1d)
        if self.use_conv1d:
            conv_dim = self.qk_dim * 2 + self.v_dim
            self.conv_dim = conv_dim
            self.conv1d = nn.Conv1d(
                conv_dim,
                conv_dim,
                kernel_size=config.linear_conv_kernel_dim,
                groups=conv_dim,
                padding=config.linear_conv_kernel_dim - 1,
                bias=True,
            )
        else:
            self.conv1d = None

        self.out_norm_weight = nn.Parameter(torch.ones(self.v_dim))
        self.out_proj = nn.Linear(self.v_dim, config.hidden_size, bias=False)
        self._mlstm_fn = None
        self._mlstm_step_fn = None

    def forward(self, hidden_states: torch.Tensor, state=None) -> tuple[torch.Tensor, tuple | None]:
        b, s, _ = hidden_states.shape
        conv_state, c, n, m = state if state is not None else (None, None, None, None)
        fused = self.in_proj(hidden_states)
        q, k, v, o_preact, i_preact, f_preact = torch.split(
            fused,
            [
                self.qk_dim,
                self.qk_dim,
                self.v_dim,
                self.v_dim,
                self.num_heads,
                self.num_heads,
            ],
            dim=-1,
        )

        if self.conv1d is not None:
            qkv = torch.cat([q, k, v], dim=-1)
            qkv, conv_state = _fla_causal_conv1d()(
                x=qkv,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation="silu",
                initial_state=conv_state,
                output_final_state=True,
                cu_seqlens=None,
            )
            q, k, v = torch.split(qkv, [self.qk_dim, self.qk_dim, self.v_dim], dim=-1)

        q = q.reshape(b, s, -1, self.key_head_dim).transpose(1, 2).contiguous()
        k = k.reshape(b, s, -1, self.key_head_dim).transpose(1, 2).contiguous()
        v = v.reshape(b, s, -1, self.value_head_dim).transpose(1, 2).contiguous()
        if self.qk_repeat > 1:
            q = q.repeat_interleave(self.qk_repeat, dim=1)
            k = k.repeat_interleave(self.qk_repeat, dim=1)

        i_preact = i_preact.float() + self.igate_bias.float()
        f_preact = f_preact.float() + self.fgate_bias.float()
        if self.gate_soft_cap is not None:
            cap = self.gate_soft_cap
            i_preact = cap * torch.tanh(i_preact / cap)
            f_preact = cap * torch.tanh(f_preact / cap)
        i_preact = i_preact.transpose(1, 2).contiguous()
        f_preact = f_preact.transpose(1, 2).contiguous()

        if s == 1:
            # Decode: one recurrent step. Kernel wants [b, nh, dh] / [b, nh].
            if self._mlstm_step_fn is None:
                self._mlstm_step_fn = _mlstm_step_kernel()
            # The step kernel wants the gates as [b, nh, 1] (not [b, nh]) —
            # keep the singleton time axis instead of indexing it away.
            h, (c, n, m) = self._mlstm_step_fn(
                q=q[:, :, 0],
                k=k[:, :, 0],
                v=v[:, :, 0],
                i=i_preact[:, :, 0:1],
                f=f_preact[:, :, 0:1],
                c=c,
                n=n,
                m=m,
            )
            h = h.unsqueeze(2)
        else:
            if self._mlstm_fn is None:
                self._mlstm_fn = _mlstm_kernel(self.backend)

            # The triton xl_chunk kernel asserts the sequence length is divisible
            # by 16 (training always fed 4096). lm-eval feeds arbitrary lengths,
            # so pad on the right up to a chunk boundary and drop the tail again:
            # the mLSTM is causal, so positions appended *after* the real tokens
            # cannot change their outputs.
            #
            # The padded steps must also leave the *returned state* untouched,
            # or cached generation would start from a corrupted state. Forcing
            # forget=1 (f_preact >> 0) and input=0 (i_preact << 0) on them makes
            # each pad step an exact no-op: C_t = C_{t-1}, n_t = n_{t-1},
            # m_t = m_{t-1}. Finite values (not +-inf) keep the log-space kernel
            # away from NaN; exp(-30) ~ 1e-13 is far below bf16 resolution.
            pad = (-s) % self.chunk_size
            if pad:
                q = F.pad(q, (0, 0, 0, pad))
                k = F.pad(k, (0, 0, 0, pad))
                v = F.pad(v, (0, 0, 0, pad))
                i_preact = F.pad(i_preact, (0, pad), value=-30.0)
                f_preact = F.pad(f_preact, (0, pad), value=30.0)

            h, (c, n, m) = self._mlstm_fn(
                q=q,
                k=k,
                v=v,
                i=i_preact,
                f=f_preact,
                c_initial=c,
                n_initial=n,
                m_initial=m,
                return_last_states=True,
                chunk_size=self.chunk_size,
            )
            if pad:
                h = h[:, :, :s]

        # Per-head RMSNorm (fp32) then the elementwise sigmoid output gate.
        dtype = h.dtype
        h = h.transpose(1, 2).float()
        h = h * torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + self.norm_eps)
        h = h.reshape(b, s, -1) * self.out_norm_weight.float()
        h = h * torch.sigmoid(o_preact.float())
        return self.out_proj(h.to(dtype)), (conv_state, c, n, m)


class OellmMamba2(nn.Module):
    """Mamba2; ``in_proj`` stays fused as ``[z | x | B | C | dt]``."""

    def __init__(self, config: OellmHybridConfig) -> None:
        super().__init__()
        self.config = config
        self.d_inner = config.mamba_d_inner
        self.nheads = config.mamba_nheads
        self.headdim = config.mamba_head_dim
        self.ngroups = config.mamba_num_groups
        self.d_state = config.mamba_state_dim
        self.chunk_size = config.mamba_chunk_size
        self.conv_dim = self.d_inner + 2 * self.ngroups * self.d_state

        in_proj_dim = self.d_inner * 2 + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(config.hidden_size, in_proj_dim, bias=False)
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            kernel_size=config.mamba_conv_kernel,
            groups=self.conv_dim,
            padding=config.mamba_conv_kernel - 1,
            bias=True,
        )
        self.dt_bias = nn.Parameter(torch.zeros(self.nheads))
        self.A_log = nn.Parameter(torch.zeros(self.nheads))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.norm = _rms_norm_gated_cls()(
            self.d_inner,
            eps=config.rms_norm_eps,
            group_size=self.d_inner // self.ngroups,
            norm_before_gate=False,
        )
        self.out_proj = nn.Linear(self.d_inner, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, state=None) -> tuple[torch.Tensor, tuple | None]:
        b, s, _ = hidden_states.shape
        conv_state, ssm_state = state if state is not None else (None, None)
        zxbcdt = self.in_proj(hidden_states)
        z, xBC, dt = torch.split(zxbcdt, [self.d_inner, self.conv_dim, self.nheads], dim=-1)

        xBC, conv_state = _fla_causal_conv1d()(
            x=xBC,
            weight=self.conv1d.weight.squeeze(1),
            bias=self.conv1d.bias,
            activation="silu",
            initial_state=conv_state,
            output_final_state=True,
            cu_seqlens=None,
        )

        x, B, C = torch.split(
            xBC,
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1,
        )
        x = x.reshape(b, s, self.nheads, self.headdim).contiguous()
        B = B.reshape(b, s, self.ngroups, self.d_state).contiguous()
        C = C.reshape(b, s, self.ngroups, self.d_state).contiguous()
        A = -torch.exp(self.A_log.float())

        if s == 1:
            # Decode: one SSM step. selective_state_update wants A/dt/D expanded
            # to (nheads, headdim, dstate) / (b, nheads, headdim), matching
            # Megatron's own `_step` path.
            try:  # inside try: transformers' remote-code import check skips it
                from mamba_ssm.ops.triton.selective_state_update import selective_state_update
            except ImportError as exc:
                raise ImportError("Mamba2 layers need mamba_ssm") from exc

            A_exp = A.view(-1, 1, 1).expand(self.nheads, self.headdim, self.d_state)
            dt_exp = dt[:, 0].unsqueeze(-1).expand(b, self.nheads, self.headdim)
            dt_bias_exp = self.dt_bias.float().unsqueeze(-1).expand(self.nheads, self.headdim)
            D_exp = self.D.float().unsqueeze(-1).expand(self.nheads, self.headdim)
            y = selective_state_update(
                ssm_state,
                x[:, 0],
                dt_exp,
                A_exp,
                B[:, 0],
                C[:, 0],
                D=D_exp,
                z=None,  # rmsnorm=True -> gating happens in self.norm below
                dt_bias=dt_bias_exp,
                dt_softplus=True,
            )
            y = y.unsqueeze(1)
        else:
            y, ssm_state = _mamba_chunk_scan_combined()(
                x,
                dt.contiguous(),
                A,
                B,
                C,
                self.chunk_size,
                D=self.D,
                z=None,  # rmsnorm=True -> gating happens in self.norm below
                dt_bias=self.dt_bias.float(),
                dt_softplus=True,
                return_final_states=True,
                initial_states=ssm_state,
            )
        y = y.reshape(b, s, self.d_inner)
        y = self.norm(y, z)
        return self.out_proj(y), (conv_state, ssm_state)


class OellmShortConv(nn.Module):
    """LFM2 gated short convolution (Megatron ``ShortConv``, the ``C`` symbol).

    ``in_proj`` stays fused as ``[B | C | x]``; ``y = C * causal_conv(B * x)``.
    The conv is depthwise with a width of ``short_conv_kernel_size`` and runs in
    plain torch: the decoding state is the last ``kernel_size - 1`` inputs.
    """

    def __init__(self, config: OellmHybridConfig) -> None:
        super().__init__()
        h = config.hidden_size
        self.kernel_size = config.short_conv_kernel_size
        self.in_proj = nn.Linear(h, 3 * h, bias=False)
        self.conv1d = nn.Conv1d(
            h, h, kernel_size=self.kernel_size, groups=h, bias=config.short_conv_bias
        )
        self.out_proj = nn.Linear(h, h, bias=False)

    def forward(
        self, hidden_states: torch.Tensor, state=None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        b, c, x = self.in_proj(hidden_states).chunk(3, dim=-1)
        bx = (b * x).transpose(1, 2)  # [batch, hidden, seq]
        if state is None:
            state = bx.new_zeros(bx.shape[0], bx.shape[1], self.kernel_size - 1)
        padded = torch.cat([state.to(bx.dtype), bx], dim=-1)
        conv = F.conv1d(padded, self.conv1d.weight, self.conv1d.bias, groups=bx.shape[1])
        new_state = padded[..., padded.shape[-1] - (self.kernel_size - 1) :]
        y = c * conv.transpose(1, 2)
        return self.out_proj(y), new_state


class OellmEngram(nn.Module):
    """Engram hashed n-gram memory (Megatron ``EngramLayer``, the ``N``
    symbol).

    Per token, every suffix n-gram (orders ``engram_ngram_orders``) is hashed by
    ``engram_num_heads`` multiplicative-XOR heads into tables of prime size; the
    looked-up rows ``e`` are gated into the residual stream::

        k = W_K e,  v = W_V e,  alpha = sigmoid(RMSNorm(h) . RMSNorm(k) / sqrt(d))
        v~ = alpha * v,  y = SiLU(causal_conv(v~)) + v~

    with a depthwise conv of dilation ``max(orders)``. The layer normalises ``h``
    itself, so the decoder layer applies no pre-norm. The table addresses depend on
    the token ids alone: ``table`` may be moved to the CPU (``offload_table``) and
    the lookup gathers only the rows it needs.
    """

    def __init__(self, config: OellmHybridConfig) -> None:
        super().__init__()
        h = config.hidden_size
        self.orders = list(config.engram_ngram_orders)
        self.num_heads = config.engram_num_heads
        self.sizes = list(config.engram_table_sizes)
        self.kernel_size = config.engram_conv_kernel_size
        self.dilation = max(self.orders)
        # Plain ints, not non-persistent buffers: from_pretrained builds the model on
        # the meta device and leaves such buffers uninitialised (garbage row indices).
        self.row_offsets = [0]
        for n in self.sizes[:-1]:
            self.row_offsets.append(self.row_offsets[-1] + n)
        self.hash_mult = [[int(a) for a in row] for row in config.engram_hash_multipliers]
        self.register_buffer(
            "token_map",
            torch.zeros(config.vocab_size, dtype=torch.long) if config.engram_token_map else None,
        )
        rows = sum(self.sizes)
        # engram_table_dtype: "fp8" (inference) stores e4m3 bytes as uint8 plus one scale
        # per row; "fp32" keeps the table fp32 under a bf16 model (continued training).
        self.table_dtype = getattr(config, "engram_table_dtype", None) or "bf16"
        if self.table_dtype == "fp8":
            self.register_buffer(
                "table", torch.zeros(rows, config.engram_embed_dim, dtype=torch.uint8)
            )
            scale_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[
                getattr(config, "engram_scale_dtype", "fp32")
            ]
            self.register_buffer("table_scale", torch.ones(rows, dtype=scale_dtype))
        else:
            # Explicit dtype: from_pretrained builds the model under its load dtype.
            table_dtype = torch.float32 if self.table_dtype == "fp32" else None
            self.table = nn.Parameter(torch.empty(rows, config.engram_embed_dim, dtype=table_dtype))
            self.table_scale = None
        mem_dim = config.engram_embed_dim * len(self.sizes)
        self.key_proj = nn.Linear(mem_dim, h, bias=False)
        self.value_proj = nn.Linear(mem_dim, h, bias=False)
        self.hidden_norm = OellmRMSNorm(h, eps=config.rms_norm_eps)
        self.key_norm = OellmRMSNorm(h, eps=config.rms_norm_eps)
        self.conv1d = nn.Conv1d(
            h, h, kernel_size=self.kernel_size, groups=h, bias=False, dilation=self.dilation
        )

    def _apply(self, fn, recurse=True):
        """Keep an fp32 table and fp8 scales out of the model's dtype casts
        (``.to(bf16)``, ``from_pretrained(dtype=...)``); device moves still
        apply."""
        keep = []
        if self.table_dtype == "fp32" and "table" in self._parameters:
            keep.append(("_parameters", "table"))
        if self.table_scale is not None and "table_scale" in self._buffers:
            keep.append(("_buffers", "table_scale"))
        probe = fn(torch.zeros(1, dtype=torch.float32, device=self.table.device))
        if not keep or probe.dtype == torch.float32:
            return super()._apply(fn, recurse)
        held = [(store, name, getattr(self, store).pop(name)) for store, name in keep]
        try:
            super()._apply(fn, recurse)
        finally:
            for store, name, tensor in held:
                if tensor.device != probe.device:
                    tensor.data = tensor.data.to(probe.device)
                getattr(self, store)[name] = tensor
        return self

    def offload_table(self, device="cpu") -> None:
        """Keep the table on ``device`` (e.g. CPU) while the rest stays where
        it is."""
        self.table.data = self.table.data.to(device)
        if self.table_scale is not None:
            self.table_scale.data = self.table_scale.data.to(device)

    def _lookup(self, rows: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Table rows ``[..., L] -> [..., L, dim]``; fp8 rows are dequantized
        here."""
        idx = rows.to(self.table.device)
        if self.table_dtype == "fp8":
            q = self.table[idx].view(torch.float8_e4m3fn).to(torch.float32)
            return (q * self.table_scale[idx].float().unsqueeze(-1)).to(dtype)
        return F.embedding(idx, self.table).to(dtype)

    def _rows(self, shifted: torch.Tensor, s: int) -> torch.Tensor:
        """Global table rows of the last ``s`` positions; ``shifted`` = ids + 1 with
        the ``max_order - 1`` previous ids (0 = before the start) in front."""
        p = max(self.orders) - 1
        history = [shifted[:, p - i : p - i + s] for i in range(max(self.orders))]
        out, lookup = [], 0
        for n in self.orders:
            for _ in range(self.num_heads):
                hsh = torch.zeros_like(history[0])
                for i in range(n):
                    hsh = hsh ^ (history[i] * self.hash_mult[lookup][i])
                out.append(torch.remainder(hsh, self.sizes[lookup]) + self.row_offsets[lookup])
                lookup += 1
        return torch.stack(out, dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.LongTensor | None,
        token_mask: torch.Tensor | None = None,
        state=None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if input_ids is None:
            raise ValueError("engram layers need input_ids (inputs_embeds alone is not enough)")
        b, s, h = hidden_states.shape
        ids = input_ids if self.token_map is None else self.token_map[input_ids]
        shifted = ids.long() + 1
        if token_mask is not None:  # padding hashes like "before the start"
            shifted = shifted * token_mask.long()
        p = max(self.orders) - 1
        id_state, conv_state = state if state is not None else (None, None)
        if id_state is None:
            id_state = shifted.new_zeros(b, p)
        shifted = torch.cat([id_state, shifted], dim=1)
        new_id_state = shifted[:, shifted.shape[1] - p :]

        rows = self._rows(shifted, s)
        e = self._lookup(rows, hidden_states.dtype)
        e = e.to(device=hidden_states.device).reshape(b, s, -1)
        k, v = self.key_proj(e), self.value_proj(e)
        gate = (self.hidden_norm(hidden_states).float() * self.key_norm(k).float()).sum(
            -1, keepdim=True
        )
        v = torch.sigmoid(gate / h**0.5).to(v.dtype) * v
        if token_mask is not None:
            v = v * token_mask[..., None].to(v.dtype)

        pad = (self.kernel_size - 1) * self.dilation
        vt = v.transpose(1, 2)  # [b, h, s]
        if conv_state is None:
            conv_state = vt.new_zeros(b, h, pad)
        padded = torch.cat([conv_state.to(vt.dtype), vt], dim=-1)
        conv = F.conv1d(padded, self.conv1d.weight, groups=h, dilation=self.dilation)
        new_conv_state = padded[..., padded.shape[-1] - pad :]
        y = F.silu(conv.transpose(1, 2)) + v
        return y, (new_id_state, new_conv_state)


class OellmFLALayer(nn.Module):
    """A complete FLA layer, built from ``{"class", "kwargs"}`` as Megatron
    built it.

    kda / signed_kda / complex_kda (and complex_kda's gated NoPE attention) wrap the
    FLA layer whole, so the checkpoint holds exactly its parameters, under
    ``mixer.layer``. Each layer keeps a private FLA ``Cache`` with ``layer_idx=0``
    as its decoding state, so the HF layer numbering never has to match FLA's.
    """

    def __init__(self, spec: dict) -> None:
        super().__init__()
        import importlib

        module, name = spec["class"].rsplit(".", 1)
        cls = getattr(importlib.import_module(module), name)
        self.layer = cls(**spec["kwargs"], layer_idx=0)

    def forward(
        self, hidden_states: torch.Tensor, state=None, attention_mask: torch.Tensor | None = None
    ):
        from fla.models.utils import Cache

        cache = state if state is not None else Cache()
        # Some of the fork's Triton kernels launch on the CURRENT device, not their inputs'. With
        # the model spread over GPUs (device_map) that is another GPU: a "cannot be accessed from
        # Triton" error, or -- once peer access is on -- unsynchronized reads of this GPU's memory
        # (NaN / wrong outputs depending on what ran before). Megatron sets one device per rank.
        with torch.cuda.device(hidden_states.device) if hidden_states.is_cuda else contextlib.nullcontext():
            out, _, cache = self.layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
            )
        return out, cache


# --------------------------------------------------------------------------
# decoder
# --------------------------------------------------------------------------


def _build_mixer(config: OellmHybridConfig, layer_type: str) -> nn.Module:
    if layer_type in ("full_attention", "sliding_attention"):
        return OellmAttention(config, layer_type)
    if layer_type == "gdn":
        return OellmGatedDeltaNet(config)
    if layer_type == "mlstm":
        return OellmMLSTM(config)
    if layer_type == "mamba2":
        return OellmMamba2(config)
    if layer_type == "short_conv":
        return OellmShortConv(config)
    if layer_type == "engram":
        return OellmEngram(config)
    if layer_type == "fla":
        return OellmFLALayer(config.fla_layer)
    if layer_type == "fla_attention":
        return OellmFLALayer(config.fla_attention_layer)
    raise ValueError(f"Unknown layer type {layer_type!r}")


class OellmHybridDecoderLayer(nn.Module):
    def __init__(self, config: OellmHybridConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_type = config.mixer_types[layer_idx]
        self.is_attention = self.layer_type in ("full_attention", "sliding_attention")
        # A HybridModel layer can be a mixer or an MLP alone (see mlp_layers).
        self.has_mixer = self.layer_type != "none"
        self.is_engram = self.layer_type == "engram"
        self.is_fla = self.layer_type in ("fla", "fla_attention")
        self.has_mlp = getattr(config, "mlp_layers", None) is None or config.mlp_layers[layer_idx]

        def norm():
            return OellmRMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
                zero_centered_gamma=getattr(config, "layernorm_zero_centered_gamma", False),
            )

        if self.has_mixer:
            if not self.is_engram:  # Engram normalises internally
                self.input_layernorm = norm()
            self.mixer = _build_mixer(config, self.layer_type)
        if self.has_mlp:
            self.post_attention_layernorm = norm()
            self.mlp = OellmMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        attn_mask: torch.Tensor | None,
        state=None,
        past_len: int = 0,
        input_ids: torch.LongTensor | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple | None]:
        new_state = None
        if self.is_engram:
            h, new_state = self.mixer(hidden_states, input_ids, token_mask, state)
            hidden_states = hidden_states + h
        elif self.has_mixer:
            residual = hidden_states
            h = self.input_layernorm(hidden_states)
            if self.is_attention:
                h, new_state = self.mixer(h, position_embeddings, attn_mask, state, past_len)
            elif self.is_fla:
                h, new_state = self.mixer(h, state, attn_mask)
            else:
                h, new_state = self.mixer(h, state)
            hidden_states = residual + h

        if self.has_mlp:
            residual = hidden_states
            h = self.post_attention_layernorm(hidden_states)
            hidden_states = residual + self.mlp(h)
        return hidden_states, new_state


class OellmHybridPreTrainedModel(PreTrainedModel):
    config_class = OellmHybridConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["OellmHybridDecoderLayer"]
    _supports_sdpa = True
    _supports_flash_attn = False

    def _init_weights(self, module: nn.Module) -> None:
        # transformers >= 5 runs _init_weights after loading and skips tensors it loaded
        # only through its guarded init functions; a raw ``.data.normal_()`` would
        # overwrite loaded weights (it did, for the Engram tables).
        try:
            from transformers import initialization as hf_init

            normal_, zeros_ = hf_init.normal_, hf_init.zeros_
        except ImportError:  # older transformers: only missing weights get here
            normal_, zeros_ = torch.nn.init.normal_, torch.nn.init.zeros_
        std = self.config.initializer_range
        with torch.no_grad():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, OellmEngram) and module.table.is_floating_point():
                normal_(module.table, mean=0.0, std=std)


class OellmHybridModel(OellmHybridPreTrainedModel):
    def __init__(self, config: OellmHybridConfig) -> None:
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            OellmHybridDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        )
        self.norm = OellmRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            zero_centered_gamma=getattr(config, "layernorm_zero_centered_gamma", False),
        )
        self.rotary_emb = (
            OellmRotaryEmbedding(
                int(config.head_dim * getattr(config, "partial_rotary_factor", 1.0)),
                config.rope_theta,
            )
            if config.position_embedding_type == "rope"
            else None
        )
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        past_key_values: OellmHybridCache | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Pass exactly one of input_ids or inputs_embeds")
        if use_cache is None:
            use_cache = self.config.use_cache
        if past_key_values is not None and not isinstance(past_key_values, OellmHybridCache):
            past_key_values = None
        if use_cache and past_key_values is None:
            past_key_values = OellmHybridCache(len(self.layers))
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        b, s, _ = inputs_embeds.shape

        past_len = past_key_values.seen_tokens if past_key_values is not None else 0
        if position_ids is None:
            position_ids = (
                torch.arange(past_len, past_len + s, device=inputs_embeds.device)
                .unsqueeze(0)
                .expand(b, s)
            )

        position_embeddings = self.rotary_emb(position_ids) if self.rotary_emb is not None else None

        # A padding mask of all ones carries no information; dropping it keeps
        # the fast is_causal SDPA path.
        if attention_mask is not None and bool(attention_mask.all()):
            attention_mask = None

        hidden_states = inputs_embeds
        token_mask = None
        if attention_mask is not None:
            # The recurrent mixers have no mask interface: zeroing padded
            # positions keeps them from contributing to the state (their
            # value/key projections are bias-free, so a zero input adds zero).
            #
            # `attention_mask` spans past + current tokens ([b, past_len + s]),
            # while hidden_states covers only the current ones ([b, s, h]). Slice
            # to the current window before multiplying: broadcasting the full
            # mask against a 1-token decode step silently EXPANDS it to
            # [b, past_len + s, h] instead of masking it, which then produced
            # `kv_len = past_len + (past_len + s)` and blew up in the attention
            # mask as "size of tensor a (2n-1) must match tensor b (n)".
            # Only reachable when a padding mask survives, i.e. left-padded
            # generation batches -- which is why every generate_until task
            # failed while log-likelihood tasks were fine.
            cur_mask = attention_mask[:, -s:] if attention_mask.shape[1] >= s else attention_mask
            hidden_states = hidden_states * cur_mask[..., None].to(hidden_states.dtype)
            token_mask = cur_mask

        all_hidden_states: list[torch.Tensor] = []
        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            state = past_key_values.states[idx] if past_key_values is not None else None
            hidden_states, new_state = layer(
                hidden_states,
                position_embeddings,
                attention_mask,
                state,
                past_len,
                input_ids=input_ids,
                token_mask=token_mask,
            )
            if past_key_values is not None:
                past_key_values.states[idx] = new_state

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        if past_key_values is not None:
            past_key_values.seen_tokens = past_len + s

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
        )


class OellmHybridForCausalLM(OellmHybridPreTrainedModel, GenerationMixin):
    # transformers 5.x expects a mapping {tied_param: source_param}, not a list.
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: OellmHybridConfig) -> None:
        super().__init__(config)
        self.model = OellmHybridModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_decoder(self, decoder: nn.Module) -> None:
        self.model = decoder

    def get_decoder(self) -> nn.Module:
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        past_key_values: OellmHybridCache | None = None,
        use_cache: bool | None = None,
        labels: torch.LongTensor | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )
        logits = self.lm_head(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, **kwargs
    ):
        if use_cache is None:
            use_cache = self.config.use_cache
        # `generate` may hand us a DynamicCache it created itself; only our cache
        # can carry conv/SSM/mLSTM state, so replace anything else with a fresh
        # one (an empty foreign cache means nothing has been cached yet anyway).
        if past_key_values is not None and not isinstance(past_key_values, OellmHybridCache):
            past_key_values = OellmHybridCache(self.config.num_hidden_layers)
        # With a warm cache only the tokens it has not consumed are new.
        if past_key_values is not None and past_key_values.seen_tokens > 0:
            input_ids = input_ids[:, past_key_values.seen_tokens :]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        if past_key_values is not None:
            past_key_values.reorder_cache(beam_idx)
        return past_key_values


__all__ = [
    "OellmHybridConfig",
    "OellmHybridForCausalLM",
    "OellmHybridModel",
    "OellmHybridPreTrainedModel",
    "OellmHybridCache",
]

"""HuggingFace port of the OpenEuroLLM hybrid architecture-scaling models.

This is a *transcription* of the Megatron modules the checkpoints were trained
with, not a reimplementation:

* ``OellmAttention``     <- ``megatron.core.transformer.attention.SelfAttention``
* ``OellmGatedDeltaNet`` <- ``megatron.core.ssm.gated_delta_net.GatedDeltaNet``
* ``OellmMLSTM``         <- ``megatron.core.ssm.mlstm.MLSTM``
* ``OellmMamba2``        <- ``megatron.core.ssm.mamba_mixer.MambaMixer``

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

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .configuration_oellm_hybrid import OellmHybridConfig

# --------------------------------------------------------------------------
# lazily-imported third-party kernels (identical to the Megatron training path)
# --------------------------------------------------------------------------


def _chunk_gated_delta_rule():
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    return chunk_gated_delta_rule


def _fused_recurrent_gated_delta_rule():
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule

    return fused_recurrent_gated_delta_rule


def _fla_causal_conv1d():
    from fla.modules.convolution import causal_conv1d

    return causal_conv1d


def _fla_l2norm():
    from fla.modules.l2norm import l2norm

    return l2norm


def _mlstm_kernel(name: str):
    from mlstm_kernels.torch import get_mlstm_kernel

    return get_mlstm_kernel(name)


def _mlstm_step_kernel(name: str = "triton"):
    from mlstm_kernels.torch import get_mlstm_step_kernel

    return get_mlstm_step_kernel(name)


def _mamba_chunk_scan_combined():
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

    return mamba_chunk_scan_combined


def _rms_norm_gated_cls():
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated

    return RMSNormGated


# --------------------------------------------------------------------------
# shared building blocks
# --------------------------------------------------------------------------


class OellmRMSNorm(nn.Module):
    """RMSNorm with fp32 reductions, matching Megatron/TE."""

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return (self.weight.float() * x).to(dtype)


class OellmMLP(nn.Module):
    """SwiGLU MLP. Megatron's fused ``linear_fc1`` is split into gate/up by the
    converter (``torch.chunk(x, 2, dim=-1)`` in ``bias_swiglu`` ⇒ first half is
    the gated branch)."""

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
    # t: [b, h, s, d]; cos/sin: [b, s, d]
    cos = cos.unsqueeze(1).to(t.dtype)
    sin = sin.unsqueeze(1).to(t.dtype)
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
            if isinstance(x, tuple):
                return tuple(_sel(i) for i in x)
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
        self.sliding_window = (
            config.sliding_window if layer_type == "sliding_attention" else None
        )

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        if config.qk_layernorm:
            self.q_norm = OellmRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = OellmRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]],
        attn_mask: Optional[torch.Tensor],
        state=None,
        past_len: int = 0,
    ) -> tuple[torch.Tensor, Optional[tuple]]:
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
                attn_mask, s, kv_len, past_len, self.sliding_window,
                hidden_states.device, q.dtype,
            )
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scaling)

        out = out.transpose(1, 2).reshape(b, s, self.num_heads * self.head_dim)
        return self.o_proj(out), new_state


def _build_attention_mask(
    padding_mask: Optional[torch.Tensor],
    q_len: int,
    kv_len: int,
    past_len: int,
    sliding_window: Optional[int],
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
    """GatedDeltaNet; ``in_proj`` stays fused as ``[q | k | v | z | beta | alpha]``."""

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
        self.out_norm = OellmRMSNorm(self.value_head_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.v_dim, config.hidden_size, bias=False)

    def forward(
        self, hidden_states: torch.Tensor, state=None
    ) -> tuple[torch.Tensor, Optional[tuple]]:
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
    """mLSTM; ``in_proj`` stays fused as ``[q | k | v | ogate | igate | fgate]``."""

    def __init__(self, config: OellmHybridConfig) -> None:
        super().__init__()
        self.config = config
        self.key_head_dim = config.linear_key_head_dim
        self.value_head_dim = config.linear_value_head_dim
        self.num_heads = config.linear_num_value_heads
        if config.linear_num_key_heads != config.linear_num_value_heads:
            raise ValueError("mLSTM requires linear_num_key_heads == linear_num_value_heads")
        self.qk_dim = self.key_head_dim * self.num_heads
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

    def forward(
        self, hidden_states: torch.Tensor, state=None
    ) -> tuple[torch.Tensor, Optional[tuple]]:
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
                q=q[:, :, 0], k=k[:, :, 0], v=v[:, :, 0],
                i=i_preact[:, :, 0:1], f=f_preact[:, :, 0:1], c=c, n=n, m=m,
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
                q=q, k=k, v=v, i=i_preact, f=f_preact,
                c_initial=c, n_initial=n, m_initial=m,
                return_last_states=True, chunk_size=self.chunk_size,
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

    def forward(
        self, hidden_states: torch.Tensor, state=None
    ) -> tuple[torch.Tensor, Optional[tuple]]:
        b, s, _ = hidden_states.shape
        conv_state, ssm_state = state if state is not None else (None, None)
        zxbcdt = self.in_proj(hidden_states)
        z, xBC, dt = torch.split(
            zxbcdt, [self.d_inner, self.conv_dim, self.nheads], dim=-1
        )

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
            from mamba_ssm.ops.triton.selective_state_update import selective_state_update

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
    raise ValueError(f"Unknown layer type {layer_type!r}")


class OellmHybridDecoderLayer(nn.Module):
    def __init__(self, config: OellmHybridConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_type = config.mixer_types[layer_idx]
        self.is_attention = self.layer_type in ("full_attention", "sliding_attention")
        self.input_layernorm = OellmRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mixer = _build_mixer(config, self.layer_type)
        self.post_attention_layernorm = OellmRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = OellmMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]],
        attn_mask: Optional[torch.Tensor],
        state=None,
        past_len: int = 0,
    ) -> tuple[torch.Tensor, Optional[tuple]]:
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        if self.is_attention:
            h, new_state = self.mixer(h, position_embeddings, attn_mask, state, past_len)
        else:
            h, new_state = self.mixer(h, state)
        hidden_states = residual + h

        residual = hidden_states
        h = self.post_attention_layernorm(hidden_states)
        return residual + self.mlp(h), new_state


class OellmHybridPreTrainedModel(PreTrainedModel):
    config_class = OellmHybridConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["OellmHybridDecoderLayer"]
    _supports_sdpa = True
    _supports_flash_attn = False

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)


class OellmHybridModel(OellmHybridPreTrainedModel):
    def __init__(self, config: OellmHybridConfig) -> None:
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            OellmHybridDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        )
        self.norm = OellmRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = (
            OellmRotaryEmbedding(config.head_dim, config.rope_theta)
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
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[OellmHybridCache] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
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

        position_embeddings = (
            self.rotary_emb(position_ids) if self.rotary_emb is not None else None
        )

        # A padding mask of all ones carries no information; dropping it keeps
        # the fast is_causal SDPA path.
        if attention_mask is not None and bool(attention_mask.all()):
            attention_mask = None

        hidden_states = inputs_embeds
        if attention_mask is not None:
            # The recurrent mixers have no mask interface: zeroing padded
            # positions keeps them from contributing to the state (their
            # value/key projections are bias-free, so a zero input adds zero).
            hidden_states = hidden_states * attention_mask[..., None].to(hidden_states.dtype)

        all_hidden_states: list[torch.Tensor] = []
        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            state = past_key_values.states[idx] if past_key_values is not None else None
            hidden_states, new_state = layer(
                hidden_states, position_embeddings, attention_mask, state, past_len
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
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[OellmHybridCache] = None,
        use_cache: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
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

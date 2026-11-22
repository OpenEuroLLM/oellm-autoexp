"""HF port of the engram_scaling models: Megatron ``HybridModel`` (``hybrid_stack_spec``), one HF layer
per pattern symbol, each a transcription of the Megatron module it was trained as:

* ``gdn``       ('L') <- ``TransformerLayer(GatedDeltaNet)``       (megatron/core/ssm/gated_delta_net/)
* ``attention`` ('*') <- ``TransformerLayer(SelfAttention)``       (megatron/core/transformer/attention.py)
* ``moe``       ('E') <- ``MoETransformerLayer(MoELayer)``         (megatron/core/transformer/moe/)
* ``engram``    ('N') <- ``EngramLayer``                           (megatron/core/ssm/engram.py)

Every layer is a residual block on the hidden states ``h`` (Megatron's bias-dropout-add, bf16)::

    gdn / attention / moe:  h = h + mixer(input_layernorm(h))
    engram:                 h = h + Engram(h, token ids)        (no pre-norm: it normalises inside)

then ``logits = final_norm(h) @ embed_tokens.weight.T`` (tied embedding).

Megatron's fused parameter layouts are kept as they are in the checkpoint -- attention ``linear_qkv``
(per KV group ``[q heads | gate heads | k | v]``), GDN ``in_proj`` (``[q | k | v | z | beta | alpha]``),
the stacked expert weights ``[E, out, in]`` -- and split at run time exactly as Megatron splits them,
so the conversion re-lays out no tensor.

Numerics follow the training forward: RMSNorms in fp32 (backbone norms zero-centred, applied as
``1 + w``; Engram's own norms plain), router in fp32, the attention output gate and RoPE in fp32. On
the GPU the gated delta rule runs the FLA kernels Megatron trains with
(``fla.ops.gated_delta_rule``); on the CPU a reference recurrence in fp32 (for tests).

Decoding state is cached per layer (:class:`OellmEngramCache`).
"""

from __future__ import annotations

import contextlib
import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .configuration_oellm_engram import OellmEngramConfig

# `fla.utils.custom_device_ctx` calls `torch.cpu.device(...)`, which NVIDIA's NGC torch builds lack.
if not hasattr(torch.cpu, "device"):  # pragma: no cover - environment shim

    class _CpuDeviceCtx(contextlib.nullcontext):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

    torch.cpu.device = _CpuDeviceCtx


def _fla_gated_delta_rule(single_step: bool):
    try:  # inside try: transformers' remote-code import check skips it
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    except ImportError as exc:
        raise ImportError("Gated DeltaNet on the GPU needs flash-linear-attention (fla)") from exc
    return fused_recurrent_gated_delta_rule if single_step else chunk_gated_delta_rule


# --------------------------------------------------------------------------- building blocks


class OellmRMSNorm(nn.Module):
    """RMSNorm in fp32. ``zero_centered_gamma`` (Megatron ``layernorm_zero_centered_gamma``, TE's
    ``zero_centered_gamma``): the weight is stored as ``gamma - 1`` and applied as ``1 + w``. The
    Engram norms (``engram._RMSNorm``) are plain: ``zero_centered_gamma=False``."""

    def __init__(self, hidden_size: int, eps: float, zero_centered_gamma: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size) if zero_centered_gamma else torch.ones(hidden_size))
        self.eps = eps
        self.zero_centered_gamma = zero_centered_gamma

    def gamma(self) -> torch.Tensor:
        w = self.weight.float()
        return w + 1.0 if self.zero_centered_gamma else w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.gamma()).to(x.dtype)


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """``fla.modules.l2norm`` (Megatron GDN, ``use_qk_l2norm``): ``x / sqrt(sum(x^2) + eps)`` in fp32."""
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).sum(-1, keepdim=True) + eps)).to(x.dtype)


def _causal_conv_silu(x: torch.Tensor, weight: torch.Tensor, state: torch.Tensor | None):
    """``fla.modules.convolution.causal_conv1d(activation="silu")``: depthwise causal conv over the
    sequence, then SiLU, in fp32. ``x`` [b, s, c], ``weight`` [c, 1, K]; ``state`` [b, c, K - 1] holds
    the previous inputs (zeros at the sequence start). Returns the output and the new state."""
    b, s, c = x.shape
    width = weight.shape[-1]
    xt = x.transpose(1, 2).float()
    if state is None:
        state = xt.new_zeros(b, c, width - 1)
    padded = torch.cat([state.float(), xt], dim=-1)
    y = F.silu(F.conv1d(padded, weight.float(), groups=c))
    return y.transpose(1, 2).to(x.dtype), padded[..., padded.shape[-1] - (width - 1):].to(x.dtype)


def _torch_gated_delta_rule(q, k, v, g, beta, initial_state):
    """Reference recurrence of ``fla.ops.gated_delta_rule`` (fp32, one token at a time). ``q``, ``k``
    [b, s, h, dk] (l2-normalised), ``v`` [b, s, h, dv], ``g`` (log decay), ``beta`` [b, s, h]; state
    [b, h, dk, dv] (FLA's layout). Per token: ``S = exp(g) S;  S += k (beta (v - k^T S))^T;
    o = (q / sqrt(dk))^T S``."""
    dtype, dk = v.dtype, q.shape[-1]
    q, k, v, g, beta = (t.float() for t in (q, k, v, g, beta))
    q = q * dk**-0.5
    b, s, h, dv = v.shape
    state = q.new_zeros(b, h, dk, dv) if initial_state is None else initial_state.float()
    out = []
    for t in range(s):
        state = state * g[:, t].exp()[..., None, None]
        u = (v[:, t] - torch.einsum("bhk,bhkv->bhv", k[:, t], state)) * beta[:, t, :, None]
        state = state + torch.einsum("bhk,bhv->bhkv", k[:, t], u)
        out.append(torch.einsum("bhk,bhkv->bhv", q[:, t], state))
    return torch.stack(out, dim=1).to(dtype), state


class OellmRotaryEmbedding(nn.Module):
    """Megatron ``RotaryEmbedding``: the angle of every rotated channel, ``[b, s, dim]``.
    ``interleaved_freqs``: the frequency layout, ``[f0, f0, f1, f1, ...]`` (``RotaryEmbedding(
    rotary_interleaved=True)``) or ``[f0 .. fn, f0 .. fn]``. HybridModel builds its RotaryEmbedding
    without ``rotary_interleaved`` (hybrid_model.py), so its layout is the second even when the
    attention rotates interleaved pairs. ``inv_freq`` is recomputed, not a buffer: from_pretrained builds
    on the meta device and would leave a non-persistent buffer uninitialised (RoPE silently the identity)."""

    def __init__(self, dim: int, theta: float, interleaved_freqs: bool) -> None:
        super().__init__()
        self.dim, self.theta, self.interleaved_freqs = int(dim), float(theta), bool(interleaved_freqs)

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        exponent = torch.arange(0, self.dim, 2, dtype=torch.float32, device=position_ids.device)
        inv_freq = 1.0 / (self.theta ** (exponent / self.dim))
        freqs = position_ids.float().unsqueeze(-1) * inv_freq  # [b, s, dim / 2]
        emb = freqs.repeat_interleave(2, dim=-1) if self.interleaved_freqs else torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def _apply_rope(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool) -> torch.Tensor:
    """``rope_utils._apply_rotary_pos_emb_bshd`` / TE's fused RoPE on ``t`` [b, h, s, d]: per channel d
    of the first ``rot``, ``t[d] cos[d] + rotate(t)[d] sin[d]``, with ``rotate`` pairing ``(2i, 2i+1)``
    (``interleaved``) or ``(i, i + rot/2)``; the remaining channels pass. Each channel uses its own
    angle, so the frequency layout (:class:`OellmRotaryEmbedding`) decides which angle a pair gets."""
    rot = cos.shape[-1]
    x, x_pass = t[..., :rot].float(), t[..., rot:]
    if interleaved:
        x1, x2 = x[..., 0::2], x[..., 1::2]
        rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
    else:
        x1, x2 = x.chunk(2, dim=-1)
        rotated = torch.cat((-x2, x1), dim=-1)
    x = x * cos.unsqueeze(1) + rotated * sin.unsqueeze(1)
    return torch.cat((x.to(t.dtype), x_pass), dim=-1)


def _attention_mask(padding_mask, q_len: int, kv_len: int, past_len: int, device, dtype) -> torch.Tensor:
    """Additive mask [b, 1, q, kv]: causal (queries at absolute positions past_len ..), and padded keys
    (``padding_mask`` [b, kv]) masked. A finite minimum, not -inf: a padded query row stays finite."""
    q_pos = torch.arange(past_len, past_len + q_len, device=device).unsqueeze(1)
    allowed = (torch.arange(kv_len, device=device).unsqueeze(0) <= q_pos)[None, None]
    if padding_mask is not None:
        allowed = allowed & padding_mask[:, None, None, :].bool()
    return torch.zeros(allowed.shape, dtype=dtype, device=device).masked_fill(~allowed, torch.finfo(dtype).min)


class OellmEngramCache:
    """Per-layer decoding state, one slot per layer:

    * attention -- ``(k, v)`` at ``num_key_value_heads`` (before the GQA expansion)
    * gdn       -- ``(conv_state, recurrent_state)``: the last ``K - 1`` conv inputs, the delta-rule state
    * engram    -- ``(id_state, conv_state)``: the last ``max_order - 1`` (shifted) token ids and the last
      ``(K - 1) * dilation`` conv inputs
    * moe       -- nothing (per token)

    Not a ``transformers.Cache``: that API is built around key / value tensors."""

    def __init__(self, num_layers: int) -> None:
        self.states: list = [None] * num_layers
        self.seen_tokens = 0

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.seen_tokens

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        def sel(x):
            if isinstance(x, torch.Tensor):
                return x.index_select(0, beam_idx.to(x.device))
            if isinstance(x, (tuple, list)):
                return type(x)(sel(i) for i in x)
            return x

        self.states = [sel(s) for s in self.states]

    def __len__(self) -> int:
        return len(self.states)


# --------------------------------------------------------------------------- mixers


class OellmGatedDeltaNet(nn.Module):
    """``GatedDeltaNet`` (gated_delta_net/gdn.py + common.py). ``in_proj`` stays fused as
    ``[q | k | v | z | beta | alpha]``, ``conv1d`` over ``[q | k | v]``. q, k: ``linear_num_key_heads``
    heads, l2-normalised, repeated (``repeat_interleave``) to the ``linear_num_value_heads`` value heads;
    ``g = -exp(A_log) softplus(alpha + dt_bias)`` (fp32), ``beta = sigmoid(beta)``; output
    ``out_proj(RMSNorm(o) * silu(z))`` per value head."""

    def __init__(self, config: OellmEngramConfig) -> None:
        super().__init__()
        h = config.hidden_size
        self.nk, self.nv = config.linear_num_key_heads, config.linear_num_value_heads
        self.dk, self.dv = config.linear_key_head_dim, config.linear_value_head_dim
        self.qk_dim, self.v_dim = self.nk * self.dk, self.nv * self.dv
        self.conv_dim = 2 * self.qk_dim + self.v_dim
        self.in_proj = nn.Linear(h, 2 * self.qk_dim + 2 * self.v_dim + 2 * self.nv, bias=False)
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, config.linear_conv_kernel_dim,
                                groups=self.conv_dim, bias=False)
        self.A_log = nn.Parameter(torch.zeros(self.nv))
        self.dt_bias = nn.Parameter(torch.ones(self.nv))
        self.out_norm = OellmRMSNorm(self.dv, config.rms_norm_eps, config.layernorm_zero_centered_gamma)
        self.out_proj = nn.Linear(self.v_dim, h, bias=False)

    def forward(self, x: torch.Tensor, state=None) -> tuple[torch.Tensor, tuple]:
        b, s, _ = x.shape
        conv_state, recurrent_state = state if state is not None else (None, None)
        qkv, z, beta, alpha = torch.split(self.in_proj(x), [self.conv_dim, self.v_dim, self.nv, self.nv], dim=-1)
        qkv, conv_state = _causal_conv_silu(qkv, self.conv1d.weight, conv_state)
        q, k, v = torch.split(qkv, [self.qk_dim, self.qk_dim, self.v_dim], dim=-1)
        q = _l2norm(q.reshape(b, s, self.nk, self.dk))
        k = _l2norm(k.reshape(b, s, self.nk, self.dk))
        v = v.reshape(b, s, self.nv, self.dv)
        if self.nv > self.nk:
            q = q.repeat_interleave(self.nv // self.nk, dim=2)
            k = k.repeat_interleave(self.nv // self.nk, dim=2)
        g = -self.A_log.float().exp() * F.softplus(alpha.float() + self.dt_bias.float())
        beta = beta.sigmoid()
        if x.is_cuda:
            o, recurrent_state = _fla_gated_delta_rule(single_step=s == 1)(
                q.contiguous(), k.contiguous(), v.contiguous(), g=g, beta=beta.contiguous(),
                initial_state=recurrent_state, output_final_state=True, use_qk_l2norm_in_kernel=False,
            )
        else:
            o, recurrent_state = _torch_gated_delta_rule(q, k, v, g, beta, recurrent_state)
        # gdn_fused_out_norm: RMSNorm(o) * w * silu(z) in fp32, rounded once
        of = o.float()
        y = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + self.out_norm.eps) * self.out_norm.gamma()
        y = (y * F.silu(z.reshape(b, s, self.nv, self.dv).float())).to(x.dtype)
        return self.out_proj(y.reshape(b, s, self.v_dim)), (conv_state, recurrent_state)


class OellmGatedAttention(nn.Module):
    """``SelfAttention`` with ``attention_output_gate``, ``qk_layernorm`` and (partial, interleaved)
    RoPE. ``qkv_proj`` is Megatron's ``linear_qkv``: per KV group ``[q heads | gate heads | k | v]``
    (``get_query_key_value_tensors``). q / k are RMS-normalised per head before RoPE; the output is
    ``o_proj(attn * sigmoid(gate))``, the gate in fp32 (``_sigmoid_output_gate``)."""

    def __init__(self, config: OellmEngramConfig) -> None:
        super().__init__()
        self.heads, self.groups, self.d = config.num_attention_heads, config.num_key_value_heads, config.head_dim
        self.per_group = self.heads // self.groups
        self.gated = config.attention_output_gate
        self.interleaved = config.rope_interleaved
        q_heads = self.per_group * (2 if self.gated else 1)
        self.qkv_proj = nn.Linear(config.hidden_size, self.groups * (q_heads + 2) * self.d, bias=False)
        self.o_proj = nn.Linear(self.heads * self.d, config.hidden_size, bias=False)
        zc = config.layernorm_zero_centered_gamma
        self.q_norm = OellmRMSNorm(self.d, config.rms_norm_eps, zc) if config.qk_layernorm else None
        self.k_norm = OellmRMSNorm(self.d, config.rms_norm_eps, zc) if config.qk_layernorm else None

    def forward(self, x, position_embeddings, padding_mask, state=None, past_len: int = 0):
        b, s, _ = x.shape
        mixed = self.qkv_proj(x).view(b, s, self.groups, -1)
        hq = self.per_group * self.d
        if self.gated:
            q, gate, k, v = torch.split(mixed, [hq, hq, self.d, self.d], dim=-1)
        else:
            (q, k, v), gate = torch.split(mixed, [hq, self.d, self.d], dim=-1), None
        q = q.reshape(b, s, self.heads, self.d)  # head g * per_group + i of group g
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # [b, heads, s, d]
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q = _apply_rope(q, cos, sin, self.interleaved)
            k = _apply_rope(k, cos, sin, self.interleaved)
        if state is not None:
            k = torch.cat([state[0], k], dim=2)
            v = torch.cat([state[1], v], dim=2)
        new_state = (k, v)
        kv_len = k.shape[2]
        k = k.repeat_interleave(self.per_group, dim=1)  # KV group g serves query heads of group g
        v = v.repeat_interleave(self.per_group, dim=1)
        if padding_mask is None and s == kv_len:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.d**-0.5)
        else:
            mask = _attention_mask(padding_mask, s, kv_len, past_len, x.device, q.dtype)
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.d**-0.5)
        o = o.transpose(1, 2).reshape(b, s, self.heads * self.d)
        if gate is not None:
            o = (o * torch.sigmoid(gate.reshape(b, s, -1).float())).to(o.dtype)
        return self.o_proj(o), new_state


class OellmRouter(nn.Module):
    """``TopKRouter`` with the sigmoid score function and the expert bias
    (``moe_utils.topk_routing_with_score_function``): ``scores = sigmoid(x W^T)`` in fp32
    (``moe_router_dtype: fp32``), the top-k experts by ``scores + expert_bias``, weighted by their
    unbiased scores normalised to sum 1 (k > 1), times the scaling factor. Kept in fp32 under any model
    dtype: a bf16 expert bias would move near-tie top-k choices."""

    def __init__(self, config: OellmEngramConfig) -> None:
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.norm_topk = config.router_norm_topk_prob
        self.scaling = config.router_scaling_factor
        self.weight = nn.Parameter(torch.zeros(config.num_experts, config.hidden_size, dtype=torch.float32))
        self.register_buffer("expert_bias", torch.zeros(config.num_experts, dtype=torch.float32))

    def _apply(self, fn, recurse=True):
        probe = fn(torch.zeros(1, dtype=torch.float32, device=self.weight.device))
        if probe.dtype == torch.float32:
            return super()._apply(fn, recurse)
        return super()._apply(lambda t: fn(t).float() if t.is_floating_point() else fn(t), recurse)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``x`` [tokens, hidden] -> (weights [tokens, k] fp32, expert ids [tokens, k])."""
        scores = torch.sigmoid(F.linear(x.float(), self.weight.float()))
        _, idx = torch.topk(scores + self.expert_bias.float(), self.top_k, dim=-1)
        w = scores.gather(-1, idx)
        if self.norm_topk and self.top_k > 1:
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
        return w * self.scaling, idx


class OellmMoE(nn.Module):
    """``MoELayer``: routed SwiGLU experts plus the sigmoid-gated shared expert, summed.

    ``experts_gate_up`` [E, 2f, h] / ``experts_down`` [E, h, f] are Megatron's grouped expert weights as
    stored (``experts.experts.linear_fc1/2.weight``, gate rows before up rows). Per expert (TEGroupedMLP):
    ``fc2((silu(gate) * up) * weight)``, the routing weight applied after the activation in fp32 and
    rounded to the model dtype. Shared expert (SharedExpertMLP): ``fc2(silu(gate) * up) * sigmoid(x g^T)``."""

    def __init__(self, config: OellmEngramConfig) -> None:
        super().__init__()
        h, e, f = config.hidden_size, config.num_experts, config.moe_intermediate_size
        self.num_experts = e
        self.router = OellmRouter(config)
        self.experts_gate_up = nn.Parameter(torch.empty(e, 2 * f, h))
        self.experts_down = nn.Parameter(torch.empty(e, h, f))
        fs = config.shared_expert_intermediate_size
        self.shared_gate_up = nn.Linear(h, 2 * fs, bias=False) if fs else None
        self.shared_down = nn.Linear(fs, h, bias=False) if fs else None
        self.shared_gate = nn.Linear(h, 1, bias=False) if fs and config.shared_expert_gate else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        weights, idx = self.router(x)
        out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        flat = idx.flatten()
        order = flat.argsort()
        counts = torch.bincount(flat, minlength=self.num_experts).tolist()
        token = order // idx.shape[1]  # token of every (token, k) slot, grouped by expert
        slot_w = weights.flatten()[order]
        start = 0
        for e, n in enumerate(counts):
            if n == 0:
                continue
            t, w = token[start:start + n], slot_w[start:start + n]
            start += n
            gate, up = F.linear(x[t], self.experts_gate_up[e]).chunk(2, dim=-1)
            act = ((F.silu(gate) * up).float() * w[:, None]).to(x.dtype)
            out.index_add_(0, t, F.linear(act, self.experts_down[e]).float())
        out = out.to(x.dtype)
        if self.shared_gate_up is not None:
            gate, up = self.shared_gate_up(x).chunk(2, dim=-1)
            shared = self.shared_down(F.silu(gate) * up)
            if self.shared_gate is not None:
                shared = shared * torch.sigmoid(self.shared_gate(x))
            out = out + shared
        return out.reshape(shape)


class OellmEngramMemory(nn.Module):
    """``EngramLayer`` (engram.py). Per token t, every suffix n-gram of the (optionally token-mapped)
    ids, n in ``engram_ngram_orders``, is hashed by ``engram_num_heads`` heads::

        row = (XOR_i (x_{t-i} + 1) * a_i) mod M  (+ the offset of that lookup's table)

    with ``x + 1 = 0`` before the sequence start. The rows of all lookups are concatenated into ``e``;
    ``k = W_K e``, ``v = W_V e``, ``alpha = sigmoid(RMSNorm(h) . RMSNorm(k) / sqrt(d))`` (optionally
    signed-sqrt of the dot product), ``v~ = alpha v``, and the layer returns
    ``Y = silu(conv(RMSNorm(v~))) + v~`` (conv: depthwise, causal, dilation = max order; the RMSNorm only
    with ``engram_conv_prenorm``). The norms are Megatron's plain ``_RMSNorm`` (not zero-centred)."""

    def __init__(self, config: OellmEngramConfig, layer_idx: int) -> None:
        super().__init__()
        h = config.hidden_size
        self.orders = list(config.engram_ngram_orders)
        self.num_heads = config.engram_num_heads
        self.sizes = list(config.engram_table_sizes)
        # plain ints, not non-persistent buffers: from_pretrained leaves those uninitialised (meta device)
        self.hash_mult = [[int(a) for a in row] for row in config.engram_hash_multipliers[str(layer_idx)]]
        self.row_offsets = [sum(self.sizes[:i]) for i in range(len(self.sizes))]
        self.kernel_size = config.engram_conv_kernel_size
        self.dilation = config.engram_conv_dilation
        self.signed_sqrt = config.engram_gate_signed_sqrt
        self.register_buffer(
            "token_map", torch.zeros(config.vocab_size, dtype=torch.long) if config.engram_token_map else None
        )
        self.table_fp32 = config.engram_table_dtype == "fp32"
        self.table = nn.Parameter(torch.empty(sum(self.sizes), config.engram_embed_dim,
                                              dtype=torch.float32 if self.table_fp32 else None))
        mem_dim = config.engram_embed_dim * len(self.sizes)
        eps = config.rms_norm_eps
        self.key_proj = nn.Linear(mem_dim, h, bias=False)
        self.value_proj = nn.Linear(mem_dim, h, bias=False)
        self.hidden_norm = OellmRMSNorm(h, eps, zero_centered_gamma=False)
        self.key_norm = OellmRMSNorm(h, eps, zero_centered_gamma=False)
        self.conv_norm = OellmRMSNorm(h, eps, zero_centered_gamma=False) if config.engram_conv_prenorm else None
        self.conv1d = nn.Conv1d(h, h, self.kernel_size, groups=h, bias=False, dilation=self.dilation)

    def _apply(self, fn, recurse=True):
        """An fp32 table (``engram_table_dtype: fp32``) follows device moves, not dtype casts."""
        probe = fn(torch.zeros(1, dtype=torch.float32, device=self.table.device))
        if not self.table_fp32 or probe.dtype == torch.float32:
            return super()._apply(fn, recurse)
        table = self._parameters.pop("table")
        try:
            super()._apply(fn, recurse)
        finally:
            table.data = table.data.to(probe.device)
            self._parameters["table"] = table
        return self

    def rows(self, shifted: torch.Tensor, s: int) -> torch.Tensor:
        """Global table rows [b, s, lookups] of the last ``s`` positions of ``shifted`` (ids + 1, with
        the ``max order - 1`` previous ones in front, 0 before the start): ``engram.engram_hash`` plus
        ``row_offsets``."""
        p = max(self.orders) - 1
        history = [shifted[:, p - i: p - i + s] for i in range(max(self.orders))]  # history[i] = x_{t-i} + 1
        out, lookup = [], 0
        for n in self.orders:
            for _ in range(self.num_heads):
                hsh = torch.zeros_like(history[0])
                for i in range(n):
                    hsh = hsh ^ (history[i] * self.hash_mult[lookup][i])
                out.append(torch.remainder(hsh, self.sizes[lookup]) + self.row_offsets[lookup])
                lookup += 1
        return torch.stack(out, dim=-1)

    def forward(self, h: torch.Tensor, input_ids: torch.LongTensor, token_mask=None, state=None):
        if input_ids is None:
            raise ValueError("Engram layers need input_ids (inputs_embeds alone is not enough)")
        b, s, d = h.shape
        ids = input_ids if self.token_map is None else self.token_map[input_ids]
        shifted = ids.long() + 1
        if token_mask is not None:  # a padded token hashes like "before the start"
            shifted = shifted * token_mask.long()
        id_state, conv_state = state if state is not None else (None, None)
        p = max(self.orders) - 1
        if id_state is None:
            id_state = shifted.new_zeros(b, p)
        shifted = torch.cat([id_state, shifted], dim=1)
        new_id_state = shifted[:, shifted.shape[1] - p:]

        e = F.embedding(self.rows(shifted, s).to(self.table.device), self.table).to(h.dtype).reshape(b, s, -1)
        k, v = self.key_proj(e), self.value_proj(e)
        gate = (self.hidden_norm(h).float() * self.key_norm(k).float()).sum(-1, keepdim=True) / math.sqrt(d)
        if self.signed_sqrt:
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        v = torch.sigmoid(gate).to(h.dtype) * v
        if token_mask is not None:
            v = v * token_mask[..., None].to(v.dtype)
        conv_in = v if self.conv_norm is None else self.conv_norm(v)

        pad = (self.kernel_size - 1) * self.dilation
        xt = conv_in.transpose(1, 2)  # [b, d, s]
        if conv_state is None:
            conv_state = xt.new_zeros(b, d, pad)
        padded = torch.cat([conv_state.to(xt.dtype), xt], dim=-1)
        conv = F.conv1d(padded, self.conv1d.weight, groups=d, dilation=self.dilation)
        return F.silu(conv.transpose(1, 2)) + v, (new_id_state, padded[..., padded.shape[-1] - pad:])


# --------------------------------------------------------------------------- model


class OellmEngramDecoderLayer(nn.Module):
    """One Megatron layer (one pattern symbol) as a residual block."""

    def __init__(self, config: OellmEngramConfig, layer_idx: int) -> None:
        super().__init__()
        self.block_type = config.block_types[layer_idx]
        if self.block_type != "engram":  # Megatron's fused / separate pre-norm of the sublayer
            self.input_layernorm = OellmRMSNorm(config.hidden_size, config.rms_norm_eps,
                                                config.layernorm_zero_centered_gamma)
        if self.block_type == "gdn":
            self.linear_attn = OellmGatedDeltaNet(config)
        elif self.block_type == "attention":
            self.self_attn = OellmGatedAttention(config)
        elif self.block_type == "moe":
            self.mlp = OellmMoE(config)
        else:
            self.engram = OellmEngramMemory(config, layer_idx)

    def forward(self, h, position_embeddings, padding_mask, token_mask, input_ids, state, past_len):
        if self.block_type == "engram":
            y, state = self.engram(h, input_ids, token_mask, state)
            return h + y, state
        x = self.input_layernorm(h)
        if self.block_type == "gdn":
            if token_mask is not None:  # a padded token adds nothing to the conv / delta-rule state
                x = x * token_mask[..., None].to(x.dtype)
            y, state = self.linear_attn(x, state)
        elif self.block_type == "attention":
            y, state = self.self_attn(x, position_embeddings, padding_mask, state, past_len)
        else:
            y = self.mlp(x)
        return h + y, state


class OellmEngramPreTrainedModel(PreTrainedModel):
    config_class = OellmEngramConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["OellmEngramDecoderLayer"]
    _keep_in_fp32_modules = ["router"]
    _supports_sdpa = True

    def _init_weights(self, module: nn.Module) -> None:
        # transformers >= 5 runs _init_weights after loading and skips only tensors its own init
        # functions see as loaded; a raw `.normal_()` would overwrite loaded weights
        try:
            from transformers import initialization as hf_init
        except ImportError:
            hf_init = nn.init
        normal_, zeros_, ones_ = hf_init.normal_, hf_init.zeros_, hf_init.ones_
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Embedding)):
            normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, OellmRMSNorm):
            (zeros_ if module.zero_centered_gamma else ones_)(module.weight)
        elif isinstance(module, OellmRouter):
            normal_(module.weight, mean=0.0, std=std)
            zeros_(module.expert_bias)
        elif isinstance(module, OellmMoE):
            normal_(module.experts_gate_up, mean=0.0, std=std)
            normal_(module.experts_down, mean=0.0, std=std)
        elif isinstance(module, OellmGatedDeltaNet):
            zeros_(module.A_log)
            ones_(module.dt_bias)
        elif isinstance(module, OellmEngramMemory):
            normal_(module.table, mean=0.0, std=std)


class OellmEngramModel(OellmEngramPreTrainedModel):
    def __init__(self, config: OellmEngramConfig) -> None:
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(OellmEngramDecoderLayer(config, i) for i in range(len(config.block_types)))
        self.norm = OellmRMSNorm(config.hidden_size, config.rms_norm_eps, config.layernorm_zero_centered_gamma)
        self.rotary_emb = (
            OellmRotaryEmbedding(config.rotary_dim, config.rope_theta, config.rope_interleaved_freqs)
            if "attention" in config.block_types else None
        )
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embed_tokens = value

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, inputs_embeds=None,
                past_key_values=None, use_cache=None, output_hidden_states=None, return_dict=None, **kwargs):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("pass exactly one of input_ids or inputs_embeds")
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if past_key_values is not None and not isinstance(past_key_values, OellmEngramCache):
            past_key_values = None
        if use_cache and past_key_values is None:
            past_key_values = OellmEngramCache(len(self.layers))
        h = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        b, s, _ = h.shape
        past_len = past_key_values.seen_tokens if past_key_values is not None else 0
        if position_ids is None and attention_mask is not None:
            # count positions from each sequence's first real token: HybridModel's RoPE gives the two
            # channels of a pair different angles, so attention depends on absolute positions too
            position_ids = (attention_mask.long().cumsum(-1) - 1).clamp_(min=0)[:, -s:]
        elif position_ids is None:
            position_ids = torch.arange(past_len, past_len + s, device=h.device).unsqueeze(0).expand(b, s)
        position_embeddings = self.rotary_emb(position_ids) if self.rotary_emb is not None else None

        # attention_mask: [b, past + s] padding mask (1 = token). All ones carries no information.
        if attention_mask is not None and bool(attention_mask.all()):
            attention_mask = None
        token_mask = None if attention_mask is None else attention_mask[:, -s:]

        hidden = []
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                hidden.append(h)
            state = past_key_values.states[i] if past_key_values is not None else None
            h, state = layer(h, position_embeddings, attention_mask, token_mask, input_ids, state, past_len)
            if past_key_values is not None:
                past_key_values.states[i] = state
        h = self.norm(h)
        if output_hidden_states:
            hidden.append(h)
        if past_key_values is not None:
            past_key_values.seen_tokens = past_len + s
        return BaseModelOutputWithPast(last_hidden_state=h, past_key_values=past_key_values,
                                       hidden_states=tuple(hidden) if output_hidden_states else None)


class OellmEngramForCausalLM(OellmEngramPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: OellmEngramConfig) -> None:
        super().__init__(config)
        self.model = OellmEngramModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def get_decoder(self) -> nn.Module:
        return self.model

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, inputs_embeds=None,
                past_key_values=None, use_cache=None, labels=None, output_hidden_states=None,
                return_dict=None, **kwargs) -> CausalLMOutputWithPast:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                         inputs_embeds=inputs_embeds, past_key_values=past_key_values, use_cache=use_cache,
                         output_hidden_states=output_hidden_states)
        logits = self.lm_head(out.last_hidden_state)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                                   labels[:, 1:].reshape(-1), ignore_index=-100)
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=out.past_key_values,
                                      hidden_states=out.hidden_states)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None,
                                      use_cache=None, **kwargs):
        use_cache = self.config.use_cache if use_cache is None else use_cache
        # generate() may create a transformers cache of its own: only ours carries the GDN / Engram state
        if past_key_values is not None and not isinstance(past_key_values, OellmEngramCache):
            past_key_values = OellmEngramCache(len(self.config.block_types))
        if past_key_values is not None and past_key_values.seen_tokens > 0:
            input_ids = input_ids[:, past_key_values.seen_tokens:]
        return {"input_ids": input_ids, "attention_mask": attention_mask,
                "past_key_values": past_key_values, "use_cache": use_cache}

    def _reorder_cache(self, past_key_values, beam_idx):
        if past_key_values is not None:
            past_key_values.reorder_cache(beam_idx)
        return past_key_values


__all__ = ["OellmEngramConfig", "OellmEngramForCausalLM", "OellmEngramModel", "OellmEngramPreTrainedModel",
           "OellmEngramCache"]

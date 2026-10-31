"""HF port of the ambitious hybrid main run, a transcription of the Megatron modules it trains with.

* residual (``residual``):
  - "mhc": mHC (``megatron.core.transformer.mhc.MHCMixer``) over n streams. The embedding is
    copied into n streams, every layer reads its input ``h_pre . streams`` and writes back
    ``h_res @ streams + h_post (x) out``, and the trunk ends with the stream mean.
  - "ldm": Linear Depth Memory, the stream form of ``megatron.core.transformer.depth_memory``
    (``DepthMemoryMixer``, ``DepthMemoryStreamBoundary``, the torch workspace's step). Per token,
    next to the stream x, a matrix memory M [G, K, h/G] and inverse key covariances P [G, K, K]
    (fp32) fit the layer outputs over depth by recursive least squares. Each layer reads
    ``x + r`` (r: the pending read), writes its output ``o`` (``x <- x + o``, RLS step keyed by the
    new x) and reads r' with the next layer's query; the embedding is written first, the last
    read is added before the final norm. The state is per token, so decoding needs no cache.
* layers, one per Megatron pattern symbol (``block_types``):
  - "attention" ``*``: NoPE softmax attention with qk-norm, exclusive self attention (XSA,
    ``fused_xsa.xsa_reference``) and the sigmoid output gate, behind ``linear_qkv``'s fused norm;
  - "ckda" ``L``: the FLA ComplexKimiDeltaAttention layer, as Megatron's ComplexKDA builds it;
  - "moe" ``E``: Megatron's sigmoid router, balanced by an expert bias (top-k on scores + bias) or
    quantile balancing (top-k on logits - qb_beta); weights = the unbiased scores normalised to 1.
    SwiGLU experts with the gpt-oss / DeepSeek-V4.1 clamp and optionally the bounded
    post-activation norm (pab, ``fused_bias_swiglu.normed_swiglu``), an ungated shared expert;
  - "engram" ``N``: ``EngramLayer`` with the signed-sqrt gate, the conv pre-norm, per-layer
    hash multipliers and optionally the sign hash (``engram_sign_hash``).

The mHC, CKDA and Engram pieces call the same math as Megatron (fp32 coefficients and Sinkhorn in
log space, the same FLA layer). Decoding keeps a per-layer state (``OellmHybridCache``).
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .configuration_oellm_ambitious import OellmAmbitiousConfig
# imported here too: the HF dynamic loader copies only the direct relative imports of this file
from .configuration_oellm_hybrid import OellmHybridConfig  # noqa: F401
from .modeling_oellm_hybrid import OellmEngram, OellmFLALayer, OellmHybridCache, OellmRMSNorm


# --------------------------------------------------------------------------- mHC
def _sinkhorn(logits: torch.Tensor, iters: int) -> torch.Tensor:
    x = logits.float()
    for _ in range(iters):
        x = x - torch.logsumexp(x, dim=-2, keepdim=True)
        x = x - torch.logsumexp(x, dim=-1, keepdim=True)
    return torch.exp(x)


class OellmMHC(nn.Module):
    """``MHCMixer``: coefficients from the RMS-normalised flattened streams."""

    def __init__(self, config: OellmAmbitiousConfig) -> None:
        super().__init__()
        n, h = config.mhc_expansion_rate, config.hidden_size
        self.n, self.iters, self.eps = n, config.mhc_sinkhorn_iters, config.mhc_eps
        self.phi = nn.Parameter(torch.zeros(n * h, 2 * n + n * n))
        self.b_pre = nn.Parameter(torch.zeros(n))
        self.b_post = nn.Parameter(torch.zeros(n))
        self.b_res = nn.Parameter(torch.zeros(n, n))
        self.alpha_pre = nn.Parameter(torch.zeros(1))
        self.alpha_post = nn.Parameter(torch.zeros(1))
        self.alpha_res = nn.Parameter(torch.zeros(1))

    def read(self, streams: torch.Tensor):
        """``[b, s, n, h]`` streams -> (layer input ``[b, s, h]``, (h_post, h_res))."""
        b, s, n, h = streams.shape
        flat = streams.reshape(b, s, n * h)
        r = torch.linalg.vector_norm(flat, dim=-1, keepdim=True).float() / (flat.shape[-1] ** 0.5)
        coeffs = torch.matmul(flat, self.phi.to(flat.dtype)).float() / r.clamp(min=self.eps)
        pre, post, res = torch.split(coeffs, [n, n, n * n], dim=-1)
        h_pre = torch.sigmoid(self.alpha_pre.float() * pre + self.b_pre.float())
        h_post = 2.0 * torch.sigmoid(self.alpha_post.float() * post + self.b_post.float())
        h_res = _sinkhorn(self.alpha_res.float() * res.view(b, s, n, n) + self.b_res.float(), self.iters)
        x = (h_pre.to(streams.dtype).unsqueeze(-1) * streams).sum(dim=-2)
        return x, (h_post, h_res)

    @staticmethod
    def write(streams: torch.Tensor, out: torch.Tensor, pending) -> torch.Tensor:
        h_post, h_res = (t.to(streams.dtype) for t in pending)
        mixed = (h_res.unsqueeze(-1) * streams.unsqueeze(-3)).sum(dim=-2)
        return mixed + h_post.unsqueeze(-1) * out.unsqueeze(-2)


# --------------------------------------------------------------------------- Linear Depth Memory
class OellmLDMNet(nn.Module):
    """``depth_memory._AddressNetwork`` as the fused step evaluates it (``_site_addresses``):
    RMS-normalise, fold the norm gain into W1 (bf16), SiLU, the L2-normalised address [G, K] and
    the identity-centred gate 2 sigmoid [G], both fp32."""

    def __init__(self, config: OellmAmbitiousConfig) -> None:
        super().__init__()
        h, hid = config.hidden_size, config.ldm_hidden_dim
        self.groups, self.key_dim, self.eps = config.ldm_groups, config.ldm_key_dim, config.rms_norm_eps
        self.norm_weight = nn.Parameter(torch.ones(h))
        self.w1 = nn.Linear(h, hid)
        self.address = nn.Linear(hid, self.groups * self.key_dim)
        self.gate = nn.Linear(hid, self.groups)

    def forward(self, x: torch.Tensor):
        xf = x.float()
        w1 = self.w1.weight
        xn = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)).to(w1.dtype)
        hidden = F.silu(F.linear(xn, w1 * self.norm_weight.to(w1.dtype), self.w1.bias))
        shape = (*x.shape[:-1], self.groups, self.key_dim)
        address = F.normalize(self.address(hidden).float().view(shape), dim=-1, eps=1e-6)
        return address, 2.0 * torch.sigmoid(self.gate(hidden).float())


class OellmLDMWrite(nn.Module):
    """``DepthMemoryWrite``: the key / weight network and the per-group forgetting."""

    def __init__(self, config: OellmAmbitiousConfig) -> None:
        super().__init__()
        self.net = OellmLDMNet(config)
        self.forget_logit = nn.Parameter(torch.zeros(config.ldm_groups))


class OellmLDMRead(nn.Module):
    """``DepthMemoryRead``: the query network and the per-group read gain."""

    def __init__(self, config: OellmAmbitiousConfig) -> None:
        super().__init__()
        self.net = OellmLDMNet(config)
        self.read_gain = nn.Parameter(torch.ones(config.ldm_groups))

    def query(self, x: torch.Tensor) -> torch.Tensor:
        q, gate = self.net(x)
        return q * (gate * self.read_gain.float()).unsqueeze(-1)


def ldm_step(write: OellmLDMWrite, read: OellmLDMRead, x, value, memory, covariance):
    """One step (``TorchDepthMemoryWorkspace.forward``): the RLS write of ``value`` under the key
    of ``x``, then the read with ``read``'s query of ``x``. Returns (read [.., h] in x's dtype,
    memory, covariance)."""
    key, weight = write.net(x)
    g = key.shape[-2]
    scale = weight.clamp_min(1e-6).sqrt()
    kbar = key * scale.unsqueeze(-1)
    vbar = value.float().reshape(*value.shape[:-1], g, value.shape[-1] // g) * scale.unsqueeze(-1)
    lam = torch.sigmoid(write.forget_logit.float()).clamp_min(1e-6)
    pk = torch.einsum("...gij,...gj->...gi", covariance, kbar)
    kappa = pk / (lam + (kbar * pk).sum(-1)).unsqueeze(-1)
    u = vbar - torch.einsum("...gkd,...gk->...gd", memory, kbar)
    kp = torch.einsum("...gi,...gij->...gj", kbar, covariance)
    memory = memory + kappa.unsqueeze(-1) * u.unsqueeze(-2)
    covariance = (covariance - kappa.unsqueeze(-1) * kp.unsqueeze(-2)) / lam[:, None, None]
    r = torch.einsum("...gkd,...gk->...gd", memory, read.query(x))
    return r.reshape(x.shape).to(x.dtype), memory, covariance


class OellmLDMMixer(nn.Module):
    """``DepthMemoryMixer``: this layer's write and the following read's query."""

    def __init__(self, config: OellmAmbitiousConfig) -> None:
        super().__init__()
        self.write_site = OellmLDMWrite(config)
        self.next_query = OellmLDMRead(config)


class OellmLDMBoundary(nn.Module):
    """``DepthMemoryStreamBoundary``: the embedding write and the first layer's query."""

    def __init__(self, config: OellmAmbitiousConfig) -> None:
        super().__init__()
        self.groups, self.key_dim, self.alpha = config.ldm_groups, config.ldm_key_dim, config.ldm_alpha
        self.embed_write = OellmLDMWrite(config)
        self.first_query = OellmLDMRead(config)

    def enter(self, x: torch.Tensor):
        """The embedding -> (x, r, memory, covariance)."""
        prefix = x.shape[:-1]
        g, k = self.groups, self.key_dim
        memory = x.new_zeros((*prefix, g, k, x.shape[-1] // g), dtype=torch.float32)
        eye = torch.eye(k, device=x.device, dtype=torch.float32) * self.alpha
        covariance = eye.expand(*prefix, g, k, k)
        r, memory, covariance = ldm_step(self.embed_write, self.first_query, x, x, memory, covariance)
        return x, r, memory, covariance


# --------------------------------------------------------------------------- attention
class OellmXSAAttention(nn.Module):
    """Megatron ``SelfAttention`` as the main run uses it: no position embedding, qk-norm over the
    head dim, XSA (subtract each head's projection onto its own value) and the sigmoid output gate
    (``exclusive_self_attention`` fuses both), then ``linear_proj``."""

    def __init__(self, config: OellmAmbitiousConfig) -> None:
        super().__init__()
        h, d = config.hidden_size, config.head_dim
        self.nh, self.nkv, self.d = config.num_attention_heads, config.num_key_value_heads, d
        self.xsa, self.xsa_eps = config.exclusive_self_attention, config.xsa_eps
        self.q_proj = nn.Linear(h, self.nh * d, bias=False)
        self.gate_proj = nn.Linear(h, self.nh * d, bias=False) if config.attention_output_gate else None
        self.k_proj = nn.Linear(h, self.nkv * d, bias=False)
        self.v_proj = nn.Linear(h, self.nkv * d, bias=False)
        self.o_proj = nn.Linear(self.nh * d, h, bias=False)
        zcg = config.layernorm_zero_centered_gamma
        self.q_norm = OellmRMSNorm(d, config.rms_norm_eps, zcg) if config.qk_layernorm else None
        self.k_norm = OellmRMSNorm(d, config.rms_norm_eps, zcg) if config.qk_layernorm else None

    def forward(self, x: torch.Tensor, state=None):
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.nh, self.d)
        k = self.k_proj(x).view(b, s, self.nkv, self.d)
        v = self.v_proj(x).view(b, s, self.nkv, self.d)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        own_v = v
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        past = 0
        if state is not None:
            past = state[0].shape[2]
            k = torch.cat([state[0], k], dim=2)
            v = torch.cat([state[1], v], dim=2)
        new_state = (k, v)
        rep = self.nh // self.nkv
        kk = k.repeat_interleave(rep, dim=1) if rep > 1 else k
        vv = v.repeat_interleave(rep, dim=1) if rep > 1 else v
        if past == 0:
            o = F.scaled_dot_product_attention(q, kk, vv, is_causal=True, scale=self.d**-0.5)
        else:  # queries at past..past+s-1 see keys 0..their position
            qp = torch.arange(past, past + s, device=x.device)[:, None]
            mask = torch.arange(k.shape[2], device=x.device)[None, :] <= qp
            o = F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask[None, None], scale=self.d**-0.5)
        o = o.transpose(1, 2)  # [b, s, nh, d]
        if self.xsa:
            vf = own_v.float().repeat_interleave(rep, dim=2) if rep > 1 else own_v.float()
            of = o.float()
            c = (of * vf).sum(-1, keepdim=True) / (vf.pow(2).sum(-1, keepdim=True) + self.xsa_eps)
            of = of - c * vf
            if self.gate_proj is not None:
                of = of * torch.sigmoid(self.gate_proj(x).view(b, s, self.nh, self.d).float())
            o = of.to(x.dtype)
        elif self.gate_proj is not None:
            o = (o.float() * torch.sigmoid(self.gate_proj(x).view(b, s, self.nh, self.d).float())).to(x.dtype)
        return self.o_proj(o.reshape(b, s, self.nh * self.d)), new_state


# --------------------------------------------------------------------------- MoE
def _swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float | None, norm=None) -> torch.Tensor:
    """SwiGLU with the clamp; ``norm`` = (eps, floor, smooth): the post-activation norm of
    ``fused_bias_swiglu.normed_swiglu`` (fp32: h / max(rms(h), floor), floor 0: h / rms(h);
    smooth: h / sqrt(1 + ms / floor^2)), cast back to the input dtype."""
    if limit is not None:  # fused_bias_swiglu.clamped_swiglu
        gate, up = gate.clamp(max=limit), up.clamp(min=-limit, max=limit)
    if norm is None:
        return F.silu(gate) * up
    eps, floor, smooth = norm
    act = F.silu(gate.float()) * up.float()
    ms = act.square().mean(-1, keepdim=True)
    if smooth:
        scale = torch.rsqrt(1.0 + ms / (floor * floor))
    else:
        scale = torch.rsqrt((ms + eps).clamp(min=floor * floor))
    return (act * scale).to(gate.dtype)


class OellmMoE(nn.Module):
    def __init__(self, config: OellmAmbitiousConfig) -> None:
        super().__init__()
        h, e, i = config.hidden_size, config.num_experts, config.moe_intermediate_size
        self.top_k, self.num_experts, self.scaling = config.num_experts_per_tok, e, config.router_scaling_factor
        self.limit = config.activation_clamp_value
        pab = (config.rms_norm_eps, config.post_act_norm_floor, config.post_act_norm_smooth)
        self.expert_norm = pab if config.moe_expert_post_act_norm else None
        self.shared_norm = pab if config.mlp_post_act_norm else None
        self.quantile = config.router_balancing == "quantile"
        self.gate = nn.Parameter(torch.zeros(e, h))  # router weight
        if self.quantile:  # Router.qb_beta: selection on logits - qb_beta
            self.register_buffer("qb_beta", torch.zeros(e, dtype=torch.float32))
        else:
            self.register_buffer("e_score_correction_bias", torch.zeros(e, dtype=torch.float32))
        self.gate_up_proj = nn.Parameter(torch.zeros(e, 2 * i, h))
        self.down_proj = nn.Parameter(torch.zeros(e, h, i))
        si = config.shared_expert_intermediate_size
        self.shared_gate_proj = nn.Linear(h, si, bias=False) if si else None
        self.shared_up_proj = nn.Linear(h, si, bias=False) if si else None
        self.shared_down_proj = nn.Linear(si, h, bias=False) if si else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        t = x.reshape(-1, shape[-1])
        logits = F.linear(t.float(), self.gate.float())
        scores = torch.sigmoid(logits)
        if self.quantile:  # moe_utils.qb_dual_update (no update at inference)
            _, idx = torch.topk(logits - self.qb_beta.float(), self.top_k, dim=-1)
        else:
            _, idx = torch.topk(scores + self.e_score_correction_bias.float(), self.top_k, dim=-1)
        w = scores.gather(-1, idx)
        if self.top_k > 1:
            w = w / (w.sum(-1, keepdim=True) + 1e-20)
        w = (w * self.scaling).to(t.dtype)
        out = torch.zeros_like(t)
        for e in torch.unique(idx).tolist():
            tok, slot = torch.where(idx == e)
            g, u = F.linear(t[tok], self.gate_up_proj[e]).chunk(2, dim=-1)
            y = F.linear(_swiglu(g, u, self.limit, self.expert_norm), self.down_proj[e])
            out.index_add_(0, tok, (y * w[tok, slot, None]).to(out.dtype))
        if self.shared_gate_proj is not None:
            out = out + self.shared_down_proj(
                _swiglu(self.shared_gate_proj(t), self.shared_up_proj(t), self.limit, self.shared_norm)
            )
        return out.view(shape)


# --------------------------------------------------------------------------- Engram
class OellmEngramV2(OellmEngram):
    """``EngramLayer`` with the main run's options: per-layer hash multipliers
    (``engram_hash_per_layer``), the signed-sqrt gate and the RMSNorm before the causal conv."""

    def __init__(self, config: OellmAmbitiousConfig, layer_idx: int) -> None:
        cfg = copy.copy(config)
        cfg.engram_hash_multipliers = config.engram_hash_multipliers[str(layer_idx)]
        super().__init__(cfg)
        self.signed_sqrt = config.engram_gate_signed_sqrt
        self.conv_norm = OellmRMSNorm(config.hidden_size, config.rms_norm_eps) if config.engram_conv_prenorm else None
        # engram_sign_hash: a +-1 pattern per (n-gram, lookup) from an independent hash into a fixed
        # table (Megatron engram_sign_table; exported as a buffer)
        self.sign_hash = bool(getattr(config, "engram_sign_hash", False))
        if self.sign_hash:
            self.sign_mult = [[int(a) for a in row] for row in config.engram_sign_multipliers[str(layer_idx)]]
            self.sign_size = int(config.engram_sign_table_size)
            self.register_buffer("sign_table", torch.ones(self.sign_size, config.engram_embed_dim))

    def _sign_rows(self, shifted: torch.Tensor, s: int) -> torch.Tensor:
        """Sign-table rows of the last ``s`` positions: ``engram_hash`` with the sign multipliers
        and one table of ``engram_sign_table_size`` rows per lookup (no offsets)."""
        p = max(self.orders) - 1
        history = [shifted[:, p - i : p - i + s] for i in range(max(self.orders))]
        out, lookup = [], 0
        for n in self.orders:
            for _ in range(self.num_heads):
                hsh = torch.zeros_like(history[0])
                for i in range(n):
                    hsh = hsh ^ (history[i] * self.sign_mult[lookup][i])
                out.append(torch.remainder(hsh, self.sign_size))
                lookup += 1
        return torch.stack(out, dim=-1)

    def forward(self, hidden_states, input_ids, token_mask=None, state=None):
        b, s, h = hidden_states.shape
        ids = input_ids if self.token_map is None else self.token_map[input_ids]
        shifted = ids.long() + 1
        if token_mask is not None:
            shifted = shifted * token_mask.long()
        p = max(self.orders) - 1
        id_state, conv_state = state if state is not None else (None, None)
        if id_state is None:
            id_state = shifted.new_zeros(b, p)
        shifted = torch.cat([id_state, shifted], dim=1)
        new_id_state = shifted[:, shifted.shape[1] - p :]
        e = self._lookup(self._rows(shifted, s), hidden_states.dtype).to(hidden_states.device)
        if self.sign_hash:
            e = e * self.sign_table[self._sign_rows(shifted, s).to(self.sign_table.device)].to(e.device, e.dtype)
        e = e.reshape(b, s, -1)
        k, v = self.key_proj(e), self.value_proj(e)
        gate = (self.hidden_norm(hidden_states).float() * self.key_norm(k).float()).sum(-1, keepdim=True) / h**0.5
        if self.signed_sqrt:
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        v = torch.sigmoid(gate).to(v.dtype) * v
        if token_mask is not None:
            v = v * token_mask[..., None].to(v.dtype)
        conv_in = v if self.conv_norm is None else self.conv_norm(v)
        pad = (self.kernel_size - 1) * self.dilation
        ct = conv_in.transpose(1, 2)
        if conv_state is None:
            conv_state = ct.new_zeros(b, h, pad)
        padded = torch.cat([conv_state.to(ct.dtype), ct], dim=-1)
        conv = F.conv1d(padded, self.conv1d.weight, groups=h, dilation=self.dilation)
        new_conv_state = padded[..., padded.shape[-1] - pad :]
        return F.silu(conv.transpose(1, 2)) + v, (new_id_state, new_conv_state)


# --------------------------------------------------------------------------- layers / model
class OellmAmbitiousLayer(nn.Module):
    def __init__(self, config: OellmAmbitiousConfig, layer_idx: int) -> None:
        super().__init__()
        self.kind = config.block_types[layer_idx]
        self.ldm = config.residual == "ldm"
        if self.ldm:
            self.ldm_mixer = OellmLDMMixer(config)
        else:
            self.mhc = OellmMHC(config)
        zcg = config.layernorm_zero_centered_gamma
        if self.kind == "attention":
            self.input_layernorm = OellmRMSNorm(config.hidden_size, config.rms_norm_eps, zcg)
            self.mixer = OellmXSAAttention(config)
        elif self.kind == "ckda":
            self.input_layernorm = OellmRMSNorm(config.hidden_size, config.rms_norm_eps, zcg)
            self.mixer = OellmFLALayer(config.fla_layer)
        elif self.kind == "moe":
            self.input_layernorm = OellmRMSNorm(config.hidden_size, config.rms_norm_eps, zcg)
            self.mixer = OellmMoE(config)
        else:  # engram normalises internally
            self.mixer = OellmEngramV2(config, layer_idx)

    def _sublayer(self, x, state, input_ids, token_mask, attention_mask):
        new_state = None
        if self.kind == "attention":
            out, new_state = self.mixer(self.input_layernorm(x), state)
        elif self.kind == "ckda":
            out, new_state = self.mixer(self.input_layernorm(x), state, attention_mask)
        elif self.kind == "moe":
            out = self.mixer(self.input_layernorm(x))
        else:
            out, new_state = self.mixer(x, input_ids, token_mask, state)
        return out, new_state

    def forward(self, streams, state=None, input_ids=None, token_mask=None, attention_mask=None):
        if self.ldm:  # streams = (x, r, memory, covariance)
            x, r, memory, covariance = streams
            out, new_state = self._sublayer(x + r, state, input_ids, token_mask, attention_mask)
            x = x + out
            r, memory, covariance = ldm_step(
                self.ldm_mixer.write_site, self.ldm_mixer.next_query, x, out, memory, covariance
            )
            return (x, r, memory, covariance), new_state
        x, pending = self.mhc.read(streams)
        new_state = None
        if self.kind == "attention":
            out, new_state = self.mixer(self.input_layernorm(x), state)
        elif self.kind == "ckda":
            out, new_state = self.mixer(self.input_layernorm(x), state, attention_mask)
        elif self.kind == "moe":
            out = self.mixer(self.input_layernorm(x))
        else:
            out, new_state = self.mixer(x, input_ids, token_mask, state)
        return self.mhc.write(streams, out, pending), new_state


class OellmAmbitiousPreTrainedModel(PreTrainedModel):
    config_class = OellmAmbitiousConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["OellmAmbitiousLayer"]
    _supports_sdpa = True

    def _init_weights(self, module: nn.Module) -> None:
        # every tensor is loaded from the export; nothing to initialise (and a raw normal_
        # would overwrite loaded weights under transformers >= 5)
        return


class OellmAmbitiousModel(OellmAmbitiousPreTrainedModel):
    def __init__(self, config: OellmAmbitiousConfig) -> None:
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.ldm_boundary = OellmLDMBoundary(config) if config.residual == "ldm" else None
        self.layers = nn.ModuleList(OellmAmbitiousLayer(config, i) for i in range(config.num_hidden_layers))
        self.norm = OellmRMSNorm(config.hidden_size, config.rms_norm_eps, config.layernorm_zero_centered_gamma)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, past_key_values=None,
                use_cache=None, output_hidden_states=None, **kwargs) -> BaseModelOutputWithPast:
        if input_ids is None:
            raise ValueError("the Engram layers need input_ids")
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if past_key_values is not None and not isinstance(past_key_values, OellmHybridCache):
            past_key_values = None
        if use_cache and past_key_values is None:
            past_key_values = OellmHybridCache(len(self.layers))
        x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        b, s, _ = x.shape
        if attention_mask is not None and bool(attention_mask.all()):
            attention_mask = None
        token_mask = None
        if attention_mask is not None:
            token_mask = attention_mask[:, -s:]
            x = x * token_mask[..., None].to(x.dtype)
        ldm = self.ldm_boundary is not None
        if ldm:
            streams = self.ldm_boundary.enter(x)
        else:
            streams = x.unsqueeze(-2).expand(b, s, self.config.mhc_expansion_rate, x.shape[-1]).contiguous()
        hidden = []
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                hidden.append(streams[0] if ldm else streams.mean(dim=-2))
            st = past_key_values.states[i] if past_key_values is not None else None
            streams, new_state = layer(streams, st, input_ids, token_mask, attention_mask)
            if past_key_values is not None:
                past_key_values.states[i] = new_state
        h = self.norm(streams[0] + streams[1] if ldm else streams.mean(dim=-2))
        if past_key_values is not None:
            past_key_values.seen_tokens += s
        if output_hidden_states:
            hidden.append(h)
        return BaseModelOutputWithPast(last_hidden_state=h, past_key_values=past_key_values,
                                       hidden_states=tuple(hidden) if output_hidden_states else None)


class OellmAmbitiousForCausalLM(OellmAmbitiousPreTrainedModel, GenerationMixin):
    def __init__(self, config: OellmAmbitiousConfig) -> None:
        super().__init__(config)
        self.model = OellmAmbitiousModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids=None, attention_mask=None, past_key_values=None, labels=None,
                use_cache=None, output_hidden_states=None, **kwargs) -> CausalLMOutputWithPast:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values,
                         use_cache=use_cache, output_hidden_states=output_hidden_states)
        logits = self.lm_head(out.last_hidden_state)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1),
                                   ignore_index=-100)
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=out.past_key_values,
                                      hidden_states=out.hidden_states)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        # `generate` hands over a DynamicCache of its own at the prefill; only ours carries the
        # CKDA / Engram state and the mHC layers' KV (an empty foreign cache holds nothing yet)
        if past_key_values is not None and not isinstance(past_key_values, OellmHybridCache):
            past_key_values = None
        if past_key_values is not None and past_key_values.seen_tokens > 0:
            input_ids = input_ids[:, past_key_values.seen_tokens :]
        return {"input_ids": input_ids, "past_key_values": past_key_values, "attention_mask": attention_mask,
                "use_cache": kwargs.get("use_cache", True)}

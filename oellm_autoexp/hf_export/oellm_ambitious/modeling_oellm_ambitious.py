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

Inference fast path (CUDA, no autograd; ``OELLM_HF_IMPL=reference`` turns it off): the kernels of
``oellm_kernels`` -- fused mHC read (+ the layer's input norm) / write, FP8 experts grouped by
expert and dequantized inside the GEMM (no BF16 copy of the experts), the sigmoid router, the
Engram hashes and a zero-copy gather from tables pinned in host memory -- and no host
synchronization, so the layers on the GPUs of a device map run concurrently on micro-batches
(``OELLM_HF_PIPELINE``). ``from_pretrained(device_map="auto")`` places whole layers on the GPUs
itself and keeps the Engram tables in pinned host memory (``OELLM_HF_PLACEMENT=accelerate``: the
stock placement). Scoring calls (no ``use_cache``) skip the decoding state.
"""

from __future__ import annotations

import contextlib
import copy
import os

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


def _fast(t: torch.Tensor) -> bool:
    """The inference kernels apply: CUDA, no autograd, not switched off."""
    return t.is_cuda and not torch.is_grad_enabled() and os.environ.get("OELLM_HF_IMPL", "fast") != "reference"


def _kernels():
    # the `from .x import` form: transformers' remote-code loader copies only those targets
    from .oellm_kernels import rms_norm  # noqa: F401
    from . import oellm_kernels

    return oellm_kernels


def _cached(module: nn.Module, name: str, make, *sources: torch.Tensor):
    """``make()`` computed once per version of ``sources`` (fused weight copies, fp32 casts)."""
    key = tuple((t.data_ptr(), t._version, t.device) for t in sources)
    hit = module.__dict__.get(name)
    if hit is None or hit[0] != key:
        hit = (key, make())
        module.__dict__[name] = hit
    return hit[1]


# --------------------------------------------------------------------------- mHC
def _sinkhorn(logits: torch.Tensor, iters: int) -> torch.Tensor:
    if (
        os.environ.get("OELLM_MHC_IMPL", "reference") == "triton"
        and logits.is_cuda
        and logits.shape[-2:] == (4, 4)
        and not logits.requires_grad
    ):
        from .mhc_kernels import sinkhorn4

        return sinkhorn4(logits, iters)
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

    def read(self, streams: torch.Tensor, norm: OellmRMSNorm | None = None):
        """``[b, s, n, h]`` streams -> (layer input ``[b, s, h]``, (h_post, h_res)); with ``norm``
        the fast path returns ``norm(input)`` (fused), the reference the input itself."""
        b, s, n, h = streams.shape
        if n == 4 and _fast(streams):
            nw = None if norm is None else (norm.weight, norm.variance_epsilon, norm.zero_centered_gamma)
            x, pending = _kernels().mhc_read(streams.reshape(b * s, n, h), self.phi, self.alpha_pre, self.alpha_post,
                                             self.alpha_res, self.b_pre, self.b_post, self.b_res, self.eps,
                                             self.iters, nw)
            return x.view(b, s, h), pending
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
        b, s, n, h = streams.shape
        if n == 4 and _fast(streams):
            return _kernels().mhc_write(streams.reshape(b * s, n, h), out.reshape(b * s, h), pending).view(b, s, n, h)
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

    def _fast_forward(self, x: torch.Tensor, keep_state: bool):
        """No past: one GEMM for q | gate | k | v, the fused qk-norm, GQA inside SDPA, XSA against
        the shared value head without expanding it."""
        b, s, _ = x.shape
        nh, nkv, d = self.nh, self.nkv, self.d
        mods = [self.q_proj] + ([self.gate_proj] if self.gate_proj is not None else []) + [self.k_proj, self.v_proj]
        w = _cached(self, "_qgkv", lambda: torch.cat([m.weight for m in mods]), *(m.weight for m in mods))
        y = F.linear(x, w)
        sizes = [nh * d] + ([nh * d] if self.gate_proj is not None else []) + [nkv * d, nkv * d]
        parts = torch.split(y, sizes, dim=-1)
        q, k, v = parts[0], parts[-2], parts[-1]
        gate = parts[1] if self.gate_proj is not None else None
        q, k, v = q.reshape(b, s, nh, d), k.reshape(b, s, nkv, d), v.reshape(b, s, nkv, d)
        if self.q_norm is not None:
            kern = _kernels()
            q = kern.rms_norm(q, self.q_norm.weight, self.q_norm.variance_epsilon, self.q_norm.zero_centered_gamma)
            k = kern.rms_norm(k, self.k_norm.weight, self.k_norm.variance_epsilon, self.k_norm.zero_centered_gamma)
        kt, vt = k.transpose(1, 2), v.transpose(1, 2)
        o = F.scaled_dot_product_attention(q.transpose(1, 2), kt, vt, is_causal=True, scale=d**-0.5,
                                           enable_gqa=nh != nkv).transpose(1, 2)
        rep = nh // nkv
        if self.xsa:
            vf = v.float().unsqueeze(-2)  # [b, s, nkv, 1, d]: head kv * rep + r reads value head kv
            of = o.float().reshape(b, s, nkv, rep, d)
            c = (of * vf).sum(-1, keepdim=True) / (vf.pow(2).sum(-1, keepdim=True) + self.xsa_eps)
            of = (of - c * vf).reshape(b, s, nh, d)
            if gate is not None:
                of = of * torch.sigmoid(gate.reshape(b, s, nh, d).float())
            o = of.to(x.dtype)
        elif gate is not None:
            o = (o.float() * torch.sigmoid(gate.reshape(b, s, nh, d).float())).to(x.dtype)
        state = (kt, vt) if keep_state else None
        return self.o_proj(o.reshape(b, s, nh * d)), state

    def forward(self, x: torch.Tensor, state=None, keep_state: bool = True):
        if state is None and _fast(x):
            return self._fast_forward(x, keep_state)
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
        # "fused" (default): the inference kernels when they apply (_fast), else the reference loop;
        # "grouped": experiment 008's torch._grouped_mm path (BF16 experts, prepare_grouped_experts)
        self.impl = os.environ.get("OELLM_HF_MOE", "fused")
        if self.impl not in ("reference", "grouped", "fused"):
            raise ValueError(f"Unknown OELLM_HF_MOE: {self.impl}")
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
        self.fp8 = getattr(config, "expert_weight_dtype", "bf16") == "fp8"
        if self.fp8:  # E4M3 bytes + one fp32 scale per `block` elements of a row, dequantized per use
            self.block = int(config.expert_scale_block)
            self.register_buffer("gate_up_proj", torch.zeros(e, 2 * i, h, dtype=torch.uint8))
            self.register_buffer("gate_up_proj_scale", torch.ones(e, 2 * i, h // self.block, dtype=torch.float32))
            self.register_buffer("down_proj", torch.zeros(e, h, i, dtype=torch.uint8))
            self.register_buffer("down_proj_scale", torch.ones(e, h, i // self.block, dtype=torch.float32))
        else:
            self.gate_up_proj = nn.Parameter(torch.zeros(e, 2 * i, h))
            self.down_proj = nn.Parameter(torch.zeros(e, h, i))
        si = config.shared_expert_intermediate_size
        self.shared_gate_proj = nn.Linear(h, si, bias=False) if si else None
        self.shared_up_proj = nn.Linear(h, si, bias=False) if si else None
        self.shared_down_proj = nn.Linear(si, h, bias=False) if si else None

    def _apply(self, fn, recurse=True):
        """Keep the fp32 expert scales out of the model's dtype casts (device moves still apply)."""
        if not self.fp8:
            return super()._apply(fn, recurse)
        held = [(n, self._buffers.pop(n)) for n in ("gate_up_proj_scale", "down_proj_scale")]
        try:
            super()._apply(fn, recurse)
        finally:
            dev = self.gate_up_proj.device
            for n, t in held:
                self._buffers[n] = t.to(dev) if t.device != dev else t
        return self

    def _expert(self, name: str, e: int, dtype: torch.dtype) -> torch.Tensor:
        """Expert e's weight in the compute dtype; fp8: Megatron's dequantization (E4M3 -> fp32,
        times the tile's fp32 scale, rounded once to bf16; fp8_param_gather.dequantize_unpack)."""
        w = getattr(self, name)[e]
        if not self.fp8:
            return w
        q = w.view(torch.float8_e4m3fn).float()
        s = getattr(self, name + "_scale")[e].float()
        return (q.view(*q.shape[:-1], -1, self.block) * s[..., None]).view(q.shape).to(torch.bfloat16).to(dtype)

    def prepare_grouped_experts(self) -> None:
        """Expand FP8 experts once, before dispatching the BF16-sized model."""
        if not self.fp8:
            return
        with torch.no_grad():
            for name in ("gate_up_proj", "down_proj"):
                q = getattr(self, name).view(torch.float8_e4m3fn).float()
                scale = getattr(self, name + "_scale").float()
                value = q.view(*q.shape[:-1], -1, self.block) * scale[..., None]
                self._buffers[name] = value.reshape(q.shape).to(torch.bfloat16)
                self._buffers.pop(name + "_scale")
        self.fp8 = False

    def _grouped_experts(self, t, idx, weights):
        if self.fp8:
            raise RuntimeError(
                "Call prepare_grouped_experts() before device dispatch; "
                "the grouped backend needs a BF16-sized device map"
            )
        if not t.is_cuda:
            raise ValueError("The grouped MoE backend requires CUDA")
        flat = idx.flatten()
        order = torch.argsort(flat, stable=True)
        tokens = order // self.top_k
        counts = torch.bincount(flat, minlength=self.num_experts)
        offsets = counts.cumsum(0).to(torch.int32)
        gu = torch._grouped_mm(
            t[tokens], self.gate_up_proj.transpose(-1, -2), offs=offsets
        )
        gate, up = gu.chunk(2, dim=-1)
        act = _swiglu(gate, up, self.limit, self.expert_norm)
        y = torch._grouped_mm(
            act, self.down_proj.transpose(-1, -2), offs=offsets
        )
        weighted = (y * weights.flatten()[order, None]).to(t.dtype)
        contributions = torch.empty_like(weighted)
        contributions[order] = weighted
        contributions = contributions.view(t.shape[0], self.top_k, t.shape[1])
        expert_order = idx.argsort(dim=-1)
        token_ids = torch.arange(t.shape[0], device=t.device)
        out = torch.zeros_like(t)
        # The reference adds experts in ascending ID order with BF16 rounding
        # after every contribution. Preserve that order without a CPU sync.
        for slot in range(self.top_k):
            out.add_(contributions[token_ids, expert_order[:, slot]])
        return out

    def _fused_forward(self, t: torch.Tensor) -> torch.Tensor:
        """The routed experts grouped by expert (FP8 dequantized in the GEMM), no host sync; the
        reference's rounding points and its ascending-expert accumulation."""
        kern = _kernels()
        gate = _cached(self, "_gate_f32", lambda: self.gate.float(), self.gate)
        logits = F.linear(t.float(), gate)
        bias = self.qb_beta if self.quantile else self.e_score_correction_bias
        w, idx = kern.route(logits, bias.float(), self.top_k, self.scaling, self.quantile)
        w = w.to(t.dtype)
        sched = kern.ExpertSchedule(idx, self.num_experts, kern.block_m_for(idx.numel(), self.num_experts))
        fp8 = self.fp8
        gu = kern.grouped_gemm(t, sched, self.gate_up_proj, self.gate_up_proj_scale if fp8 else None, gather=True)
        act = kern.swiglu_norm(gu, self.limit, self.expert_norm)
        y = kern.grouped_gemm(act, sched, self.down_proj, self.down_proj_scale if fp8 else None, gather=False)
        shared = None
        if self.shared_gate_proj is not None:
            g, u = self.shared_gate_proj.weight, self.shared_up_proj.weight
            wgu = _cached(self, "_shared_gu", lambda: torch.cat([g, u]), g, u)
            shared = self.shared_down_proj(kern.swiglu_norm(F.linear(t, wgu), self.limit, self.shared_norm))
        return kern.moe_combine(y, sched, idx, w, shared)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        t = x.reshape(-1, shape[-1])
        if self.impl == "fused" and _fast(t) and t.dtype == torch.bfloat16:
            return self._fused_forward(t).view(shape)
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
        if self.impl == "grouped":
            out = self._grouped_experts(t, idx, w)
        else:
            out = torch.zeros_like(t)
            for e in torch.unique(idx).tolist():
                tok, slot = torch.where(idx == e)
                g, u = F.linear(t[tok], self._expert("gate_up_proj", e, t.dtype)).chunk(2, dim=-1)
                y = F.linear(_swiglu(g, u, self.limit, self.expert_norm), self._expert("down_proj", e, t.dtype))
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

    def _apply(self, fn, recurse=True):
        """A table pinned for the zero-copy gather stays in host memory through device moves."""
        if self.__dict__.get("_uva") is None:
            return super()._apply(fn, recurse)
        held = [(n, self._parameters.pop(n) if n in self._parameters else self._buffers.pop(n))
                for n in ("table", "table_scale") if getattr(self, n, None) is not None]
        try:
            nn.Module._apply(self, fn, recurse)  # not OellmEngram._apply: it reads the held table
        finally:
            for n, t in held:
                (self._parameters if isinstance(t, nn.Parameter) else self._buffers)[n] = t
        return self

    def pin_table(self) -> bool:
        """Register the host table for the zero-copy GPU gather (in place; a file-backed or
        non-contiguous table is copied into anonymous memory first). False: not applicable."""
        if self.__dict__.get("_uva") is not None:
            return True
        t = self.table
        if t.device.type != "cpu" or self.table_dtype == "fp8" or not torch.cuda.is_available():
            return False
        kern = _kernels()
        try:
            if not t.is_contiguous():
                t.data = t.data.contiguous()
            try:
                uva = kern.UvaTable(t.data)
            except RuntimeError:  # e.g. a read-only file mapping: copy into anonymous memory
                fresh = torch.empty(t.shape, dtype=t.dtype)
                fresh.copy_(t.data)
                t.data = fresh
                uva = kern.UvaTable(t.data)
        except Exception as exc:  # no pinning: the host lookup of the reference
            import warnings

            warnings.warn(f"Engram table not pinned ({exc}); using the host lookup")
            return False
        self.__dict__["_uva"] = uva
        return True

    def _hashes(self, device):
        """(row hash, sign hash or None) on ``device``."""
        cache = self.__dict__.setdefault("_hash_dev", {})
        key = (device, self.sign_hash)
        if key not in cache:
            kern = _kernels()
            per = [n for n in self.orders for _ in range(self.num_heads)]
            rows = kern.EngramHash(per, self.hash_mult, self.sizes, self.row_offsets, device)
            sign = None
            if self.sign_hash:
                sign = kern.EngramHash(per, self.sign_mult, [self.sign_size] * len(per), [0] * len(per), device)
            cache[key] = (rows, sign)
        return cache[key]

    def _fast_lookup(self, shifted: torch.Tensor, s: int, dtype: torch.dtype):
        """Rows (times the sign rows) gathered on the GPU from a table on this device or pinned in
        host memory; None when neither applies (an fp8 table, a table elsewhere)."""
        table = self.table
        if self.table_dtype == "fp8":
            return None
        dev = shifted.device
        if table.device != dev and not (table.device.type == "cpu" and os.environ.get("OELLM_HF_ENGRAM_UVA", "1") != "0"
                                        and self.pin_table()):
            return None
        kern = _kernels()
        rows_hash, sign_hash = self._hashes(dev)
        rows = rows_hash(shifted, s)
        sign = None
        if sign_hash is not None:
            st = self.sign_table if self.sign_table.device == dev else self.sign_table.to(dev)
            sign = (st.to(dtype), sign_hash(shifted, s))
        src = table if table.device == dev else self.__dict__["_uva"]._host
        return kern.gather(src, table.shape[-1], rows, dtype, sign)

    def forward(self, hidden_states, input_ids, token_mask=None, state=None):
        b, s, h = hidden_states.shape
        fast = _fast(hidden_states)
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
        e = self._fast_lookup(shifted, s, hidden_states.dtype) if fast else None
        if e is None:
            e = self._lookup(self._rows(shifted, s), hidden_states.dtype).to(hidden_states.device)
            if self.sign_hash:
                e = e * self.sign_table[self._sign_rows(shifted, s).to(self.sign_table.device)].to(e.device, e.dtype)
        e = e.reshape(b, s, -1)
        if fast:
            kw, vw = self.key_proj.weight, self.value_proj.weight
            k, v = F.linear(e, _cached(self, "_kv", lambda: torch.cat([kw, vw]), kw, vw)).split(h, dim=-1)
            kern = _kernels()
            hn = kern.rms_norm(hidden_states, self.hidden_norm.weight, self.hidden_norm.variance_epsilon)
            kn = kern.rms_norm(k, self.key_norm.weight, self.key_norm.variance_epsilon)
            gate = (hn.float() * kn.float()).sum(-1, keepdim=True) / h**0.5
        else:
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

    def _mix(self, xin, state, input_ids, token_mask, attention_mask, keep_state):
        """The mixer on its (normalised, except Engram) input."""
        new_state = None
        if self.kind == "attention":
            out, new_state = self.mixer(xin, state, keep_state)
        elif self.kind == "ckda":
            # stateless (scoring, fast path): FLA's fused input projections, no final state
            stateless = not keep_state and state is None and _fast(xin)
            out, new_state = self.mixer(xin, state, attention_mask, not stateless)
        elif self.kind == "moe":
            out = self.mixer(xin)
        else:
            out, new_state = self.mixer(xin, input_ids, token_mask, state)
        return out, new_state

    def _norm(self, x):
        return x if self.kind == "engram" else self.input_layernorm(x)

    def forward(self, streams, state=None, input_ids=None, token_mask=None, attention_mask=None, keep_state=True):
        if self.ldm:  # streams = (x, r, memory, covariance)
            x, r, memory, covariance = streams
            out, new_state = self._mix(self._norm(x + r), state, input_ids, token_mask, attention_mask, keep_state)
            x = x + out
            r, memory, covariance = ldm_step(
                self.ldm_mixer.write_site, self.ldm_mixer.next_query, x, out, memory, covariance
            )
            return (x, r, memory, covariance), new_state
        norm = None if self.kind == "engram" else self.input_layernorm
        if _fast(streams) and streams.shape[-2] == 4:
            xin, pending = self.mhc.read(streams, norm)  # the input norm fused into the read
        else:
            x, pending = self.mhc.read(streams)
            xin = self._norm(x)
        out, new_state = self._mix(xin, state, input_ids, token_mask, attention_mask, keep_state)
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

    def _apply(self, fn, recurse=True):
        out = super()._apply(fn, recurse)
        for m in self.modules():  # device moves invalidate the cached layer devices
            m.__dict__.pop("_oellm_device", None)
        return out


class OellmEmbedNorm(nn.Module):
    """Megatron's embedding_norm: an fp32 RMSNorm of the embedding output with a learnable gain (a
    module, so device maps place it)."""

    def __init__(self, hidden: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x.float(), (x.shape[-1],), self.weight.float(), self.eps).to(x.dtype)


class OellmAmbitiousModel(OellmAmbitiousPreTrainedModel):
    def __init__(self, config: OellmAmbitiousConfig) -> None:
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_scale = float(getattr(config, "embedding_multiplier", 1.0))
        # LanguageModelEmbedding.embedding_norm_weight: an fp32 RMSNorm of the (scaled) embedding output
        self.embed_norm = OellmEmbedNorm(config.hidden_size, config.rms_norm_eps) if getattr(config, "embedding_norm", False) else None
        self.ldm_boundary = OellmLDMBoundary(config) if config.residual == "ldm" else None
        self.layers = nn.ModuleList(OellmAmbitiousLayer(config, i) for i in range(config.num_hidden_layers))
        self.norm = OellmRMSNorm(config.hidden_size, config.rms_norm_eps, config.layernorm_zero_centered_gamma)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # ----------------------------------------------------------------------- devices
    @staticmethod
    def _device_of(module: nn.Module):
        """The CUDA device a layer computes on (its first CUDA tensor; an Engram table may stay in
        host memory), None when it has none (CPU, or weights offloaded by accelerate hooks)."""
        cached = module.__dict__.get("_oellm_device")
        if cached is not None:
            return cached
        for t in module.parameters():
            if t.device.type == "cuda":
                module.__dict__["_oellm_device"] = t.device
                return t.device
        return None

    def _embed(self, input_ids, inputs_embeds, token_mask):
        x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if self.embed_scale != 1.0:
            x = x * self.embed_scale
        if self.embed_norm is not None:
            x = self.embed_norm(x)
        if token_mask is not None:
            x = x * token_mask[..., None].to(x.dtype)
        return x

    def _enter(self, x):
        if self.ldm_boundary is not None:
            return self.ldm_boundary.enter(x)
        b, s, h = x.shape
        return x.unsqueeze(-2).expand(b, s, self.config.mhc_expansion_rate, h).contiguous()

    def _exit(self, streams):
        ldm = self.ldm_boundary is not None
        dev = self._device_of(self.norm)
        if dev is not None:
            streams = _to(streams, dev)
        return self.norm(streams[0] + streams[1] if ldm else streams.mean(dim=-2))

    def _stages(self):
        """Contiguous runs of layers on one CUDA device: [(device, first, end)]."""
        stages = []
        for i, layer in enumerate(self.layers):
            dev = self._device_of(layer)
            if stages and stages[-1][0] == dev:
                stages[-1][2] = i + 1
            else:
                stages.append([dev, i, i + 1])
        return stages

    def _run_layers(self, streams, first, end, input_ids, token_mask, attention_mask, cache, keep_state,
                    hidden=None):
        for i in range(first, end):
            layer = self.layers[i]
            dev = self._device_of(layer)
            if dev is not None and _device(streams) != dev:
                streams = _to(streams, dev)
                input_ids = input_ids.to(dev)
                token_mask = None if token_mask is None else token_mask.to(dev)
                attention_mask = None if attention_mask is None else attention_mask.to(dev)
            if hidden is not None:
                hidden.append(streams[0] if self.ldm_boundary is not None else streams.mean(dim=-2))
            st = cache.states[i] if cache is not None else None
            streams, new_state = layer(streams, st, input_ids, token_mask, attention_mask, keep_state)
            if cache is not None:
                cache.states[i] = new_state
        return streams

    def _pipelined(self, input_ids, inputs_embeds, token_mask, attention_mask, head):
        """Prefill without a cache, the batch split into micro-batches that flow through the
        per-device stages in a wavefront: with no host synchronization in the layers, the GPUs of
        a device map compute concurrently. Returns (hidden, head(hidden)) concatenated over rows,
        or None when it does not apply."""
        if os.environ.get("OELLM_HF_PIPELINE", "1") == "0":
            return None
        stages = self._stages()
        b = input_ids.shape[0]
        if len(stages) < 2 or b < 2 or any(d is None for d, _, _ in stages):
            return None
        per = int(os.environ.get("OELLM_HF_PIPELINE_CHUNKS", "0")) or 2 * len(stages)
        n = min(b, per)
        bounds = [round(i * b / n) for i in range(n + 1)]
        rows = [slice(bounds[i], bounds[i + 1]) for i in range(n)]

        def part(t, r):
            return None if t is None else t[r]

        carry = [None] * n
        outs = [None] * n
        for step in range(len(stages) + n - 1):
            for c in range(n):
                st = step - c
                if not 0 <= st < len(stages):
                    continue
                dev, first, end = stages[st]
                if st == 0:
                    ids = input_ids[rows[c]]
                    x = self._embed(ids, part(inputs_embeds, rows[c]), part(token_mask, rows[c]))
                    carry[c] = (self._enter(x), ids, part(token_mask, rows[c]), part(attention_mask, rows[c]))
                streams, ids, tm, am = carry[c]
                streams = self._run_layers(streams, first, end, ids, tm, am, None, False)
                carry[c] = (streams, ids, tm, am)
                if st == len(stages) - 1:
                    h = self._exit(streams)
                    outs[c] = (h, head(h) if head is not None else None)
                    carry[c] = None
        h = torch.cat([o[0] for o in outs])
        return h, (torch.cat([o[1] for o in outs]) if head is not None else None)

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, past_key_values=None,
                use_cache=None, output_hidden_states=None, _head=None, **kwargs) -> BaseModelOutputWithPast:
        """``_head``: applied to the final hidden states (per micro-batch when pipelined); its
        result is returned as ``logits`` of the output."""
        if input_ids is None:
            raise ValueError("the Engram layers need input_ids")
        if past_key_values is not None and not isinstance(past_key_values, OellmHybridCache):
            past_key_values = None
        # generate() always passes use_cache; a plain call (scoring) keeps no decoding state
        use_cache = past_key_values is not None if use_cache is None else use_cache
        if use_cache and past_key_values is None:
            past_key_values = OellmHybridCache(len(self.layers))
        dev = self._device_of(self.embed_tokens)
        if dev is not None and input_ids.device != dev:
            input_ids = input_ids.to(dev)
            attention_mask = None if attention_mask is None else attention_mask.to(dev)
        s = input_ids.shape[1]
        if attention_mask is not None and bool(attention_mask.all()):
            attention_mask = None
        token_mask = None if attention_mask is None else attention_mask[:, -s:]
        keep_state = past_key_values is not None
        if not keep_state and not output_hidden_states and _fast(self.embed_tokens.weight):
            done = self._pipelined(input_ids, inputs_embeds, token_mask, attention_mask, _head)
            if done is not None:
                out = BaseModelOutputWithPast(last_hidden_state=done[0], past_key_values=None, hidden_states=None)
                out["logits"] = done[1]
                return out
        x = self._embed(input_ids, inputs_embeds, token_mask)
        streams = self._enter(x)
        hidden = [] if output_hidden_states else None
        streams = self._run_layers(streams, 0, len(self.layers), input_ids, token_mask, attention_mask,
                                   past_key_values, keep_state, hidden)
        h = self._exit(streams)
        if past_key_values is not None:
            past_key_values.seen_tokens += s
        if output_hidden_states:
            hidden.append(h)
        out = BaseModelOutputWithPast(last_hidden_state=h, past_key_values=past_key_values,
                                      hidden_states=tuple(hidden) if output_hidden_states else None)
        if _head is not None:
            out["logits"] = _head(h)
        return out


def _device(streams):
    return (streams[0] if isinstance(streams, tuple) else streams).device


def _to(streams, dev):
    if isinstance(streams, tuple):
        return tuple(t.to(dev) for t in streams)
    return streams.to(dev)


def _parse_bytes(v) -> int:
    if isinstance(v, (int, float)):
        return int(v)
    v = str(v).strip().upper()
    for unit, mult in (("TIB", 2**40), ("GIB", 2**30), ("MIB", 2**20), ("TB", 10**12), ("GB", 10**9), ("MB", 10**6)):
        if v.endswith(unit):
            return int(float(v[: -len(unit)]) * mult)
    return int(float(v))


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

    # ----------------------------------------------------------------------- placement
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """``device_map="auto"`` (or "balanced", "sequential") on CUDA: load into host memory and
        place whole layers on the visible GPUs (``place_on_gpus``) with the Engram tables pinned in
        host memory. accelerate's own map cannot hold an 80 GB table layer on one GPU and offloads
        it (weights copied per forward). ``OELLM_HF_PLACEMENT=accelerate``: the stock path."""
        dm = kwargs.get("device_map")
        auto = isinstance(dm, str) and dm in ("auto", "balanced", "balanced_low_0", "sequential")
        if not (auto and torch.cuda.is_available() and os.environ.get("OELLM_HF_PLACEMENT", "oellm") == "oellm"):
            return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        kwargs.pop("device_map")
        max_memory = kwargs.pop("max_memory", None)
        for k in ("offload_folder", "offload_state_dict", "offload_buffers"):
            kwargs.pop(k, None)
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        model.place_on_gpus(max_memory=max_memory)
        return model

    def place_on_gpus(self, max_memory: dict | None = None, gpus: list[int] | None = None):
        """Whole layers on ``gpus`` (default: the visible GPUs, ``OELLM_HF_GPUS`` = how many), in
        order and balanced by weight bytes; the embedding on the first, the final norm and the head
        on the last; Engram tables pinned in host memory (zero-copy gather). Returns the map."""
        if gpus is None:
            count = torch.cuda.device_count()
            want = int(os.environ.get("OELLM_HF_GPUS", "0")) or count
            gpus = list(range(min(want, count)))
        tables = []
        for mod in self.modules():
            if isinstance(mod, OellmEngramV2):
                if not mod.pin_table():  # an fp8 table: no zero-copy gather; it goes with its layer
                    continue
                tables += [t for t in (mod.table, getattr(mod, "table_scale", None)) if t is not None]
        skip = {id(t) for t in tables}

        def size(m):
            return sum(t.numel() * t.element_size() for t in [*m.parameters(), *m.buffers()] if id(t) not in skip)

        base = self.model
        head = [base.norm] + ([] if self.config.tie_word_embeddings else [self.lm_head])
        front = [base.embed_tokens] + [m for m in (base.embed_norm, base.ldm_boundary) if m is not None]
        sizes = [size(layer) for layer in base.layers]
        total = sum(sizes) + sum(size(m) for m in front + head)
        n = len(gpus)
        caps = []
        for g in gpus:
            free, whole = torch.cuda.mem_get_info(g)
            cap = whole * 0.9
            if max_memory is not None and (g in max_memory or str(g) in max_memory):
                cap = min(cap, _parse_bytes(max_memory.get(g, max_memory.get(str(g)))))
            caps.append(cap)
        used = [0.0] * n
        used[0] += sum(size(m) for m in front)
        used[-1] += sum(size(m) for m in head)
        target = total / n
        gpu, plan = 0, []
        for i, b in enumerate(sizes):
            while gpu < n - 1 and (used[gpu] + b > caps[gpu] or (used[gpu] + b / 2 > target * 1.0 and used[gpu] > 0)):
                gpu += 1
            if used[gpu] + b > caps[gpu]:
                raise RuntimeError(f"layer {i}: the model does not fit {n} GPUs ({[round(c / 2**30) for c in caps]} GiB)")
            plan.append(gpu)
            used[gpu] += b
        dmap = {}
        for m, name in zip(front, ["embed_tokens", "embed_norm", "ldm_boundary"][: len(front)]):
            m.to(f"cuda:{gpus[0]}")
            dmap[f"model.{name}"] = gpus[0]
        for i, (layer, g) in enumerate(zip(base.layers, plan)):
            layer.to(f"cuda:{gpus[g]}")
            layer.__dict__.pop("_oellm_device", None)
            dmap[f"model.layers.{i}"] = gpus[g]
        for m in head:
            m.to(f"cuda:{gpus[-1]}")
        dmap["model.norm"] = dmap["lm_head"] = gpus[-1]
        for m in [base.embed_tokens, base.norm, *front]:
            m.__dict__.pop("_oellm_device", None)
        self.__dict__["oellm_device_map"] = dmap
        print("[oellm] weights per GPU (GiB):", [round(u / 2**30, 1) for u in used],
              f"; {len(tables)} Engram tensors pinned in host memory", flush=True)
        return dmap

    def forward(self, input_ids=None, attention_mask=None, past_key_values=None, labels=None,
                use_cache=None, output_hidden_states=None, **kwargs) -> CausalLMOutputWithPast:
        # eval-mode calls without labels need no autograd graph (lighteval scores with grad enabled);
        # OELLM_HF_GRAD=1 keeps it
        no_grad = not self.training and labels is None and os.environ.get("OELLM_HF_GRAD", "0") != "1"
        with torch.no_grad() if no_grad else contextlib.nullcontext():
            in_dev = input_ids.device if input_ids is not None else None

            def head(h):
                dev = self.model._device_of(self.lm_head)
                return self.lm_head(h if dev is None else h.to(dev))

            out = self.model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values,
                             use_cache=use_cache, output_hidden_states=output_hidden_states, _head=head)
            logits = out["logits"]
            if in_dev is not None and logits.device != in_dev:  # like accelerate's io_same_device
                logits = logits.to(in_dev)
            loss = None
            if labels is not None:
                loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                       labels[:, 1:].reshape(-1).to(logits.device), ignore_index=-100)
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

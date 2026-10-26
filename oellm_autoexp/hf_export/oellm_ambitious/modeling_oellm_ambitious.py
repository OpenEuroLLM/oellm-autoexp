"""HF port of the ambitious hybrid main run, a transcription of the Megatron modules it trains with.

* residual: mHC (``megatron.core.transformer.mhc.MHCMixer``) over n streams. The embedding is copied
  into n streams, every layer reads its input ``h_pre . streams`` and writes back
  ``h_res @ streams + h_post (x) out``, and the trunk ends with the stream mean.
* layers, one per Megatron pattern symbol (``block_types``):
  - "attention" ``*``: NoPE softmax attention with qk-norm, exclusive self attention (XSA,
    ``fused_xsa.xsa_reference``) and the sigmoid output gate, behind ``linear_qkv``'s fused norm;
  - "ckda" ``L``: the FLA ComplexKimiDeltaAttention layer, as Megatron's ComplexKDA builds it;
  - "moe" ``E``: Megatron's sigmoid router with expert bias (top-k on scores + bias, weights =
    the unbiased scores normalised to 1), SwiGLU experts with the gpt-oss / DeepSeek-V4.1 clamp,
    an ungated shared expert;
  - "engram" ``N``: ``EngramLayer`` with the signed-sqrt gate, the conv pre-norm and per-layer
    hash multipliers.

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
def _swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float | None) -> torch.Tensor:
    if limit is not None:  # fused_bias_swiglu.clamped_swiglu
        gate, up = gate.clamp(max=limit), up.clamp(min=-limit, max=limit)
    return F.silu(gate) * up


class OellmMoE(nn.Module):
    def __init__(self, config: OellmAmbitiousConfig) -> None:
        super().__init__()
        h, e, i = config.hidden_size, config.num_experts, config.moe_intermediate_size
        self.top_k, self.num_experts, self.scaling = config.num_experts_per_tok, e, config.router_scaling_factor
        self.limit = config.activation_clamp_value
        self.gate = nn.Parameter(torch.zeros(e, h))  # router weight
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
        _, idx = torch.topk(scores + self.e_score_correction_bias.float(), self.top_k, dim=-1)
        w = scores.gather(-1, idx)
        if self.top_k > 1:
            w = w / (w.sum(-1, keepdim=True) + 1e-20)
        w = (w * self.scaling).to(t.dtype)
        out = torch.zeros_like(t)
        for e in torch.unique(idx).tolist():
            tok, slot = torch.where(idx == e)
            g, u = F.linear(t[tok], self.gate_up_proj[e]).chunk(2, dim=-1)
            y = F.linear(_swiglu(g, u, self.limit), self.down_proj[e])
            out.index_add_(0, tok, (y * w[tok, slot, None]).to(out.dtype))
        if self.shared_gate_proj is not None:
            out = out + self.shared_down_proj(_swiglu(self.shared_gate_proj(t), self.shared_up_proj(t), self.limit))
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
        e = self._lookup(self._rows(shifted, s), hidden_states.dtype).to(hidden_states.device).reshape(b, s, -1)
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

    def forward(self, streams, state=None, input_ids=None, token_mask=None, attention_mask=None):
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
        streams = x.unsqueeze(-2).expand(b, s, self.config.mhc_expansion_rate, x.shape[-1]).contiguous()
        hidden = []
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                hidden.append(streams.mean(dim=-2))
            st = past_key_values.states[i] if past_key_values is not None else None
            streams, new_state = layer(streams, st, input_ids, token_mask, attention_mask)
            if past_key_values is not None:
                past_key_values.states[i] = new_state
        h = self.norm(streams.mean(dim=-2))
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

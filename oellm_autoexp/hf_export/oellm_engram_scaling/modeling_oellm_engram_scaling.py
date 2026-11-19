"""HF port of the Engram scaling runs (Megatron HybridModel), a transcription of the modules they train with.

One HF layer per Megatron pattern symbol (``block_types``), each a plain pre-norm residual block
``x = x + mixer(norm(x))`` (Engram: ``x = x + engram(x)``, it normalises internally):

* "gdn" ``L``: ``GatedDeltaNet`` behind ``in_proj``'s fused norm (``oellm_hybrid.OellmGatedDeltaNet``);
* "attention" ``*``: softmax attention with qk-norm, partial RoPE (``rotary_dim`` of ``head_dim``
  channels; ``rotary_interleaved``: Megatron's interleaved pairs) and the sigmoid output gate, behind
  ``linear_qkv``'s fused norm;
* "moe" ``E``: Megatron's sigmoid router with the expert bias (top-k on scores + bias, weights = the
  unbiased scores normalised to 1), SwiGLU experts, a shared expert scaled by ``sigmoid(x . w_gate)``
  (``moe_shared_expert_gate``);
* "engram" ``N``: ``EngramLayer`` with per-layer hashes, the signed-sqrt gate and the conv pre-norm
  (``oellm_ambitious.OellmEngramV2``).

No decoding state is kept (every call recomputes the full sequence): enough for scoring, slow for
generation. Padding: right padding (scoring) is exact; with left padding the GDN layers, which take no
mask, see the pad tokens.
"""

from __future__ import annotations

import contextlib
import os

import torch
import torch.nn.functional as F
from torch import nn
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .configuration_oellm_engram_scaling import OellmEngramScalingConfig
# imported here too: the HF dynamic loader copies only the direct relative imports of this file
from .configuration_oellm_ambitious import OellmAmbitiousConfig  # noqa: F401
from .configuration_oellm_hybrid import OellmHybridConfig  # noqa: F401
from .modeling_oellm_ambitious import OellmEngramV2, OellmMoE, _swiglu
from .modeling_oellm_hybrid import OellmGatedDeltaNet, OellmRMSNorm


# --------------------------------------------------------------------------- attention
def _apply_rope(t: torch.Tensor, rotary_dim: int, theta: float, interleaved: bool) -> torch.Tensor:
    """HybridModel's RoPE on the first ``rotary_dim`` channels of ``t`` [b, s, heads, d] (positions
    0..s-1), in fp32 like TE's fused kernel.

    HybridModel builds its ``RotaryEmbedding`` without ``rotary_interleaved`` (GPTModel passes it), so
    the frequencies always come in the halves layout ``cat(f, f)`` (channel c: f[c mod rotary_dim/2]),
    while the attention rotates with ``config.rotary_interleaved``: pairs (x0, x1) -> (-x1, x0) when
    set, else the halves. With ``rotary_interleaved`` this is not textbook interleaved RoPE, but it is
    what these runs trained with (checked against the Megatron forward per attention layer)."""
    s = t.shape[1]
    inv = 1.0 / theta ** (torch.arange(0, rotary_dim, 2, device=t.device, dtype=torch.float32) / rotary_dim)
    freqs = torch.outer(torch.arange(s, device=t.device, dtype=torch.float32), inv)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos, sin = emb.cos()[None, :, None, :], emb.sin()[None, :, None, :]
    rot, rest = t[..., :rotary_dim].float(), t[..., rotary_dim:]
    if interleaved:
        half = torch.stack([-rot[..., 1::2], rot[..., 0::2]], dim=-1).flatten(-2)
    else:
        x1, x2 = rot.chunk(2, dim=-1)
        half = torch.cat([-x2, x1], dim=-1)
    return torch.cat([(rot * cos + half * sin).to(t.dtype), rest], dim=-1)


class OellmRopeAttention(nn.Module):
    """Megatron ``SelfAttention``: q/k RMSNorm over the head dim, then RoPE, causal GQA softmax attention,
    ``o * sigmoid(gate)`` in fp32 (``_sigmoid_output_gate``), then ``linear_proj``."""

    def __init__(self, config: OellmEngramScalingConfig) -> None:
        super().__init__()
        h, d = config.hidden_size, config.head_dim
        self.nh, self.nkv, self.d = config.num_attention_heads, config.num_key_value_heads, d
        self.rotary_dim, self.theta = config.rotary_dim, config.rope_theta
        self.interleaved = config.rotary_interleaved
        self.q_proj = nn.Linear(h, self.nh * d, bias=False)
        self.gate_proj = nn.Linear(h, self.nh * d, bias=False) if config.attention_output_gate else None
        self.k_proj = nn.Linear(h, self.nkv * d, bias=False)
        self.v_proj = nn.Linear(h, self.nkv * d, bias=False)
        self.o_proj = nn.Linear(self.nh * d, h, bias=False)
        zcg = config.layernorm_zero_centered_gamma
        self.q_norm = OellmRMSNorm(d, config.rms_norm_eps, zcg) if config.qk_layernorm else None
        self.k_norm = OellmRMSNorm(d, config.rms_norm_eps, zcg) if config.qk_layernorm else None

    def forward(self, x: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.nh, self.d)
        k = self.k_proj(x).view(b, s, self.nkv, self.d)
        v = self.v_proj(x).view(b, s, self.nkv, self.d)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q = _apply_rope(q, self.rotary_dim, self.theta, self.interleaved)
        k = _apply_rope(k, self.rotary_dim, self.theta, self.interleaved)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        gqa = self.nh != self.nkv
        if token_mask is None:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.d**-0.5, enable_gqa=gqa)
        else:  # causal and the key padding; the diagonal stays visible (no all-masked row)
            pos = torch.arange(s, device=x.device)
            causal = pos[None, :] <= pos[:, None]
            mask = (causal[None] & token_mask[:, None, :].bool()) | torch.eye(s, dtype=torch.bool, device=x.device)
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None], scale=self.d**-0.5, enable_gqa=gqa)
        o = o.transpose(1, 2).reshape(b, s, self.nh * self.d)
        if self.gate_proj is not None:
            o = (o.float() * torch.sigmoid(self.gate_proj(x).float())).to(o.dtype)
        return self.o_proj(o)


# --------------------------------------------------------------------------- MoE
class OellmGatedSharedMoE(OellmMoE):
    """``MoELayer`` with the sigmoid router + expert bias and the gated shared expert
    (``SharedExpertMLP``: ``shared(x) * sigmoid(x . gate_weight)``). Experts: the reference loop, or
    ``OELLM_HF_MOE=grouped`` (``torch._grouped_mm``, CUDA bf16)."""

    def __init__(self, config: OellmEngramScalingConfig) -> None:
        super().__init__(config)
        gated = config.shared_expert_gate and config.shared_expert_intermediate_size
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False) if gated else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        t = x.reshape(-1, shape[-1])
        scores = torch.sigmoid(F.linear(t.float(), self.gate.float()))
        _, idx = torch.topk(scores + self.e_score_correction_bias.float(), self.top_k, dim=-1)
        w = scores.gather(-1, idx)
        if self.top_k > 1:
            w = w / (w.sum(-1, keepdim=True) + 1e-20)
        w = (w * self.scaling).to(t.dtype)
        if self.impl == "grouped" and t.is_cuda:
            out = self._grouped_experts(t, idx, w)
        else:
            out = torch.zeros_like(t)
            for e in torch.unique(idx).tolist():
                tok, slot = torch.where(idx == e)
                g, u = F.linear(t[tok], self.gate_up_proj[e]).chunk(2, dim=-1)
                y = F.linear(_swiglu(g, u, None), self.down_proj[e])
                out.index_add_(0, tok, (y * w[tok, slot, None]).to(out.dtype))
        if self.shared_gate_proj is not None:
            shared = self.shared_down_proj(_swiglu(self.shared_gate_proj(t), self.shared_up_proj(t), None))
            if self.shared_expert_gate is not None:
                shared = shared * torch.sigmoid(self.shared_expert_gate(t))
            out = out + shared
        return out.view(shape)


# --------------------------------------------------------------------------- layers / model
class OellmEngramScalingLayer(nn.Module):
    def __init__(self, config: OellmEngramScalingConfig, layer_idx: int) -> None:
        super().__init__()
        self.kind = config.block_types[layer_idx]
        if self.kind != "engram":  # Engram normalises internally
            self.input_layernorm = OellmRMSNorm(config.hidden_size, config.rms_norm_eps,
                                                config.layernorm_zero_centered_gamma)
        if self.kind == "gdn":
            self.mixer = OellmGatedDeltaNet(config)
        elif self.kind == "attention":
            self.mixer = OellmRopeAttention(config)
        elif self.kind == "moe":
            self.mixer = OellmGatedSharedMoE(config)
        else:
            self.mixer = OellmEngramV2(config, layer_idx)

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor, token_mask: torch.Tensor | None) -> torch.Tensor:
        if self.kind == "engram":
            return x + self.mixer(x, input_ids, token_mask)[0]
        xin = self.input_layernorm(x)
        if self.kind == "gdn":
            out = self.mixer(xin)[0]
        elif self.kind == "attention":
            out = self.mixer(xin, token_mask)
        else:
            out = self.mixer(xin)
        return x + out


class OellmEngramScalingPreTrainedModel(PreTrainedModel):
    config_class = OellmEngramScalingConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["OellmEngramScalingLayer"]
    _supports_sdpa = True

    def _init_weights(self, module: nn.Module) -> None:
        # every tensor is loaded from the export; nothing to initialise
        return


class OellmEngramScalingModel(OellmEngramScalingPreTrainedModel):
    def __init__(self, config: OellmEngramScalingConfig) -> None:
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(OellmEngramScalingLayer(config, i) for i in range(config.num_hidden_layers))
        self.norm = OellmRMSNorm(config.hidden_size, config.rms_norm_eps, config.layernorm_zero_centered_gamma)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=None, **kwargs):
        if input_ids is None:
            raise ValueError("the Engram layers need input_ids")
        if attention_mask is not None and bool(attention_mask.all()):
            attention_mask = None
        token_mask = attention_mask
        x = self.embed_tokens(input_ids)
        if token_mask is not None:
            x = x * token_mask[..., None].to(x.dtype)
        hidden = [] if output_hidden_states else None
        for layer in self.layers:
            if hidden is not None:
                hidden.append(x)
            x = layer(x, input_ids, token_mask)
        x = self.norm(x)
        if hidden is not None:
            hidden.append(x)
        return BaseModelOutputWithPast(last_hidden_state=x, hidden_states=tuple(hidden) if hidden is not None else None)


class OellmEngramScalingForCausalLM(OellmEngramScalingPreTrainedModel, GenerationMixin):
    def __init__(self, config: OellmEngramScalingConfig) -> None:
        super().__init__(config)
        self.model = OellmEngramScalingModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids=None, attention_mask=None, labels=None, output_hidden_states=None,
                **kwargs) -> CausalLMOutputWithPast:
        # eval-mode calls without labels need no autograd graph (lighteval scores with grad enabled)
        no_grad = not self.training and labels is None and os.environ.get("OELLM_HF_GRAD", "0") != "1"
        with torch.no_grad() if no_grad else contextlib.nullcontext():
            out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             output_hidden_states=output_hidden_states)
            logits = self.lm_head(out.last_hidden_state)
            loss = None
            if labels is not None:
                loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                       labels[:, 1:].reshape(-1), ignore_index=-100)
        return CausalLMOutputWithPast(loss=loss, logits=logits, hidden_states=out.hidden_states)

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        # no decoding state: every step scores the whole sequence
        return {"input_ids": input_ids, "attention_mask": attention_mask}

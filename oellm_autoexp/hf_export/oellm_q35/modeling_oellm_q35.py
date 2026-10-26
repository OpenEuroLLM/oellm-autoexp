"""Model of the conservative main run: stock Qwen3-Next, the router replaced by Megatron's
(``megatron.core.transformer.moe.moe_utils.topk_routing_with_score_function`` with score_function
"sigmoid" and an expert bias): scores = sigmoid(logits) in fp32; the top-k experts by
scores + expert_bias; their unbiased scores, normalised to sum 1 (top-k > 1), times the scaling
factor. Everything else -- gated attention, Gated DeltaNet, the fused experts, the gated shared
expert, zero-centred norms, partial RoPE -- is transformers' own Qwen3-Next."""

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextForCausalLM,
    Qwen3NextSparseMoeBlock,
)

from .configuration_oellm_q35 import OellmQ35Config


class OellmSigmoidTopKRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.router_score_function != "sigmoid":
            raise ValueError(f"unsupported router_score_function {config.router_score_function!r}")
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.scaling = config.router_scaling_factor
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))
        self.register_buffer("e_score_correction_bias", torch.zeros(self.num_experts, dtype=torch.float32))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        # Megatron: router weight and logits in fp32 (moe_router_dtype fp32)
        router_logits = F.linear(hidden_states.float(), self.weight.float())
        scores = torch.sigmoid(router_logits)
        _, router_indices = torch.topk(scores + self.e_score_correction_bias.float(), self.top_k, dim=-1)
        weights = scores.gather(-1, router_indices)
        if self.top_k > 1:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.scaling
        return router_logits, weights.to(hidden_states.dtype), router_indices


class OellmQ35ForCausalLM(Qwen3NextForCausalLM):
    config_class = OellmQ35Config

    def __init__(self, config):
        super().__init__(config)
        for layer in self.model.layers:
            if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
                layer.mlp.gate = OellmSigmoidTopKRouter(config)
        self.post_init()

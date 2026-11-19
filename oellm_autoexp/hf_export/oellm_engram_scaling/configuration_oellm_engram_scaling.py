"""Configuration of the Engram scaling runs (Megatron HybridModel, ``torch_dist`` checkpoints).

One HF layer per Megatron pattern symbol, ``block_types`` in {"gdn", "attention", "moe", "engram"}:
``L`` Gated DeltaNet, ``*`` gated softmax attention with partial (interleaved) RoPE, ``E`` sigmoid-routed
MoE with a gated shared expert, ``N`` Engram. Plain pre-norm residual (no mHC / LDM).

The MoE and Engram modules are the ambitious export's (``oellm_ambitious``) and read the attributes it
reads; the fixed values below are the ones this family trains with and the exporter checks.
"""

from transformers import PretrainedConfig

LAYER_TYPES = ("gdn", "attention", "moe", "engram")


class OellmEngramScalingConfig(PretrainedConfig):
    model_type = "oellm_engram_scaling"

    def __init__(
        self,
        vocab_size: int = 262144,
        hidden_size: int = 640,
        block_types: list | None = None,
        rms_norm_eps: float = 1e-6,
        layernorm_zero_centered_gamma: bool = True,
        # attention
        num_attention_heads: int = 10,
        num_key_value_heads: int = 2,
        head_dim: int = 128,
        qk_layernorm: bool = True,
        attention_output_gate: bool = True,
        rope_theta: float = 10_000_000.0,
        rotary_dim: int = 32,
        rotary_interleaved: bool = True,
        max_position_embeddings: int = 4096,
        # Gated DeltaNet
        linear_num_key_heads: int = 10,
        linear_num_value_heads: int = 20,
        linear_key_head_dim: int = 64,
        linear_value_head_dim: int = 64,
        linear_conv_kernel_dim: int = 4,
        # MoE
        num_experts: int = 40,
        num_experts_per_tok: int = 4,
        moe_intermediate_size: int = 320,
        shared_expert_intermediate_size: int = 320,
        shared_expert_gate: bool = True,
        # Engram
        engram_ngram_orders: list | None = None,
        engram_num_heads: int = 8,
        engram_embed_dim: int = 64,
        engram_conv_kernel_size: int = 4,
        engram_table_sizes: list | None = None,
        engram_hash_multipliers: dict | None = None,  # {layer index (str): [[...]]}
        engram_token_map: bool = False,
        engram_gate_signed_sqrt: bool = True,
        engram_conv_prenorm: bool = True,
        engram_table_dtype: str | None = None,
        tie_word_embeddings: bool = False,
        use_cache: bool = False,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
        **kwargs,
    ):
        block_types = list(block_types or [])
        bad = sorted(set(block_types) - set(LAYER_TYPES))
        if bad:
            raise ValueError(f"unknown block_types {bad}")
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.block_types = block_types
        self.num_hidden_layers = len(block_types)
        self.rms_norm_eps = rms_norm_eps
        self.layernorm_zero_centered_gamma = layernorm_zero_centered_gamma
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.qk_layernorm = qk_layernorm
        self.attention_output_gate = attention_output_gate
        self.rope_theta = rope_theta
        self.rotary_dim = rotary_dim
        self.rotary_interleaved = rotary_interleaved
        self.max_position_embeddings = max_position_embeddings
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.shared_expert_gate = shared_expert_gate
        self.engram_ngram_orders = list(engram_ngram_orders or [2, 3])
        self.engram_num_heads = engram_num_heads
        self.engram_embed_dim = engram_embed_dim
        self.engram_conv_kernel_size = engram_conv_kernel_size
        self.engram_table_sizes = engram_table_sizes
        self.engram_hash_multipliers = engram_hash_multipliers or {}
        self.engram_token_map = engram_token_map
        self.engram_gate_signed_sqrt = engram_gate_signed_sqrt
        self.engram_conv_prenorm = engram_conv_prenorm
        self.engram_table_dtype = engram_table_dtype
        # fixed in this family (read by the reused oellm_ambitious MoE / Engram modules)
        self.router_scaling_factor = 1.0
        self.activation_clamp_value = None
        self.router_balancing = "expert_bias"
        self.moe_expert_post_act_norm = False
        self.mlp_post_act_norm = False
        self.post_act_norm_floor = 0.0
        self.post_act_norm_smooth = False
        self.expert_weight_dtype = "bf16"
        self.engram_sign_hash = False
        self.use_cache = use_cache
        self.bos_token_id, self.eos_token_id, self.pad_token_id = bos_token_id, eos_token_id, pad_token_id
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

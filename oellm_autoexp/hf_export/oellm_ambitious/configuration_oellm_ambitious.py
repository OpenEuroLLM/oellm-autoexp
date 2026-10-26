"""Configuration of the ambitious hybrid main run (Megatron HybridModel, mHC-4).

Unlike ``oellm_hybrid`` (one HF layer = mixer + MLP), the layers here are Megatron's own: one per
pattern symbol, ``block_types`` in {"attention", "ckda", "moe", "engram"} (not ``layer_types``:
transformers validates that name against its own attention kinds), each wrapped by its mHC
read / write over ``mhc_expansion_rate`` residual streams.
"""

from transformers import PretrainedConfig

LAYER_TYPES = ("attention", "ckda", "moe", "engram")


class OellmAmbitiousConfig(PretrainedConfig):
    model_type = "oellm_ambitious"

    def __init__(
        self,
        vocab_size: int = 262144,
        hidden_size: int = 2048,
        block_types: list | None = None,
        rms_norm_eps: float = 1e-5,
        layernorm_zero_centered_gamma: bool = False,
        initializer_range: float = 0.02,
        # mHC
        mhc_expansion_rate: int = 4,
        mhc_sinkhorn_iters: int = 20,
        mhc_eps: float = 1e-5,
        # attention (NoPE, qk-norm, output gate, exclusive self attention)
        num_attention_heads: int = 16,
        num_key_value_heads: int = 16,
        head_dim: int = 128,
        qk_layernorm: bool = True,
        attention_output_gate: bool = True,
        exclusive_self_attention: bool = True,
        xsa_eps: float = 1e-6,
        # CKDA: the FLA layer spec ({"class", "kwargs"}) as Megatron built it
        fla_layer: dict | None = None,
        # MoE (sigmoid router + expert bias, SwiGLU experts, ungated shared expert)
        num_experts: int = 216,
        num_experts_per_tok: int = 5,
        moe_intermediate_size: int = 768,
        shared_expert_intermediate_size: int = 768,
        router_scaling_factor: float = 1.0,
        activation_clamp_value: float | None = 10.0,
        # Engram
        engram_ngram_orders: list | None = None,
        engram_num_heads: int = 8,
        engram_embed_dim: int = 64,
        engram_conv_kernel_size: int = 4,
        engram_table_sizes: list | None = None,
        engram_hash_multipliers: dict | None = None,  # {layer index (str): [[...]]}
        engram_token_map: bool = False,
        engram_gate_signed_sqrt: bool = False,
        engram_conv_prenorm: bool = False,
        engram_table_dtype: str | None = None,
        engram_scale_dtype: str = "fp32",
        tie_word_embeddings: bool = False,
        use_cache: bool = True,
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
        self.initializer_range = initializer_range
        self.mhc_expansion_rate = mhc_expansion_rate
        self.mhc_sinkhorn_iters = mhc_sinkhorn_iters
        self.mhc_eps = mhc_eps
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.qk_layernorm = qk_layernorm
        self.attention_output_gate = attention_output_gate
        self.exclusive_self_attention = exclusive_self_attention
        self.xsa_eps = xsa_eps
        self.fla_layer = fla_layer
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.router_scaling_factor = router_scaling_factor
        self.activation_clamp_value = activation_clamp_value
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
        self.engram_scale_dtype = engram_scale_dtype
        self.sliding_window = None
        self.use_cache = use_cache
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

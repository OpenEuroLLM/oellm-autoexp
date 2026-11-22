"""Configuration of the engram_scaling models (Megatron HybridModel, one HF layer per pattern symbol).

``block_types`` (one per Megatron layer, in {"gdn", "attention", "moe", "engram"}) is not called
``layer_types``: transformers validates that name against its own attention kinds. The values are
written by ``engram_scaling_clariden/hf_config.py`` from the training run's resolved config.
"""

from transformers import PretrainedConfig

BLOCK_TYPES = ("gdn", "attention", "moe", "engram")


class OellmEngramConfig(PretrainedConfig):
    model_type = "oellm_engram"

    def __init__(
        self,
        vocab_size: int = 262144,
        hidden_size: int = 640,
        num_hidden_layers: int | None = None,
        block_types: list | None = None,
        rms_norm_eps: float = 1e-6,
        layernorm_zero_centered_gamma: bool = True,
        initializer_range: float = 0.02,
        # gated attention
        num_attention_heads: int = 10,
        num_key_value_heads: int = 2,
        head_dim: int = 128,
        qk_layernorm: bool = True,
        attention_output_gate: bool = True,
        rope_theta: float = 10000000.0,
        partial_rotary_factor: float = 0.25,
        rotary_dim: int | None = None,
        rope_interleaved: bool = True,  # the rotation's pairing: (2i, 2i+1), else (i, i + rotary_dim / 2)
        rope_interleaved_freqs: bool = False,  # the frequency layout; HybridModel: never interleaved
        max_position_embeddings: int = 4096,
        # Gated DeltaNet
        linear_num_key_heads: int = 10,
        linear_num_value_heads: int = 20,
        linear_key_head_dim: int = 64,
        linear_value_head_dim: int = 64,
        linear_conv_kernel_dim: int = 4,
        # MoE
        num_experts: int = 96,
        num_experts_per_tok: int = 4,
        moe_intermediate_size: int = 320,
        shared_expert_intermediate_size: int = 320,
        shared_expert_gate: bool = True,
        router_score_function: str = "sigmoid",
        router_norm_topk_prob: bool = True,
        router_scaling_factor: float = 1.0,
        hidden_act: str = "silu",
        # Engram
        engram_layers: list | None = None,
        engram_ngram_orders: list | None = None,
        engram_num_heads: int = 8,
        engram_embed_dim: int = 64,
        engram_conv_kernel_size: int = 4,
        engram_conv_dilation: int | None = None,
        engram_table_sizes: list | None = None,
        engram_hash_multipliers: dict | None = None,  # {layer index (str): [[...]]}
        engram_gate_signed_sqrt: bool = False,
        engram_conv_prenorm: bool = False,
        engram_token_map: bool = False,
        engram_table_dtype: str = "bf16",  # "fp32": the table stays fp32 under a bf16 model
        tie_word_embeddings: bool = True,
        use_cache: bool = True,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
        **kwargs,
    ):
        block_types = list(block_types or [])
        bad = sorted(set(block_types) - set(BLOCK_TYPES))
        if bad:
            raise ValueError(f"unknown block_types {bad} (expected {BLOCK_TYPES})")
        if router_score_function != "sigmoid" or hidden_act != "silu":
            raise ValueError("expects the sigmoid router and SwiGLU experts")
        if engram_table_dtype not in ("bf16", "fp32"):
            raise ValueError(f"engram_table_dtype {engram_table_dtype!r}")
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.block_types = block_types
        self.num_hidden_layers = len(block_types) if num_hidden_layers is None else num_hidden_layers
        self.rms_norm_eps = rms_norm_eps
        self.layernorm_zero_centered_gamma = layernorm_zero_centered_gamma
        self.initializer_range = initializer_range
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.qk_layernorm = qk_layernorm
        self.attention_output_gate = attention_output_gate
        self.rope_theta = rope_theta
        self.partial_rotary_factor = partial_rotary_factor
        self.rotary_dim = int(head_dim * partial_rotary_factor) if rotary_dim is None else rotary_dim
        self.rope_interleaved = rope_interleaved
        self.rope_interleaved_freqs = rope_interleaved_freqs
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
        self.router_score_function = router_score_function
        self.router_norm_topk_prob = router_norm_topk_prob
        self.router_scaling_factor = router_scaling_factor
        self.hidden_act = hidden_act
        self.engram_layers = list(engram_layers or [])
        self.engram_ngram_orders = list(engram_ngram_orders or [2, 3])
        self.engram_num_heads = engram_num_heads
        self.engram_embed_dim = engram_embed_dim
        self.engram_conv_kernel_size = engram_conv_kernel_size
        self.engram_conv_dilation = (
            max(self.engram_ngram_orders) if engram_conv_dilation is None else engram_conv_dilation
        )
        self.engram_table_sizes = engram_table_sizes
        self.engram_hash_multipliers = engram_hash_multipliers or {}
        self.engram_gate_signed_sqrt = engram_gate_signed_sqrt
        self.engram_conv_prenorm = engram_conv_prenorm
        self.engram_token_map = engram_token_map
        self.engram_table_dtype = engram_table_dtype
        self.use_cache = use_cache
        # transformers 5: no token-id defaults in PretrainedConfig, each config declares its own
        self.bos_token_id, self.eos_token_id, self.pad_token_id = bos_token_id, eos_token_id, pad_token_id
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

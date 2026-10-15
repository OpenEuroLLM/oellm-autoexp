"""HF config for the OpenEuroLLM hybrid architecture-scaling models.

One config covers every variant of the multilingual architecture comparison
(``config/experiments/architecture_scaling_variants/multilingual``). All five
variants share the same Qwen3-style backbone — RMSNorm, SwiGLU MLP, QK-layernorm,
no biases, tied embeddings — and differ only in *which token mixer* sits in each
layer. That is expressed by :attr:`mixer_types`, one entry per layer:

===================  ==========================================================
``full_attention``   softmax attention, optional RoPE
``sliding_attention``softmax attention restricted to ``[i - sliding_window, i]``
``gdn``              GatedDeltaNet (``fla.ops.gated_delta_rule``)
``mlstm``            mLSTM / xLSTM (``mlstm_kernels.torch``)
``mamba2``           Mamba2 (``mamba_ssm.ops.triton.ssd_combined``)
``short_conv``       LFM2 gated short convolution (plain torch)
``engram``           Engram hashed n-gram memory (plain torch, no pre-norm)
``fla``              a complete FLA layer: kda / signed_kda / complex_kda
                     (``fla_layer``: its class and constructor kwargs)
``fla_attention``    FLA's gated NoPE attention, complex_kda's softmax layers
                     (``fla_attention_layer``)
``none``             no mixer: the layer is only an MLP
===================  ==========================================================

So e.g. the ``gdn7_nope`` variant (``linear_attention_freq=8``) becomes
``["gdn"]*7 + ["full_attention"] + ["gdn"]*7 + ["full_attention"]`` for 16
layers, and ``fullattn`` is ``["full_attention"] * num_hidden_layers``.

A Megatron ``HybridModel`` places mixers and MLPs as separate pattern symbols
(``L-L-*-``, ``C-C-*-``); :attr:`mlp_layers` (one bool per layer, default all
``True``) says which HF layers carry an MLP, so a mixer followed by an MLP is
one HF layer, and a mixer or MLP standing alone is a layer with ``mlp=False``
or mixer ``none``. The GPT-path models, every layer mixer + MLP, are unchanged.

The field names mirror the Megatron ``TransformerConfig`` knobs they were
trained with so that the conversion in ``convert_megatron_to_hf.py`` is a
transcription rather than a translation.
"""

from __future__ import annotations

from transformers.configuration_utils import PretrainedConfig

ATTENTION_LAYER_TYPES = ("full_attention", "sliding_attention")
LINEAR_LAYER_TYPES = ("gdn", "mlstm", "mamba2", "short_conv", "fla")
# Complete FLA layers, rebuilt from {"class": "fla.layers.<module>.<Class>", "kwargs": {...}}.
FLA_LAYER_TYPES = ("fla", "fla_attention")
ENGRAM = "engram"
NO_MIXER = "none"
ALL_LAYER_TYPES = ATTENTION_LAYER_TYPES + LINEAR_LAYER_TYPES + ("fla_attention", ENGRAM, NO_MIXER)


class OellmHybridConfig(PretrainedConfig):
    """Configuration for :class:`OellmHybridForCausalLM`."""

    model_type = "oellm_hybrid"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 262272,
        hidden_size: int = 512,
        intermediate_size: int = 1536,
        num_hidden_layers: int = 16,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        max_position_embeddings: int = 4096,
        rms_norm_eps: float = 1e-5,
        tie_word_embeddings: bool = True,
        # ---- attention layers ----
        # "rope" or "none" (NoPE). Applies to the softmax-attention layers only.
        position_embedding_type: str = "rope",
        rope_theta: float = 10000.0,
        # Megatron ``window_size=(left, right)`` with right=0; a query at position
        # i attends to j in [i - sliding_window, i], i.e. sliding_window + 1 keys.
        sliding_window: int | None = None,
        qk_layernorm: bool = True,
        # ---- Qwen3.5 / Qwen3-Next block options (gdn3_qwen35 arm) ----
        # Sigmoid output gate on the softmax layers: attn_out * sigmoid(gate),
        # with `gate` a fourth projection the same shape as q.
        attention_output_gate: bool = False,
        # Zero-centred RMSNorm: the stored gamma is (gamma - 1), so the norm
        # applies (1 + w) rather than w. Wrong setting = silently wrong outputs.
        layernorm_zero_centered_gamma: bool = False,
        # Partial RoPE: rotate only the first int(head_dim * factor) dims and
        # pass the rest through (Megatron's `rotary_percent`). 1.0 = full RoPE.
        partial_rotary_factor: float = 1.0,
        # ---- per-layer mixer assignment ----
        mixer_types: list[str] | None = None,
        mlp_layers: list[bool] | None = None,
        # ---- GatedDeltaNet / mLSTM shared head geometry ----
        linear_key_head_dim: int = 128,
        linear_value_head_dim: int = 128,
        linear_num_key_heads: int = 8,
        linear_num_value_heads: int = 8,
        linear_conv_kernel_dim: int = 4,
        # ---- mLSTM ----
        mlstm_chunk_size: int = 128,
        mlstm_gate_soft_cap: float | None = 15.0,
        mlstm_backend: str = "chunkwise--triton_xl_chunk",
        mlstm_conv1d: bool = False,
        # ---- Mamba2 ----
        mamba_state_dim: int = 128,
        mamba_head_dim: int = 64,
        mamba_num_groups: int = 8,
        mamba_num_heads: int | None = None,
        mamba_expand: int = 2,
        mamba_conv_kernel: int = 4,
        mamba_chunk_size: int = 128,
        # ---- short_conv (LFM2) ----
        short_conv_kernel_size: int = 3,
        short_conv_bias: bool = False,
        # ---- engram (Megatron `N`) ----
        # The hash geometry is part of the model: the converter copies the table
        # sizes (one prime per order and head) and the multipliers from Megatron.
        engram_ngram_orders: list[int] | None = None,
        engram_num_heads: int = 4,
        engram_embed_dim: int = 64,
        engram_conv_kernel_size: int = 4,
        engram_table_sizes: list[int] | None = None,
        engram_hash_multipliers: list[list[int]] | None = None,
        # Whether the weights carry a token -> canonical id map (tokenizer compression).
        engram_token_map: bool = False,
        # ---- complete FLA layers (kda / signed_kda / complex_kda) ----
        fla_layer: dict | None = None,
        fla_attention_layer: dict | None = None,
        # ---- misc ----
        initializer_range: float = 0.02,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps

        self.position_embedding_type = position_embedding_type
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window
        self.qk_layernorm = qk_layernorm
        self.attention_output_gate = attention_output_gate
        self.layernorm_zero_centered_gamma = layernorm_zero_centered_gamma
        self.partial_rotary_factor = partial_rotary_factor

        if mixer_types is None:
            mixer_types = ["full_attention"] * num_hidden_layers
        if len(mixer_types) != num_hidden_layers:
            raise ValueError(
                f"mixer_types has {len(mixer_types)} entries but num_hidden_layers="
                f"{num_hidden_layers}"
            )
        unknown = sorted(set(mixer_types) - set(ALL_LAYER_TYPES))
        if unknown:
            raise ValueError(f"Unknown mixer_types {unknown}; allowed: {list(ALL_LAYER_TYPES)}")
        if "sliding_attention" in mixer_types and sliding_window is None:
            raise ValueError("mixer_types contains 'sliding_attention' but sliding_window is None")
        self.mixer_types = list(mixer_types)
        if mlp_layers is None:
            mlp_layers = [True] * num_hidden_layers
        if len(mlp_layers) != num_hidden_layers:
            raise ValueError(
                f"mlp_layers has {len(mlp_layers)} entries but num_hidden_layers="
                f"{num_hidden_layers}"
            )
        empty = [
            i for i, (m, f) in enumerate(zip(mixer_types, mlp_layers)) if m == NO_MIXER and not f
        ]
        if empty:
            raise ValueError(f"layers {empty} have neither a mixer nor an MLP")
        self.mlp_layers = [bool(f) for f in mlp_layers]

        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_conv_kernel_dim = linear_conv_kernel_dim

        self.mlstm_chunk_size = mlstm_chunk_size
        self.mlstm_gate_soft_cap = mlstm_gate_soft_cap
        self.mlstm_backend = mlstm_backend
        self.mlstm_conv1d = mlstm_conv1d

        self.mamba_state_dim = mamba_state_dim
        self.mamba_head_dim = mamba_head_dim
        self.mamba_num_groups = mamba_num_groups
        self.mamba_num_heads = mamba_num_heads
        self.mamba_expand = mamba_expand
        self.mamba_conv_kernel = mamba_conv_kernel
        self.mamba_chunk_size = mamba_chunk_size

        self.short_conv_kernel_size = short_conv_kernel_size
        self.short_conv_bias = short_conv_bias

        self.engram_ngram_orders = list(engram_ngram_orders or [2, 3])
        self.engram_num_heads = engram_num_heads
        self.engram_embed_dim = engram_embed_dim
        self.engram_conv_kernel_size = engram_conv_kernel_size
        self.engram_table_sizes = engram_table_sizes
        self.engram_hash_multipliers = engram_hash_multipliers
        self.engram_token_map = engram_token_map
        if ENGRAM in mixer_types:
            lookups = len(self.engram_ngram_orders) * engram_num_heads
            if engram_table_sizes is None or len(engram_table_sizes) != lookups:
                raise ValueError(f"engram layers need engram_table_sizes ({lookups} entries)")
            if engram_hash_multipliers is None or len(engram_hash_multipliers) != lookups:
                raise ValueError(f"engram layers need engram_hash_multipliers ({lookups} rows)")

        self.fla_layer = fla_layer
        self.fla_attention_layer = fla_attention_layer
        for kind, spec in (("fla", fla_layer), ("fla_attention", fla_attention_layer)):
            if kind in mixer_types and not (spec and "class" in spec and "kwargs" in spec):
                raise ValueError(
                    f"{kind} layers need {kind}_layer = {{'class': ..., 'kwargs': ...}}"
                )

        self.initializer_range = initializer_range
        self.use_cache = use_cache

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    # -- derived Mamba2 geometry (mirrors MambaMixer.__init__) --------------

    @property
    def mamba_d_inner(self) -> int:
        if self.mamba_num_heads is not None:
            return self.mamba_num_heads * self.mamba_head_dim
        return self.mamba_expand * self.hidden_size

    @property
    def mamba_nheads(self) -> int:
        if self.mamba_num_heads is not None:
            return self.mamba_num_heads
        return self.mamba_d_inner // self.mamba_head_dim


__all__ = [
    "OellmHybridConfig",
    "ALL_LAYER_TYPES",
    "ATTENTION_LAYER_TYPES",
    "LINEAR_LAYER_TYPES",
    "ENGRAM",
    "FLA_LAYER_TYPES",
    "NO_MIXER",
]

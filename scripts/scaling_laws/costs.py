"""Cost model for the multilingual architecture-scaling sweep.

Maps each (architecture, size, tokens) checkpoint to:
  - N   : non-embedding parameter count (the scaling-law model size)
  - D   : training tokens seen
  - FLOPs      : Chinchilla 6ND (optionally + attention FLOPs)
  - GPU-hours  : D / (measured tok/s/GPU * 3600)

The architectures are NOT param-matched: every size shares the Qwen3 dense
geometry (hidden, ffn, layers), but the hybrid mixers are heavier than the
attention block they replace — +9-12% non-embedding params for gdn7/mlstm7 and
up to +23% for mamba2 at 0.2B/0.9B (expand-2 d_inner = 2*hidden, vs GDN's
heads*kv_channels). N is therefore the MEASURED per-(arch, size) count from
Megatron's "number of parameters" log line (`N_TOTAL_MEASURED`) by default;
`n_mode="nominal"` reproduces the old shared-geometry accounting. The GPU-hour
axis uses the per-(arch,size) MEASURED throughput.
"""
from __future__ import annotations

# Per-size Qwen3 geometry shared by every architecture (the mixer differs, so
# the parameter count does not — see N_TOTAL_MEASURED). h=hidden, f=ffn,
# L=layers, heads/qgroups for (G)QA.
SIZE_ARCH: dict[str, dict[str, int]] = {
    "0.1B": dict(h=512, f=1536, L=16, heads=8, qgroups=8),
    "0.2B": dict(h=768, f=2304, L=22, heads=8, qgroups=8),
    "0.4B": dict(h=1024, f=3072, L=28, heads=16, qgroups=8),
    "0.9B": dict(h=1536, f=4608, L=28, heads=16, qgroups=8),
}
KV_CHANNELS = 128
VOCAB = 262272          # padded oellm 256k tokenizer
SEQ_LEN = 4096
DECAY_FRACTION = 0.2    # WSD: cooldowns branch at 0.8*D and decay the last 20%

# Measured tok/s/GPU per (architecture, size) at production MBS (main_scaling_all
# aux.tok_s). The throughput invariant — independent of gbs/GA/nodes.
TOK_S: dict[str, dict[str, float]] = {
    "fullattn":      {"0.1B": 193000, "0.2B": 115000, "0.4B": 59000, "0.9B": 36000},
    "swa7_rope100k": {"0.1B": 206000, "0.2B": 124000, "0.4B": 67000, "0.9B": 38000},
    "gdn7_nope":     {"0.1B": 165000, "0.2B": 101000, "0.4B": 52000, "0.9B": 34000},
    # MEASURED from the gdn3 sweep stables (steady-state ms/iter over the last
    # 30 logged steps, converted as gbs*4096/(ms/1000)/(nodes*4)). Within 2-4%
    # of gdn7 everywhere: at seq 4096 the 3:1 vs 7:1 interleave barely moves
    # throughput, because the quadratic term is still cheap at this context.
    "gdn3_nope":     {"0.1B": 163455, "0.2B": 100825, "0.4B": 53232, "0.9B": 35235},
    # Qwen3.5-style 3:1 GDN, measured the SAME way from its own sweep stables
    # (median ms/iter over the last 30 logged steps, gbs*4096/(ms/1000)/(nodes*4);
    # the two ladders per size averaged). Costs 1-2% vs gdn3_nope everywhere —
    # the sigmoid output gate is a pointwise op on the 1-in-4 attention layers,
    # so it barely registers next to the GDN kernels.
    "gdn3_qwen35":   {"0.1B": 161584, "0.2B": 99340, "0.4B": 52015, "0.9B": 34611},
    "mlstm7_nope":   {"0.1B": 188000, "0.2B": 115000, "0.4B": 56000, "0.9B": 34000},
    "mamba2_nope":   {"0.1B": 172000, "0.2B": 94000,  "0.4B": 55000, "0.9B": 32000},
}

# Fraction of layers using full (quadratic) attention. SWA/linear variants keep
# full attention on every 8th layer (linear_attention_freq / window_attn_skip_freq 8).
FULL_ATTN_FRACTION: dict[str, float] = {
    "fullattn": 1.0,
    "swa7_rope100k": 1.0 / 8,   # 7/8 windowed, 1/8 full
    "gdn7_nope": 1.0 / 8,
    "gdn3_nope": 1.0 / 4,   # 3 linear : 1 full attention
    "gdn3_qwen35": 1.0 / 4,  # same interleave; the gate does not change the count
    "mlstm7_nope": 1.0 / 8,
    "mamba2_nope": 1.0 / 8,
}
SWA_WINDOW = 1024

# Total trainable params (tied embedding counted once) per (arch, size), from the
# "number of parameters on (tensor, pipeline) model parallel rank (0, 0)" line of
# every main_7to1 eval log (TP=PP=1, so rank 0 holds the whole model; identical
# across all tiers/budgets of an (arch, size)). Collected 2026-09-29.
N_TOTAL_MEASURED: dict[str, dict[str, int]] = {
    "fullattn":      {"0.1B": 205607424, "0.2B": 387456256, "0.4B": 709033984, "0.9B": 1261728256},
    "swa7_rope100k": {"0.1B": 205607424, "0.2B": 387456256, "0.4B": 709033984, "0.9B": 1261728256},
    "gdn7_nope":     {"0.1B": 213232608, "0.2B": 403674176, "0.4B": 762689184, "0.9B": 1342007456},
    "gdn3_nope":     {"0.1B": 212143296, "0.2B": 401241488, "0.4B": 754104352, "0.9B": 1329162784},
    "gdn3_qwen35":   {"0.1B": 214240448, "0.2B": 405173648, "0.4B": 768784416, "0.9B": 1351182880},
    "mlstm7_nope":   {"0.1B": 213073120, "0.2B": 403446336, "0.4B": 814756384, "0.9B": 1420289056},
    "mamba2_nope":   {"0.1B": 213288608, "0.2B": 427531936, "0.4B": 762841184, "0.9B": 1460893456},
}
N_MODES = ("measured", "nominal")


def n_embedding(size: str) -> int:
    """Tied token-embedding params."""
    return VOCAB * SIZE_ARCH[size]["h"]


def n_ne(size: str, arch: str | None = None, n_mode: str = "measured") -> int:
    """Non-embedding parameters — the scaling-law N.

    n_mode="measured" (default, needs `arch`): the arch's own Megatron count
    minus the tied embedding. Unknown archs raise instead of silently falling
    back — crediting a heavier hybrid with the dense count is exactly the
    confound this replaces (it read ~0.015-0.035 nats of pure parameter gain as
    architecture gain).
    n_mode="nominal": the shared dense Qwen3 reference (kv_channels 128), i.e.
    fullattn's own count up to the norm weights, for every arch.
    """
    if n_mode == "measured" and arch is not None:
        try:
            return N_TOTAL_MEASURED[arch][size] - n_embedding(size)
        except KeyError:
            raise KeyError(
                f"no measured param count for ({arch}, {size}) — add it to "
                f"costs.N_TOTAL_MEASURED from the eval log, or use n_mode='nominal'"
            ) from None
    if n_mode not in N_MODES:
        raise ValueError(f"n_mode must be one of {N_MODES}, got {n_mode!r}")
    a = SIZE_ARCH[size]
    d = KV_CHANNELS
    # attn: q(h*H*d)+o(H*d*h)+k(h*G*d)+v(h*G*d); mlp swiglu: 3*h*f
    per_layer = a["h"] * d * (2 * a["heads"] + 2 * a["qgroups"]) + 3 * a["h"] * a["f"]
    return per_layer * a["L"]


def n_total(size: str, arch: str | None = None, n_mode: str = "measured") -> int:
    """N_ne + tied token embedding."""
    return n_ne(size, arch, n_mode) + n_embedding(size)


def flops(n: float, d: float) -> float:
    """Chinchilla training FLOPs: 6 * N * D (matmul FLOPs; attention-agnostic)."""
    return 6.0 * n * d


def attention_flops(size: str, arch: str, d: float) -> float:
    """Additive self-attention FLOPs over D tokens for the full-attention layers.

    Per full-attn layer, ~ 12 * h * seq (QK^T + AV) FLOPs per token; windowed
    layers cap the context at SWA_WINDOW. Small vs 6ND but arch-dependent, so it
    is the term that makes 'FLOPs' distinguish full vs linear attention.
    """
    a = SIZE_ARCH[size]
    full_ctx = SEQ_LEN
    frac_full = FULL_ATTN_FRACTION[arch]
    per_tok_full = 12 * a["h"] * a["L"] * frac_full * full_ctx
    # non-full layers: windowed (swa) => SWA_WINDOW ctx; linear => ~O(1), ignore.
    win_ctx = SWA_WINDOW if arch == "swa7_rope100k" else 0
    per_tok_win = 12 * a["h"] * a["L"] * (1 - frac_full) * win_ctx
    return (per_tok_full + per_tok_win) * d


def total_flops(size: str, arch: str, d: float, mode: str = "6nd",
                n_mode: str = "measured") -> float:
    """Training FLOPs. mode='6nd' (Chinchilla) or 'full' (6ND + attention)."""
    c = flops(n_ne(size, arch, n_mode), d)
    if mode == "full":
        c += attention_flops(size, arch, d)
    return c


def gpu_hours(d: float, arch: str, size: str) -> float:
    """GPU-hours to train D tokens at the measured per-GPU throughput."""
    return d / (TOK_S[arch][size] * 3600.0)


def throughput_powerlaw(arch: str, n_mode: str = "measured") -> tuple[float, float]:
    """Fit tok/s/GPU ~ s0 * N**(-sigma) across the measured sizes. Returns (s0, sigma).

    sigma comes out near 0.68, NOT 1.0: doubling N does not halve throughput,
    because bigger models use the GPU better. So GPU-hours ~ D * N**0.68 while
    FLOPs ~ D * N — the hardware makes large models cheaper than 6ND accounting
    says, and the GPU-hour-optimal model is correspondingly bigger than the
    FLOP-optimal one. Fits every architecture to within ~10%.
    """
    import math
    sizes = list(TOK_S[arch])
    x = [math.log(n_ne(s, arch, n_mode)) for s in sizes]
    y = [math.log(TOK_S[arch][s]) for s in sizes]
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    slope = sxy / sxx
    return math.exp(my - slope * mx), -slope


def tok_s_ratio(arch: str, size: str, baseline: str = "fullattn") -> float:
    """Throughput of `arch` relative to `baseline` (>1 = arch is faster per token).

    At SEQ_LEN 4096 every linear/gated variant is SLOWER than full attention
    (0.82-0.97x) — the quadratic term is still cheap at this context length, so
    the hybrids pay their extra state/gating cost without collecting the
    subquadratic win. Any GPU-hour advantage they show here therefore comes
    purely from loss quality, and is a floor on what longer contexts would give.
    """
    return TOK_S[arch][size] / TOK_S[baseline][size]


def tokens_from_iter(iteration: int, gbs: int) -> float:
    """Tokens seen at a given iteration."""
    return float(iteration) * gbs * SEQ_LEN

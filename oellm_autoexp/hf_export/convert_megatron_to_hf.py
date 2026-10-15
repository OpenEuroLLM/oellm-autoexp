"""Convert a Megatron ``torch_dist`` checkpoint of the multilingual
architecture-scaling runs into a self-contained HuggingFace model directory.

The checkpoints are PyTorch DCP, so the full (TP/PP/DP-gathered) tensors can be
read without a Megatron runtime, without CUDA and from a single process — the
per-tensor global shapes live in the checkpoint metadata.

Usage::

    python -m oellm_autoexp.hf_export.convert_megatron_to_hf \\
        --run-dir  $OUTPUT_DIR/<group>/qwen3_gdn7_nope_0.1B_ne_lr0.002_gbsz128_firstcd_decay20BT \\
        --out-dir  $OUTPUT_DIR/<group>/hf/qwen3_gdn7_nope_..._decay20BT \\
        --tokenizer /e/data1/datasets/products/openeurollm/tokenizers/tokenizer-256k

``--run-dir`` must contain ``current.yaml`` (the resolved oellm-autoexp config,
the authoritative source for the architecture) plus the ``iter_NNNNNNN``
directories. By default the iteration in ``latest_checkpointed_iteration.txt``
is converted; override with ``--iter``.

The output directory is loadable with ``AutoModelForCausalLM.from_pretrained(...,
trust_remote_code=True)``: the two ``oellm_hybrid`` modules are copied in next to
the weights and referenced from ``config.json``'s ``auto_map``.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import torch
import yaml

LOGGER = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_PKG = _HERE / "oellm_hybrid"

# Megatron ``in_proj``/``conv1d`` fused tensors are stored split into named
# sub-tensors by the sharded-state-dict factories. These are the concatenation
# orders the forward passes assume.
FUSED_SECTIONS: dict[str, dict[str, list[str]]] = {
    "gdn": {
        "in_proj.weight": ["query", "key", "value", "z", "beta", "alpha"],
        "conv1d.weight": ["query", "key", "value"],
    },
    "mlstm": {
        "in_proj.weight": ["query", "key", "value", "ogate", "igate", "fgate"],
        "conv1d.weight": ["query", "key", "value"],
        "conv1d.bias": ["query", "key", "value"],
    },
    "mamba2": {
        "in_proj.weight": ["z", "x", "B", "C", "dt"],
        "conv1d.weight": ["x", "B", "C"],
        "conv1d.bias": ["x", "B", "C"],
    },
    "short_conv": {
        "in_proj.weight": ["B", "C", "x"],
    },
}

# experimental_attention_variant -> HF mixer type (GPT path and HybridModel `L`).
# kda / signed_kda / complex_kda are complete FLA layers (megatron/core/ssm/fla_hybrid.py,
# complex_kda.py): the HF model builds the same FLA class and loads its weights as is.
_VARIANT_MIXERS = {
    "gated_delta_net": "gdn",
    "mlstm": "mlstm",
    "mamba": "mamba2",
    "kda": "fla",
    "signed_kda": "fla",
    "complex_kda": "fla",
}

# TransformerConfig defaults of the fields the FLA layers are built from, for a
# current.yaml that predates a field.
_FLA_DEFAULTS = {
    "linear_fla_expand_v": 1.0,
    "linear_fla_allow_neg_eigval": True,
    "linear_fla_lower_bound": -5.0,
    "linear_fla_gate": "signed_sigmoid2",
    "linear_fla_output_gate": "linear",
    "linear_fla_gate_init_style": "spread",
    "linear_fla_drop_silu": True,
    "linear_gate_activation": "signed_sigmoid2",
    "linear_gate_init_style": "shipped",
    "linear_gate_lower_bound": -5.0,
    "linear_output_gate": "lowrank",
    "linear_drop_qkv_silu": False,
    "linear_beta_max": 1.0,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 16,
}


def _fla_layer_spec(mcfg: dict[str, Any]) -> dict[str, Any]:
    """The FLA class and constructor kwargs of a kda / signed_kda / complex_kda
    layer, as the Megatron wrappers build it (``layer_idx`` is set by the HF
    model)."""

    def get(key):
        value = mcfg.get(key)
        return _FLA_DEFAULTS[key] if value in (None, "None") else value

    variant = mcfg["experimental_attention_variant"]
    eps = float(mcfg["layernorm_epsilon"])
    conv = int(get("linear_conv_kernel_dim"))
    if variant in ("kda", "signed_kda"):  # fla_hybrid.py: KDA, SignedKDA
        common = dict(
            hidden_size=mcfg["hidden_size"],
            expand_v=float(get("linear_fla_expand_v")),
            head_dim=int(get("linear_key_head_dim")),
            num_heads=int(get("linear_num_key_heads")),
            num_v_heads=None,
            mode="chunk",
            use_short_conv=True,
            conv_size=conv,
            conv_bias=False,
            allow_neg_eigval=bool(get("linear_fla_allow_neg_eigval")),
            norm_eps=eps,
        )
        lower = mcfg.get("linear_fla_lower_bound", _FLA_DEFAULTS["linear_fla_lower_bound"])
        lower = None if lower in (None, "None") else float(lower)
        if variant == "kda":
            return {
                "class": "fla.layers.kda.KimiDeltaAttention",
                "kwargs": {**common, "safe_gate": lower is not None, "lower_bound": lower},
            }
        return {
            "class": "fla.layers.complex_kda_layer.ComplexKimiDeltaAttention",
            "kwargs": {
                **common,
                "gate": get("linear_fla_gate"),
                "output_gate": get("linear_fla_output_gate"),
                "lower_bound": lower,
                # init only (the weights are loaded), kept for a faithful rebuild
                "gate_init_style": get("linear_fla_gate_init_style"),
                "drop_silu": bool(get("linear_fla_drop_silu")),
            },
        }
    if variant == "complex_kda":  # complex_kda.py: ComplexKDA
        head_dim = mcfg.get("linear_key_head_dim") or mcfg.get("kv_channels")
        head_dim = int(head_dim or mcfg["hidden_size"] // mcfg["num_attention_heads"])
        lower = mcfg.get("linear_gate_lower_bound", _FLA_DEFAULTS["linear_gate_lower_bound"])
        return {
            "class": "fla.layers.complex_kda_layer.ComplexKimiDeltaAttention",
            "kwargs": dict(
                hidden_size=mcfg["hidden_size"],
                num_heads=mcfg["hidden_size"] // head_dim,
                head_dim=head_dim,
                allow_neg_eigval=float(get("linear_beta_max")) == 2.0,
                gate=get("linear_gate_activation"),
                gate_init_style=get("linear_gate_init_style"),
                lower_bound=None if lower in (None, "None") else float(lower),
                output_gate=get("linear_output_gate"),
                drop_silu=bool(get("linear_drop_qkv_silu")),
                conv_size=conv,
                norm_eps=eps,
            ),
        }
    raise ValueError(f"no FLA layer for experimental_attention_variant={variant!r}")


def _fla_attention_spec(mcfg: dict[str, Any], num_kv_heads: int) -> dict[str, Any]:
    """complex_kda's gated NoPE attention (ComplexKDAHybridAttention), an FLA
    Attention."""
    return {
        "class": "fla.layers.attn.Attention",
        "kwargs": dict(
            hidden_size=mcfg["hidden_size"],
            num_heads=mcfg["num_attention_heads"],
            num_kv_heads=num_kv_heads,
            qkv_bias=bool(mcfg.get("add_qkv_bias", False)),
            qk_norm=bool(mcfg.get("qk_layernorm", False)),
            output_gate=True,
            use_rope=False,
        ),
    }


def _hybrid_layout(mcfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Group a HybridModel pattern into HF layers (optional mixer + optional
    MLP).

    Each Megatron pattern symbol is one layer: a mixer (``*``, ``L``, ``G``,
    ``C``, ``M``, ``N``) or an MLP (``-``). A mixer directly followed by ``-`` becomes
    one HF layer with both; a mixer or ``-`` standing alone becomes a layer
    without the other half. Returns, per HF layer, the HF mixer type (or
    ``"none"``), the Megatron symbol, and the Megatron layer indices of the
    mixer and the MLP (``None`` when absent).
    """
    pattern = mcfg["hybrid_layer_pattern"]
    if "/" in pattern:
        raise ValueError(f"MTP layers ({pattern!r}) cannot be exported to HF")
    symbols = pattern.replace("|", "")
    variant = mcfg.get("experimental_attention_variant")
    mixers = {"*": "full_attention", "G": "gdn", "C": "short_conv", "M": "mamba2", "N": "engram"}
    layers: list[dict[str, Any]] = []
    for idx, sym in enumerate(symbols):
        if sym == "-":
            if layers and layers[-1]["mlp_src"] is None and layers[-1]["mixer"] != "none":
                layers[-1]["mlp_src"] = idx
            else:
                layers.append({"mixer": "none", "symbol": None, "mixer_src": None, "mlp_src": idx})
            continue
        if sym == "L":
            mixer = _VARIANT_MIXERS.get(variant)
            if mixer is None:
                raise ValueError(f"Unsupported experimental_attention_variant={variant!r} for 'L'")
        elif sym in mixers:
            mixer = mixers[sym]
        else:
            raise ValueError(
                f"Hybrid pattern symbol {sym!r} has no HF export (pattern {pattern!r})"
            )
        layers.append({"mixer": mixer, "symbol": sym, "mixer_src": idx, "mlp_src": None})
    return layers


# --------------------------------------------------------------------------
# config derivation
# --------------------------------------------------------------------------

# megatron/core/ssm/engram.py: the hash geometry is fixed by the config and this
# seed, so it is recomputed here rather than read from the checkpoint.
_ENGRAM_HASH_SEED = 0x5EED_E46A
_ENGRAM_DEFAULTS = {
    "engram_ngram_orders": [2, 3],
    "engram_num_heads": 4,
    "engram_table_size": 262144,
    "engram_embed_dim": 64,
    "engram_conv_kernel_size": 4,
}


def _engram_geometry(mcfg: dict[str, Any]) -> tuple[list[int], list[list[int]]]:
    """Table sizes and hash multipliers, as ``engram_table_sizes`` and
    ``engram_hash_multipliers`` compute them in Megatron."""
    import numpy as np

    orders = mcfg.get("engram_ngram_orders") or _ENGRAM_DEFAULTS["engram_ngram_orders"]
    heads = mcfg.get("engram_num_heads") or _ENGRAM_DEFAULTS["engram_num_heads"]
    lookups = len(orders) * heads

    def is_prime(n: int) -> bool:
        return n >= 2 and all(n % d for d in range(2, int(n**0.5) + 1))

    sizes, candidate = [], mcfg.get("engram_table_size") or _ENGRAM_DEFAULTS["engram_table_size"]
    for _ in range(lookups):
        while not is_prime(candidate):
            candidate += 1
        sizes.append(candidate)
        candidate += 1
    rng = np.random.default_rng(_ENGRAM_HASH_SEED)
    mult = rng.integers(1 << 20, 1 << 31, size=(lookups, max(orders))) | 1
    return sizes, mult.tolist()


def _megatron_cfg(run_dir: Path) -> dict[str, Any]:
    cfg_path = run_dir / "current.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"{cfg_path} not found (needed to reconstruct the architecture)")
    doc = yaml.safe_load(cfg_path.read_text())
    return doc["config"]["backend"]["megatron"]


def _mixer_types(mcfg: dict[str, Any]) -> list[str]:
    """Reproduce Megatron's per-layer mixer assignment.

    Linear-attention variants (``experimental_attention_variant`` +
    ``linear_attention_freq=F``) place a *softmax* layer wherever
    ``(i + 1) % F == 0`` and the linear mixer everywhere else
    (``_get_linear_attention_pattern``). Sliding-window attention uses the
    identical rule via ``window_attn_skip_freq`` (``layer_number % F != 0``
    ⇒ windowed, ``layer_number`` being 1-based).
    """
    n = mcfg["num_layers"]
    variant = mcfg.get("experimental_attention_variant")
    freq = mcfg.get("linear_attention_freq")
    skip_freq = mcfg.get("window_attn_skip_freq")

    if variant:
        mixer = _VARIANT_MIXERS.get(variant)
        if mixer is None:
            raise ValueError(f"Unsupported experimental_attention_variant={variant!r}")
        if not isinstance(freq, int):
            raise ValueError(f"Expected an integer linear_attention_freq, got {freq!r}")
        # complex_kda can swap the softmax layers for FLA's gated NoPE attention.
        softmax = (
            "fla_attention"
            if variant == "complex_kda" and mcfg.get("linear_hybrid_attention") == "gated_nope"
            else "full_attention"
        )
        return [softmax if (i + 1) % freq == 0 else mixer for i in range(n)]

    if skip_freq:
        if not isinstance(skip_freq, int):
            raise ValueError(f"Expected an integer window_attn_skip_freq, got {skip_freq!r}")
        return [
            "full_attention" if (i + 1) % skip_freq == 0 else "sliding_attention" for i in range(n)
        ]

    return ["full_attention"] * n


def _sliding_window(mcfg: dict[str, Any]) -> int | None:
    """Megatron ``window_size`` is ``(left, right)``; we only ever trained
    ``right=0`` (causal)."""
    ws = mcfg.get("window_size")
    if ws in (None, "None"):
        return None
    if isinstance(ws, str):
        left, right = (int(x) for x in ws.split(","))
    else:
        left, right = int(ws[0]), int(ws[1])
    if right != 0:
        raise ValueError(f"Only causal windows (right=0) are supported, got {ws!r}")
    return left


def build_hf_config(mcfg: dict[str, Any], vocab_size: int) -> dict[str, Any]:
    if mcfg.get("hybrid_layer_pattern"):
        layout = _hybrid_layout(mcfg)
        mixer_types = [layer["mixer"] for layer in layout]
        mlp_layers = [layer["mlp_src"] is not None for layer in layout]
    else:
        layout = None
        mixer_types = _mixer_types(mcfg)
        mlp_layers = [True] * len(mixer_types)
    engram = {}
    if "engram" in mixer_types:
        sizes, mult = _engram_geometry(mcfg)
        engram = {
            "engram_ngram_orders": mcfg.get("engram_ngram_orders")
            or _ENGRAM_DEFAULTS["engram_ngram_orders"],
            "engram_num_heads": mcfg.get("engram_num_heads")
            or _ENGRAM_DEFAULTS["engram_num_heads"],
            "engram_embed_dim": mcfg.get("engram_embed_dim")
            or _ENGRAM_DEFAULTS["engram_embed_dim"],
            "engram_conv_kernel_size": mcfg.get("engram_conv_kernel_size")
            or _ENGRAM_DEFAULTS["engram_conv_kernel_size"],
            "engram_table_sizes": sizes,
            "engram_hash_multipliers": mult,
            "engram_token_map": mcfg.get("engram_token_map") not in (None, "None"),
            # Not a model buffer in Megatron: remap_state reads the .npy file.
            "_engram_token_map_path": mcfg.get("engram_token_map"),
        }
    head_dim = mcfg.get("kv_channels") or mcfg["hidden_size"] // mcfg["num_attention_heads"]
    num_kv_heads = (
        mcfg["num_query_groups"]
        if mcfg.get("group_query_attention")
        else mcfg["num_attention_heads"]
    )
    fla = {}
    if "fla" in mixer_types:
        fla["fla_layer"] = _fla_layer_spec(mcfg)
    if "fla_attention" in mixer_types:
        fla["fla_attention_layer"] = _fla_attention_spec(mcfg, num_kv_heads)
    return {
        "architectures": ["OellmHybridForCausalLM"],
        "model_type": "oellm_hybrid",
        "auto_map": {
            "AutoConfig": "configuration_oellm_hybrid.OellmHybridConfig",
            "AutoModelForCausalLM": "modeling_oellm_hybrid.OellmHybridForCausalLM",
        },
        "torch_dtype": "bfloat16",
        "vocab_size": vocab_size,
        "hidden_size": mcfg["hidden_size"],
        "intermediate_size": mcfg["ffn_hidden_size"],
        "num_hidden_layers": len(mixer_types),
        "num_attention_heads": mcfg["num_attention_heads"],
        "num_key_value_heads": num_kv_heads,
        "head_dim": head_dim,
        "max_position_embeddings": mcfg["seq_length"],
        "rms_norm_eps": float(mcfg["layernorm_epsilon"]),
        "tie_word_embeddings": not mcfg.get("untie_embeddings_and_output_weights", False),
        "position_embedding_type": mcfg.get("position_embedding_type", "rope"),
        "rope_theta": float(mcfg.get("rotary_base") or 10000.0),
        "sliding_window": _sliding_window(mcfg),
        "qk_layernorm": bool(mcfg.get("qk_layernorm", False)),
        # Qwen3.5 block options. Read from the training run's own current.yaml,
        # so a run that did not use them exports exactly as before.
        "attention_output_gate": bool(mcfg.get("attention_output_gate", False)),
        "layernorm_zero_centered_gamma": bool(mcfg.get("layernorm_zero_centered_gamma", False)),
        "partial_rotary_factor": float(mcfg.get("rotary_percent") or 1.0),
        "mixer_types": mixer_types,
        "mlp_layers": mlp_layers,
        # Megatron layer indices behind each HF layer (HybridModel runs only); read
        # by remap_state and dropped from the saved config.
        "_hybrid_layout": layout,
        "linear_key_head_dim": mcfg.get("linear_key_head_dim", 128),
        "linear_value_head_dim": mcfg.get("linear_value_head_dim", 128),
        "linear_num_key_heads": mcfg.get("linear_num_key_heads", 8),
        "linear_num_value_heads": mcfg.get("linear_num_value_heads", 8),
        "linear_conv_kernel_dim": mcfg.get("linear_conv_kernel_dim", 4),
        "mlstm_chunk_size": mcfg.get("mlstm_chunk_size", 128),
        "mlstm_gate_soft_cap": mcfg.get("mlstm_gate_soft_cap", 15.0),
        "mlstm_backend": mcfg.get("mlstm_backend", "chunkwise--triton_xl_chunk"),
        "mlstm_conv1d": bool(mcfg.get("mlstm_conv1d", False)),
        "mamba_state_dim": mcfg.get("mamba_state_dim", 128),
        "mamba_head_dim": mcfg.get("mamba_head_dim", 64),
        "mamba_num_groups": mcfg.get("mamba_num_groups", 8),
        "mamba_num_heads": mcfg.get("mamba_num_heads"),
        "mamba_expand": 2,
        "mamba_conv_kernel": 4,
        "mamba_chunk_size": 128,
        "short_conv_kernel_size": mcfg.get("short_conv_kernel_size", 3),
        "short_conv_bias": bool(mcfg.get("short_conv_bias", False)),
        **engram,
        **fla,
        "use_cache": True,
    }


# --------------------------------------------------------------------------
# checkpoint loading
# --------------------------------------------------------------------------


def load_megatron_state(ckpt_dir: Path) -> dict[str, torch.Tensor]:
    """Read every model tensor of a DCP checkpoint into a plain dict.

    ``_load_state_dict_from_keys`` cannot be used: Megatron writes these
    checkpoints without ``planner_data`` in the metadata, which that path
    dereferences unconditionally. Instead we build an explicit destination
    state dict from the per-tensor metadata (global shape + dtype are recorded
    there, so TP/PP/DP sharding is resolved for us) and let the default load
    planner fill it in place.
    """
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    reader = dcp.FileSystemReader(str(ckpt_dir))
    md = reader.read_metadata()

    template: dict[str, torch.Tensor] = {}
    for key, entry in md.state_dict_metadata.items():
        if not key.startswith(("embedding.", "decoder.", "output_layer.")):
            continue
        if "_extra_state" in key or not isinstance(entry, TensorStorageMetadata):
            continue
        template[key] = torch.empty(tuple(entry.size), dtype=entry.properties.dtype)

    LOGGER.info("Reading %d model tensors from %s", len(template), ckpt_dir)
    dcp.load(template, storage_reader=reader)
    return template


def unstack_layers(state: dict[str, torch.Tensor], num_layers: int) -> dict[str, torch.Tensor]:
    """Expand Megatron's *stacked* layer representation into per-layer keys.

    When every layer shares one spec — the ``fullattn`` and ``swa7_rope100k``
    runs — ``TransformerBlock`` writes one tensor per parameter with a leading
    layer axis (``decoder.layers.mlp.linear_fc1.weight`` of shape
    ``[16, 3072, 512]``). The hybrid runs have heterogeneous specs and are
    already stored per layer (``decoder.layers.0.…``). Normalise the former to
    the latter so the remapping below only has one case to handle.
    """
    out: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        if not key.startswith("decoder.layers."):
            out[key] = tensor
            continue
        rest = key[len("decoder.layers.") :]
        if rest.split(".", 1)[0].isdigit():  # already per-layer
            out[key] = tensor
            continue
        if tensor.shape[0] != num_layers:
            raise ValueError(
                f"{key}: expected a leading layer axis of {num_layers}, got {tuple(tensor.shape)}"
            )
        for i in range(num_layers):
            out[f"decoder.layers.{i}.{rest}"] = tensor[i]
    return out


def _gather(state: dict[str, torch.Tensor], prefix: str, sections: list[str]) -> torch.Tensor:
    """Concatenate a fused tensor that the checkpoint stores as named
    sections."""
    if prefix in state:  # already fused
        return state[prefix]
    missing = [s for s in sections if f"{prefix}.{s}" not in state]
    if missing:
        raise KeyError(f"{prefix}: missing sections {missing}")
    return torch.cat([state[f"{prefix}.{s}"] for s in sections], dim=0)


def _split_qkv(
    w: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    output_gate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """De-interleave Megatron's grouped ``linear_qkv`` weight.

    Megatron lays the rows out per query group as
    ``[q_1 … q_{h/g}, k, v]``, each block ``head_dim`` rows tall — not as three
    contiguous q/k/v blocks.

    With ``attention_output_gate`` the projection is widened to
    ``[q_1 … q_{h/g}, gate_1 … gate_{h/g}, k, v]`` — the gate blocks sit BETWEEN
    q and k, so a 3-way split would silently return the gate as k and the k as v
    (see ``attention.py::get_query_key_value_tensors``, whose split_arg_list is
    ``[q_per_group*d, q_per_group*d, d, d]``). Shapes still line up for a
    non-gated model, which is why this has to be keyed off the config flag
    rather than inferred from the tensor.
    """
    hidden = w.shape[1]
    heads_per_group = num_heads // num_kv_heads
    if not output_gate:
        w = w.reshape(num_kv_heads, heads_per_group + 2, head_dim, hidden)
        q = w[:, :heads_per_group].reshape(num_heads * head_dim, hidden)
        k = w[:, heads_per_group].reshape(num_kv_heads * head_dim, hidden)
        v = w[:, heads_per_group + 1].reshape(num_kv_heads * head_dim, hidden)
        return q, k, v, None

    w = w.reshape(num_kv_heads, 2 * heads_per_group + 2, head_dim, hidden)
    q = w[:, :heads_per_group].reshape(num_heads * head_dim, hidden)
    gate = w[:, heads_per_group : 2 * heads_per_group].reshape(num_heads * head_dim, hidden)
    k = w[:, 2 * heads_per_group].reshape(num_kv_heads * head_dim, hidden)
    v = w[:, 2 * heads_per_group + 1].reshape(num_kv_heads * head_dim, hidden)
    return q, k, v, gate


def assert_finite(state: dict[str, torch.Tensor], where: str) -> None:
    """Refuse to export non-finite weights.

    A checkpoint read while it is still being written (or already being removed)
    can deserialize into all-NaN tensors WITHOUT raising: the file is complete
    enough to parse, the values are garbage. That produced a silently dead HF
    model for mamba2 0.9B/300BT -- lm-eval scored it at chance (0.20 vs ~0.40)
    and nothing anywhere flagged it. Failing here turns that into a loud error
    at conversion time, which is the only place it is cheap to catch.
    """
    bad = []
    for name, tensor in state.items():
        if not hasattr(tensor, "isfinite"):
            continue
        if not bool(tensor.isfinite().all()):
            n_nan = int(tensor.isnan().sum())
            n_inf = int(tensor.isinf().sum())
            bad.append(f"{name}: {n_nan} NaN, {n_inf} Inf of {tensor.numel()}")
        if len(bad) >= 5:
            break
    if bad:
        raise ValueError(
            f"non-finite weights in {where} -- refusing to export.\n  "
            + "\n  ".join(bad)
            + "\nThe source checkpoint is corrupt or was read mid-write; "
            "re-run the stage that produced it."
        )


def remap_state(state: dict[str, torch.Tensor], hf_cfg: dict[str, Any]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    out["model.embed_tokens.weight"] = state["embedding.word_embeddings.weight"]
    # GPTModel names the final norm final_layernorm, HybridModel final_norm.
    final_norm = "decoder.final_layernorm.weight"
    out["model.norm.weight"] = state.get(final_norm, state.get("decoder.final_norm.weight"))
    if out["model.norm.weight"] is None:
        raise KeyError("neither decoder.final_layernorm.weight nor decoder.final_norm.weight")
    if not hf_cfg["tie_word_embeddings"]:
        out["lm_head.weight"] = state["output_layer.weight"]

    ffn = hf_cfg["intermediate_size"]
    layout = hf_cfg.get("_hybrid_layout")
    for i, layer_type in enumerate(hf_cfg["mixer_types"]):
        dst = f"model.layers.{i}"
        if layout is None:  # GPTModel: mixer and MLP share the Megatron layer
            mixer_src = mlp_src = f"decoder.layers.{i}"
            symbol = None
        else:
            entry = layout[i]
            mixer_src = (
                None if entry["mixer_src"] is None else f"decoder.layers.{entry['mixer_src']}"
            )
            mlp_src = None if entry["mlp_src"] is None else f"decoder.layers.{entry['mlp_src']}"
            symbol = entry["symbol"]

        if mlp_src is not None:
            # MLP (identical for every layer type); linear_fc1 is [gate | up].
            fc1 = state[f"{mlp_src}.mlp.linear_fc1.weight"]
            out[f"{dst}.mlp.gate_proj.weight"] = fc1[:ffn]
            out[f"{dst}.mlp.up_proj.weight"] = fc1[ffn:]
            out[f"{dst}.mlp.down_proj.weight"] = state[f"{mlp_src}.mlp.linear_fc2.weight"]
            out[f"{dst}.post_attention_layernorm.weight"] = state[
                f"{mlp_src}.mlp.linear_fc1.layer_norm_weight"
            ]
        if mixer_src is None:
            continue
        src = mixer_src

        if layer_type == "engram":
            # The Megatron table is padded to a multiple of 4096 rows; drop the padding.
            rows = sum(hf_cfg["engram_table_sizes"])
            out[f"{dst}.mixer.table"] = state[f"{src}.table"][:rows]
            for name in (
                "key_proj.weight",
                "value_proj.weight",
                "hidden_norm.weight",
                "key_norm.weight",
                "conv1d.weight",
            ):
                out[f"{dst}.mixer.{name}"] = state[f"{src}.{name}"]
            if hf_cfg["engram_token_map"]:
                import numpy as np

                token_map = torch.from_numpy(np.load(hf_cfg["_engram_token_map_path"]))
                out[f"{dst}.mixer.token_map"] = token_map.to(torch.int64)
            continue

        if layer_type in ("fla", "fla_attention"):
            # A whole FLA layer behind a separate pre-norm: copy its parameters as is.
            out[f"{dst}.input_layernorm.weight"] = state[f"{src}.input_layernorm.weight"]
            prefix = f"{src}.self_attention.layer."
            names = [k[len(prefix) :] for k in state if k.startswith(prefix)]
            if not names:
                raise KeyError(f"no FLA layer parameters under {prefix}")
            for name in names:
                out[f"{dst}.mixer.layer.{name}"] = state[prefix + name]
            continue

        if layer_type in ("full_attention", "sliding_attention"):
            attn = f"{src}.self_attention"
            out[f"{dst}.input_layernorm.weight"] = state[f"{attn}.linear_qkv.layer_norm_weight"]
            q, k, v, gate = _split_qkv(
                state[f"{attn}.linear_qkv.weight"],
                hf_cfg["num_attention_heads"],
                hf_cfg["num_key_value_heads"],
                hf_cfg["head_dim"],
                output_gate=hf_cfg.get("attention_output_gate", False),
            )
            out[f"{dst}.mixer.q_proj.weight"] = q
            out[f"{dst}.mixer.k_proj.weight"] = k
            out[f"{dst}.mixer.v_proj.weight"] = v
            if gate is not None:
                out[f"{dst}.mixer.gate_proj.weight"] = gate
            out[f"{dst}.mixer.o_proj.weight"] = state[f"{attn}.linear_proj.weight"]
            if hf_cfg["qk_layernorm"]:
                out[f"{dst}.mixer.q_norm.weight"] = state[f"{attn}.q_layernorm.weight"]
                out[f"{dst}.mixer.k_norm.weight"] = state[f"{attn}.k_layernorm.weight"]
            continue

        # Mamba2 hangs its mixer one level deeper (Mamba2Attention -> .mixer); a
        # native HybridModel Mamba layer (`M`) keeps it at .mixer directly.
        if symbol == "M":
            base = f"{src}.mixer"
        else:
            base = f"{src}.self_attention" + (".mixer" if layer_type == "mamba2" else "")
        sections = FUSED_SECTIONS[layer_type]
        out[f"{dst}.input_layernorm.weight"] = state[f"{base}.in_proj.layer_norm_weight"]
        out[f"{dst}.mixer.in_proj.weight"] = _gather(
            state, f"{base}.in_proj.weight", sections["in_proj.weight"]
        )
        out[f"{dst}.mixer.out_proj.weight"] = state[f"{base}.out_proj.weight"]

        if layer_type == "short_conv":
            out[f"{dst}.mixer.conv1d.weight"] = state[f"{base}.conv1d.weight"]
            if hf_cfg["short_conv_bias"]:
                out[f"{dst}.mixer.conv1d.bias"] = state[f"{base}.conv1d.bias"]
        elif layer_type == "gdn":
            out[f"{dst}.mixer.conv1d.weight"] = _gather(
                state, f"{base}.conv1d.weight", sections["conv1d.weight"]
            )
            out[f"{dst}.mixer.A_log"] = state[f"{base}.A_log"]
            out[f"{dst}.mixer.dt_bias"] = state[f"{base}.dt_bias"]
            out[f"{dst}.mixer.out_norm.weight"] = state[f"{base}.out_norm.weight"]
        elif layer_type == "mlstm":
            out[f"{dst}.mixer.igate_bias"] = state[f"{base}.igate_bias"]
            out[f"{dst}.mixer.fgate_bias"] = state[f"{base}.fgate_bias"]
            out[f"{dst}.mixer.out_norm_weight"] = state[f"{base}.out_norm_weight"]
            if hf_cfg["mlstm_conv1d"]:
                out[f"{dst}.mixer.conv1d.weight"] = _gather(
                    state, f"{base}.conv1d.weight", sections["conv1d.weight"]
                )
                out[f"{dst}.mixer.conv1d.bias"] = _gather(
                    state, f"{base}.conv1d.bias", sections["conv1d.bias"]
                )
        elif layer_type == "mamba2":
            out[f"{dst}.mixer.conv1d.weight"] = _gather(
                state, f"{base}.conv1d.weight", sections["conv1d.weight"]
            )
            out[f"{dst}.mixer.conv1d.bias"] = _gather(
                state, f"{base}.conv1d.bias", sections["conv1d.bias"]
            )
            out[f"{dst}.mixer.A_log"] = state[f"{base}.A_log"]
            out[f"{dst}.mixer.D"] = state[f"{base}.D"]
            out[f"{dst}.mixer.dt_bias"] = state[f"{base}.dt_bias"]
            out[f"{dst}.mixer.norm.weight"] = state[f"{base}.norm.weight"]

    return out


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def _resolve_iter(run_dir: Path, iteration: int | None) -> Path:
    if iteration is not None:
        return run_dir / f"iter_{iteration:07d}"
    marker = run_dir / "latest_checkpointed_iteration.txt"
    if not marker.exists():
        raise FileNotFoundError(f"{marker} not found; pass --iter explicitly")
    return run_dir / f"iter_{int(marker.read_text().strip()):07d}"


def convert(
    run_dir: Path, out_dir: Path, tokenizer_dir: Path | None, iteration: int | None = None
) -> Path:
    from safetensors.torch import save_file

    ckpt_dir = _resolve_iter(run_dir, iteration)
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"checkpoint {ckpt_dir} does not exist")

    mcfg = _megatron_cfg(run_dir)
    state = load_megatron_state(ckpt_dir)
    assert_finite(state, f"source checkpoint {ckpt_dir}")
    vocab_size = state["embedding.word_embeddings.weight"].shape[0]
    hf_cfg = build_hf_config(mcfg, vocab_size)
    LOGGER.info(
        "%s: %d layers %s",
        run_dir.name,
        len(hf_cfg["mixer_types"]),
        sorted(set(hf_cfg["mixer_types"])),
    )

    state = unstack_layers(state, hf_cfg["num_hidden_layers"])
    tensors = remap_state(state, hf_cfg)
    # Re-check after remapping: a bad split/concat could also introduce
    # non-finite values that were not present in the source tensors.
    assert_finite(tensors, f"remapped export for {ckpt_dir}")
    tensors = {
        k: (v if not v.is_floating_point() else v.to(torch.bfloat16)).contiguous()
        for k, v in tensors.items()
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    saved_cfg = {k: v for k, v in hf_cfg.items() if not k.startswith("_")}
    (out_dir / "config.json").write_text(json.dumps(saved_cfg, indent=2) + "\n")

    for mod in ("configuration_oellm_hybrid.py", "modeling_oellm_hybrid.py"):
        # Copy verbatim: transformers' dynamic-module loader resolves *relative*
        # sibling imports (`from .configuration_oellm_hybrid import …`) by pulling
        # the sibling file in too. Rewriting it to an absolute import makes
        # `check_imports` treat it as a missing third-party package.
        shutil.copy2(_PKG / mod, out_dir / mod)

    if tokenizer_dir is not None:
        # `tokenizer.model` matters: some transformers versions construct the fast
        # tokenizer by way of the slow LlamaTokenizer, which requires the
        # sentencepiece file and dies with "expected str, bytes or os.PathLike
        # object, not NoneType" without it. transformers 5.8.1 loads from
        # tokenizer.json alone, the eval container's version does not.
        for name in (
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
        ):
            src = tokenizer_dir / name
            if src.exists():
                shutil.copy2(src, out_dir / name)

    # Record provenance so a results file can be traced back to a run/iteration.
    (out_dir / "megatron_source.json").write_text(
        json.dumps(
            {"run_dir": str(run_dir), "checkpoint": str(ckpt_dir), "iteration": ckpt_dir.name},
            indent=2,
        )
        + "\n"
    )
    LOGGER.info("wrote %s (%d tensors)", out_dir, len(tensors))
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, default=None)
    p.add_argument("--iter", type=int, default=None)
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.skip_existing and (args.out_dir / "model.safetensors").exists():
        LOGGER.info("%s already converted, skipping", args.out_dir)
        return
    convert(args.run_dir, args.out_dir, args.tokenizer, args.iter)


if __name__ == "__main__":
    main()

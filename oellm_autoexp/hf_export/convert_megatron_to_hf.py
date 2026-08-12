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
}


# --------------------------------------------------------------------------
# config derivation
# --------------------------------------------------------------------------


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
        mixer = {"gated_delta_net": "gdn", "mlstm": "mlstm", "mamba": "mamba2"}.get(variant)
        if mixer is None:
            raise ValueError(f"Unsupported experimental_attention_variant={variant!r}")
        if not isinstance(freq, int):
            raise ValueError(f"Expected an integer linear_attention_freq, got {freq!r}")
        return ["full_attention" if (i + 1) % freq == 0 else mixer for i in range(n)]

    if skip_freq:
        if not isinstance(skip_freq, int):
            raise ValueError(f"Expected an integer window_attn_skip_freq, got {skip_freq!r}")
        return [
            "full_attention" if (i + 1) % skip_freq == 0 else "sliding_attention"
            for i in range(n)
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
    mixer_types = _mixer_types(mcfg)
    head_dim = mcfg.get("kv_channels") or mcfg["hidden_size"] // mcfg["num_attention_heads"]
    num_kv_heads = (
        mcfg["num_query_groups"]
        if mcfg.get("group_query_attention")
        else mcfg["num_attention_heads"]
    )
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
        "num_hidden_layers": mcfg["num_layers"],
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
        "mixer_types": mixer_types,
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
    """Concatenate a fused tensor that the checkpoint stores as named sections."""
    if prefix in state:  # already fused
        return state[prefix]
    missing = [s for s in sections if f"{prefix}.{s}" not in state]
    if missing:
        raise KeyError(f"{prefix}: missing sections {missing}")
    return torch.cat([state[f"{prefix}.{s}"] for s in sections], dim=0)


def _split_qkv(
    w: torch.Tensor, num_heads: int, num_kv_heads: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """De-interleave Megatron's grouped ``linear_qkv`` weight.

    Megatron lays the rows out per query group as
    ``[q_1 … q_{h/g}, k, v]``, each block ``head_dim`` rows tall — not as three
    contiguous q/k/v blocks.
    """
    hidden = w.shape[1]
    heads_per_group = num_heads // num_kv_heads
    w = w.reshape(num_kv_heads, heads_per_group + 2, head_dim, hidden)
    q = w[:, :heads_per_group].reshape(num_heads * head_dim, hidden)
    k = w[:, heads_per_group].reshape(num_kv_heads * head_dim, hidden)
    v = w[:, heads_per_group + 1].reshape(num_kv_heads * head_dim, hidden)
    return q, k, v


def remap_state(
    state: dict[str, torch.Tensor], hf_cfg: dict[str, Any]
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    out["model.embed_tokens.weight"] = state["embedding.word_embeddings.weight"]
    out["model.norm.weight"] = state["decoder.final_layernorm.weight"]
    if not hf_cfg["tie_word_embeddings"]:
        out["lm_head.weight"] = state["output_layer.weight"]

    ffn = hf_cfg["intermediate_size"]
    for i, layer_type in enumerate(hf_cfg["mixer_types"]):
        src = f"decoder.layers.{i}"
        dst = f"model.layers.{i}"

        # MLP (identical for every layer type); linear_fc1 is [gate | up].
        fc1 = state[f"{src}.mlp.linear_fc1.weight"]
        out[f"{dst}.mlp.gate_proj.weight"] = fc1[:ffn]
        out[f"{dst}.mlp.up_proj.weight"] = fc1[ffn:]
        out[f"{dst}.mlp.down_proj.weight"] = state[f"{src}.mlp.linear_fc2.weight"]
        out[f"{dst}.post_attention_layernorm.weight"] = state[
            f"{src}.mlp.linear_fc1.layer_norm_weight"
        ]

        if layer_type in ("full_attention", "sliding_attention"):
            attn = f"{src}.self_attention"
            out[f"{dst}.input_layernorm.weight"] = state[f"{attn}.linear_qkv.layer_norm_weight"]
            q, k, v = _split_qkv(
                state[f"{attn}.linear_qkv.weight"],
                hf_cfg["num_attention_heads"],
                hf_cfg["num_key_value_heads"],
                hf_cfg["head_dim"],
            )
            out[f"{dst}.mixer.q_proj.weight"] = q
            out[f"{dst}.mixer.k_proj.weight"] = k
            out[f"{dst}.mixer.v_proj.weight"] = v
            out[f"{dst}.mixer.o_proj.weight"] = state[f"{attn}.linear_proj.weight"]
            if hf_cfg["qk_layernorm"]:
                out[f"{dst}.mixer.q_norm.weight"] = state[f"{attn}.q_layernorm.weight"]
                out[f"{dst}.mixer.k_norm.weight"] = state[f"{attn}.k_layernorm.weight"]
            continue

        # Mamba2 hangs its mixer one level deeper (Mamba2Attention -> .mixer).
        base = f"{src}.self_attention" + (".mixer" if layer_type == "mamba2" else "")
        sections = FUSED_SECTIONS[layer_type]
        out[f"{dst}.input_layernorm.weight"] = state[f"{base}.in_proj.layer_norm_weight"]
        out[f"{dst}.mixer.in_proj.weight"] = _gather(
            state, f"{base}.in_proj.weight", sections["in_proj.weight"]
        )
        out[f"{dst}.mixer.out_proj.weight"] = state[f"{base}.out_proj.weight"]

        if layer_type == "gdn":
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
    vocab_size = state["embedding.word_embeddings.weight"].shape[0]
    hf_cfg = build_hf_config(mcfg, vocab_size)
    LOGGER.info(
        "%s: %d layers %s", run_dir.name, len(hf_cfg["mixer_types"]),
        sorted(set(hf_cfg["mixer_types"])),
    )

    state = unstack_layers(state, hf_cfg["num_hidden_layers"])
    tensors = remap_state(state, hf_cfg)
    tensors = {k: v.to(torch.bfloat16).contiguous() for k, v in tensors.items()}

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    (out_dir / "config.json").write_text(json.dumps(hf_cfg, indent=2) + "\n")

    for mod in ("configuration_oellm_hybrid.py", "modeling_oellm_hybrid.py"):
        # Copy verbatim: transformers' dynamic-module loader resolves *relative*
        # sibling imports (`from .configuration_oellm_hybrid import …`) by pulling
        # the sibling file in too. Rewriting it to an absolute import makes
        # `check_imports` treat it as a missing third-party package.
        shutil.copy2(_PKG / mod, out_dir / mod)

    if tokenizer_dir is not None:
        for name in (
            "tokenizer.json",
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

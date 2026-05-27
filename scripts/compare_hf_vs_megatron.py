#!/usr/bin/env python3
"""
Diagnostic: compare HF Qwen3.5-4B-Base vs the Megatron checkpoint.

──────────────────────────────────────────────────────────────────────
Phase 1 – weight check (1 GPU needed for dist init)
  Loads the torch_dist checkpoint using PyTorch DCP into a flat dict,
  then compares each tensor against the corresponding HF weight.
  ✓ Catches Bridge conversion bugs and checkpoint-save issues.
  ✗ Does NOT test the Megatron checkpoint loader itself.

Phase 2 – activation check (1 GPU, needs FLA + TransformerEngine)
  Builds a Megatron model in-memory via the Bridge (HF weights loaded
  directly, no disk checkpoint round-trip).  Registers forward hooks
  on both models, runs the same token sequence, and compares per-layer
  activations.
  ✓ Catches forward-pass numerical issues even when Phase 1 passes.

inspect_model_weights() – call at the training.py breakpoint
  After load_checkpoint() returns (breakpoint at training.py:1721),
  this compares the *actually-loaded* Megatron model parameters against
  HF.  If Phase 1 passes but this fails, the checkpoint *loading* code
  is broken (not the conversion).

──────────────────────────────────────────────────────────────────────
Usage (from a salloc'd GPU node, inside the container):

  # Phase 1 + 2
  python scripts/compare_hf_vs_megatron.py

  # Phase 1 only (checkpoint weight check)
  python scripts/compare_hf_vs_megatron.py --phase 1

  # Phase 2 only (activation comparison)
  python scripts/compare_hf_vs_megatron.py --phase 2

  # From the training.py breakpoint:
  #   import sys; sys.path.insert(0, '/path/to/oellm-autoexp')
  #   import scripts.compare_hf_vs_megatron as cmp
  #   cmp.inspect_model_weights(model[0])   # model is the list from training
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional

import torch

# ── Adjust these paths ────────────────────────────────────────────────────────
HF_MODEL_ID = "Qwen/Qwen3.5-4B-Base"
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
MEGATRON_CKPT = (
    "/shared_silo/scratch/rluukkon/oellm/Megatron-Bridge"
    "/megatron_ckpt/Qwen3.5-4B-text_fix1"
)
MEGATRON_LM_PATH = f"{_REPO_ROOT}/submodules/Megatron-LM"
MEGATRON_BRIDGE_SRC = f"{_REPO_ROOT}/submodules/Megatron-Bridge/src"

DEVICE = "cuda"
DTYPE = torch.bfloat16
SEQ_LEN = 64
# bfloat16 has ~1e-2 rounding; anything larger is a real numerical difference
ATOL = 2e-2


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def _setup_paths():
    for p in [MEGATRON_LM_PATH, MEGATRON_BRIDGE_SRC]:
        if p not in sys.path:
            sys.path.insert(0, p)


def _init_distributed():
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29501")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(0)


def _fmt(t: torch.Tensor, ref: torch.Tensor, key: str, atol: float = ATOL) -> bool:
    t, ref = t.detach().float().cpu(), ref.detach().float().cpu()
    if t.shape != ref.shape:
        print(f"  ✗  {key:<65}  SHAPE {t.shape} vs {ref.shape}")
        return False
    diff = (t - ref).abs()
    max_e, mean_e = diff.max().item(), diff.mean().item()
    ok = max_e <= atol
    print(f"  {'✓' if ok else '✗'}  {key:<65}  max={max_e:.3e}  mean={mean_e:.3e}")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# HF reference helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_hf_sd(model_id: str = HF_MODEL_ID) -> Dict[str, torch.Tensor]:
    from transformers import AutoModelForCausalLM
    print(f"  Loading HF state dict from {model_id} …")
    # Use dtype (not torch_dtype) for newer transformers; avoid device_map so
    # we get a plain CPU model without accelerate dispatch hooks.
    try:
        m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    m = m.cpu()
    sd = {k: v.clone() for k, v in m.state_dict().items()}
    del m; gc.collect()
    return sd


def _hf_fused_qkv(hf_sd: dict, i: int) -> torch.Tensor:
    p = f"model.language_model.layers.{i}.self_attn"
    return torch.cat([hf_sd[f"{p}.q_proj.weight"],
                      hf_sd[f"{p}.k_proj.weight"],
                      hf_sd[f"{p}.v_proj.weight"]], dim=0).float()


def _hf_fused_fc1(hf_sd: dict, i: int) -> torch.Tensor:
    p = f"model.language_model.layers.{i}.mlp"
    return torch.cat([hf_sd[f"{p}.gate_proj.weight"],
                      hf_sd[f"{p}.up_proj.weight"]], dim=0).float()


def _hf_fused_in_proj(hf_sd: dict, i: int) -> torch.Tensor:
    """Reconstruct Megatron in_proj.weight from the 4 HF sub-weights."""
    p = f"model.language_model.layers.{i}.linear_attn"
    return torch.cat([
        hf_sd[f"{p}.in_proj_qkv.weight"],
        hf_sd[f"{p}.in_proj_z.weight"],
        hf_sd[f"{p}.in_proj_b.weight"],
        hf_sd[f"{p}.in_proj_a.weight"],
    ], dim=0).float()


def _hf_conv1d(hf_sd: dict, i: int) -> torch.Tensor:
    return hf_sd[f"model.language_model.layers.{i}.linear_attn.conv1d.weight"].float()


# ─────────────────────────────────────────────────────────────────────────────
# Megatron ↔ HF parameter mapping (matches Qwen35DenseTextBridge)
# ─────────────────────────────────────────────────────────────────────────────

def _meg_to_hf_direct(num_layers: int = 32) -> Dict[str, str]:
    """Direct 1-to-1 Megatron key → HF key (no fusing)."""
    m: Dict[str, str] = {
        "embedding.word_embeddings.weight":
            "model.language_model.embed_tokens.weight",
        "decoder.final_layernorm.weight":
            "model.language_model.norm.weight",
    }
    for i in range(num_layers):
        is_gdn = ((i + 1) % 4 != 0)
        lm = f"model.language_model.layers.{i}"
        dc = f"decoder.layers.{i}"
        m[f"{dc}.mlp.linear_fc1.layer_norm_weight"] = f"{lm}.post_attention_layernorm.weight"
        m[f"{dc}.mlp.linear_fc2.weight"]            = f"{lm}.mlp.down_proj.weight"
        if is_gdn:
            m[f"{dc}.self_attention.in_proj.layer_norm_weight"] = f"{lm}.input_layernorm.weight"
            m[f"{dc}.self_attention.out_proj.weight"]           = f"{lm}.linear_attn.out_proj.weight"
            m[f"{dc}.self_attention.A_log"]                     = f"{lm}.linear_attn.A_log"
            m[f"{dc}.self_attention.dt_bias"]                   = f"{lm}.linear_attn.dt_bias"
            m[f"{dc}.self_attention.out_norm.weight"]           = f"{lm}.linear_attn.norm.weight"
        else:
            m[f"{dc}.self_attention.linear_qkv.layer_norm_weight"] = f"{lm}.input_layernorm.weight"
            m[f"{dc}.self_attention.linear_proj.weight"]           = f"{lm}.self_attn.o_proj.weight"
            m[f"{dc}.self_attention.q_layernorm.weight"]           = f"{lm}.self_attn.q_norm.weight"
            m[f"{dc}.self_attention.k_layernorm.weight"]           = f"{lm}.self_attn.k_norm.weight"
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Generic comparison of a flat {Megatron-key: tensor} dict against HF SD
# ─────────────────────────────────────────────────────────────────────────────

def _compare_megatron_sd_vs_hf(
    meg_sd: Dict[str, torch.Tensor],
    hf_sd: Dict[str, torch.Tensor],
    num_layers: int = 32,
    label: str = "",
    atol: float = ATOL,
) -> int:
    """
    Check each Megatron parameter against its HF counterpart.
    Returns the number of mismatching / missing parameters.
    """
    direct = _meg_to_hf_direct(num_layers)
    n_ok = n_bad = 0

    def _check(meg_key: str, hf_t: Optional[torch.Tensor]) -> bool:
        nonlocal n_ok, n_bad
        if meg_key not in meg_sd:
            print(f"  ?  {meg_key:<65}  NOT IN MEGATRON SD")
            n_bad += 1
            return False
        if hf_t is None:
            print(f"  ?  {meg_key:<65}  HF source missing")
            n_bad += 1
            return False
        ok = _fmt(meg_sd[meg_key], hf_t, meg_key, atol)
        n_ok += ok; n_bad += not ok
        return ok

    # Direct mappings
    for meg_key, hf_key in sorted(direct.items()):
        _check(meg_key, hf_sd.get(hf_key, None) if hf_key else None)

    # Fused mappings (layer-by-layer)
    for i in range(num_layers):
        is_gdn = ((i + 1) % 4 != 0)
        dc = f"decoder.layers.{i}"
        _check(f"{dc}.mlp.linear_fc1.weight",  _hf_fused_fc1(hf_sd, i))
        if is_gdn:
            _check(f"{dc}.self_attention.in_proj.weight", _hf_fused_in_proj(hf_sd, i))
            _check(f"{dc}.self_attention.conv1d.weight",  _hf_conv1d(hf_sd, i))
        else:
            _check(f"{dc}.self_attention.linear_qkv.weight", _hf_fused_qkv(hf_sd, i))

    print(f"\n  {label}: {n_ok} OK,  {n_bad} MISMATCH/MISSING")
    return n_bad


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 – load checkpoint tensors via PyTorch DCP and compare vs HF
# ─────────────────────────────────────────────────────────────────────────────

def _load_ckpt_as_flat_dict(ckpt_dir: str) -> Dict[str, torch.Tensor]:
    """
    Use torch.distributed.checkpoint to load every tensor in the checkpoint
    into a plain Python dict, WITHOUT needing a Megatron model or distributed
    process group (torch.distributed is initialised for this but no NCCL ops
    are issued for a single-rank load).

    Factory-split keys (in_proj.weight.query, in_proj.weight.key, …) are left
    as-is here and reconstructed afterwards by _reconstruct_factories().
    """
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    ckpt_path = Path(ckpt_dir) / "iter_0000000"
    print(f"  Reading checkpoint metadata …")

    try:
        reader = dcp.FileSystemReader(str(ckpt_path))
    except AttributeError:
        from torch.distributed.checkpoint.filesystem import FileSystemReader
        reader = FileSystemReader(str(ckpt_path))

    meta = reader.read_metadata()

    # Build a flat dict pre-populated with zero tensors of the right shape.
    flat: Dict[str, torch.Tensor] = {}
    for key, smd in meta.state_dict_metadata.items():
        if isinstance(smd, TensorStorageMetadata):
            flat[key] = torch.empty(smd.size, dtype=smd.properties.dtype)

    print(f"  Found {len(flat)} tensor keys; loading …")
    dcp.load(
        state_dict=flat,
        checkpoint_id=str(ckpt_path),
    )
    print(f"  Loaded {len(flat)} tensors.")
    return {k: v.float() for k, v in flat.items()}


def _reconstruct_factories(
    flat: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Cat factory sub-keys back into their fused tensors along dim 0.

      in_proj.weight.{query,key,value,z,beta,alpha} → in_proj.weight
      conv1d.weight.{query,key,value}               → conv1d.weight
    """
    groups = {
        "in_proj.weight":  ["query", "key", "value", "z", "beta", "alpha"],
        "conv1d.weight":   ["query", "key", "value"],
    }
    consumed: set = set()
    merged: Dict[str, torch.Tensor] = {}

    for base_suffix, parts in groups.items():
        for key in list(flat.keys()):
            if not key.endswith(f".{base_suffix}.{parts[0]}"):
                continue
            prefix = key[: -len(f".{parts[0]}")]  # e.g. decoder.layers.0.self_attention.in_proj.weight
            sub_keys = [f"{prefix}.{p}" for p in parts]
            if any(sk not in flat for sk in sub_keys):
                continue
            merged[prefix] = torch.cat([flat[sk] for sk in sub_keys], dim=0)
            consumed.update(sub_keys)

    result = {k: v for k, v in flat.items() if k not in consumed}
    result.update(merged)
    return result


def run_phase1(
    hf_model_id: str = HF_MODEL_ID,
    ckpt_dir: str = MEGATRON_CKPT,
    num_layers: int = 32,
) -> int:
    print("\n" + "=" * 70)
    print("  PHASE 1 – Weight check: HF model vs checkpoint tensors")
    print("=" * 70)

    _init_distributed()
    hf_sd = _load_hf_sd(hf_model_id)

    flat = _load_ckpt_as_flat_dict(ckpt_dir)
    meg_sd = _reconstruct_factories(flat)
    print(f"  Checkpoint has {len(meg_sd)} reconstructed keys.\n")

    n_bad = _compare_megatron_sd_vs_hf(meg_sd, hf_sd, num_layers,
                                        label="Checkpoint vs HF")
    return n_bad


# ─────────────────────────────────────────────────────────────────────────────
# Breakpoint helper – inspect the training model AFTER load_checkpoint()
# ─────────────────────────────────────────────────────────────────────────────

def inspect_model_weights(
    megatron_model,
    hf_model_id: str = HF_MODEL_ID,
    num_layers: int = 32,
    atol: float = ATOL,
) -> int:
    """
    Call this at the breakpoint on training.py:1721 (after load_checkpoint).

    Usage inside pdb / breakpoint():
        import sys; sys.path.insert(0, '/path/to/oellm-autoexp')
        import scripts.compare_hf_vs_megatron as cmp
        cmp.inspect_model_weights(model[0])
        # 'model' is the list passed to load_checkpoint() in training.py
    """
    _setup_paths()
    print("\n[inspect] Comparing just-loaded Megatron model vs HF weights …")

    hf_sd = _load_hf_sd(hf_model_id)

    # Unwrap DDP/FSDP layers by following .module until we reach the GPTModel.
    raw = megatron_model
    while hasattr(raw, "module") and not hasattr(raw, "decoder"):
        raw = raw.module

    # Build flat Megatron state dict (strip 'module.' prefix from DDP if present)
    meg_sd: Dict[str, torch.Tensor] = {}
    for name, param in raw.named_parameters():
        if "_extra_state" in name:
            continue
        key = name.replace("module.", "")
        meg_sd[key] = param.detach().float().cpu()

    n_bad = _compare_megatron_sd_vs_hf(
        meg_sd, hf_sd, num_layers,
        label="Loaded training model vs HF",
        atol=atol,
    )

    if n_bad == 0:
        print("\n  ✓ All weights match HF. The model loaded correctly.")
        print("    The training loss issue is elsewhere (data, LR, config).")
    else:
        print(f"\n  ✗ {n_bad} parameter(s) differ from HF.")
        print("    This means checkpoint loading is broken or conversion was wrong.")
        print("    Run Phase 1 next to narrow down which of the two it is.")
    return n_bad


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 – per-layer activation comparison
# ─────────────────────────────────────────────────────────────────────────────

def _make_hook(store: dict, key: str, transpose_seq_batch: bool = False):
    def _h(module, inp, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        if transpose_seq_batch and t.dim() == 3:
            t = t.transpose(0, 1)   # [S, B, H] → [B, S, H]
        store[key] = t.detach().float().cpu()
    return _h


def _register_hf_hooks(model) -> dict:
    acts: Dict[str, torch.Tensor] = OrderedDict()
    model.model.embed_tokens.register_forward_hook(_make_hook(acts, "embedding"))
    for i, layer in enumerate(model.model.layers):
        layer.register_forward_hook(            _make_hook(acts, f"layer_{i:02d}"))
        layer.self_attn.register_forward_hook(  _make_hook(acts, f"layer_{i:02d}_attn"))
        layer.mlp.register_forward_hook(        _make_hook(acts, f"layer_{i:02d}_mlp"))
    model.model.norm.register_forward_hook(_make_hook(acts, "final_norm"))
    return acts


def _register_megatron_hooks(model) -> dict:
    acts: Dict[str, torch.Tensor] = OrderedDict()
    raw = model
    while hasattr(raw, "module") and not hasattr(raw, "decoder"):
        raw = raw.module

    raw.embedding.register_forward_hook(
        _make_hook(acts, "embedding", transpose_seq_batch=True))
    for i, layer in enumerate(raw.decoder.layers):
        layer.register_forward_hook(
            _make_hook(acts, f"layer_{i:02d}", transpose_seq_batch=True))
        layer.self_attention.register_forward_hook(
            _make_hook(acts, f"layer_{i:02d}_attn", transpose_seq_batch=True))
        layer.mlp.register_forward_hook(
            _make_hook(acts, f"layer_{i:02d}_mlp", transpose_seq_batch=True))
    raw.decoder.final_layernorm.register_forward_hook(
        _make_hook(acts, "final_norm", transpose_seq_batch=True))
    return acts


def _build_megatron_via_bridge(hf_model_id: str):
    """
    Build a Megatron GPTModel with HF weights loaded directly (no disk ckpt).
    This bypasses checkpoint save/load so we can isolate forward-pass issues.
    """
    from transformers import AutoModelForCausalLM
    from megatron.bridge.models.qwen.qwen3_5_dense_bridge import (
        Qwen35DenseTextBridge,
    )
    from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM

    print("  Loading HF model for Bridge …")
    hf_raw = AutoModelForCausalLM.from_pretrained(
        hf_model_id, torch_dtype=DTYPE, device_map="cpu"
    )
    hf_pre = PreTrainedCausalLM(hf_raw)

    bridge = Qwen35DenseTextBridge()
    provider = bridge.provider_bridge(hf_pre)

    print("  Initialising Megatron (single GPU) …")
    provider.initialize_megatron(extra_args_provider=None)

    print("  Building GPTModel …")
    meg = provider.provide().to(DEVICE).to(DTYPE)
    meg.eval()

    print("  Copying HF weights into Megatron …")
    bridge.load_weights_hf_to_megatron(hf_pre, meg)
    return meg


def _print_act_comparison(hf_acts: dict, meg_acts: dict, atol: float) -> int:
    print(f"\n  {'Key':<30}  {'MaxAbsErr':>12}  {'MeanAbsErr':>12}  {'OK?':>5}")
    print(f"  {'-'*30}  {'-'*12}  {'-'*12}  {'-'*5}")
    n_bad = 0
    for key in hf_acts:
        if key not in meg_acts:
            print(f"  {'?':<5}  {key:<30}  MISSING in Megatron")
            n_bad += 1; continue
        a, b = hf_acts[key], meg_acts[key]
        if a.shape != b.shape:
            print(f"  {'✗':<5}  {key:<30}  shape {a.shape} vs {b.shape}")
            n_bad += 1; continue
        diff = (a - b).abs()
        max_e, mean_e = diff.max().item(), diff.mean().item()
        ok = max_e <= atol
        n_bad += not ok
        print(f"  {'✓' if ok else '✗':<5}  {key:<30}  {max_e:>12.4e}  {mean_e:>12.4e}")
    return n_bad


def run_phase2(hf_model_id: str = HF_MODEL_ID) -> int:
    print("\n" + "=" * 70)
    print("  PHASE 2 – Activation check: Bridge-loaded Megatron vs HF")
    print("=" * 70)

    _setup_paths()
    _init_distributed()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"  Loading HF model {hf_model_id} …")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_id, torch_dtype=DTYPE, device_map=DEVICE
    )
    hf_model.eval()
    hf_acts = _register_hf_hooks(hf_model)

    meg_model = _build_megatron_via_bridge(hf_model_id)
    meg_acts  = _register_megatron_hooks(meg_model)

    # Shared input
    text = "The quick brown fox jumps over the lazy dog. " * 4
    tokens = tokenizer(text, return_tensors="pt",
                       max_length=SEQ_LEN, truncation=True)
    input_ids  = tokens["input_ids"].to(DEVICE)
    B, S = input_ids.shape
    position_ids = torch.arange(S, device=DEVICE).unsqueeze(0).expand(B, -1)

    print("  HF forward pass …")
    with torch.no_grad():
        hf_model(input_ids=input_ids, position_ids=position_ids,
                 use_cache=False)

    print("  Megatron forward pass …")
    with torch.no_grad():
        meg_model(input_ids=input_ids, position_ids=position_ids,
                  attention_mask=None)

    print("\n  Layer-by-layer comparison:")
    n_bad = _print_act_comparison(hf_acts, meg_acts, atol=ATOL)

    print(f"\n  Phase 2 summary: {len(hf_acts) - n_bad} OK,  {n_bad} MISMATCH")
    if n_bad == 0:
        print("\n  ✓ Activations match – Bridge conversion and forward pass are correct.")
        print("    The loss issue is therefore in the checkpoint *loading* pipeline.")
        print("    → Run inspect_model_weights() at the training.py breakpoint.")
    else:
        print("\n  ✗ Activations diverge – there is a forward-pass or weight-mapping bug.")
        print("    Identify the FIRST diverging layer; that module contains the bug.")
    return n_bad


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", type=int, choices=[1, 2], default=0,
                        help="Phase to run (default: both)")
    parser.add_argument("--hf-model", default=HF_MODEL_ID)
    parser.add_argument("--ckpt",     default=MEGATRON_CKPT)
    parser.add_argument("--layers",   type=int, default=32)
    args = parser.parse_args()

    _setup_paths()
    results: dict = {}

    if args.phase in (0, 1):
        results["phase1"] = run_phase1(args.hf_model, args.ckpt, args.layers)
    if args.phase in (0, 2):
        results["phase2"] = run_phase2(args.hf_model)

    print("\n" + "=" * 70)
    for k, v in results.items():
        print(f"  {k}: {'PASS ✓' if v == 0 else f'FAIL ✗  ({v} issue(s))'}")
    print("=" * 70)

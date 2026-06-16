#!/usr/bin/env python3
"""Compute MoE routing / coactivation / activation-norm metrics for a single
released HF MoE checkpoint (Qwen3-MoE, Step-3.5-Flash, OLMoE, Mixtral, DBRX...).

Mirrors the Megatron `compute_moe_metrics.py` semantics so cross-model plots
can be overlaid:

  * Coactivation: OLMoE-style adjacent-rank pair counting within each token's
    top-k indices (NOT adjacent-token-in-sequence). Normalized by single
    counts; diagonal zeroed.
  * Token counts: per-layer per-expert hit count from the top-k indices.
  * Activation norms: per-expert L2/RMS of the expert FFN output across the
    tokens routed to it, averaged across batches.
  * Routing maps: saved as a packed .npz so a later aggregator can compute
    saturation curves across OLMoE revisions.

Data
----
Default ("megatron_indexed_text"): read the same Megatron-indexed validation
set used for the Megatron jobs, decode GPT-NeoX-20B token IDs back to UTF-8,
then re-tokenize with the HF model's own tokenizer. Same source bytes,
model-native vocab.

Alternative ("hf_dataset"): pull a HF text dataset directly (e.g.
`nvidia/Nemotron-CC-v2-HQ`).

Outputs (match Megatron paths so the dashboard picks them up):
  coactivation.json, token_counts.json, expert_activation_norms.json,
  routing_maps.npz, coactivation_layer_<N>.png (if --moe-plot).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def _log(msg: str) -> None:
    print(f"[hf-moe] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_megatron_indexed_text(
    path_prefix: str,
    decode_tokenizer_path: str,
    num_chars_target: int,
    seed: int,
) -> List[str]:
    """Read documents from a Megatron IndexedDataset, decode to text via the
    GPT-NeoX-20B BPE so the same source bytes can be re-tokenized by any HF
    tokenizer downstream.

    Returns a list of doc strings (shuffled deterministically by `seed`),
    enough to cover roughly `num_chars_target` characters.
    """
    try:
        from megatron.core.datasets.indexed_dataset import IndexedDataset
    except ImportError as e:
        raise RuntimeError(
            "megatron.core not importable; this path requires the Megatron "
            "container (or pip-installed megatron-core)."
        ) from e

    from transformers import AutoTokenizer

    base = Path(path_prefix).expanduser()
    if not Path(str(base) + ".bin").is_file() or not Path(str(base) + ".idx").is_file():
        raise FileNotFoundError(f"Missing .bin/.idx at {base}")

    _log(f"Opening Megatron IndexedDataset: {base}")
    ds = IndexedDataset(str(base), multimodal=False, mmap=True)
    n_docs = len(ds)
    _log(f"{n_docs:,} documents")

    _log(f"Loading decode tokenizer: {decode_tokenizer_path}")
    # Local GPT-NeoX-20B dump only has vocab.json + merges.txt — the fast path
    # demands tokenizer.json (or sentencepiece/tiktoken to convert). Slow BPE
    # works directly and is plenty fast for the decode step.
    try:
        decode_tok = AutoTokenizer.from_pretrained(decode_tokenizer_path)
    except (ValueError, ImportError):
        decode_tok = AutoTokenizer.from_pretrained(decode_tokenizer_path, use_fast=False)

    rng = np.random.default_rng(seed)
    order = rng.permutation(n_docs)

    texts: List[str] = []
    char_count = 0
    docs_used = 0
    for doc_id in order:
        tokens = np.asarray(ds.get(int(doc_id)), dtype=np.int64)
        if tokens.size == 0:
            continue
        text = decode_tok.decode(tokens.tolist(), skip_special_tokens=True)
        if not text:
            continue
        texts.append(text)
        char_count += len(text)
        docs_used += 1
        if char_count >= num_chars_target:
            break

    _log(f"Decoded {docs_used:,} docs, {char_count:,} chars total")
    return texts


def _load_hf_text_dataset(
    name: str,
    split: str,
    text_field: str,
    num_chars_target: int,
    seed: int,
    cache_dir: Optional[str],
) -> List[str]:
    from datasets import load_dataset

    _log(f"Loading HF dataset {name} split={split}")
    ds = load_dataset(name, split=split, cache_dir=cache_dir)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ds))
    texts: List[str] = []
    char_count = 0
    for i in order:
        row = ds[int(i)]
        text = row.get(text_field)
        if not isinstance(text, str) or not text:
            continue
        texts.append(text)
        char_count += len(text)
        if char_count >= num_chars_target:
            break
    _log(f"Loaded {len(texts):,} docs, {char_count:,} chars")
    return texts


def _pack_batches(
    texts: List[str],
    tokenizer,
    batch_size: int,
    seq_length: int,
    num_batches: int,
    device: torch.device,
) -> List[torch.Tensor]:
    """Re-tokenize and pack into [num_batches, batch_size, seq_length] tensors.

    Concatenates with the tokenizer's EOS between docs to mirror the Megatron
    packing path; last partial chunk is padded with EOS.
    """
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = tokenizer.pad_token_id
    if eos_id is None:
        eos_id = 0

    needed = num_batches * batch_size * seq_length
    buf = np.full(needed + seq_length, eos_id, dtype=np.int64)
    cursor = 0
    for text in texts:
        if cursor >= needed:
            break
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        ids_arr = np.asarray(ids, dtype=np.int64)
        n = min(ids_arr.size, buf.size - cursor)
        buf[cursor : cursor + n] = ids_arr[:n]
        cursor += n
        if cursor < buf.size:
            buf[cursor] = eos_id
            cursor += 1

    if cursor < needed:
        _log(f"WARNING: only filled {cursor:,}/{needed:,} tokens — padding with EOS")
        buf[cursor:needed] = eos_id

    flat = torch.from_numpy(buf[:needed]).long().view(num_batches, batch_size, seq_length)
    return [flat[i].to(device) for i in range(num_batches)]


# ---------------------------------------------------------------------------
# Model loading and hook discovery
# ---------------------------------------------------------------------------

_MOE_BLOCK_CLASS_HINTS = (
    "SparseMoeBlock",   # Mixtral / Qwen3MoE / OLMoE
    "MoeBlock",
    "MoELayer",
    "MoEMLP",           # stepfun Step3p5MoEMLP (fused MoELinear experts)
    "FFN",              # DBRX
)


def _is_moe_block(module) -> bool:
    name = type(module).__name__
    return any(hint in name for hint in _MOE_BLOCK_CLASS_HINTS) and (
        hasattr(module, "experts") or hasattr(module, "ffn") or hasattr(module, "gate")
    )


def _enable_router_logits(model) -> None:
    """Best-effort flip of output_router_logits on the model config."""
    if hasattr(model, "config"):
        try:
            model.config.output_router_logits = True
        except Exception:
            pass


def _get_top_k(model) -> int:
    cfg = getattr(model, "config", None)
    for attr in ("num_experts_per_tok", "moe_top_k", "router_top_k", "top_k"):
        if cfg is not None and hasattr(cfg, attr):
            v = getattr(cfg, attr)
            if isinstance(v, int) and v > 0:
                return v
    return 2


# ---------------------------------------------------------------------------
# Metric collectors
# ---------------------------------------------------------------------------

class RouterCollector:
    """Captures router logits from each MoE block. Works whether the block
    returns (hidden, router_logits) or just hidden — in the second case the
    user must set output_router_logits=True so logits show up in the model
    output's router_logits tuple. We hook the block directly so we don't
    depend on that contract.
    """

    def __init__(self, top_k: int):
        self.top_k = top_k
        # layer_key -> list of LongTensor[N_tokens, top_k]
        self.topk_indices: Dict[str, List[torch.Tensor]] = {}
        self._debugged: bool = False

    def hook_gate(self, layer_key: str):
        """Hook the router/gate Linear: its output IS router_logits."""
        def _fn(module, inputs, outputs):
            router_logits = outputs[0] if isinstance(outputs, tuple) else outputs
            if not isinstance(router_logits, torch.Tensor):
                return
            if not self._debugged:
                self._debugged = True
                _log(f"[RouterCollector] gate output shape={tuple(router_logits.shape)}")
            if router_logits.dim() == 3:
                router_logits = router_logits.reshape(-1, router_logits.shape[-1])
            n_experts = router_logits.shape[-1]
            k = min(self.top_k, n_experts)
            probs = torch.softmax(router_logits.float(), dim=-1)
            idx = torch.topk(probs, k=k, dim=-1).indices  # [N, k]
            self.topk_indices.setdefault(layer_key, []).append(idx.detach().cpu())

        return _fn


class ExpertNormCollector:
    """Per-expert L2/RMS of the expert MLP output, reduced over its tokens.

    Hooks each individual expert (a small MLP module inside the MoE block's
    `experts` ModuleList). Works for Mixtral / Qwen3MoE / OLMoE; DBRX uses a
    fused experts kernel which we skip.
    """

    def __init__(self, vector_norm: str = "l2", token_reduce: str = "mean"):
        self.vector_norm = vector_norm
        self.token_reduce = token_reduce
        # layer_key -> list of [n_experts] tensors, one per batch
        self._per_batch: Dict[str, List[torch.Tensor]] = {}
        # tmp state filled by hooks during a single forward
        self._scratch: Dict[Tuple[str, int], List[float]] = {}
        self._n_experts: Dict[str, int] = {}

    def _per_token(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if self.vector_norm == "rms":
            return x.pow(2).mean(dim=-1).sqrt()
        return x.norm(dim=-1)

    def _reduce(self, v: torch.Tensor) -> float:
        if v.numel() == 0:
            return float("nan")
        if self.token_reduce == "rms":
            return v.pow(2).mean().sqrt().item()
        if self.token_reduce == "max":
            return v.max().item()
        return v.mean().item()

    def hook_expert(self, layer_key: str, expert_idx: int, total_experts: int):
        self._n_experts[layer_key] = total_experts

        def _fn(module, inputs, outputs):
            out = outputs[0] if isinstance(outputs, tuple) else outputs
            if not isinstance(out, torch.Tensor):
                return
            if out.dim() == 3:
                out = out.reshape(-1, out.shape[-1])
            if out.numel() == 0:
                return
            val = self._reduce(self._per_token(out))
            self._scratch.setdefault((layer_key, expert_idx), []).append(val)

        return _fn

    def flush_batch(self):
        """Bucket the per-expert values collected during one forward into one
        [n_experts] row per layer; reset scratch.
        """
        per_layer: Dict[str, torch.Tensor] = {}
        for (layer_key, e_idx), vals in self._scratch.items():
            n = self._n_experts[layer_key]
            if layer_key not in per_layer:
                per_layer[layer_key] = torch.full((n,), float("nan"))
            if vals:
                per_layer[layer_key][e_idx] = float(np.nanmean(vals))
        for k, v in per_layer.items():
            self._per_batch.setdefault(k, []).append(v)
        self._scratch.clear()

    def finalize(self) -> Dict[str, List[float]]:
        out: Dict[str, List[float]] = {}
        for layer_key, batches in self._per_batch.items():
            stacked = torch.stack(batches, dim=0)
            mask = ~torch.isnan(stacked)
            denom = mask.float().sum(dim=0).clamp(min=1.0)
            stacked = torch.where(mask, stacked, torch.zeros_like(stacked))
            out[layer_key] = (stacked.sum(dim=0) / denom).tolist()
        return out


def _wrap_olmoe_experts(module, layer_key: str, norms: "ExpertNormCollector") -> None:
    """OLMoE's OlmoeExperts is a fused-weights module (no per-expert nn.Module),
    so forward_hooks on individual experts are impossible. We monkey-patch its
    forward to mirror the original loop while recording per-expert norms.
    Norm is taken AFTER applying routing weights to match the Megatron pipeline.
    """
    import types
    F = torch.nn.functional
    n_experts = module.num_experts
    norms._n_experts[layer_key] = n_experts

    def instrumented_forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            if current_hidden_states.numel() > 0:
                val = norms._reduce(norms._per_token(current_hidden_states))
                norms._scratch.setdefault((layer_key, int(expert_idx)), []).append(val)
            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
            )
        return final_hidden_states

    module.forward = types.MethodType(instrumented_forward, module)


def _is_step3p5_moe(module) -> bool:
    """stepfun Step-3.5-Flash MoE block: fused MoELinear experts + an fp32 gate
    bypass that reads `gate.weight` directly (so a forward-hook on `.gate` never
    fires). Detected by its fused expert projections and no `experts` ModuleList.
    """
    return (
        type(module).__name__ == "Step3p5MoEMLP"
        and hasattr(module, "gate")
        and hasattr(module, "gate_proj")
        and hasattr(module, "down_proj")
        and not hasattr(module, "experts")
    )


def _wrap_step3p5_moe(module, layer_key: str, router: "RouterCollector",
                      norms: "Optional[ExpertNormCollector]") -> None:
    """Monkey-patch Step3p5MoEMLP.forward to feed BOTH collectors from one pass.

    The gate hook can't be used (fp32 bypass) and there is no per-expert
    nn.Module to hook (fused MoELinear), so we mirror the block's own forward,
    recording top-k expert indices (for coactivation/token counts) and per-expert
    output norms (after routing weights, matching the Megatron pipeline).
    """
    import types
    F = torch.nn.functional
    if norms is not None:
        norms._n_experts[layer_key] = int(module.num_experts)

    def instrumented_forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hs = hidden_states.view(-1, hidden_dim)
        if self.need_fp32_gate:
            router_logits = torch.matmul(
                hs.to(torch.float32), self.gate.weight.t().to(torch.float32)
            )
        else:
            router_logits = self.gate(hs)
        if self.custom_routing_function:
            routing_weights, selected_experts = self.custom_routing_function(
                router_logits, self.top_k, renormalize=True
            )
        else:
            rw = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(rw, self.top_k, dim=-1)
        routing_weights = routing_weights * self.routed_scaling_factor

        # Record top-k indices (same [N_tokens, top_k] format as hook_gate).
        router.topk_indices.setdefault(layer_key, []).append(
            selected_experts.detach().cpu()
        )

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hs.dtype, device=hs.device,
        )
        expert_mask = F.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)
        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            current_state = hs[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = (
                self.get_expert_output(current_state, expert_idx)
                * routing_weights[top_x, idx, None]
            )
            if norms is not None and current_hidden_states.numel() > 0:
                val = norms._reduce(norms._per_token(current_hidden_states))
                norms._scratch.setdefault((layer_key, int(expert_idx)), []).append(val)
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hs.dtype)
            )
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

    module.forward = types.MethodType(instrumented_forward, module)


def _attach_hooks(model, top_k: int, do_act_norms: bool):
    router = RouterCollector(top_k=top_k)
    norms = ExpertNormCollector() if do_act_norms else None
    handles = []
    wrapped_blocks = 0

    for name, module in model.named_modules():
        if not _is_moe_block(module):
            continue
        # Step-3.5-Flash: fp32 gate bypass + fused experts. One block-level
        # monkey-patch feeds both the router and norm collectors; the standard
        # gate/experts hooks below would silently no-op here.
        if _is_step3p5_moe(module):
            _wrap_step3p5_moe(module, name, router, norms)
            wrapped_blocks += 1
            _log(f"[step3p5] wrapped MoE block at {name}")
            continue
        gate = getattr(module, "gate", None) or getattr(module, "router", None)
        if isinstance(gate, torch.nn.Module):
            handles.append(gate.register_forward_hook(router.hook_gate(name)))
        else:
            _log(f"WARNING: MoE block {name} has no .gate or .router submodule; skipping router hook")
        if do_act_norms:
            experts = getattr(module, "experts", None)
            if experts is None:
                continue
            # Case 1: ModuleList of per-expert nn.Module (Mixtral, Qwen3MoE, old OLMoE).
            exp_list = []
            if hasattr(experts, "__iter__"):
                try:
                    exp_list = list(experts)
                except TypeError:
                    exp_list = []
                exp_list = [e for e in exp_list if isinstance(e, torch.nn.Module)]
            if exp_list:
                for i, e in enumerate(exp_list):
                    handles.append(
                        e.register_forward_hook(norms.hook_expert(name, i, len(exp_list)))
                    )
            # Case 2: fused-weights OlmoeExperts — monkey-patch forward.
            elif hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj"):
                _wrap_olmoe_experts(experts, name, norms)
                _log(f"[norms] wrapped fused experts at {name}")
            else:
                _log(f"WARNING: cannot hook experts for {name} (unknown structure)")

    if not router.topk_indices and not handles and not wrapped_blocks:
        _log("WARNING: no MoE blocks detected — check class-name heuristics")
    else:
        _log(f"Hooked {sum(1 for n, m in model.named_modules() if _is_moe_block(m))} MoE blocks")

    return router, norms, handles


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate_coactivation(
    topk_indices: Dict[str, List[torch.Tensor]],
    n_experts_by_layer: Dict[str, int],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """OLMoE-style adjacent-rank coactivation within each token's top-k list,
    matching `compute_moe_metrics.py:_collect_final_routing_and_coactivation`.
    """
    coact: Dict[str, np.ndarray] = {}
    singles: Dict[str, np.ndarray] = {}
    for layer, batches in topk_indices.items():
        if not batches:
            continue
        n_e = n_experts_by_layer[layer]
        co = np.zeros((n_e, n_e), dtype=np.float64)
        sg = np.zeros((n_e,), dtype=np.float64)
        for b in batches:
            arr = b.numpy()  # [N, k]
            if arr.shape[1] < 2:
                # top-1: only single counts; no pairs
                sg += np.bincount(arr.ravel(), minlength=n_e)
                continue
            sg += np.bincount(arr.ravel(), minlength=n_e)
            a = arr[:, :-1].ravel()
            c = arr[:, 1:].ravel()
            np.add.at(co, (a, c), 1.0)
            np.add.at(co, (c, a), 1.0)
        coact[layer] = co
        singles[layer] = sg
    return coact, singles


def _normalize_coact(co: np.ndarray, sg: np.ndarray) -> np.ndarray:
    denom = np.maximum(sg, 1.0)[:, None]
    norm = co / denom
    np.fill_diagonal(norm, 0.0)
    return norm


def _select_top_experts_pairwise(norm: np.ndarray, top_n: int) -> List[int]:
    n_e = norm.shape[0]
    if n_e <= top_n:
        return list(range(n_e))
    flat = norm.copy()
    np.fill_diagonal(flat, 0.0)
    order = np.argsort(flat, axis=None)[::-1]
    selected: List[int] = []
    for flat_idx in order:
        v = float(flat.ravel()[flat_idx])
        if v <= 0:
            break
        a = flat_idx // n_e
        b = flat_idx % n_e
        if a not in selected:
            selected.append(int(a))
        if b not in selected:
            selected.append(int(b))
        if len(selected) >= top_n:
            return selected[:top_n]
    fill = np.argsort(norm.max(axis=1))[::-1].tolist()
    for e in fill:
        if e not in selected:
            selected.append(int(e))
        if len(selected) >= top_n:
            break
    return selected[:top_n]


def _save_coactivation(
    coact: Dict[str, np.ndarray],
    singles: Dict[str, np.ndarray],
    out_path: Path,
    top_n: int,
    model_id: str,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "_meta": {
            "coactivation_mode": "pairwise-adjacent",
            "top_n": top_n,
            "model": model_id,
        },
        "layers": {},
    }
    for layer, co in coact.items():
        sg = singles[layer]
        norm = _normalize_coact(co, sg)
        sel = _select_top_experts_pairwise(norm, top_n)
        sub = norm[np.ix_(sel, sel)]
        summary["layers"][layer] = {
            "expert_ids": sel,
            "coactivation_matrix": sub.tolist(),
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    _log(f"Wrote {out_path}")
    return summary


def _save_token_counts(
    singles: Dict[str, np.ndarray],
    out_path: Path,
    model_id: str,
    step: Optional[str],
) -> None:
    payload = {
        "_meta": {"model": model_id, "step": step or "final"},
        "layers": {layer: sg.astype(int).tolist() for layer, sg in singles.items()},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    _log(f"Wrote {out_path}")


def _save_activation_norms(
    norms: Dict[str, List[float]],
    out_path: Path,
    model_id: str,
    vector_norm: str,
    token_reduce: str,
) -> None:
    payload = {
        "_meta": {
            "model": model_id,
            "vector_norm": vector_norm,
            "token_reduce": token_reduce,
        },
        "layers": norms,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    _log(f"Wrote {out_path}")


def _save_routing_maps_npz(
    topk_indices: Dict[str, List[torch.Tensor]],
    n_experts_by_layer: Dict[str, int],
    out_path: Path,
) -> None:
    """Pack per-layer routing maps as a boolean [N_tokens, n_experts] array.

    Saved fields: <layer>::routing (bool), <layer>::n_experts (int).
    Aggregator script will reuse these for cross-revision saturation.
    """
    payload: Dict[str, np.ndarray] = {}
    for layer, batches in topk_indices.items():
        if not batches:
            continue
        n_e = n_experts_by_layer[layer]
        n_tokens = sum(b.shape[0] for b in batches)
        rmap = np.zeros((n_tokens, n_e), dtype=bool)
        off = 0
        for b in batches:
            idx = b.numpy()
            rows = np.arange(idx.shape[0])[:, None].repeat(idx.shape[1], axis=1).ravel()
            rmap[off + rows, idx.ravel()] = True
            off += idx.shape[0]
        # npz keys can't contain dots reliably; replace.
        key = layer.replace(".", "__")
        payload[key + "::routing"] = rmap
        payload[key + "::n_experts"] = np.array([n_e], dtype=np.int64)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    _log(f"Wrote {out_path} ({len(topk_indices)} layers)")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

import re


def _layer_index(name: str) -> Optional[int]:
    m = re.search(r"layers?[._](\d+)", name)
    return int(m.group(1)) if m else None


def _layer_short_label(name: str) -> str:
    li = _layer_index(name)
    return f"layer_{li}" if li is not None else name.replace(".", "_")


def _plot_coactivation(summary: Dict[str, Any], plot_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    for layer, info in summary["layers"].items():
        mat = np.asarray(info["coactivation_matrix"]) * 100.0  # to %
        ids = info["expert_ids"]
        plt.figure(figsize=(5, 4))
        plt.imshow(mat, cmap="RdPu", aspect="auto", vmin=0, vmax=60)
        plt.colorbar(ticks=[0, 15, 30, 45, 60])
        short = _layer_short_label(layer)
        plt.title(f"Co-activation: {short}")
        plt.xlabel("Expert")
        plt.ylabel("Expert")
        tick_positions = list(range(len(ids)))
        plt.xticks(tick_positions, [str(i) for i in ids], rotation=90)
        plt.yticks(tick_positions, [str(i) for i in ids])
        plt.tight_layout()
        li = _layer_index(layer)
        fname = f"coactivation_layer_{li}.png" if li is not None else f"coactivation_{layer.replace('.', '_')}.png"
        plt.savefig(plot_dir / fname, dpi=200)
        plt.close()
    _log(f"Wrote coactivation plots to {plot_dir}")


def _plot_token_counts(token_counts_path: Path, plot_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(token_counts_path.read_text())
    layers = data.get("layers", {})
    if not layers:
        _log(f"[plot] no layers in {token_counts_path}; skipping token-count plots")
        return
    plot_dir.mkdir(parents=True, exist_ok=True)
    step = data.get("_meta", {}).get("step", "")
    for layer, counts in layers.items():
        counts = list(counts)
        n_experts = len(counts)
        uniform = sum(counts) / n_experts if n_experts else 0.0
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(n_experts), counts, color="#1f77b4")
        ax.axhline(uniform, color="gray", linestyle="--", linewidth=1, label="uniform")
        li = _layer_index(layer)
        title = f"Layer {li} — per-expert token count" if li is not None else f"{layer} — per-expert token count"
        if step:
            title += f" (step {step})"
        ax.set_title(title)
        ax.set_xlabel("Expert ID")
        ax.set_ylabel("Token count")
        ax.legend()
        fig.tight_layout()
        fname = f"token_counts_layer_{li}.png" if li is not None else f"token_counts_{layer.replace('.', '_')}.png"
        fig.savefig(plot_dir / fname, dpi=200)
        plt.close(fig)
    _log(f"Wrote token-count plots to {plot_dir}")


def _plot_routing_maps(npz_path: Path, plot_dir: Path, max_tokens: int = 1024) -> None:
    """Per-layer heatmap: token-position (y) × expert (x), boolean routing mask.

    Routing maps can be huge (millions of tokens); subsample rows to max_tokens
    for plotting.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not npz_path.exists():
        _log(f"[plot] {npz_path} missing; skipping routing-map plots")
        return
    plot_dir.mkdir(parents=True, exist_ok=True)
    with np.load(npz_path) as npz:
        keys = list(npz.keys())
        layers = sorted({k.split("::")[0] for k in keys if k.endswith("::routing")})
        for layer_key in layers:
            rmap = npz[f"{layer_key}::routing"]
            n_tokens, n_experts = rmap.shape
            if n_tokens > max_tokens:
                idx = np.linspace(0, n_tokens - 1, max_tokens).astype(int)
                rmap = rmap[idx]
            layer = layer_key.replace("__", ".")
            li = _layer_index(layer)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.imshow(rmap.astype(np.uint8), cmap="Greys", aspect="auto", interpolation="nearest")
            ax.set_xlabel("Expert")
            ax.set_ylabel("Token position (subsampled)" if rmap.shape[0] < n_tokens else "Token position")
            title = f"Routing map: layer {li}" if li is not None else f"Routing map: {layer}"
            ax.set_title(title)
            fig.tight_layout()
            fname = f"routing_map_layer_{li}.png" if li is not None else f"routing_map_{layer.replace('.', '_')}.png"
            fig.savefig(plot_dir / fname, dpi=150)
            plt.close(fig)
    _log(f"Wrote routing-map plots to {plot_dir}")


def _plot_activation_norms(norms_path: Path, plot_dir: Path) -> None:
    """Single-snapshot version of the Megatron max-vs-median plot.

    X = layer index, two lines: solid=max-over-experts, dashed=median-over-experts.
    Companion plot: max/median ratio per layer (log y).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(norms_path.read_text())
    layers = data.get("layers", {})
    if not layers:
        _log(f"[plot] no layers in {norms_path}; skipping activation-norm plots")
        return
    plot_dir.mkdir(parents=True, exist_ok=True)
    meta = data.get("_meta", {})
    vn = meta.get("vector_norm", "l2")
    tr = meta.get("token_reduce", "mean")

    items = []
    for layer, vals in layers.items():
        li = _layer_index(layer)
        if li is None:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            continue
        items.append((li, float(arr.max()), float(np.median(arr))))
    if not items:
        _log("[plot] activation-norm plots: no usable layers")
        return
    items.sort(key=lambda x: x[0])
    xs = [li for li, _, _ in items]
    maxes = [m for _, m, _ in items]
    meds = [md for _, _, md in items]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs, maxes, color="#d62728", linewidth=2.0, marker="o", label="max")
    ax.plot(xs, meds, color="#1f77b4", linewidth=1.6, marker="o",
            linestyle="--", label="median")
    ax.set_yscale("log")
    ax.set_xlabel("Layer index", fontweight="bold")
    ax.set_ylabel(f"Per-expert activation norm ({vn}/{tr})", fontweight="bold")
    ax.set_title("Expert activation norms — solid: max, dashed: median")
    ax.grid(True, which="both", alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "expert_activation_norms.png", dpi=200)
    plt.close(fig)

    ratios = [m / md for m, md in zip(maxes, meds) if md > 0]
    rxs = [li for (li, m, md) in items if md > 0]
    if rxs:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(rxs, ratios, color="#2ca02c", linewidth=1.8, marker="o")
        ax.set_yscale("log")
        ax.set_xlabel("Layer index", fontweight="bold")
        ax.set_ylabel("max / median", fontweight="bold")
        ax.set_title("Max-to-median expert activation-norm ratio per layer")
        ax.grid(True, which="both", alpha=0.25, linestyle="--")
        fig.tight_layout()
        fig.savefig(plot_dir / "expert_activation_max_over_median.png", dpi=200)
        plt.close(fig)
    _log(f"Wrote activation-norm plots to {plot_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HF MoE metrics — coactivation, norms, token counts.")
    p.add_argument("--model", required=True, help="HF model id (e.g. allenai/OLMoE-1B-7B-0924)")
    p.add_argument("--revision", default="main")
    p.add_argument("--hf-cache-dir", default=None)
    p.add_argument("--torch-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device-map", default="auto")

    p.add_argument("--moe-data-type", default="megatron_indexed_text",
                   choices=["megatron_indexed_text", "hf_dataset"])
    p.add_argument("--moe-megatron-data-path", default=None,
                   help="Prefix to .bin/.idx (megatron_indexed_text)")
    p.add_argument("--moe-decode-tokenizer", default="EleutherAI/gpt-neox-20b",
                   help="Tokenizer used to decode the .bin bytes (must match the bin tokenizer)")
    p.add_argument("--moe-hf-dataset", default=None)
    p.add_argument("--moe-hf-split", default="train")
    p.add_argument("--moe-hf-text-field", default="text")

    p.add_argument("--moe-batch-size", type=int, default=1)
    p.add_argument("--moe-seq-length", type=int, default=4096)
    p.add_argument("--moe-num-batches", type=int, default=240)
    p.add_argument("--moe-seed", type=int, default=1234)

    p.add_argument("--moe-coactivation-output", default=None)
    p.add_argument("--moe-coactivation-top-n", type=int, default=32)
    p.add_argument("--moe-token-count-output", default=None)
    p.add_argument("--moe-activation-norm-output", default=None)
    p.add_argument("--moe-act-norm-token-reduce", default="mean", choices=["mean", "rms", "max"])
    p.add_argument("--moe-act-norm-vector-norm", default="l2", choices=["l2", "rms"])
    p.add_argument("--moe-routing-maps-output", default=None,
                   help=".npz path for raw routing maps (used by cross-revision saturation aggregator).")
    p.add_argument("--moe-plot", action="store_true")
    p.add_argument("--moe-plot-dir", default=None)

    return p.parse_args(argv)


def _build_text_corpus(args: argparse.Namespace) -> List[str]:
    # Aim for ~5 chars/token on average; pad to 2x to absorb tokenizer differences.
    target_tokens = args.moe_num_batches * args.moe_batch_size * args.moe_seq_length
    target_chars = target_tokens * 8

    if args.moe_data_type == "megatron_indexed_text":
        if not args.moe_megatron_data_path:
            raise ValueError("--moe-megatron-data-path required for megatron_indexed_text")
        return _load_megatron_indexed_text(
            args.moe_megatron_data_path,
            args.moe_decode_tokenizer,
            target_chars,
            args.moe_seed,
        )
    if not args.moe_hf_dataset:
        raise ValueError("--moe-hf-dataset required for hf_dataset")
    return _load_hf_text_dataset(
        args.moe_hf_dataset,
        args.moe_hf_split,
        args.moe_hf_text_field,
        target_chars,
        args.moe_seed,
        args.hf_cache_dir,
    )


def _resolve_cache_dir(arg_value: Optional[str]) -> Optional[str]:
    """transformers/huggingface_hub expects cache_dir to be the directory that
    *directly contains* the `models--<org>--<repo>/` entries. That's `HF_HOME/hub`,
    not `HF_HOME`. If the caller passed `HF_HOME` (or `HF_HOME/hub` already),
    figure out which level they meant and return the right one.
    """
    if not arg_value:
        return None
    p = Path(arg_value)
    if (p / "hub").is_dir() and not p.name == "hub":
        return str(p / "hub")
    return str(p)


def _load_model_and_tokenizer(args: argparse.Namespace):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        args.torch_dtype
    ]
    cache_dir = _resolve_cache_dir(args.hf_cache_dir)
    _log(
        f"Loading model {args.model}@{args.revision} dtype={args.torch_dtype} "
        f"cache_dir={cache_dir} device_map={args.device_map}"
    )
    tok = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, cache_dir=cache_dir, trust_remote_code=True
    )

    # `device_map=none` skips the accelerate-backed loader and just puts the
    # model on a single GPU. Works for OLMoE-1B-7B (~14 GB bf16). For Qwen3-30B
    # and Step-3.5-Flash you still need `accelerate` installed and device_map=auto.
    common = dict(
        revision=args.revision,
        cache_dir=cache_dir,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    # Step-3.5-Flash's custom modeling reads `config.pad_token_id` in __init__
    # but the published config doesn't set one. Inject from the tokenizer
    # (falling back to eos, the usual decoder-only convention) so the model
    # builds. Harmless for models that already have it set.
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if pad_id is not None:
        common["pad_token_id"] = pad_id
    if args.device_map and args.device_map.lower() != "none":
        model = AutoModelForCausalLM.from_pretrained(
            args.model, device_map=args.device_map, **common
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, **common)
        if torch.cuda.is_available():
            model = model.cuda()
    model.eval()
    _enable_router_logits(model)
    return model, tok


def main(argv: List[str]) -> int:
    args = _parse_args(argv)

    model, tok = _load_model_and_tokenizer(args)
    top_k = _get_top_k(model)
    _log(f"Detected top_k={top_k}")

    texts = _build_text_corpus(args)
    if not texts:
        _log("No text loaded; aborting")
        return 1

    device = next(model.parameters()).device
    batches = _pack_batches(
        texts, tok, args.moe_batch_size, args.moe_seq_length, args.moe_num_batches, device
    )
    _log(f"Built {len(batches)} batches of [{args.moe_batch_size}, {args.moe_seq_length}]")

    do_norms = bool(args.moe_activation_norm_output)
    router, norms, handles = _attach_hooks(model, top_k, do_norms)

    n_experts_by_layer: Dict[str, int] = {}
    try:
        with torch.no_grad():
            for i, tokens in enumerate(batches):
                # OLMoE forward chokes on attention_mask=None paths in some versions; pass explicit.
                attn = torch.ones_like(tokens, device=tokens.device)
                try:
                    _ = model(input_ids=tokens, attention_mask=attn, use_cache=False)
                except TypeError:
                    _ = model(input_ids=tokens, use_cache=False)
                if norms is not None:
                    norms.flush_batch()
                if (i + 1) % max(1, len(batches) // 10) == 0:
                    _log(f"  batch {i + 1}/{len(batches)}")

        # Infer n_experts per layer from collected indices.
        for layer, lst in router.topk_indices.items():
            if not lst:
                continue
            n_experts_by_layer[layer] = int(lst[0].max().item()) + 1
        # Better: read from model config (more reliable).
        cfg = model.config
        cfg_n_e = (
            getattr(cfg, "num_experts", None)
            or getattr(cfg, "num_local_experts", None)
            or getattr(cfg, "moe_num_experts", None)
        )
        if isinstance(cfg_n_e, int):
            for layer in n_experts_by_layer:
                n_experts_by_layer[layer] = cfg_n_e
        _log(f"n_experts per layer (sample): {list(n_experts_by_layer.values())[:3]}")

    finally:
        for h in handles:
            h.remove()

    coact, singles = _aggregate_coactivation(router.topk_indices, n_experts_by_layer)

    if args.moe_coactivation_output:
        summary = _save_coactivation(
            coact,
            singles,
            Path(args.moe_coactivation_output),
            args.moe_coactivation_top_n,
            f"{args.model}@{args.revision}",
        )
    else:
        summary = None

    if args.moe_token_count_output:
        _save_token_counts(
            singles, Path(args.moe_token_count_output),
            f"{args.model}@{args.revision}", args.revision
        )

    if args.moe_activation_norm_output and norms is not None:
        _save_activation_norms(
            norms.finalize(),
            Path(args.moe_activation_norm_output),
            f"{args.model}@{args.revision}",
            args.moe_act_norm_vector_norm,
            args.moe_act_norm_token_reduce,
        )

    if args.moe_routing_maps_output:
        _save_routing_maps_npz(
            router.topk_indices, n_experts_by_layer, Path(args.moe_routing_maps_output)
        )

    if args.moe_plot:
        plot_dir = Path(args.moe_plot_dir) if args.moe_plot_dir else Path(
            args.moe_coactivation_output
        ).parent
        if summary is not None:
            _plot_coactivation(summary, plot_dir)
        if args.moe_token_count_output:
            _plot_token_counts(Path(args.moe_token_count_output), plot_dir)
        if args.moe_activation_norm_output:
            _plot_activation_norms(Path(args.moe_activation_norm_output), plot_dir)

    _log("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

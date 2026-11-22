"""Check :mod:`hf_config` on training runs, without weights. Per run:

1. **Engram geometry against Megatron.** The table sizes and every layer's hash multipliers, against
   Megatron's own ``engram_table_sizes`` / ``engram_hash_multipliers`` (megatron/core/ssm/engram.py) on
   the same arguments; must be equal. Also shows how many multipliers a builder without the
   ``engram_hash_per_layer`` seed offset would get wrong.
2. **Shapes from the config alone.** The parameter count of every layer, computed from the HF config,
   against the tensors in the checkpoint metadata; must be equal, layer by layer.
3. **Guards.** Turning on unported features must make :func:`hf_config.hf_config` refuse.

    python -m oellm_autoexp.hf_export.engram_scaling_clariden.check_hf_config output/engram_*
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from ..export_q35 import megatron_args
from .hf_config import hf_config, vocab_from_checkpoint

_LAYER = re.compile(r"^decoder\.layers\.(\d+)\.")


def run_paths(run_dir: Path) -> tuple[Path, Path]:
    """(latest resolved config, latest checkpoint) of a run directory."""
    config = sorted(run_dir.glob("config-*.yaml"), key=lambda p: p.stat().st_mtime)[-1]
    it = int((run_dir / "ckpt" / "latest_checkpointed_iteration.txt").read_text().strip())
    return config, run_dir / "ckpt" / f"iter_{it:07d}"


def checkpoint_numel(ckpt_dir: Path) -> dict:
    """{layer index or None (non-layer): number of model elements} from the DCP metadata."""
    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    md = dcp.FileSystemReader(str(ckpt_dir)).read_metadata().state_dict_metadata
    out: dict = defaultdict(int)
    for key, e in md.items():
        if not key.startswith(("embedding.", "decoder.", "output_layer.")) or "_extra_state" in key:
            continue
        if isinstance(e, TensorStorageMetadata):
            m = _LAYER.match(key)
            out[int(m.group(1)) if m else None] += torch.Size(e.size).numel()
    return out


def config_numel(c: dict) -> dict:
    """{layer index or None: number of elements} implied by the HF config (router expert_bias included)."""
    h = c["hidden_size"]
    out = {None: c["vocab_size"] * h + h + (0 if c["tie_word_embeddings"] else c["vocab_size"] * h)}
    qk = c["linear_num_key_heads"] * c["linear_key_head_dim"]
    v = c["linear_num_value_heads"] * c["linear_value_head_dim"]
    nv = c["linear_num_value_heads"]
    heads, kv, d = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    for i, t in enumerate(c["block_types"]):
        if t == "gdn":  # norm, in_proj [q k v z beta alpha], conv over q k v, A_log, dt_bias, out_norm, out_proj
            n = h + h * (2 * qk + 2 * v + 2 * nv) + (2 * qk + v) * c["linear_conv_kernel_dim"] + 2 * nv
            n += c["linear_value_head_dim"] + v * h
        elif t == "attention":  # norm, qkv (+ gate), q/k norms, output projection
            q_rows = heads * d * (2 if c["attention_output_gate"] else 1)
            n = h + h * (q_rows + 2 * kv * d) + (2 * d if c["qk_layernorm"] else 0) + heads * d * h
        elif t == "moe":  # norm, router + expert bias, routed SwiGLU experts, shared expert (+ gate)
            e, f, fs = c["num_experts"], c["moe_intermediate_size"], c["shared_expert_intermediate_size"]
            n = h + e * h + e + e * 3 * h * f + 3 * h * fs + (h if fs and c["shared_expert_gate"] else 0)
        else:  # engram: table, key / value projections, hidden / key (/ conv) norms, depthwise conv
            mem = c["engram_embed_dim"] * len(c["engram_table_sizes"])
            n = sum(c["engram_table_sizes"]) * c["engram_embed_dim"] + 2 * h * mem + 2 * h
            n += (h if c["engram_conv_prenorm"] else 0) + h * c["engram_conv_kernel_size"]
        out[i] = n
    return out


def check_engram(m: dict, c: dict) -> list[str]:
    if "engram_layers" not in c:
        return []
    from megatron.core.ssm import engram as meg

    ns = SimpleNamespace(**m)
    errors = []
    if c["engram_table_sizes"] != meg.engram_table_sizes(ns):
        errors.append("engram_table_sizes differ from Megatron")
    for i in c["engram_layers"]:
        ref = meg.engram_hash_multipliers(ns, i + 1).tolist()
        if c["engram_hash_multipliers"][str(i)] != ref:
            errors.append(f"layer {i}: hash multipliers differ from Megatron")
        no_offset = meg.engram_hash_multipliers(SimpleNamespace(**{**m, "engram_hash_per_layer": False}), i + 1)
        wrong = sum(a != b for ra, rb in zip(ref, no_offset.tolist()) for a, b in zip(ra, rb))
        total = sum(len(r) for r in ref)
        print(f"    layer {i}: multipliers == Megatron ({total} values); "
              f"without the per-layer seed offset {wrong}/{total} would differ")
    return errors


def check_guards(m: dict, vocab: int) -> list[str]:
    errors = []
    for flag, value in (("mhc", True), ("engram_sign_hash", True), ("mtp_num_layers", 1),
                        ("moe_router_score_function", "softmax"), ("hybrid_layer_pattern", "LE-E")):
        try:
            hf_config({**m, flag: value}, vocab)
            errors.append(f"guard: {flag}={value!r} was accepted")
        except ValueError:
            pass
    return errors


def check_run(run_dir: Path) -> bool:
    config, ckpt = run_paths(run_dir)
    m = megatron_args(config)
    have = checkpoint_numel(ckpt)
    c = hf_config(m, vocab_from_checkpoint(ckpt))
    print(f"\n== {run_dir.name}  ({config.name}, {ckpt.name})")
    print(f"    {c['num_hidden_layers']} layers, Engram layers {c.get('engram_layers', [])}, "
          f"token map {c.get('engram_token_map', False)}")
    errors = check_engram(m, c)
    want = config_numel(c)
    for k in sorted(set(want) | set(have), key=lambda k: -1 if k is None else k):
        if want.get(k) != have.get(k):
            errors.append(f"layer {k}: config implies {want.get(k)} elements, checkpoint has {have.get(k)}")
    bias = c["num_experts"] * c["block_types"].count("moe")
    print(f"    parameters: config {sum(want.values()):,} == checkpoint {sum(have.values()):,} "
          f"(incl. {bias:,} router expert_bias; without: {sum(want.values()) - bias:,})")
    errors += check_guards(m, c["vocab_size"])
    for e in errors:
        print(f"    !! {e}")
    print("    OK" if not errors else f"    FAILED ({len(errors)})")
    return not errors


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", type=Path, nargs="+", help="run directories (with config-*.yaml and ckpt/)")
    a = p.parse_args()
    ok = [check_run(r) for r in a.runs]
    print(f"\n{sum(ok)}/{len(ok)} runs OK")
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()

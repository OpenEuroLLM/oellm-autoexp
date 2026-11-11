#!/usr/bin/env python3
"""Export a native 1.7B ladder checkpoint -- including the recall-surgery INTERVENTION arms -- to the fla signed_kda HF
layout (docs/recall_surgery_ambitious.md, section 8). Wraps lm_scaling/megatron_ext/export_megatron.py (its loader,
config, renames and cross-checks) and adds:

* the AddBranchMixer branches: decoder.layers.N.self_attention.branch.X -> model.layers.N.attn.branch.X, their
  layers read off the checkpoint and checked against SKDA_INTERVENTION's layer tokens, and config.recall_branches
  (the hybrid sibling's fla attention spec + add_rope / add_swa window); loaded by hf_branch.load_exported;
* attn_rope: use_rope on the existing attention spec;
* STRICT KEY ACCOUNTING: every checkpoint tensor must be written (TE extra state and a tied output layer aside);
  an unmapped key is an error -- export_megatron.rename returns [] for anything it does not know, which silently
  dropped the branches (and would have evaluated the control under the arm's name).

    export_native.py --run <run ckpt dir | iter dir | .pt> --arm <ladder arm> --out <dir> [--intervention <spec>]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HYBRID_OF = {"skda-shipped-lowrank": "skda-shipped-hybrid-lowrank", "kda-sig-lowrank": "kda-sig-hybrid-lowrank"}
HYBRID_EVERY = 4


def parse_intervention(spec: str) -> dict:
    out = {"add": [], "attn_rope": False, "add_rope": False}
    spec = (spec or "").strip()
    if not spec or spec in ("none", "ctrl"):
        return out
    for item in spec.split("+"):
        p = item.split("@")
        if p[0] == "add_attn":
            out["add"].append((p[1], None))
        elif p[0] == "add_swa":
            out["add"].append((p[2], int(p[1])))
        elif p[0] == "attn_rope":
            out["attn_rope"] = True
        elif p[0] == "add_rope":
            out["add_rope"] = True
        else:
            raise SystemExit(f"intervention item {item!r}: unknown")
    return out


def layer_set(token: str, num_layers: int, every):
    """1-based layer numbers, as native_patch/interventions.layer_set."""
    lin = [i for i in range(1, num_layers + 1) if not every or i % every != 0]
    if token == "lin":
        return lin
    if token == "lin_late":
        return [i for i in lin if i > num_layers // 2]
    if token.startswith("every"):
        k = int(token[len("every"):])
        return [i for i in range(1, num_layers + 1) if i % k == 0]
    return sorted(int(x) for x in token.split(","))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--geometry", default="1.7B")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--intervention", default=os.environ.get("SKDA_INTERVENTION", ""))
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    ap.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    a = ap.parse_args(argv)

    import importlib.util

    import export_hf
    import torch
    from safetensors.torch import save_file

    # export_megatron by FILE: megatron_ext's package __init__ may import Megatron (absent in the eval env)
    lm = Path(export_hf.__file__).resolve().parent
    spec_ = importlib.util.spec_from_file_location("_export_megatron", lm / "megatron_ext" / "export_megatron.py")
    em = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(em)

    import hf_branch

    fla_arms, campaign = em.mixer_constants()
    if a.arm not in fla_arms:
        raise SystemExit(f"{a.arm!r}: not an fla arm of this ladder")
    iv = parse_intervention(a.intervention)
    pt, _ = em.find_checkpoint(a.run)
    model, args, it = em.load_model_tensors(pt)

    cfg = export_hf.hf_config(a.arm, a.geometry)
    cfg["norm_eps"] = float(getattr(args, "layernorm_epsilon", cfg["norm_eps"]))
    cfg["drop_silu"] = bool(campaign["drop_silu"])
    cfg["vocab_size"] = int(args.padded_vocab_size)
    cfg["max_position_embeddings"] = int(args.seq_length)
    cfg["tie_word_embeddings"] = not bool(args.untie_embeddings_and_output_weights)
    em._check_args(args, a.arm, a.geometry, cfg, a.tokenizer)
    n_layers = cfg["num_hidden_layers"]
    if cfg.get("attn"):
        em._check_hybrid(model, cfg, n_layers)

    # the branches: on disk vs the intervention's layer tokens
    on_disk = sorted({int(k.split(".")[2]) for k in model if ".self_attention.branch." in k})
    every = HYBRID_EVERY if cfg.get("attn") else None
    want, windows = set(), set()
    for token, window in iv["add"]:
        for layer in layer_set(token, n_layers, every):
            want.add(layer - 1)
            windows.add(window)
    if sorted(want) != on_disk:
        raise SystemExit(f"branch layers on disk {on_disk} != the intervention's {sorted(want)} "
                         f"(SKDA_INTERVENTION={a.intervention!r})")
    if len(windows) > 1:
        raise SystemExit(f"one branch spec per export: windows {windows}")
    if on_disk:
        sib = a.arm if cfg.get("attn") else HYBRID_OF.get(a.arm)
        if not sib:
            raise SystemExit(f"{a.arm}: no hybrid sibling to take the branch attention flavor from")
        spec = dict(export_hf.hf_config(sib, a.geometry)["attn"])
        spec.pop("layers", None)
        spec["use_rope"] = bool(iv["add_rope"])
        spec["window_size"] = next(iter(windows))
        cfg["recall_branches"] = {"layers": on_disk, "attn": spec}
    if iv["attn_rope"]:
        if not cfg.get("attn"):
            raise SystemExit("attn_rope on a stack without attention layers")
        cfg["attn"] = dict(cfg["attn"], use_rope=True)
    if on_disk:
        cfg["recall_intervention"] = a.intervention

    out, unmapped = {}, []
    for k, v in model.items():
        if ".self_attention.branch." in k:
            n, inner = k.split(".")[2], k.split(".self_attention.branch.", 1)[1]
            pairs = [(f"model.layers.{n}.attn.branch.{inner}", v)]
        else:
            pairs = em.rename(k, cfg["intermediate_size"], v)
        if not pairs:
            if k.endswith("_extra_state") or (k == "output_layer.weight" and cfg["tie_word_embeddings"]):
                continue
            unmapped.append(k)
        for name, t in pairs:
            out[name] = t.to(getattr(torch, a.dtype)).contiguous()
    if unmapped:
        raise SystemExit(f"{len(unmapped)} checkpoint tensors have no HF name (they would be DROPPED): {unmapped[:6]}")
    if cfg["tie_word_embeddings"]:
        out.pop("lm_head.weight", None)

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    save_file(out, str(a.out / "model.safetensors"), metadata={"format": "pt"})
    n = sum(t.numel() for t in out.values())
    print(f"[export_native] {a.out}: {len(out)} tensors, {n:,} parameters; {a.arm} iteration {it}; "
          f"intervention {a.intervention!r}: branches {on_disk}, attn_rope {iv['attn_rope']}")
    # structural gate: strict load (branches included) through the eval's own loader
    hf_branch.install()
    export_hf.load_exported(a.out)
    print("[export_native] load check: strict, every tensor placed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

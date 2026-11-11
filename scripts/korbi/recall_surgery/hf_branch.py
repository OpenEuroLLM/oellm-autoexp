"""HF side of the recall-surgery intervention arms (docs/recall_surgery_ambitious.md, section 8).

The native AddBranchMixer (scripts/korbi/recall_surgery/native_patch/interventions.py) is the CKDA mixer plus a
parallel fla attention branch on the same normalized input, out = mixer(x) + branch(x). The export
(export_native.py) writes its tensors as model.layers.N.attn.branch.* and the config field

    "recall_branches": {"layers": [0-based layer indices], "attn": {<fla Attention spec: num_heads, num_kv_heads,
                        qkv_bias, qk_norm, output_gate, use_rope, window_size, rope_theta>}}

`load_exported` builds the stock fla SignedKDAForCausalLM (upstream fla untouched), attaches an fla Attention as
`attn.branch` to the named blocks' mixers, wraps their forward to add its output, and loads STRICTLY: any missing or
unexpected tensor is an error. attn_rope needs nothing here (the export sets `use_rope` in the existing attn spec).

KV cache: the branch keeps a cache of its own (an fla Cache, layer slot 0), reset whenever the model's cache is new
for this layer -- one fla Cache cannot hold two attention states for one layer index.
"""
from __future__ import annotations

import json
import types
from pathlib import Path

RECALL_KEYS = ("recall_branches", "recall_intervention")


def _attach(mixer, spec: dict, hidden: int, max_pos: int, layer_idx: int):
    from fla.layers.attn import Attention
    from fla.models.utils import Cache

    if hasattr(mixer, "branch"):
        raise SystemExit(f"layer {layer_idx}: the mixer has a `branch` attribute already")
    mixer.add_module("branch", Attention(
        hidden_size=hidden, num_heads=spec["num_heads"], num_kv_heads=spec["num_kv_heads"],
        qkv_bias=spec["qkv_bias"], qk_norm=spec.get("qk_norm", False), output_gate=spec.get("output_gate", False),
        use_rope=spec.get("use_rope", False), window_size=spec.get("window_size"),
        rope_theta=spec.get("rope_theta", 10000.0), max_position_embeddings=max_pos, layer_idx=0))
    state = {"main": None, "cache": None}
    orig = mixer.forward

    def forward(self, hidden_states, attention_mask=None, past_key_values=None, use_cache=False,
                output_attentions=False, **kwargs):
        bcache = None
        if past_key_values is not None:
            fresh = past_key_values is not state["main"] or len(past_key_values) <= layer_idx
            if fresh:
                state["main"], state["cache"] = past_key_values, Cache.from_legacy_cache(None)
            bcache = state["cache"]
        b, _, _ = self.branch(hidden_states=hidden_states, attention_mask=attention_mask, past_key_values=bcache,
                              use_cache=use_cache, output_attentions=False)
        o, attn, pkv = orig(hidden_states=hidden_states, attention_mask=attention_mask,
                            past_key_values=past_key_values, use_cache=use_cache,
                            output_attentions=output_attentions, **kwargs)
        return o + b, attn, pkv

    mixer.forward = types.MethodType(forward, mixer)


def load_exported(path, dtype="bfloat16", _orig=None):
    """export_hf.load_exported, plus the recall_branches; falls through to the original for other exports."""
    import torch
    from safetensors.torch import load_file

    path = Path(path)
    raw = json.loads((path / "config.json").read_text())
    if not raw.get("recall_branches"):
        clean = {k: v for k, v in raw.items() if k not in RECALL_KEYS}
        if clean != raw:  # an intervention export without branches (attn_rope): the config minus our keys
            raise SystemExit(f"{path}: recall_* keys without branches are handled by the plain loader -- the "
                             f"exporter should not write them")
        return _orig(path, dtype)
    from fla.models.signed_kda import SignedKDAForCausalLM
    from fla.models.signed_kda.configuration_signed_kda import SignedKDAConfig

    tied = bool(raw.get("tie_word_embeddings", False))
    cfg = SignedKDAConfig(**{k: (False if k == "tie_word_embeddings" else v) for k, v in raw.items()
                             if k not in ("architectures", "model_type") + RECALL_KEYS})
    model = SignedKDAForCausalLM(cfg)
    br = raw["recall_branches"]
    for i in br["layers"]:
        _attach(model.model.layers[i].attn, br["attn"], raw["hidden_size"], raw["max_position_embeddings"], i)
    missing, unexpected = model.load_state_dict(load_file(str(path / "model.safetensors")), strict=False)
    if tied:
        model.lm_head.weight = model.model.embeddings.weight
        missing = [k for k in missing if k != "lm_head.weight"]
    if missing or unexpected:
        raise SystemExit(f"{path}: {len(missing)} missing and {len(unexpected)} unexpected tensors with "
                         f"{len(br['layers'])} branches -- missing[:4]={missing[:4]} unexpected[:4]={unexpected[:4]}")
    for k in raw:
        if k in RECALL_KEYS:
            setattr(cfg, k, raw[k])
    print(f"[hf_branch] {path.name}: {len(br['layers'])} attention branches at layers {br['layers']} "
          f"(use_rope {br['attn'].get('use_rope')}, window {br['attn'].get('window_size')})", flush=True)
    return model.to(getattr(torch, dtype)).eval(), cfg


def install():
    """Route export_hf.load_exported (used by eval_downstream.build_lm) through this loader."""
    import export_hf
    if getattr(export_hf.load_exported, "_recall_branches", False):
        return
    orig = export_hf.load_exported

    def patched(path, dtype="bfloat16"):
        return load_exported(path, dtype, _orig=orig)
    patched._recall_branches = True
    export_hf.load_exported = patched

"""Check an export of :mod:`export_hf`, cheapest first:

1. **Structure** (no weights loaded): the config loads with ``trust_remote_code``, the model builds on the
   meta device, and its tensor names / shapes equal the safetensors headers; every file dtype follows
   the export's policy (router fp32, token map int64, Engram table as configured, the rest bf16).
2. **Conservation** (reads the source checkpoint again): every Megatron model tensor is found in the
   export at the place an *independent* inverse of the name table says (the GDN sections at their row
   offsets), with the same values (bitwise, after the dtype cast); every exported element is covered
   exactly once; the token map equals the ``.npy`` it came from.
3. **Strict load**: ``from_pretrained(dtype=bf16)`` reports no missing / unexpected / mismatched tensors.
4. **Forward**: mean NLL of a short multilingual text far below ``log(vocab)`` (a layout mistake lands
   near it), and greedy continuations of a few prompts (cached == uncached).

    python -m oellm_autoexp.hf_export.engram_scaling_clariden.check_hf_export <export dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch

from ..validate_hf_export import PROBE_TEXT
from .export_hf import iter_model_tensors

RESULTS: list[tuple[str, bool, str]] = []
_ST_DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "I64": torch.int64}
PROMPTS = ("The capital of France is", "Die Hauptstadt von Deutschland ist", "def fibonacci(n):")


def report(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, ok, detail))
    print(f"  [{'ok' if ok else 'FAIL'}] {name}: {detail}")


class Export:
    """The export's tensors by name, read lazily from its shards."""

    def __init__(self, d: Path) -> None:
        from safetensors import safe_open

        self.dir = d
        weight_map = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
        self.files = {shard: safe_open(str(d / shard), framework="pt") for shard in set(weight_map.values())}
        self.where = weight_map
        self.shape = {k: tuple(self.files[s].get_slice(k).get_shape()) for k, s in weight_map.items()}
        self.dtype = {k: _ST_DTYPES[self.files[s].get_slice(k).get_dtype()] for k, s in weight_map.items()}

    def get(self, name: str) -> torch.Tensor:
        return self.files[self.where[name]].get_tensor(name)


def expected_dtype(name: str, cfg) -> torch.dtype:
    if ".mlp.router." in name:
        return torch.float32
    if name.endswith("engram.token_map"):
        return torch.int64
    if name.endswith("engram.table"):
        return {"bf16": torch.bfloat16, "fp32": torch.float32}[cfg.engram_table_dtype]
    return torch.bfloat16


def check_structure(d: Path, ex: Export):
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(d, trust_remote_code=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    sd = model.state_dict()
    want = {k: tuple(v.shape) for k, v in sd.items() if not (cfg.tie_word_embeddings and k == "lm_head.weight")}
    missing, unexpected = sorted(set(want) - set(ex.shape)), sorted(set(ex.shape) - set(want))
    wrong = sorted(k for k in set(want) & set(ex.shape) if want[k] != ex.shape[k])
    report(f"structure: {type(model).__name__}, {len(cfg.block_types)} layers, {len(want)} tensors",
           not (missing or unexpected or wrong),
           f"missing {missing[:3]}, unexpected {unexpected[:3]}, shape mismatches {wrong[:3]}")
    bad = sorted(k for k in ex.dtype if ex.dtype[k] != expected_dtype(k, cfg))
    counts: dict[str, int] = {}
    for k in ex.dtype:
        counts[str(ex.dtype[k]).replace("torch.", "")] = counts.get(str(ex.dtype[k]).replace("torch.", ""), 0) + 1
    report("file dtypes: router fp32, token map int64, Engram table as configured, the rest bf16", not bad,
           f"{counts}{'' if not bad else f', unexpected: {bad[:3]}'}")
    return cfg


# --------------------------------------------------------------------------- conservation


def hf_place(meg: str, cfg) -> tuple[str, int]:
    """Megatron name (within ``decoder.layers.i.`` for layer tensors) -> (HF name, first row in it).
    Written independently of export_hf.convert_layer, from the README's stage 3 table."""
    m = re.match(r"^decoder\.layers\.(\d+)\.(.+)$", meg)
    if not m:
        return {"embedding.word_embeddings.weight": "model.embed_tokens.weight",
                "decoder.final_norm.weight": "model.norm.weight", "output_layer.weight": "lm_head.weight"}[meg], 0
    i, rest = int(m.group(1)), m.group(2)
    p = f"model.layers.{i}."
    qk = cfg.linear_num_key_heads * cfg.linear_key_head_dim
    v = cfg.linear_num_value_heads * cfg.linear_value_head_dim
    nv = cfg.linear_num_value_heads
    in_proj_rows = {"query": 0, "key": qk, "value": 2 * qk, "z": 2 * qk + v, "beta": 2 * qk + 2 * v,
                    "alpha": 2 * qk + 2 * v + nv}
    conv_rows = {"query": 0, "key": qk, "value": 2 * qk}
    rules = [
        (r"self_attention\.in_proj\.layer_norm_weight", "input_layernorm.weight"),
        (r"self_attention\.linear_qkv\.layer_norm_weight", "input_layernorm.weight"),
        (r"pre_mlp_layernorm\.weight", "input_layernorm.weight"),
        (r"self_attention\.(A_log|dt_bias|out_norm\.weight|out_proj\.weight)", r"linear_attn.\1"),
        (r"self_attention\.linear_qkv\.weight", "self_attn.qkv_proj.weight"),
        (r"self_attention\.linear_proj\.weight", "self_attn.o_proj.weight"),
        (r"self_attention\.q_layernorm\.weight", "self_attn.q_norm.weight"),
        (r"self_attention\.k_layernorm\.weight", "self_attn.k_norm.weight"),
        (r"mlp\.router\.(weight|expert_bias)", r"mlp.router.\1"),
        (r"mlp\.experts\.experts\.linear_fc1\.weight", "mlp.experts_gate_up"),
        (r"mlp\.experts\.experts\.linear_fc2\.weight", "mlp.experts_down"),
        (r"mlp\.shared_experts\.linear_fc1\.weight", "mlp.shared_gate_up.weight"),
        (r"mlp\.shared_experts\.linear_fc2\.weight", "mlp.shared_down.weight"),
        (r"mlp\.shared_experts\.gate_weight", "mlp.shared_gate.weight"),
        (r"(table|(?:key_proj|value_proj|hidden_norm|key_norm|conv_norm|conv1d)\.weight)", r"engram.\1"),
    ]
    sec = re.fullmatch(r"self_attention\.(in_proj|conv1d)\.weight\.(\w+)", rest)
    if sec:
        rows = in_proj_rows if sec.group(1) == "in_proj" else conv_rows
        return p + f"linear_attn.{sec.group(1)}.weight", rows[sec.group(2)]
    for pattern, target in rules:
        if re.fullmatch(pattern, rest):
            return p + re.sub(pattern, target, rest), 0
    raise KeyError(f"no HF place for {meg}")


def check_conservation(ex: Export, cfg, source: dict) -> None:
    covered = {k: 0 for k in ex.shape}
    n_tensors = n_elements = 0
    mismatched = []
    for i, tensors in iter_model_tensors(Path(source["checkpoint"])):
        for name, t in tensors.items():
            meg = name if i is None else f"decoder.layers.{i}.{name}"
            hf, row = hf_place(meg, cfg)
            if hf not in ex.shape:
                mismatched.append(f"{meg} -> {hf} (not in the export)")
                continue
            got = ex.get(hf)[row: row + t.shape[0]]
            if got.shape != t.shape or not torch.equal(got, t.to(got.dtype)):
                mismatched.append(f"{meg} -> {hf}[{row}:]")
            covered[hf] += t.numel()
            n_tensors += 1
            n_elements += t.numel()
    exported = {k: math.prod(s) for k, s in ex.shape.items()}
    token_maps = [k for k in exported if k.endswith("engram.token_map")]
    uncovered = sorted(k for k in exported if k not in token_maps and covered[k] != exported[k])
    report(f"conservation: {n_tensors} Megatron tensors, {n_elements:,} elements -> the export",
           not mismatched and not uncovered,
           f"values bitwise equal after the dtype cast, every exported element covered once"
           if not (mismatched or uncovered) else f"mismatched {mismatched[:3]}, coverage wrong {uncovered[:3]}")
    if token_maps:
        tm = source["engram_token_map"]
        raw = Path(tm["path"]).read_bytes()
        ref = torch.from_numpy(np.load(tm["path"])[: cfg.vocab_size].astype(np.int64))
        ok = hashlib.sha256(raw).hexdigest() == tm["sha256"] and all(torch.equal(ex.get(k), ref) for k in token_maps)
        report(f"token map: {len(token_maps)} Engram layer(s) == {Path(tm['path']).name} (sha256 recorded)", ok,
               f"{ref.numel()} ids -> {int(ref.unique().numel())} canonical ids")


# --------------------------------------------------------------------------- load and forward


def check_load_and_forward(d: Path, cfg, device: str) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model, info = AutoModelForCausalLM.from_pretrained(d, trust_remote_code=True, dtype=torch.bfloat16,
                                                       output_loading_info=True)
    problems = {k: v for k, v in info.items() if v}
    report("strict load: from_pretrained(dtype=bf16)", not problems, str(problems) if problems else
           "no missing / unexpected / mismatched tensors")
    model.to(device).eval()
    moe = next(layer.mlp for layer in model.model.layers if layer.block_type == "moe")
    report("loaded dtypes: router fp32, weights bf16",
           moe.router.expert_bias.dtype == torch.float32 and moe.experts_down.dtype == torch.bfloat16,
           f"router {moe.router.weight.dtype}, expert bias {moe.router.expert_bias.dtype}, experts {moe.experts_down.dtype}")

    tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
    ids = tok(PROBE_TEXT, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        logits = model(ids, use_cache=False).logits.float()
    nll = torch.nn.functional.cross_entropy(logits[0, :-1], ids[0, 1:]).item()
    chance = math.log(cfg.vocab_size)
    report(f"forward: mean NLL on a {ids.shape[1]}-token multilingual text", nll < chance - 3,
           f"{nll:.3f} nats (chance: log V = {chance:.2f})")

    # Cached decoding (prefill + one-token steps) and a full forward over the same tokens round
    # differently in bf16, so greedy tokens can differ at near ties; compare the distributions instead,
    # and require every token where the two argmaxes differ to be a near tie of the full forward.
    kls, flips, worst_gap = [], 0, 0.0
    for prompt in PROMPTS:
        p = tok(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.generate(p, max_new_tokens=24, do_sample=False, pad_token_id=cfg.pad_token_id,
                                 use_cache=True, output_logits=True, return_dict_in_generate=True)
            step = torch.cat(out.logits).float()  # [new tokens, vocab], one per cached step
            full = model(out.sequences, use_cache=False).logits[0, p.shape[1] - 1: -1].float()
        ls, lf = torch.log_softmax(step, -1), torch.log_softmax(full, -1)
        kls.append((lf.exp() * (lf - ls)).sum(-1))
        chosen = out.sequences[0, p.shape[1]:]
        for t in torch.nonzero(chosen != full.argmax(-1)).flatten().tolist():
            flips += 1
            worst_gap = max(worst_gap, (lf[t].max() - lf[t, chosen[t]]).item())
        print(f"      {prompt!r} -> {tok.decode(chosen)!r}")
    kl = torch.cat(kls)
    report(f"generate: cached steps vs a full forward over the same {kl.numel()} tokens",
           kl.mean().item() < 5e-3 and worst_gap < 0.5,
           f"mean KL {kl.mean():.1e} (max {kl.max():.1e}) nats; {flips} argmax flips, "
           f"largest log-prob gap at a flip {worst_gap:.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("export_dir", type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-conservation", action="store_true", help="do not re-read the source checkpoint")
    a = p.parse_args()

    source = json.loads((a.export_dir / "megatron_source.json").read_text())
    print(f"== {a.export_dir}  (from {Path(source['run_dir']).name}, iteration {source['iteration']})")
    ex = Export(a.export_dir)
    cfg = check_structure(a.export_dir, ex)
    if not a.skip_conservation:
        check_conservation(ex, cfg, source)
    check_load_and_forward(a.export_dir, cfg, a.device)
    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

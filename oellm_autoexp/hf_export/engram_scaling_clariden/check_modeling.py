"""Checks of the HF model code (``oellm_engram/``), without a trained checkpoint. Run on a GPU node.

1. **Operations against Megatron's own functions** on random inputs: RoPE (``RotaryEmbedding`` +
   ``_apply_rotary_pos_emb_bshd``, interleaved, partial), the Engram hash rows (``engram.engram_hash``
   + table offsets, with a token map), the router (``moe_utils.topk_routing_with_score_function``).
2. **Kernels against the reference:** a GDN layer on the GPU (FLA chunk / recurrent kernels) against
   the same layer on the CPU (fp32 recurrence), prefill and one-token steps.
3. **A tiny random model** with every block type (two Engram layers, a token map), fp32 on the CPU:
   cached decoding, chunked prefill and left padding each give the logits of one full forward pass;
   ``save_pretrained`` + ``AutoModelForCausalLM.from_pretrained(trust_remote_code=True)`` gives the same
   logits (and keeps the router fp32 under bf16); cached ``generate`` equals uncached; the GPU bf16
   forward stays close to the CPU fp32 one.
4. **The real architecture:** the model built (on the meta device) from an exported ``config.json`` has
   exactly the checkpoint's number of elements; one layer of each type is listed with its tensors.

    python -m oellm_autoexp.hf_export.engram_scaling_clariden.check_modeling \\
        --config-json <dir with config.json from hf_config> --expect-numel 1174548320
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import shutil
import sys
import tempfile
from pathlib import Path

import torch

from .oellm_engram import OellmEngramConfig, OellmEngramForCausalLM
from .oellm_engram.modeling_oellm_engram import (
    OellmGatedDeltaNet,
    OellmRotaryEmbedding,
    _apply_rope,
)

_PKG = Path(__file__).resolve().parent / "oellm_engram"
RESULTS: list[tuple[str, bool, str]] = []


def report(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, ok, detail))
    print(f"  [{'ok' if ok else 'FAIL'}] {name}: {detail}")


def maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def tiny_config(vocab: int = 512) -> OellmEngramConfig:
    import numpy as np

    pattern = "LENLE*ELENE*E"  # gdn, moe, engram, attention; two Engram layers
    kinds = {"L": "gdn", "E": "moe", "N": "engram", "*": "attention"}
    types = [kinds[c] for c in pattern]
    engram = [i for i, t in enumerate(types) if t == "engram"]
    rng = np.random.default_rng(0)
    orders, heads = [2, 3], 2
    return OellmEngramConfig(
        vocab_size=vocab, hidden_size=64, block_types=types, rms_norm_eps=1e-6,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32, rope_theta=1e7, partial_rotary_factor=0.25,
        rope_interleaved=True, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
        linear_value_head_dim=16, num_experts=8, num_experts_per_tok=3, moe_intermediate_size=32,
        shared_expert_intermediate_size=32, engram_layers=engram, engram_ngram_orders=orders,
        engram_num_heads=heads, engram_embed_dim=8, engram_table_sizes=[101, 103, 107, 109],
        engram_hash_multipliers={str(i): (rng.integers(1 << 20, 1 << 31, size=(len(orders) * heads, 3)) | 1).tolist()
                                 for i in engram},
        engram_gate_signed_sqrt=True, engram_conv_prenorm=True, engram_token_map=True,
        bos_token_id=1, eos_token_id=2, pad_token_id=2,
    )


def randomize(model: torch.nn.Module, seed: int = 0) -> None:
    """Every float parameter / buffer random (norm weights near their identity), the token map a
    random many-to-one map: trained-like values, so that layout mistakes show."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, t in list(model.named_parameters()) + list(model.named_buffers()):
            if name.endswith("token_map"):
                t.copy_(torch.randint(0, t.numel() // 2, t.shape, generator=g))
            elif t.is_floating_point():
                scale = 0.3 if "norm" in name or "A_log" in name or "dt_bias" in name or "bias" in name else 0.1
                t.copy_(torch.randn(t.shape, generator=g) * scale)


# --------------------------------------------------------------------------- 1. operations vs Megatron


def check_rope() -> None:
    """RoPE as HybridModel applies it: its RotaryEmbedding is built without ``rotary_interleaved``
    (hybrid_model.py), the attention applies it with ``config.rotary_interleaved`` through
    ``rope_utils.apply_rotary_pos_emb`` -- TE's fused kernel under ``apply_rope_fusion`` (the training
    setting), the unfused one otherwise."""
    from types import SimpleNamespace

    from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

    s, b, heads, d = 37, 2, 3, 128
    meg = RotaryEmbedding(kv_channels=d, rotary_percent=0.25, rotary_base=10000000, use_cpu_initialization=True)
    freqs = meg.get_emb(s)  # [s, 1, 1, 32] on the GPU (forward() = this + the context-parallel split)
    t = torch.randn(s, b, heads, d, device=freqs.device, dtype=torch.bfloat16)
    cos, sin = OellmRotaryEmbedding(32, 1e7, interleaved_freqs=False)(torch.arange(s, device=t.device)[None].expand(b, s))
    for interleaved in (True, False):
        ours = _apply_rope(t.permute(1, 2, 0, 3), cos, sin, interleaved).permute(2, 0, 1, 3).float()
        for fused in (True, False):
            cfg = SimpleNamespace(apply_rope_fusion=fused, rotary_interleaved=interleaved, mrope_section=None,
                                  multi_latent_attention=False)
            ref = apply_rotary_pos_emb(t, freqs, cfg, cp_group=object()).float()
            diff = maxdiff(ours, ref)
            report(f"RoPE vs HybridModel's (rotary_interleaved={interleaved}, "
                   f"{'TE fused' if fused else 'unfused'}, 32 of 128 dims, bf16)", diff < 2e-2,
                   f"max |diff| {diff:.2e} (inputs ~N(0, 1), bf16 rounding ~1e-2)")
    wrong = _apply_rope(t.permute(1, 2, 0, 3), *OellmRotaryEmbedding(32, 1e7, True)(
        torch.arange(s, device=t.device)[None].expand(b, s)), True).permute(2, 0, 1, 3).float()
    ref = apply_rotary_pos_emb(t, freqs, SimpleNamespace(apply_rope_fusion=True, rotary_interleaved=True,
                                                         mrope_section=None, multi_latent_attention=False),
                               cp_group=object()).float()
    report("negative control: the textbook interleaved layout [f0, f0, f1, f1, ...] does NOT match",
           maxdiff(wrong, ref) > 0.1, f"max |diff| {maxdiff(wrong, ref):.2e}")


def check_engram_hash(config: OellmEngramConfig, model: OellmEngramForCausalLM) -> None:
    from megatron.core.ssm.engram import engram_hash

    layer = config.engram_layers[0]
    mod = model.model.layers[layer].engram
    ids = torch.randint(0, config.vocab_size, (3, 50))
    mapped = mod.token_map[ids]
    mult = torch.tensor(config.engram_hash_multipliers[str(layer)], dtype=torch.int64)
    offsets = torch.tensor(mod.row_offsets)
    ref = engram_hash(mapped, config.engram_ngram_orders, config.engram_num_heads, mult,
                      config.engram_table_sizes) + offsets
    shifted = torch.cat([torch.zeros(3, max(config.engram_ngram_orders) - 1, dtype=torch.long), mapped + 1], dim=1)
    ours = mod.rows(shifted, ids.shape[1])
    report("Engram hash rows vs Megatron engram_hash (token map, 2 orders x 2 heads)", torch.equal(ours, ref),
           f"{ref.numel()} rows, {int((ours != ref).sum())} differ")


def check_router(model: OellmEngramForCausalLM) -> None:
    from megatron.core.transformer.moe.moe_utils import topk_routing_with_score_function

    moe = next(layer.mlp for layer in model.model.layers if layer.block_type == "moe")
    x = torch.randn(200, moe.router.weight.shape[1])
    w, idx = moe.router(x)
    logits = torch.nn.functional.linear(x.float(), moe.router.weight.float())
    probs, ref_idx = topk_routing_with_score_function(
        logits, moe.router.top_k, score_function="sigmoid", expert_bias=moe.router.expert_bias,
        fused=False, dense_output=True,
    )
    dense = torch.zeros_like(logits).scatter(1, idx, w)
    ref_dense = torch.zeros_like(logits).scatter(1, ref_idx, probs.float())
    diff = maxdiff(dense, ref_dense)
    report("router (sigmoid + expert bias, top-3 of 8) vs Megatron topk_routing_with_score_function",
           diff < 1e-6, f"max |diff| of the dense routing weights {diff:.2e}")


# --------------------------------------------------------------------------- 2. kernels vs reference


@contextlib.contextmanager
def recurrent_gdn_kernel():
    """Run every GDN call on FLA's recurrent kernel (plain fp32 arithmetic) instead of the chunk kernel
    (whose intra-chunk matmuls use TF32 tensor cores)."""
    from .oellm_engram import modeling_oellm_engram as mod

    kernel = mod._fla_gated_delta_rule
    mod._fla_gated_delta_rule = lambda single_step: kernel(True)
    try:
        yield
    finally:
        mod._fla_gated_delta_rule = kernel


def check_gdn_kernels(config: OellmEngramConfig) -> None:
    """The recurrent kernel computes in plain fp32 (it must match the reference recurrence closely);
    the chunk kernel's intra-chunk matmuls run on tensor cores in TF32 (~1e-3 relative)."""
    torch.manual_seed(0)
    layer = OellmGatedDeltaNet(config)
    randomize(layer)
    x = torch.randn(2, 70, config.hidden_size)
    ref, (_, ref_state) = layer(x)
    scale = ref.abs().max().item()

    with recurrent_gdn_kernel():
        out, _ = layer.cuda()(x.cuda())
    d = maxdiff(out.cpu(), ref) / scale
    report("GDN: FLA recurrent kernel over 70 tokens vs fp32 recurrence (same math)", d < 1e-4,
           f"max |diff| / max |out| {d:.1e}")

    out, (conv_state, state) = layer.cuda()(x.cuda())
    d_out, d_state = maxdiff(out.cpu(), ref) / scale, maxdiff(state.cpu(), ref_state) / ref_state.abs().max().item()
    report("GDN: FLA chunk kernel (TF32 tensor cores) vs fp32 recurrence, 70 tokens", d_out < 1e-2 and d_state < 1e-2,
           f"max |diff| / max: output {d_out:.1e}, final state {d_state:.1e}")

    step_ref, _ = layer.cpu()(x[:, :1], (conv_state.cpu(), state.cpu()))
    step, _ = layer.cuda()(x[:, :1].cuda(), (conv_state, state))
    d = maxdiff(step.cpu(), step_ref) / step_ref.abs().max().item()
    report("GDN: one-token step from the chunk kernel's state vs reference", d < 1e-2, f"max |diff| / max {d:.1e}")


# --------------------------------------------------------------------------- 3. tiny model


def check_tiny_model(config: OellmEngramConfig, model: OellmEngramForCausalLM) -> None:
    torch.manual_seed(1)
    ids = torch.randint(3, config.vocab_size, (2, 24))
    with torch.no_grad():
        full = model(ids, use_cache=False).logits
        report("forward finite, logits [2, 24, vocab]", bool(full.isfinite().all()) and full.shape[1] == 24,
               f"logit std {full.std():.3f}")

        out = model(ids[:, :10], use_cache=True)
        steps, cache = [out.logits], out.past_key_values
        for t in range(10, 24):
            out = model(ids[:, t:t + 1], past_key_values=cache, use_cache=True)
            steps.append(out.logits)
        d = maxdiff(torch.cat(steps, 1), full)
        report("cached decoding (prefill 10 + 14 one-token steps) == full forward", d < 1e-4, f"max |diff| {d:.1e}")

        out = model(ids[:, :13], use_cache=True)
        rest = model(ids[:, 13:], past_key_values=out.past_key_values, use_cache=True)
        d = maxdiff(torch.cat([out.logits, rest.logits], 1), full)
        report("chunked prefill (13 + 11 tokens) == full forward", d < 1e-4, f"max |diff| {d:.1e}")

        pad = 5
        padded = torch.cat([torch.full((2, pad), config.pad_token_id), ids], 1)
        mask = torch.cat([torch.zeros(2, pad, dtype=torch.long), torch.ones_like(ids)], 1)
        d = maxdiff(model(padded, attention_mask=mask, use_cache=False).logits[:, pad:], full)
        report("left padding (5 pad tokens, attention_mask) == unpadded", d < 1e-4, f"max |diff| {d:.1e}")


def check_roundtrip(config: OellmEngramConfig, model: OellmEngramForCausalLM) -> None:
    from transformers import AutoModelForCausalLM

    ids = torch.randint(3, config.vocab_size, (2, 20))
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        cfg = json.loads((Path(tmp) / "config.json").read_text())
        cfg["auto_map"] = {"AutoConfig": "configuration_oellm_engram.OellmEngramConfig",
                           "AutoModelForCausalLM": "modeling_oellm_engram.OellmEngramForCausalLM"}
        (Path(tmp) / "config.json").write_text(json.dumps(cfg))
        for f in ("configuration_oellm_engram.py", "modeling_oellm_engram.py"):
            shutil.copy2(_PKG / f, Path(tmp) / f)
        loaded = AutoModelForCausalLM.from_pretrained(tmp, trust_remote_code=True, dtype=torch.float32)
        with torch.no_grad():
            d = maxdiff(loaded(ids, use_cache=False).logits, model(ids, use_cache=False).logits)
        report("save_pretrained + from_pretrained(trust_remote_code) == the saved model", d == 0.0, f"max |diff| {d:.1e}")

        bf16 = AutoModelForCausalLM.from_pretrained(tmp, trust_remote_code=True, dtype=torch.bfloat16)
        moe = next(layer.mlp for layer in bf16.model.layers if layer.block_type == "moe")
        dtypes = (moe.router.weight.dtype, moe.router.expert_bias.dtype, moe.experts_down.dtype)
        report("from_pretrained(dtype=bf16): router weight / expert bias fp32, experts bf16",
               dtypes == (torch.float32, torch.float32, torch.bfloat16), str(dtypes))

        prompt = ids[:1, :8]
        kw = dict(max_new_tokens=12, do_sample=False, pad_token_id=config.pad_token_id)
        with torch.no_grad():
            cached = loaded.generate(prompt, use_cache=True, **kw)
            uncached = loaded.generate(prompt, use_cache=False, **kw)
        report("greedy generate: cached == uncached (12 new tokens)", torch.equal(cached, uncached),
               f"{cached[0, 8:].tolist()}")


def check_gpu_bf16(config: OellmEngramConfig, model: OellmEngramForCausalLM) -> None:
    """The GPU path (FLA kernels, SDPA, cuBLAS) against the CPU reference, whole model. With the GDN on
    the recurrent kernel the two compute the same fp32 arithmetic. The chunk kernel's TF32 noise and bf16
    rounding are small per layer, but in a random tiny model they can flip near-tie top-k expert choices;
    both are measured against the bf16 rounding of the CPU reference itself."""
    ids = torch.randint(3, config.vocab_size, (2, 64))

    def logp(device, dtype):  # on a copy: a cast to bf16 and back would round the fp32 weights
        m = copy.deepcopy(model).to(device, dtype)
        return torch.log_softmax(m(ids.to(device), use_cache=False).logits.float(), -1).cpu()

    with torch.no_grad():
        ref, cpu_bf16, gpu_fp32, gpu_bf16 = (logp(d, t) for d, t in (
            ("cpu", torch.float32), ("cpu", torch.bfloat16), ("cuda", torch.float32), ("cuda", torch.bfloat16)))
        with recurrent_gdn_kernel():
            gpu_exact = logp("cuda", torch.float32)

    def kl(p):
        return (ref.exp() * (ref - p)).sum(-1).mean().item()

    report("GPU fp32, GDN on the recurrent kernel, vs CPU fp32: same fp32 arithmetic",
           maxdiff(gpu_exact, ref) < 1e-4,
           f"max |log-prob diff| {maxdiff(gpu_exact, ref):.1e}, KL {kl(gpu_exact):.1e} nats")
    report("GPU fp32, GDN on the chunk kernel (TF32), vs CPU fp32: below the bf16 rounding baseline",
           kl(gpu_fp32) < kl(cpu_bf16), f"KL {kl(gpu_fp32):.1e} nats (CPU bf16: {kl(cpu_bf16):.1e})")
    report("GPU bf16 vs CPU fp32: within 3x the bf16 rounding baseline (CPU bf16)",
           kl(gpu_bf16) < 3 * kl(cpu_bf16), f"GPU bf16 {kl(gpu_bf16):.1e}, CPU bf16 {kl(cpu_bf16):.1e} nats")


# --------------------------------------------------------------------------- 4. the real architecture


def check_real_config(config_json: Path, expect_numel: int | None) -> None:
    cfg = json.loads((config_json / "config.json").read_text())
    for k in ("architectures", "auto_map", "model_type", "torch_dtype"):
        cfg.pop(k, None)
    config = OellmEngramConfig(**cfg)
    with torch.device("meta"):
        model = OellmEngramForCausalLM(config)
    state = model.state_dict()
    numel = sum(t.numel() for k, t in state.items() if k != "lm_head.weight")  # tied: counted once
    ok = expect_numel is None or numel == expect_numel
    report(f"{config_json.name}: model built from config.json, {len(config.block_types)} layers",
           ok, f"{numel:,} elements (tied embedding once){'' if expect_numel is None else f', checkpoint {expect_numel:,}'}")
    seen = set()
    for name, t in state.items():
        parts = name.split(".")
        if name.startswith("model.layers."):
            kind = config.block_types[int(parts[2])]
            if kind in seen and parts[2] != str(next(i for i, b in enumerate(config.block_types) if b == kind)):
                continue
            seen.add(kind)
        print(f"      {name:58s} {str(list(t.shape)):20s} {str(t.dtype).replace('torch.', '')}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-json", type=Path, help="a directory with an exported config.json")
    p.add_argument("--expect-numel", type=int, help="the checkpoint's model elements (check_hf_config)")
    a = p.parse_args()

    # the NGC containers set TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1: fp32 matmuls / convs on the GPU would
    # run in TF32 (~1e-3 relative), which the fp32 GPU-vs-CPU comparisons must not measure
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    config = tiny_config()
    torch.manual_seed(0)
    model = OellmEngramForCausalLM(config).eval()
    randomize(model)
    print("== 1. operations vs Megatron")
    check_rope()
    check_engram_hash(config, model)
    check_router(model)
    print("== 2. GDN kernels vs reference")
    check_gdn_kernels(config)
    print("== 3. tiny model (pattern LENLE*ELENE*E, fp32, CPU)")
    check_tiny_model(config, model)
    check_roundtrip(config, model)
    check_gpu_bf16(config, model)
    if a.config_json:
        print("== 4. real architecture")
        check_real_config(a.config_json, a.expect_numel)
    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

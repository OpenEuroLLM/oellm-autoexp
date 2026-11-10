"""vLLM port of oellm_ambitious (submodules/vllm, vllm/model_executor/models/oellm_ambitious.py) against
the HF export it transcribes, on tiny random models or an existing export.

    # tiny random models (both residuals; sign hash, quantile balancing, pab), local GPU:
    python -m oellm_autoexp.hf_export.vllm_parity tiny --out <dir>
    # an export:
    python -m oellm_autoexp.hf_export.vllm_parity export <export dir> [--tp 4]

Checks per model: prompt log-probabilities of a batch of prompts of different lengths (varlen prefill),
the same with a small token budget per step (chunked prefill: the CKDA / Engram states carried between
chunks), and greedy continuations (decode: the per-request states), each against the HF model.
Run in the vLLM environment with the FLA fork (flash-linear-attention) and this repo on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_PKG = _HERE / "oellm_ambitious"
_HYB = _HERE / "oellm_hybrid"


def _next_prime(n: int) -> int:
    def ok(x):
        return x > 1 and all(x % d for d in range(2, int(x**0.5) + 1))

    while not ok(n):
        n += 1
    return n


def tiny_config(residual: str, sign_hash: bool, quantile: bool, pab: bool, blocks: list | None = None,
                head_dim: int = 64, token_map: bool = False, embedding_norm: bool = False,
                expert_fp8: bool = False) -> dict:
    h, vocab = 256, 1024
    orders, heads = [2, 3, 4], 2
    sizes, c = [], 997
    for _ in range(len(orders) * heads):
        c = _next_prime(c)
        sizes.append(c)
        c += 1
    blocks = blocks or ["attention", "engram", "ckda", "moe", "ckda", "moe", "attention", "moe", "engram", "ckda", "moe"]
    rng = random.Random(0)
    mult = {str(i): [[rng.randrange(1, 1 << 20) * 2 + 1 for _ in range(max(orders))] for _ in sizes]
            for i, b in enumerate(blocks) if b == "engram"}
    cfg = {
        "architectures": ["OellmAmbitiousForCausalLM"],
        "model_type": "oellm_ambitious",
        "auto_map": {"AutoConfig": "configuration_oellm_ambitious.OellmAmbitiousConfig",
                     "AutoModelForCausalLM": "modeling_oellm_ambitious.OellmAmbitiousForCausalLM"},
        "vocab_size": vocab, "hidden_size": h, "block_types": blocks, "rms_norm_eps": 1e-5,
        "residual": residual, "mhc_expansion_rate": 4, "mhc_sinkhorn_iters": 20, "mhc_eps": 1e-5,
        "ldm_groups": 4, "ldm_key_dim": 8, "ldm_hidden_dim": 32, "ldm_alpha": 1.0,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": head_dim, "qk_layernorm": True,
        "attention_output_gate": True, "exclusive_self_attention": True, "xsa_eps": 1e-6,
        "fla_layer": {"class": "fla.layers.complex_kda_layer.ComplexKimiDeltaAttention",
                      "kwargs": {"hidden_size": h, "num_heads": 4, "head_dim": head_dim, "allow_neg_eigval": True,
                                 "gate": "signed_sigmoid2", "gate_init_style": "shipped", "lower_bound": -5.0,
                                 "output_gate": "lowrank", "drop_silu": True, "conv_size": 4, "norm_eps": 1e-5}},
        # kronlstm blocks (a ckda_branch KronLSTM): the production arm's settings (signed gate, s = 2,
        # decoupled erase, no unit write, normalized v, no conv SiLU, low-rank output gate)
        "fla_layer_kronlstm": {"class": "fla.layers.kronlstm.KronLSTM",
                               "kwargs": {"hidden_size": h, "k_head_dim": head_dim, "v_head_dim": head_dim,
                                          "num_k_heads": 4, "num_v_heads": 4, "mode": "chunk", "use_short_conv": True,
                                          "conv_size": 4, "conv_bias": False, "normalize_v": True,
                                          "allow_neg_eigval": True, "use_target_gate": True, "decouple_erase": True,
                                          "neg_eigval_init_bias": False, "unit_write": False, "drop_silu": True,
                                          "output_gate_rank": "lowrank", "decay_gate": "vector",
                                          "gate": "signed_sigmoid2", "lower_bound": -5.0, "norm_eps": 1e-5}},
        "num_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 128,
        "shared_expert_intermediate_size": 128, "router_scaling_factor": 1.0, "activation_clamp_value": 10.0,
        "router_balancing": "quantile" if quantile else "expert_bias",
        "moe_expert_post_act_norm": pab, "mlp_post_act_norm": pab, "post_act_norm_floor": 1.0 if pab else 0.0,
        "engram_ngram_orders": orders, "engram_num_heads": heads, "engram_embed_dim": 16,
        "engram_conv_kernel_size": 4, "engram_table_sizes": sizes, "engram_hash_multipliers": mult,
        "engram_token_map": token_map, "engram_gate_signed_sqrt": True, "engram_conv_prenorm": True,
        "engram_sign_hash": sign_hash, "engram_sign_table_size": 251,
        "engram_sign_multipliers": mult, "engram_table_dtype": "bf16",
        "embedding_norm": embedding_norm, "expert_weight_dtype": "fp8" if expert_fp8 else "bf16",
        "tie_word_embeddings": False, "torch_dtype": "bfloat16",
        "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 2,
    }
    return cfg


def _copy_code(out: Path) -> None:
    for f in ("configuration_oellm_ambitious.py", "modeling_oellm_ambitious.py", "mhc_kernels.py", "oellm_kernels.py"):
        shutil.copy(_PKG / f, out / f)
    for f in ("configuration_oellm_hybrid.py", "modeling_oellm_hybrid.py"):
        shutil.copy(_HYB / f, out / f)


def build_tiny(out: Path, router_std: float = 0.05, **kw) -> Path:
    """A random tiny export (HF classes, save_pretrained, the code files next to it)."""
    from transformers import AutoConfig, AutoModelForCausalLM

    out.mkdir(parents=True, exist_ok=True)
    tc = tiny_config(**kw)
    fp8 = tc["expert_weight_dtype"] == "fp8"
    # fp8 experts: a bf16 model is initialised, its experts quantized (export_ambitious.fp8_tiles)
    (out / "config.json").write_text(json.dumps(dict(tc, expert_weight_dtype="bf16"), indent=1))
    _copy_code(out)
    cfg = AutoConfig.from_pretrained(out, trust_remote_code=True)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, dtype=torch.bfloat16)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("forget_logit"):
                p.uniform_(2.0, 5.0)
            elif name.endswith(("norm_weight", "read_gain")) or "norm.weight" in name or name.endswith("layernorm.weight"):
                p.uniform_(0.7, 1.3)
            elif name.endswith("A_log"):
                p.uniform_(-1.0, 1.0)
            elif name.endswith("dt_bias"):
                p.uniform_(-1.0, 1.0)
            elif name.endswith(("alpha_pre", "alpha_post", "alpha_res")):
                p.fill_(0.01)
            elif name.endswith(".table"):
                p.normal_(0.0, 1.0)
            elif name.endswith("mixer.gate") and p.dim() == 2:
                p.normal_(0.0, router_std)  # large: wide top-k margins (no routing flips from noise)
            elif name.startswith("lm_head"):
                p.normal_(0.0, 0.5)  # peaked next-token distributions: greedy decoding is meaningful
            else:
                p.normal_(0.0, 0.05)
        for name, b in model.named_buffers():
            if name.endswith("sign_table"):
                b.copy_(torch.randint(0, 2, b.shape).float() * 2 - 1)
            elif name.endswith(("qb_beta", "e_score_correction_bias")):
                b.normal_(0.0, 0.1)
            elif name.endswith("token_map"):
                b.copy_(torch.randperm(b.numel()))
    if fp8:
        from .export_ambitious import fp8_tiles

        sd = {}
        for k, v in model.state_dict().items():
            if k.endswith(("mixer.gate_up_proj", "mixer.down_proj")):
                sd[k], sd[k + "_scale"] = fp8_tiles(v.to(torch.bfloat16))
            else:
                sd[k] = v
        (out / "config.json").write_text(json.dumps(tc, indent=1))
        cfg = AutoConfig.from_pretrained(out, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, dtype=torch.bfloat16)
        model.load_state_dict(sd, strict=True)
    model.save_pretrained(out, safe_serialization=True)
    _copy_code(out)
    return out


def _hf_logprobs(model, prompts: list[list[int]]) -> list[torch.Tensor]:
    """log p(token_t | tokens_<t) for t >= 1, one prompt at a time (no padding)."""
    out = []
    with torch.no_grad():
        for p in prompts:
            ids = torch.tensor([p], device=model.device)
            logits = model(input_ids=ids, use_cache=False).logits.float()
            lp = torch.log_softmax(logits[0, :-1], -1).gather(-1, ids[0, 1:, None])[:, 0]
            out.append(lp.cpu())
    return out


def _hf_greedy(model, prompt: list[int], n: int) -> list[int]:
    with torch.no_grad():
        ids = torch.tensor([prompt], device=model.device)
        gen = model.generate(input_ids=ids, max_new_tokens=n, do_sample=False, use_cache=True)
    return gen[0, len(prompt):].tolist()


def _vllm_logprobs(llm, prompts) -> list[torch.Tensor]:
    from vllm import SamplingParams

    sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0)
    res = llm.generate([{"prompt_token_ids": p} for p in prompts], sp, use_tqdm=False)
    out = []
    for p, r in zip(prompts, res):
        lps = r.prompt_logprobs[1:]
        out.append(torch.tensor([d[t].logprob for d, t in zip(lps, p[1:])]))
    return out


def _vllm_greedy(llm, prompt, n) -> list[int]:
    from vllm import SamplingParams

    r = llm.generate([{"prompt_token_ids": prompt}], SamplingParams(max_tokens=n, temperature=0.0,
                                                                    ignore_eos=True), use_tqdm=False)
    return list(r[0].outputs[0].token_ids)


def check(export: Path, tp: int = 1, vocab: int | None = None, lengths=(5, 37, 130, 300), n_gen: int = 16,
          tol: float = 0.05, hf_device_map: str | None = None, dtype: str = "bfloat16") -> dict:
    from transformers import AutoConfig, AutoModelForCausalLM
    from vllm import LLM

    cfg = AutoConfig.from_pretrained(export, trust_remote_code=True)
    vocab = vocab or cfg.vocab_size
    g = torch.Generator().manual_seed(1)
    prompts = [torch.randint(3, vocab, (n,), generator=g).tolist() for n in lengths]
    if hf_device_map:  # a real export over several GPUs (needs accelerate)
        hf = AutoModelForCausalLM.from_pretrained(export, trust_remote_code=True, dtype=getattr(torch, dtype),
                                                  device_map=hf_device_map)
    else:
        hf = AutoModelForCausalLM.from_pretrained(export, trust_remote_code=True, dtype=getattr(torch, dtype)).cuda()
    ref = _hf_logprobs(hf, prompts)
    ref_gen = _hf_greedy(hf, prompts[1], n_gen)
    del hf
    torch.cuda.empty_cache()
    report = {}
    for label, budget in (("prefill", None), ("chunked", 64)):
        kw = dict(model=str(export), trust_remote_code=True, dtype=dtype, enforce_eager=True,
                  skip_tokenizer_init=True, enable_prefix_caching=False, tensor_parallel_size=tp, enable_expert_parallel=tp > 1,
                  max_model_len=max(lengths) + n_gen + 8, gpu_memory_utilization=0.5, max_num_seqs=8)
        if budget:
            kw.update(enable_chunked_prefill=True, max_num_batched_tokens=budget)
        llm = LLM(**kw)
        got = _vllm_logprobs(llm, prompts)
        diffs = [float((a - b).abs().max()) for a, b in zip(ref, got)]
        means = [float((a - b).mean()) for a, b in zip(ref, got)]
        worst = [[int(i) for i in torch.topk((a - b).abs(), min(5, a.numel())).indices] for a, b in zip(ref, got)]
        report[label] = {"max_abs_diff": diffs, "mean_diff": means, "worst_positions": worst,
                         "n_over_0.5": [int(((a - b).abs() > 0.5).sum()) for a, b in zip(ref, got)],
                         "nll_hf": [float(-a.mean()) for a in ref], "nll_vllm": [float(-b.mean()) for b in got]}
        if label == "prefill":
            gen = _vllm_greedy(llm, prompts[1], n_gen)
            report["greedy"] = {"hf": ref_gen, "vllm": gen,
                                "first_mismatch": next((i for i, (a, b) in enumerate(zip(ref_gen, gen)) if a != b), None)}
        del llm
        torch.cuda.empty_cache()
    worst = max(max(report["prefill"]["max_abs_diff"]), max(report["chunked"]["max_abs_diff"]))
    report["pass"] = worst < tol and report["greedy"]["first_mismatch"] in (None,) + tuple(range(n_gen // 2, n_gen))
    return report


def score(export: Path, datamix: Path, tp: int, num_docs: int = 64, seq_len: int = 4096, min_len: int = 256,
          seed: int = 0, reference: Path | None = None, gpu_mem: float = 0.85) -> dict:
    """``score_on_flag`` in vLLM: the same documents (its sampler and seed, BOS prepended, cut at
    ``seq_len``), the mean next-token NLL from prompt log-probabilities; against the export's HF
    score (``score_flag.json``) when given -- that one sits next to the Megatron run's training loss.
    ``gpu_mem`` leaves room outside the KV budget for the prompt log-probs (1024-token chunks of
    gathered [chunk, vocab] logits: 0.5 GiB bf16 + their fp32 softmax at 262k vocab; 0.92 OOMed)."""
    import numpy as np
    from transformers import AutoConfig
    from vllm import LLM, SamplingParams

    from .score_on_flag import sample_docs

    cfg = AutoConfig.from_pretrained(export, trust_remote_code=True)
    bos = cfg.bos_token_id
    docs = sample_docs(datamix, num_docs, min_len, seed)
    prompts = [np.concatenate([[bos], d[: seq_len - 1]]).astype(np.int64).tolist() for d in docs]
    llm = LLM(model=str(export), trust_remote_code=True, dtype="bfloat16", enforce_eager=True,
              skip_tokenizer_init=True, enable_prefix_caching=False, tensor_parallel_size=tp, enable_expert_parallel=tp > 1,
              max_model_len=seq_len + 8, gpu_memory_utilization=gpu_mem, max_num_seqs=4,
              max_num_batched_tokens=seq_len)
    res = llm.generate([{"prompt_token_ids": p} for p in prompts],
                       SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0), use_tqdm=False)
    per_doc, total, count = [], 0.0, 0
    for p, r in zip(prompts, res):
        lp = [d[t].logprob for d, t in zip(r.prompt_logprobs[1:], p[1:])]
        per_doc.append(-float(np.mean(lp)))
        total -= float(np.sum(lp))
        count += len(lp)
    out = {"export": str(export), "docs": len(docs), "tokens": count, "mean_nll_vllm": total / count,
           "doc_lengths": [len(p) for p in prompts], "doc_nll_vllm": per_doc}
    if reference and Path(reference).exists():
        ref = json.loads(Path(reference).read_text())
        out["mean_nll_hf"] = ref["mean_nll"]
        out["diff_vllm_minus_hf"] = out["mean_nll_vllm"] - ref["mean_nll"]
        out["same_documents"] = ref.get("doc_lengths") == out["doc_lengths"] and ref.get("tokens") == count
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tiny")
    t.add_argument("--out", required=True)
    t.add_argument("--components", action="store_true",
                   help="one model per layer kind (fp32): which component carries a difference")
    t.add_argument("--only", default="", help="--components: a comma list of them")
    t.add_argument("--tp", type=int, default=1, help="vLLM tensor parallelism (the HF reference on one GPU)")
    e = sub.add_parser("export")
    e.add_argument("export")
    e.add_argument("--tp", type=int, default=1)
    e.add_argument("--hf-device-map", default=None)
    sc = sub.add_parser("score")
    sc.add_argument("export")
    sc.add_argument("--datamix", required=True)
    sc.add_argument("--tp", type=int, default=4)
    sc.add_argument("--reference", default=None, help="the export's HF score json (score_on_flag)")
    a = ap.parse_args()
    if a.cmd == "tiny" and a.components:
        base = dict(residual="mhc", sign_hash=True, quantile=True, pab=True)
        # a model needs one cached layer (attention / CKDA / Engram), else vLLM allocates no KV cache
        # and hangs at startup: the MoE models carry one attention layer (compare with "attention")
        comps = {"attention": dict(base, blocks=["attention"] * 3), "ckda": dict(base, blocks=["ckda"] * 3),
                 "moe": dict(base, blocks=["attention", "moe", "moe"]), "engram": dict(base, blocks=["engram"] * 2),
                 # a ckda_branch KronLSTM (kronlstm blocks), alone and in the v4 mix with attention and CKDA
                 "kronlstm": dict(base, blocks=["ckda", "kronlstm", "kronlstm"]),
                 "ckdab_kronlstm_mix": dict(base, quantile=False, head_dim=128,
                                            blocks=["attention", "engram", "ckda", "moe", "kronlstm", "moe",
                                                    "attention", "moe", "engram", "ckda", "moe"], router_std=1.0),
                 "ldm_attention": dict(base, residual="ldm", blocks=["attention"] * 3),
                 "ldm_moe": dict(base, residual="ldm", blocks=["attention", "moe", "moe"]),
                 # routing flips vs a mismatch: the MoE model with a router whose top-k margins are wide
                 "moe_wide_router": dict(base, blocks=["attention", "moe", "moe"], router_std=2.0),
                 # the v3 recipe in small: LDM, expert bias, pab, sign hash, token map, head dim 128, GQA 2
                 "v3_attention_hd128": dict(base, residual="ldm", quantile=False, head_dim=128,
                                            blocks=["attention"] * 3),
                 "v3_recipe": dict(base, residual="ldm", quantile=False, head_dim=128, token_map=True,
                                   blocks=["attention", "engram", "ckda", "moe", "ckda", "moe", "attention",
                                           "moe", "engram", "ckda", "moe"], router_std=1.0)}
        only = [c for c in (a.only or "").split(",") if c]
        for name, kw in comps.items():
            if only and name not in only:
                continue
            r = check(build_tiny(Path(a.out) / f"comp_{name}", **kw), dtype="float32", tol=1e-2, tp=a.tp)
            print(f"component/{name}", json.dumps(r), flush=True)
        return
    if a.cmd == "tiny":
        variants = {"mhc_bias": dict(residual="mhc", sign_hash=False, quantile=False, pab=False),
                    "ldm_qb_pab_sign": dict(residual="ldm", sign_hash=True, quantile=True, pab=True),
                    # v4_ldm in small: + embedding_norm, and the experts as fp8 tiles
                    "v4ldm_fp8": dict(residual="ldm", sign_hash=True, quantile=True, pab=True,
                                      embedding_norm=True, expert_fp8=True),
                    "mhc_bias_fp8": dict(residual="mhc", sign_hash=True, quantile=False, pab=True,
                                         expert_fp8=True)}
        results = {}
        for name, kw in variants.items():
            d = build_tiny(Path(a.out) / name, **kw)
            # fp32 decides (a correct port agrees to ~1e-4); bf16 shows the production precision's spread
            for dtype, tol in (("float32", 1e-2), ("bfloat16", 0.05)):
                results[f"{name}/{dtype}"] = r = check(d, dtype=dtype, tol=tol)
                print(f"{name}/{dtype}", json.dumps(r), flush=True)
        print("PASS" if all(r["pass"] for r in results.values()) else "FAIL")
    elif a.cmd == "score":
        r = score(Path(a.export), Path(a.datamix), a.tp, reference=a.reference)
        print(json.dumps({k: v for k, v in r.items() if k not in ("doc_lengths", "doc_nll_vllm")}, indent=1))
        print(json.dumps(r))
    else:
        r = check(Path(a.export), tp=a.tp, hf_device_map=a.hf_device_map)
        print(json.dumps(r, indent=1))
        print("PASS" if r["pass"] else "FAIL")


if __name__ == "__main__":
    main()

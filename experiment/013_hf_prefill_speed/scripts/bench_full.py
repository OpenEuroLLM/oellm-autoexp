"""Prefill / scoring speed and numerical parity of the real export on one node (4 GPUs).

Variants (one loaded model per process):
  fast    from_pretrained(device_map="auto"): the export code's own placement (whole layers on the
          GPUs, Engram tables pinned in host memory) and the inference kernels; in the same
          process also with OELLM_HF_PIPELINE=0, a few micro-batch counts, and the reference
          implementation (OELLM_HF_IMPL=reference) on a subset
  hf008   experiment 008's fast HF path: score_on_flag.layer_device_map + load_engram_tables_cpu,
          BF16-expanded experts with torch._grouped_mm, the fused Sinkhorn, sorted host lookup

Workloads (token ids from 008's fixed prompts, 256 tokens each):
  w1  32 prompts one by one (b=1, 256)  -- 008's protocol
  w2  one batch of 32 x 256             -- a lighteval batch
  w3  4 x 2048 (8 prompts concatenated per row)
  w4  1 x 8192 (32 prompts concatenated)
Logprobs of the next tokens are saved per workload (``<out>/<variant>.<w>.pt``) for parity.

    python bench_full.py --model <export> --variant fast --prompts prompts.json --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch


def sync_all():
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)


def logprobs(logits, ids):
    """Next-token logprobs [b, s-1] (fp32), computed in row chunks."""
    out = []
    for r in range(logits.shape[0]):
        row = []
        for a in range(0, logits.shape[1] - 1, 1024):  # 1024 positions at a time (memory)
            e = min(a + 1024, logits.shape[1] - 1)
            lp = torch.log_softmax(logits[r, a:e].float(), dim=-1)
            row.append(lp.gather(-1, ids[r, a + 1:e + 1].to(lp.device)[:, None])[:, 0].cpu())
            del lp
        out.append(torch.cat(row))
    return torch.stack(out)


def workloads(prompts):
    p = [torch.tensor(x[:256]) for x in prompts]
    return {
        "w1": [x[None] for x in p],
        "w2": [torch.stack(p)],
        "w3": [torch.stack([torch.cat(p[8 * r: 8 * r + 8]) for r in range(4)])],
        "w4": [torch.cat(p)[None]],
    }


def run(model, batches, dev, reps):
    times, lps = [], None
    with torch.no_grad():
        for rep in range(reps + 1):  # the first one warms up (autotune, cuBLAS heuristics)
            sync_all()
            t0 = time.perf_counter()
            outs = [model(input_ids=b.to(dev)).logits for b in batches]
            sync_all()
            dt = time.perf_counter() - t0
            if rep == 0:
                lps = [logprobs(o, b) for o, b in zip(outs, batches)]
            else:
                times.append(dt)
            del outs
    tokens = sum(b.numel() for b in batches)
    med = sorted(times)[len(times) // 2]
    return {"tokens": tokens, "seconds": med, "all_seconds": times, "tok_s": tokens / med}, lps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--variant", choices=["fast", "hf008"], required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--work", nargs="+", default=["w1", "w2", "w3", "w4"])
    p.add_argument("--reference-subset", type=int, default=8, help="w1 prompts for the reference run (0: skip)")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoConfig, AutoModelForCausalLM

    t0 = time.perf_counter()
    if a.variant == "hf008":
        os.environ.update(OELLM_HF_IMPL="reference", OELLM_HF_MOE="grouped", OELLM_MHC_IMPL="triton",
                          OELLM_HF_ENGRAM_LOOKUP="sorted")
        from oellm_autoexp.hf_export.score_on_flag import layer_device_map, load_engram_tables_cpu

        cfg = AutoConfig.from_pretrained(a.model, trust_remote_code=True)
        model = load_engram_tables_cpu(Path(a.model), layer_device_map(cfg, 54, True)).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, dtype=torch.bfloat16,
                                                     device_map="auto").eval()
    load_s = time.perf_counter() - t0
    dev = model.model.embed_tokens.weight.device
    mem = {d: round(torch.cuda.memory_allocated(d) / 2**30, 1) for d in range(torch.cuda.device_count())}
    print(json.dumps({"variant": a.variant, "load_seconds": load_s, "gib_per_gpu": mem}), flush=True)
    work = {k: v for k, v in workloads(json.loads(Path(a.prompts).read_text())).items() if k in a.work}
    cases = [(a.variant, {})]
    if a.variant == "fast":
        cases += [("fast_nopipe", {"OELLM_HF_PIPELINE": "0"}), ("fast_chunks4", {"OELLM_HF_PIPELINE_CHUNKS": "4"}),
                  ("fast_chunks16", {"OELLM_HF_PIPELINE_CHUNKS": "16"})]
    results = {"variant": a.variant, "load_seconds": load_s, "gib_per_gpu": mem,
               "device_map": getattr(model, "oellm_device_map", None) or getattr(model, "hf_device_map", None),
               "cases": {}}
    for name, env in cases:
        saved = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        for w, batches in work.items():
            if name != a.variant and w == "w4":  # a single row: no pipeline, nothing new
                continue
            r, lps = run(model, batches, dev, a.reps)
            results["cases"].setdefault(name, {})[w] = r
            if name == a.variant:
                torch.save(lps, out / f"{a.variant}.{w}.pt")
            torch.cuda.empty_cache()
            print(json.dumps({"case": name, "work": w, **r, "peak_gib": round(torch.cuda.max_memory_allocated(0) / 2**30, 1)}),
                  flush=True)
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    if a.variant == "fast" and a.reference_subset:
        os.environ.update(OELLM_HF_IMPL="reference")
        sub = {"w1": work["w1"][: a.reference_subset]} if "w1" in work else {}
        for w, batches in sub.items():
            r, lps = run(model, batches, dev, 1)
            results["cases"].setdefault("reference", {})[w] = r
            torch.save(lps, out / f"reference.{w}.pt")
            print(json.dumps({"case": "reference", "work": w, **r}), flush=True)
        os.environ.pop("OELLM_HF_IMPL")
    (out / f"{a.variant}.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()

"""Re-score lm-eval multiple-choice requests of a vLLM run with the HF port of the same export.

Reads lm-eval ``samples_*.jsonl`` files (the exact (context, continuation) strings of every request and
vLLM's log-likelihoods), encodes each pair the way lm-eval 0.4.13 does (trailing context spaces moved to
the continuation; whole = tok(context + continuation), context = tok(context), both with the tokenizer's
BOS; continuation = whole[len(context):]), scores the continuation tokens with the HF model at batch 1,
and writes per-request HF vs vLLM log-likelihoods plus a summary.

``--impls fast,reference`` scores with both HF paths in one load (OELLM_HF_IMPL is read per forward);
``--limit-reference N`` restricts the slow reference path to the first N documents of each file.

    python experiment/016_winogrande_hf_check/scripts/score_mc_hf.py --model-dir <export> \
        --samples <samples_winogrande_*.jsonl> [<samples_wsc273_*.jsonl> ...] --out <dir> \
        --gpu-mem-gib 54 --engram-tables-cpu --impls fast,reference --limit-reference 150
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def encode_pair(tok, context: str, continuation: str) -> tuple[list[int], list[int]]:
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces > 0:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]
    whole = tok(context + continuation).input_ids
    ctx = tok(context).input_ids
    return ctx, whole[len(ctx):]


@torch.no_grad()
def continuation_ll(model, ids: list[int], n_cont: int, dev) -> tuple[float, bool]:
    x = torch.tensor([ids], device=dev)
    hidden = model.model(input_ids=x, use_cache=False).last_hidden_state[0, -n_cont - 1 : -1]
    head = model.lm_head
    logits = head(hidden.to(head.weight.device)).float()
    target = x[0, -n_cont:].to(logits.device)
    lp = F.log_softmax(logits, dim=-1)
    ll = lp.gather(1, target[:, None]).sum().item()
    greedy = bool((logits.argmax(-1) == target).all().item())
    return ll, greedy


def closer_option(doc: dict) -> int | None:
    """Winogrande: index of the option mentioned last before the blank (None if not both found)."""
    if "sentence" not in doc or "option1" not in doc:
        return None
    pre = doc["sentence"][: doc["sentence"].index("_")]
    p1, p2 = pre.rfind(doc["option1"]), pre.rfind(doc["option2"])
    if p1 < 0 or p2 < 0:
        return None
    return 0 if p1 > p2 else 1


def summarize(rows: list[dict], key: str) -> dict:
    docs: dict[int, list[dict]] = {}
    for r in rows:
        if key in r:
            docs.setdefault(r["doc_id"], []).append(r)
    acc, closer, agree, gap_c, gap_f = [], [], [], [], []
    for reqs in docs.values():
        reqs.sort(key=lambda r: r["choice"])
        ll = [r[key] for r in reqs]
        pick = max(range(len(ll)), key=lambda j: ll[j])
        vpick = max(range(len(ll)), key=lambda j: reqs[j]["vllm_ll"])
        gold = reqs[0]["gold"]
        acc.append(pick == gold)
        agree.append(pick == vpick)
        c = reqs[0]["closer"]
        if c is not None:
            closer.append(pick == c)
            (gap_c if gold == c else gap_f).append(pick == gold)
    d = [abs(r[key] - r["vllm_ll"]) for r in rows if key in r]
    ntok = [r["n_cont"] for r in rows if key in r]
    out = {"docs": len(docs), "acc": st.mean(acc), "agree_with_vllm_pick": st.mean(agree),
           "mean_abs_dll_vs_vllm": st.mean(d), "max_abs_dll_vs_vllm": max(d),
           "mean_abs_dll_per_token": sum(d) / sum(ntok)}
    if closer:
        out.update(picks_closer=st.mean(closer), acc_gold_closer=st.mean(gap_c), acc_gold_farther=st.mean(gap_f))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--samples", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--impls", default="fast,reference")
    p.add_argument("--limit", type=int, default=0, help="documents per file (0: all)")
    p.add_argument("--limit-reference", type=int, default=150)
    p.add_argument("--gpu-mem-gib", type=float, default=None)
    p.add_argument("--engram-tables-cpu", action="store_true")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from oellm_autoexp.hf_export.score_on_flag import layer_device_map, load_engram_tables_cpu

    tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(a.model_dir, trust_remote_code=True)
    t0 = time.time()
    dmap = layer_device_map(cfg, a.gpu_mem_gib, a.engram_tables_cpu) if a.gpu_mem_gib else "auto"
    if not torch.cuda.is_available():  # local plumbing tests on the CPU
        dmap = None
    if a.engram_tables_cpu:
        model = load_engram_tables_cpu(a.model_dir, dmap).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(a.model_dir, trust_remote_code=True, dtype=torch.bfloat16,
                                                     device_map=dmap).eval()
    print(f"loaded {a.model_dir.name} in {time.time() - t0:.0f}s", flush=True)
    dev = model.model.embed_tokens.weight.device

    summary = {"model": str(a.model_dir), "env": {k: v for k, v in os.environ.items() if k.startswith("OELLM_")},
               "files": {}}
    for f in a.samples:
        docs = [json.loads(line) for line in open(f)]
        docs.sort(key=lambda d: d["doc_id"])
        if a.limit:
            docs = docs[: a.limit]
        task = f.name.split("_")[1] if f.name.count("_") >= 2 else f.stem
        rows = []
        for d in docs:
            gold = d["target"] if isinstance(d["target"], int) else None
            if gold is None and "answer" in d["doc"]:  # winogrande: target is the shared continuation
                gold = int(d["doc"]["answer"]) - 1
            if gold is None:
                gold = d["doc"].get("label")
            args = [d["arguments"][k] for k in sorted(d["arguments"], key=lambda k: int(k.split("_")[-1]))]
            for j, (g, resp) in enumerate(zip(args, d["resps"])):
                ctx, cont = encode_pair(tok, g["arg_0"], g["arg_1"])
                rows.append({"doc_id": d["doc_id"], "choice": j, "gold": gold, "closer": closer_option(d["doc"]),
                             "n_ctx": len(ctx), "n_cont": len(cont), "ids": ctx + cont,
                             "vllm_ll": float(resp[0][0]), "vllm_acc": d.get("acc", d.get("acc_norm"))})
        for impl in a.impls.split(","):
            os.environ["OELLM_HF_IMPL"] = impl
            limit = a.limit_reference if impl == "reference" else 0
            keep = {d["doc_id"] for d in (docs[:limit] if limit else docs)}
            t1 = time.time()
            for n, r in enumerate(rows):
                if r["doc_id"] in keep:
                    r[f"hf_{impl}"], r[f"hf_{impl}_greedy"] = continuation_ll(model, r["ids"], r["n_cont"], dev)
                if n % 200 == 0:
                    print(f"[{task} {impl}] {n}/{len(rows)} {time.time() - t1:.0f}s", flush=True)
            print(f"[{task} {impl}] done in {time.time() - t1:.0f}s", flush=True)
        res = {"vllm_all": summarize([{**r, "vllm": r["vllm_ll"]} for r in rows], "vllm")}
        for impl in a.impls.split(","):
            res[f"hf_{impl}"] = summarize(rows, f"hf_{impl}")
            sub = [r for r in rows if f"hf_{impl}" in r]
            res[f"vllm_on_{impl}_docs"] = summarize([{**r, "vllm": r["vllm_ll"]} for r in sub], "vllm")
        summary["files"][str(f)] = res
        with open(a.out / f"{task}_requests.jsonl", "w") as fo:
            for r in rows:
                fo.write(json.dumps({k: v for k, v in r.items() if k != "ids"} | {"ids_head": r["ids"][:8]}) + "\n")
        print(json.dumps({task: res}, indent=1), flush=True)
        (a.out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

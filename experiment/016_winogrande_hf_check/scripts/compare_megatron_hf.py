"""Megatron vs HF export on the very same samples, per token and per request.

Inputs: Megatron's eval token dump (MEGATRON_EVAL_TOKEN_DUMP: r<rank>_<k>.pt with tokens / labels / losses of
each eval micro-batch, possibly several sets) and paired_valid_check --dump-tokens files of the HF export.
Samples are matched by their token content (data-parallel duplicates collapse). Per set: Megatron vs HF
loss, per-sample and per-token differences. With ``--requests`` (make_mc_set.py's <prefix>.requests.json)
and ``--vllm`` (score_mc_hf.py's *_requests.jsonl, which carry the vLLM log-likelihoods): every request
that lies whole in one document segment gets its continuation log-likelihood from Megatron and HF, and the
multiple-choice decisions of Megatron / HF / vLLM are compared (accuracy, agreement, closer-mention bias).

    python compare_megatron_hf.py --megatron-dump <dir> --hf <hf_all_tokens.pt> <hf_mc_tokens.pt> \
        --requests <mc_text_document.requests.json> --vllm <winogrande_requests.jsonl> <wsc273_requests.jsonl> \
        --out <summary.json>
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import statistics as st
from pathlib import Path

import numpy as np
import torch


def key(tokens: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(tokens, dtype=np.int64).tobytes()).hexdigest()


def load_megatron(d: Path, lengths: set[int]) -> dict[str, dict]:
    out = {}
    for f in sorted(glob.glob(str(d / "r*_*.pt"))):
        x = torch.load(f, weights_only=False)
        L = x.get("seq_length") or x["tokens"].shape[-1]
        if L not in lengths:
            continue
        toks = x["tokens"].reshape(-1, L).numpy()
        losses = x["losses"].reshape(-1, L).numpy()
        mask = x["loss_mask"].reshape(-1, L).numpy()
        for t, l, m in zip(toks, losses, mask):
            out.setdefault(key(t), {"tokens": t, "losses": l, "mask": m})
    return out


def summarize_pairs(meg: dict, hf: list[dict]) -> tuple[dict, list[tuple[dict, dict]]]:
    pairs, missing = [], 0
    for h in hf:
        m = meg.get(key(h["tokens"].numpy()))
        if m is None:
            missing += 1
            continue
        pairs.append((m, h))
    mask = [h["loss_mask"].numpy().astype(bool) for _, h in pairs]
    lm = np.concatenate([m["losses"][k] for (m, _), k in zip(pairs, mask)]) if pairs else np.zeros(0)
    lh = np.concatenate([h["losses"].numpy()[k] for (_, h), k in zip(pairs, mask)]) if pairs else np.zeros(0)
    d = lh - lm
    res = {"hf_samples": len(hf), "matched": len(pairs), "unmatched": missing, "tokens": int(lm.size),
           "megatron_loss": float(lm.mean()) if lm.size else None, "hf_loss": float(lh.mean()) if lh.size else None,
           "hf_minus_megatron": float(d.mean()) if d.size else None,
           "mean_abs_token_diff": float(np.abs(d).mean()) if d.size else None,
           "p99_abs_token_diff": float(np.quantile(np.abs(d), 0.99)) if d.size else None,
           "max_abs_token_diff": float(np.abs(d).max()) if d.size else None,
           "token_corr": float(np.corrcoef(lm, lh)[0, 1]) if d.size > 1 else None}
    per_sample = [float((h["losses"].numpy()[k] - m["losses"][k]).mean()) for (m, h), k in zip(pairs, mask)]
    if per_sample:
        res["per_sample_mean_diff_range"] = [min(per_sample), max(per_sample)]
    return res, pairs


def request_lls(pairs: list[tuple[dict, dict]], req_map: list[dict]) -> dict[tuple, dict]:
    by_ids = {}
    for r in req_map:
        by_ids.setdefault(tuple(r["ids"]), []).append(r)
    out = {}
    for m, h in pairs:
        toks = h["tokens"].numpy()
        for s, e in h["segments"]:
            for r in by_ids.get(tuple(toks[s:e].tolist()), []):
                a, b = s + r["n_ctx"] - 1, s + len(r["ids"]) - 1
                out[(r["task"], r["doc_id"], r["choice"])] = {
                    "megatron": -float(m["losses"][a:b].sum()), "hf": -float(h["losses"].numpy()[a:b].sum()),
                    "n_cont": r["n_cont"]}
    return out


def decisions(lls: dict, vllm: dict) -> dict:
    res = {}
    for task in sorted({k[0] for k in lls}):
        docs = {}
        for (t, doc, c), v in lls.items():
            if t == task and (t, doc, c) in vllm:
                docs.setdefault(doc, {})[c] = {**v, "vllm": vllm[(t, doc, c)]["vllm_ll"],
                                               "gold": vllm[(t, doc, c)]["gold"], "closer": vllm[(t, doc, c)]["closer"]}
        nchoice = max(len(v) for v in docs.values())
        docs = {d: v for d, v in docs.items() if len(v) == nchoice}
        stats = {"docs_complete": len(docs)}
        picks = {}
        for src in ("megatron", "hf", "vllm"):
            picks[src] = {d: max(v, key=lambda c: v[c][src]) for d, v in docs.items()}
            gold = {d: v[0]["gold"] for d, v in docs.items()}
            stats[f"acc_{src}"] = st.mean(picks[src][d] == gold[d] for d in docs)
            closer = {d: v[0]["closer"] for d, v in docs.items() if v[0]["closer"] is not None}
            if closer:
                stats[f"picks_closer_{src}"] = st.mean(picks[src][d] == closer[d] for d in closer)
        for a, b in (("megatron", "hf"), ("megatron", "vllm"), ("hf", "vllm")):
            stats[f"agree_{a}_{b}"] = st.mean(picks[a][d] == picks[b][d] for d in docs)
            dl = [abs(v[c][a] - v[c][b]) / v[c]["n_cont"] for v in docs.values() for c in v]
            stats[f"mean_abs_dll_per_token_{a}_{b}"] = st.mean(dl)
        res[task] = stats
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--megatron-dump", type=Path, required=True)
    p.add_argument("--hf", type=Path, nargs="+", required=True)
    p.add_argument("--requests", type=Path)
    p.add_argument("--vllm", type=Path, nargs="*", default=[])
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    summary = {}
    for f in a.hf:
        hf = torch.load(f, weights_only=False)
        L = int(hf[0]["tokens"].shape[0])
        meg = load_megatron(a.megatron_dump, {L})
        res, pairs = summarize_pairs(meg, hf)
        res["megatron_samples"] = len(meg)
        if a.requests and "mc" in f.name:
            req = json.loads(a.requests.read_text())
            lls = request_lls(pairs, req["map"])
            res["requests_whole"] = len(lls)
            res["requests_total"] = len(req["map"])
            vllm = {}
            for v in a.vllm:
                task = v.name.split("_")[0]
                for line in open(v):
                    r = json.loads(line)
                    vllm[(task, r["doc_id"], r["choice"])] = r
            res["decisions"] = decisions(lls, vllm)
        summary[f.name] = res
        print(json.dumps({f.name: res}, indent=1), flush=True)
    a.out.write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()

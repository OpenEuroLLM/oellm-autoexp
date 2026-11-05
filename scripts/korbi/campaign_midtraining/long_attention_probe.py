"""T1 / T2 of the midtraining campaign (docs/campaign_3b_hybrid_midtraining.md): long-context behaviour of an HF
export of oellm_ambitious (v4_2ph, trained to 64k) beyond its training length, with dense or sparse attention.

The 6 softmax-attention layers run through Megatron's page_sparse module (the same code the training uses;
submodules/Megatron-LM/megatron/core/transformer/experimental_attention_variant/page_sparse.py, loaded by path).
Contexts are prefilled in chunks with the HF cache (constant activation memory at any length); a policy decides
how the attention layers attend:

  dense                          full causal attention
  logn[:base=65536]              dense, queries scaled by max(1, log n / log base) (Megatron's attention_length_scaling
                                 "logn"; base = the training length)
  <selector>[:k=v,...]           page_sparse with that selector (window, page_mean, quest, oracle, token_topk,
                                 compress_mean) and PageSparseConfig fields, e.g. page_mean:topk=32,page=64

Tasks:
  nll        per-position NLL (bins) on long held-out documents (v8ml-valid long_64k: 89 docs >= 128k, 23 >= 256k,
             10 >= 512k), prefilled up to each length
  retrieval  passkey (depths 0.1 / 0.5 / 0.9), 64-fact multi-key retrieval in natural text, variable tracking
             (scripts/korbi/downstream/prepare_tasks.py), scored as 10-way choices by the continuation logprob

    python long_attention_probe.py --model <export> --out <dir> --tasks nll,retrieval \\
        --policies "dense;logn;window;page_mean:topk=32" --lengths 65536,131072,262144 --docs <prefix> --raw <raw.jsonl>
    python long_attention_probe.py --tiny --out /tmp/x        # local check: reduced random model, random tokens
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import sys
import time
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import router_probe as rp  # noqa: E402

os.environ.setdefault("OELLM_HF_PIPELINE", "0")


def load_page_sparse():
    path = REPO / "submodules/Megatron-LM/megatron/core/transformer/experimental_attention_variant/page_sparse.py"
    spec = importlib.util.spec_from_file_location("page_sparse", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["page_sparse"] = mod
    spec.loader.exec_module(mod)
    return mod


ps = load_page_sparse()


def parse_policy(text):
    name, _, rest = text.partition(":")
    kw = {}
    for item in filter(None, rest.split(",")):
        key, val = item.split("=")
        kw[key] = int(val) if val.lstrip("-").isdigit() else float(val)
    if name == "logn":
        return {"name": text, "kind": "logn", "base": kw.get("base", 65536)}
    if name == "dense":
        return {"name": text, "kind": "dense"}
    return {"name": text, "kind": "sparse", "cfg": ps.PageSparseConfig(selector=name, **kw)}


class AttentionPolicy:
    """Replaces every attention mixer's forward: chunk-wise prefill / decode against the HF cache's (k, v), with
    the current policy."""

    def __init__(self, model):
        self.policy = {"kind": "dense"}
        self.n = 0
        for layer in model.model.layers:
            if layer.kind == "attention":
                layer.mixer.forward = types.MethodType(self._forward(), layer.mixer)
                self.n += 1

    def _forward(self):
        pol = self

        def forward(att, x, state=None, keep_state=True):
            b, s, _ = x.shape
            nh, nkv, d = att.nh, att.nkv, att.d
            q = att.q_proj(x).view(b, s, nh, d)
            k = att.k_proj(x).view(b, s, nkv, d)
            v = att.v_proj(x).view(b, s, nkv, d)
            if att.q_norm is not None:
                q, k = att.q_norm(q), att.k_norm(k)
            own_v = v
            q, k, v = (t.transpose(1, 2) for t in (q, k, v))
            past = 0
            if state is not None:
                past = state[0].shape[2]
                k = torch.cat([state[0], k], dim=2)
                v = torch.cat([state[1], v], dim=2)
            p = pol.policy
            if p["kind"] == "logn":
                pos = past + torch.arange(s, device=x.device)
                f = torch.clamp(torch.log(pos.float() + 1) / math.log(p["base"]), min=1.0)
                q = (q.float() * f[None, None, :, None]).to(q.dtype)
            with torch.cuda.device(x.device):
                if p["kind"] == "sparse":
                    o = ps.page_sparse_attention(q, k, v, p["cfg"], q_offset=past, scale=d**-0.5)
                else:
                    o = ps.dense_attention(q, k, v, q_offset=past, scale=d**-0.5)
            o = o.transpose(1, 2)  # [b, s, nh, d]
            rep = nh // nkv
            if att.xsa:
                vf = own_v.float().unsqueeze(-2)
                of = o.float().reshape(b, s, nkv, rep, d)
                c = (of * vf).sum(-1, keepdim=True) / (vf.pow(2).sum(-1, keepdim=True) + att.xsa_eps)
                of = (of - c * vf).reshape(b, s, nh, d)
                if att.gate_proj is not None:
                    of = of * torch.sigmoid(att.gate_proj(x).view(b, s, nh, d).float())
                o = of.to(x.dtype)
            elif att.gate_proj is not None:
                o = (o.float() * torch.sigmoid(att.gate_proj(x).view(b, s, nh, d).float())).to(x.dtype)
            return att.o_proj(o.reshape(b, s, nh * d)), ((k, v) if keep_state else None)

        return forward


@torch.no_grad()
def prefill(model, ids, chunk, cache=None, want_nll=False):
    """ids [1, L] -> (cache, last hidden [1, h], per-token NLL of ids[1:] or None). Chunks of `chunk` tokens."""
    dev = next(model.parameters()).device
    nll = [] if want_nll else None
    last = None
    L = ids.shape[1]
    for a in range(0, L, chunk):
        part = ids[:, a:a + chunk].to(dev)
        out = model.model(input_ids=part, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        h = out.last_hidden_state
        if want_nll:
            tgt = ids[0, a + 1:a + 1 + part.shape[1]].to(h.device)
            n = tgt.numel()
            for c0 in range(0, n, 4096):
                c1 = min(n, c0 + 4096)
                lp = torch.log_softmax(model.lm_head(h[0, c0:c1]).float(), -1)
                nll.append(-lp.gather(-1, tgt[c0:c1, None])[:, 0].cpu())
        last = h[:, -1]
    return cache, last, (torch.cat(nll) if want_nll else None)


@torch.no_grad()
def score_choice(model, ctx, conts, chunk):
    """Index of the highest-logprob continuation after ctx (and the logprobs)."""
    cache, last, _ = prefill(model, ctx, chunk)
    lp0 = torch.log_softmax(model.lm_head(last).float(), -1)[0]
    scores = []
    dev = next(model.parameters()).device
    for cont in conts:
        s = float(lp0[cont[0]])
        if len(cont) > 1:
            c = copy.deepcopy(cache)
            out = model.model(input_ids=torch.tensor([cont[:-1]], device=dev), past_key_values=c, use_cache=True)
            lp = torch.log_softmax(model.lm_head(out.last_hidden_state[0]).float(), -1)
            s += float(lp.gather(-1, torch.tensor(cont[1:], device=lp.device)[:, None]).sum())
            del c
        scores.append(s)
    del cache
    return int(np.argmax(scores)), scores


def long_docs(prefix, min_len, n, bos, seed=0):
    docs = rp.read_indexed(prefix)
    order = [i for i in np.random.default_rng(seed).permutation(len(docs)) if docs.lengths[i] >= min_len - 1]
    return [torch.tensor([[bos] + docs[int(i)][: min_len - 1].tolist()]) for i in order[:n]]


def run_nll(model, pol, policies, args, out):
    res = {}
    for L in args.lengths:
        docs = long_docs(args.docs, L, args.nll_docs, args.bos) if args.docs else [
            torch.randint(3, 50_000, (1, L), generator=torch.Generator().manual_seed(i)) for i in range(2)]
        if not docs:
            continue
        for p in policies:
            pol.policy = p
            t0 = time.time()
            nll = torch.stack([prefill(model, d, args.chunk, want_nll=True)[2] for d in docs])  # [n, L - 1]
            bins = {}
            edges = [0, 4096, 16384, 32768, 65536] + [x for x in (131072, 262144, 524288, 1048576) if x <= L]
            edges = sorted(set(e for e in edges if e < L)) + [L - 1]
            for lo, hi in zip(edges[:-1], edges[1:]):
                bins[f"{lo // 1024}k-{hi // 1024}k"] = float(nll[:, lo:hi].mean())
            res[f"{L}|{p['name']}"] = {"docs": len(docs), "mean": float(nll.mean()), "bins": bins,
                                       "seconds": time.time() - t0}
            print(f"[nll] L={L} {p['name']}: {float(nll.mean()):.4f} {bins} {time.time() - t0:.0f}s", flush=True)
            (out / "nll.json").write_text(json.dumps(res, indent=1))
    return res


def retrieval_rows(tok, args, L):
    sys.path.insert(0, str(REPO / "scripts/korbi/downstream"))
    import prepare_tasks as pt  # noqa: E402

    rows = pt.passkey_rows(tok, lengths=(L,), depths=(0.1, 0.5, 0.9), n=max(1, args.n // 3), seed=0)
    rows += [dict(r, task=f"kvret64_{L // 1024}k") for r in pt.kv_rows(tok, args.raw, lengths=(L,), n=args.n,
                                                                         pairs=64, seed=4)]
    rows += pt.vt_rows(tok, args.raw, lengths=(L,), n=args.n, seed=2)
    return rows


def run_retrieval(model, pol, policies, args, out, tok):
    res = defaultdict(dict)
    for L in args.lengths:
        rows = retrieval_rows(tok, args, L)
        for p in policies:
            pol.policy = p
            t0 = time.time()
            acc = defaultdict(list)
            for r in rows:
                ctx = torch.tensor([r["items"][0]["ctx"]])
                pick, _ = score_choice(model, ctx, [it["cont"] for it in r["items"]], args.chunk)
                acc[r["task"].split("_")[0]].append(pick == r["gold"])
            key = f"{L}|{p['name']}"
            for task, v in acc.items():
                res[key][task] = float(np.mean(v))
            print(f"[ret] L={L} {p['name']}: {dict(res[key])} {time.time() - t0:.0f}s", flush=True)
            (out / "retrieval.json").write_text(json.dumps(res, indent=1))
    return res


def write_summary(out, nll, ret):
    lines = ["# Long-context attention probe (T1 / T2)", ""]
    if nll:
        lines += ["## NLL by position (held-out long documents)", "", "| length | policy | mean | bins |",
                  "|---|---|---:|---|"]
        for key, r in nll.items():
            L, name = key.split("|")
            lines.append(f"| {int(L) // 1024}k | {name} | {r['mean']:.4f} | "
                         + ", ".join(f"{b} {v:.3f}" for b, v in r["bins"].items()) + " |")
    if ret:
        lines += ["", "## Retrieval accuracy (10-way)", "", "| length | policy | passkey | kvret64 | vt |",
                  "|---|---|---:|---:|---:|"]
        for key, r in ret.items():
            L, name = key.split("|")
            lines.append(f"| {int(L) // 1024}k | {name} | {r.get('passkey', float('nan')):.2f} | "
                         f"{r.get('kvret64', float('nan')):.2f} | {r.get('vt', float('nan')):.2f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model")
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tasks", default="nll,retrieval")
    p.add_argument("--policies", default="dense;logn;window;page_mean", help="';'-separated policies")
    p.add_argument("--lengths", default="65536,131072,262144")
    p.add_argument("--chunk", type=int, default=16384)
    p.add_argument("--docs", default="", help="indexed dataset prefix of long held-out documents")
    p.add_argument("--nll-docs", type=int, default=4)
    p.add_argument("--raw", default="", help="raw.jsonl (natural filler for the retrieval tasks)")
    p.add_argument("--n", type=int, default=6, help="questions per retrieval task and length")
    p.add_argument("--bos", type=int, default=1)
    p.add_argument("--parity", type=int, default=32768, help="parity check length (0: off)")
    args = p.parse_args()
    args.lengths = [int(x) for x in args.lengths.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "args.json").write_text(json.dumps(vars(args), default=str, indent=1))
    t0 = time.time()
    model = rp.load_model(args.model, args.tiny)
    pol = AttentionPolicy(model)
    print(f"[probe] model loaded in {time.time() - t0:.0f}s; {pol.n} attention layers", flush=True)
    policies = [parse_policy(x) for x in args.policies.split(";")]
    torch.backends.cuda.matmul.allow_tf32 = True  # the oracle / token_topk dense score blocks (fp32 einsums)
    nll = ret = None
    if args.docs and args.parity:  # chunked prefill with the cache vs one stateless forward, dense
        doc = long_docs(args.docs, args.parity, 1, args.bos)[0]
        pol.policy = parse_policy("dense")
        _, _, chunked = prefill(model, doc, args.chunk, want_nll=True)
        full, n = rp.nll(model, doc, next(model.parameters()).device)
        print(f"[probe] parity at {args.parity}: chunked-prefill NLL {float(chunked.mean()):.6f}, one forward "
              f"{full / n:.6f}", flush=True)
        (args.out / "parity.json").write_text(json.dumps({"tokens": args.parity, "chunked": float(chunked.mean()),
                                                          "one_forward": full / n}))
    if "nll" in args.tasks:
        nll = run_nll(model, pol, policies, args, args.out)
    if "retrieval" in args.tasks and not args.tiny:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model)
        ret = run_retrieval(model, pol, policies, args, args.out, tok)
    if args.tiny:  # chunked prefill with the cache == one forward; and the choice scoring path
        ids = torch.randint(3, 50_000, (1, 3000), generator=torch.Generator().manual_seed(1))
        pol.policy = parse_policy("dense")
        _, _, chunked = prefill(model, ids, 700, want_nll=True)
        full, n = rp.nll(model, ids, next(model.parameters()).device)
        print(f"[probe] tiny chunked-prefill NLL {float(chunked.mean()):.6f} vs one forward {full / n:.6f}", flush=True)
        pick, sc = score_choice(model, torch.randint(3, 50_000, (1, 3000)), [[5, 6, 7], [8, 9], [10]], args.chunk)
        print(f"[probe] tiny choice scoring ok: {pick} {[round(x, 3) for x in sc]}", flush=True)
    write_summary(args.out, nll, ret)
    print(f"[probe] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

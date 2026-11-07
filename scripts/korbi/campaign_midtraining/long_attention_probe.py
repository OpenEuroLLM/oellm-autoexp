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
                                 compress_mean, indexer) and PageSparseConfig fields, e.g. page_mean:topk=32
                                 (indexer: --train-indexer N trains one PageIndexer per attention layer first,
                                 KL to the dense page mass on N held-out long documents: the C2 warm-up)
                                 hierarchical selection: any page selector with hier_group=G,hier_topk=K
                                 block-wise selection: select_block=B[,select_score=prev_max]; decode=token
                                 selects per token where inference can afford it (the retrieval continuations
                                 and the prompt's final block; prefill stays block-wise)
  lsa[:pool=4,block=512,...]     LSA (sparse_variants.LSA, no parameters: zero-shot); the bucket index is built
                                 incrementally per complete cluster block, like a KV-cache index
  nsa[:branch=1,...]             NSA's selection branch (branch=1: pages ranked by the mean-compressed blocks'
                                 attention mass, training-free) or its compression branch alone (branch=0)
  csa2[:compress=2,...]          CSA2 with mean compression; the lightning indexer is trained first by
                                 --train-csa2 N (KL to the dense mass per compressed entry) or loaded (--csa2)

Tasks:
  nll        per-position NLL (bins) on long held-out documents (v8ml-valid long_64k: 89 docs >= 128k, 23 >= 256k,
             10 >= 512k), prefilled up to each length
  retrieval  passkey (depths 0.1 / 0.5 / 0.9), 64-fact multi-key retrieval in natural text, variable tracking
             (scripts/korbi/downstream/prepare_tasks.py), scored as 10-way choices by the continuation logprob

    python long_attention_probe.py --model <export> --out <dir> --tasks nll,retrieval \\
        --policies dense/logn/window/page_mean:topk=32 --lengths 65536,131072,262144 --docs <prefix> --raw <raw.jsonl>
    python long_attention_probe.py --tiny --out /tmp/x        # local check: reduced random model, random tokens
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
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


def load_sparse_variants():
    path = REPO / "submodules/Megatron-LM/megatron/core/transformer/experimental_attention_variant/sparse_variants.py"
    spec = importlib.util.spec_from_file_location("sparse_variants", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sparse_variants"] = mod
    spec.loader.exec_module(mod)
    return mod


sv = load_sparse_variants()
VARIANTS = {"lsa": sv.LSAConfig, "nsa": sv.NSAConfig, "csa2": sv.CSA2Config}


def parse_policy(text):
    name, _, rest = text.partition(":")
    kw = {}
    for item in filter(None, rest.split(",")):
        key, val = item.split("=")
        try:
            kw[key] = int(val) if val.lstrip("-").isdigit() else float(val)
        except ValueError:  # string options (select_score=prev_max)
            kw[key] = val
    if name == "logn":
        return {"name": text, "kind": "logn", "base": kw.get("base", 65536)}
    if name == "dense":
        return {"name": text, "kind": "dense"}
    decode = kw.pop("decode", None)
    if name in VARIANTS:
        branch = kw.pop("branch", 1)
        if name == "nsa":  # one branch alone (gates saturated): a converted model without trained gates
            kw["gate_init"] = tuple(30.0 if i == branch else -30.0 for i in range(3))
        return {"name": text, "kind": name, "cfg": VARIANTS[name](**kw)}
    cfg = ps.PageSparseConfig(selector=name, **kw)
    out = {"name": text, "kind": "sparse", "cfg": cfg}
    if decode == "token":
        out["decode_cfg"] = dataclasses.replace(cfg, select_block=1)
    return out


class AttentionPolicy:
    """Replaces every attention mixer's forward: chunk-wise prefill / decode against the HF cache's (k, v), with
    the current policy. With ``indexers`` (one page_sparse.PageIndexer per attention layer) the ``indexer``
    selector scores pages from the layer input; ``train_indexer`` trains them meanwhile (dense forward, KL to the
    dense attention's page mass of every chunk's queries: the C2 warm-up)."""

    def __init__(self, model):
        self.policy = {"kind": "dense"}
        self.n = 0
        self.indexers, self.opts, self.index_keys = {}, {}, {}
        self.train_indexer = False
        self.modules = {}  # (policy name, layer) -> NSA / CSA2 module
        self.csa2_state = None  # trained CSA2 indexers {layer: state_dict}, loaded into every csa2 policy
        self.train_csa2 = False
        self.decode_mode = False  # the continuation tokens of a choice (decode)
        self.tail_end = None  # absolute end of a prompt whose final selection block selects per token
        self.losses = defaultdict(list)
        for i, layer in enumerate(model.model.layers):
            if layer.kind == "attention":
                layer.mixer.forward = types.MethodType(self._forward(i), layer.mixer)
                self.n += 1

    def add_indexers(self, model, heads=4, dim=64, page=64, lr=1e-3):
        for i, layer in enumerate(model.model.layers):
            if layer.kind == "attention":
                dev = layer.mixer.q_proj.weight.device
                idx = ps.PageIndexer(layer.mixer.q_proj.weight.shape[1], heads, dim, page).to(dev)
                self.indexers[i] = idx
                self.opts[i] = torch.optim.Adam(idx.parameters(), lr=lr)

    def sparse(self, p, q, k, v, past, idx_scores):
        """page_sparse with the policy's config; with decode=token the decode queries and the prompt's final
        selection block select per token (inference re-selects there at no bandwidth cost)."""
        scale = q.shape[-1] ** -0.5
        cfg, dcfg = p["cfg"], p.get("decode_cfg")
        if dcfg is None:
            return ps.page_sparse_attention(q, k, v, cfg, q_offset=past, scale=scale, indexer_scores=idx_scores)
        if self.decode_mode:
            return ps.page_sparse_attention(q, k, v, dcfg, q_offset=past, scale=scale, indexer_scores=idx_scores)
        s = q.shape[2]
        end = past + s
        if self.tail_end != end:
            return ps.page_sparse_attention(q, k, v, cfg, q_offset=past, scale=scale, indexer_scores=idx_scores)
        t0 = max(((end - 1) // cfg.select_block) * cfg.select_block - past, 0)  # the final block's first row
        parts = []
        if t0 > 0:
            parts.append(ps.page_sparse_attention(q[:, :, :t0], k, v, cfg, q_offset=past, scale=scale,
                                                  indexer_scores=None if idx_scores is None else idx_scores[:, :t0]))
        parts.append(ps.page_sparse_attention(q[:, :, t0:], k, v, dcfg, q_offset=past + t0, scale=scale,
                                              indexer_scores=None if idx_scores is None else idx_scores[:, t0:]))
        return torch.cat(parts, 2)

    def module(self, li, p, att_x, d, nh, nkv):
        """The NSA / CSA2 module of policy p at layer li (created on first use; csa2 gets the trained indexer)."""
        key = (p["name"], li)
        if key not in self.modules:
            h = att_x.shape[-1]
            if p["kind"] == "nsa":
                m = sv.NSA(h, nh, d, p["cfg"])
            else:
                m = sv.CSA2(h, nkv, d, p["cfg"])
                if self.csa2_state is not None and li in self.csa2_state:
                    m.load_state_dict(self.csa2_state[li])
            self.modules[key] = m.to(att_x.device).float()
        return self.modules[key]

    def variant(self, li, p, q, k, v, x, past, fresh):
        """One chunk of queries (q [1, nh, s, d] at past..) over the cached k / v with an LSA / NSA / CSA2 policy.
        LSA's bucket index grows by complete cluster blocks (kept in index_keys, restored per choice like the
        indexer keys); CSA2 keeps its key-side compression features the same way."""
        scale = q.shape[-1] ** -0.5
        cfg = p["cfg"]
        if p["kind"] == "lsa":
            mod = sv.LSA(cfg)
            ikey = ("lsa", p["name"], li)
            idx = None if fresh else self.index_keys.get(ikey)
            span = cfg.pool * cfg.block
            done = 0 if idx is None else idx[4]
            nblk = k.shape[2] // span
            if nblk > done:  # the new complete blocks, indexed on their own (blocks are independent)
                c, mem, ok, last = mod.index(k[:, :, done * span: nblk * span])
                mem, last = mem + done * span, last + done * span
                if idx is not None:
                    c, mem, ok, last = self._cat_index(idx[:4], (c, mem, ok, last))
                idx = (c, mem, ok, last, nblk)
                self.index_keys[ikey] = idx
            return mod(q, k, v, past, scale, None if idx is None else idx[:4])
        mod = self.module(li, p, x, q.shape[-1], q.shape[1], k.shape[1])
        if p["kind"] == "nsa":
            return mod(q.float(), k.float(), v.float(), x.float(), past, scale).to(q.dtype)
        zkey = ("csa2z", p["name"], li)
        with torch.no_grad():
            z = mod.token_features(x.float())
        prev = None if fresh else self.index_keys.get(zkey)
        z = z if prev is None else torch.cat([prev, z], 1)
        self.index_keys[zkey] = z
        if self.train_csa2:  # the warm-up: dense output, the indexer learns the dense mass per entry
            with torch.enable_grad():
                loss = mod.indexer_loss(q.float(), k.float(), x.float(), z, past, scale)
                if loss.requires_grad:
                    opt = self.opts.setdefault(("csa2", li), torch.optim.Adam(
                        [prm for n, prm in mod.named_parameters() if n.startswith("idx_")], lr=self.csa2_lr))
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    self.losses[li].append(float(loss))
            return ps.dense_attention(q, k, v, q_offset=past, scale=scale)
        return mod(q.float(), k.float(), v.float(), x.float(), z, past, scale, chunk=256).to(q.dtype)

    @staticmethod
    def _cat_index(a, b):
        """Two LSA indexes (centroids, members [nb, M], valid [nb, M], last) -> one (members padded to max M)."""
        M = max(a[1].shape[1], b[1].shape[1])
        mem = [torch.nn.functional.pad(t[1], (0, M - t[1].shape[1])) for t in (a, b)]
        ok = [torch.nn.functional.pad(t[2], (0, M - t[2].shape[1])) for t in (a, b)]
        return (torch.cat([a[0], b[0]]), torch.cat(mem), torch.cat(ok), torch.cat([a[3], b[3]]))

    def _forward(self, li):
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
            idx_scores = None
            if li in pol.indexers:
                idxr = pol.indexers[li]
                if state is None:
                    pol.index_keys[li] = None
                with torch.enable_grad() if pol.train_indexer else torch.no_grad():
                    xf = x.float()  # the indexer trains in fp32
                    ki = idxr.wk(xf)  # [b, s, dim]
                    prev = pol.index_keys[li]
                    keys = ki if prev is None else torch.cat([prev, ki], 1)
                    pk = ps.page_pool(keys.unsqueeze(1), idxr.page, "mean")[:, 0]
                    sc = idxr(xf, pk)  # [b, s, P]
                    if pol.train_indexer:
                        oracle = ps.PageSparseConfig(selector="oracle", page=idxr.page, window=512, sink_pages=1)
                        with torch.no_grad():
                            target = ps.page_scores(q, k, oracle, past, d**-0.5)
                        cand = ps.candidate_mask(s, pk.shape[1], past, oracle, x.device)
                        if bool(cand.any()):
                            loss = ps.indexer_kl_loss(sc, target, cand)
                            pol.opts[li].zero_grad(set_to_none=True)
                            loss.backward()
                            pol.opts[li].step()
                            pol.losses[li].append(float(loss))
                pol.index_keys[li] = keys.detach()
                idx_scores = sc.detach()
            if p["kind"] == "logn":
                pos = past + torch.arange(s, device=x.device)
                f = torch.clamp(torch.log(pos.float() + 1) / math.log(p["base"]), min=1.0)
                q = (q.float() * f[None, None, :, None]).to(q.dtype)
            with torch.cuda.device(x.device):
                if p["kind"] in VARIANTS:
                    o = pol.variant(li, p, q, k, v, x, past, state is None)
                elif p["kind"] == "sparse":
                    o = pol.sparse(p, q, k, v, past, idx_scores)
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
def score_choice(model, ctx, conts, chunk, pol=None):
    """Index of the highest-logprob continuation after ctx (and the logprobs). ``pol``: its indexers' key cache
    is restored for every continuation, like the copied model cache."""
    if pol is not None:
        pol.tail_end = ctx.shape[1]  # the prompt's final block: per token under decode=token
    cache, last, _ = prefill(model, ctx, chunk)
    saved = dict(pol.index_keys) if pol is not None else None
    if pol is not None:
        pol.tail_end, pol.decode_mode = None, True
    lp0 = torch.log_softmax(model.lm_head(last).float(), -1)[0]
    scores = []
    dev = next(model.parameters()).device
    for cont in conts:
        s = float(lp0[cont[0]])
        if len(cont) > 1:
            if saved is not None:
                pol.index_keys = dict(saved)
            c = copy.deepcopy(cache)
            out = model.model(input_ids=torch.tensor([cont[:-1]], device=dev), past_key_values=c, use_cache=True)
            lp = torch.log_softmax(model.lm_head(out.last_hidden_state[0]).float(), -1)
            s += float(lp.gather(-1, torch.tensor(cont[1:], device=lp.device)[:, None]).sum())
            del c
        scores.append(s)
    if pol is not None:
        pol.decode_mode = False
    del cache
    return int(np.argmax(scores)), scores


def long_docs(prefix, min_len, n, bos, seed=0):
    if not prefix:  # the local check: random tokens
        return [torch.randint(3, 50_000, (1, min_len), generator=torch.Generator().manual_seed(seed + i))
                for i in range(n)]
    docs = rp.read_indexed(prefix)
    order = [i for i in np.random.default_rng(seed).permutation(len(docs)) if docs.lengths[i] >= min_len - 1]
    return [torch.tensor([[bos] + docs[int(i)][: min_len - 1].tolist()]) for i in order[:n]]


def train_indexers(model, pol, args, out):
    """The indexer warm-up on long held-out documents after the evaluation ones (dense forward)."""
    pol.add_indexers(model, args.indexer_heads, args.indexer_dim, 64, args.indexer_lr)
    docs = long_docs(args.docs, args.indexer_len, args.nll_docs + args.train_indexer, args.bos)[args.nll_docs:]
    pol.policy, pol.train_indexer = parse_policy("dense"), True
    t0 = time.time()
    for j, doc in enumerate(docs):
        prefill(model, doc, args.chunk)
        if j % 8 == 7 or j == len(docs) - 1:
            mean = {li: round(float(np.mean(v[-16:])), 4) for li, v in pol.losses.items()}
            print(f"[indexer] {j + 1}/{len(docs)} docs, KL (last 16 chunks) {mean} {time.time() - t0:.0f}s",
                  flush=True)
    pol.train_indexer = False
    torch.save({li: {k: v.cpu() for k, v in m.state_dict().items()} for li, m in pol.indexers.items()},
               out / "indexers.pt")
    (out / "indexer_losses.json").write_text(json.dumps({str(k): v for k, v in pol.losses.items()}))


def train_csa2(model, pol, policy, args, out):
    """CSA2's lightning-indexer warm-up (dense forward, KL to the dense attention mass per compressed entry) on
    long held-out documents after the evaluation ones; the indexers then serve every csa2 policy."""
    docs = long_docs(args.docs, args.indexer_len, args.nll_docs + args.train_csa2, args.bos)[args.nll_docs:]
    pol.policy, pol.train_csa2, pol.csa2_lr = policy, True, args.indexer_lr
    t0 = time.time()
    for j, doc in enumerate(docs):
        prefill(model, doc, args.chunk)
        if j % 8 == 7 or j == len(docs) - 1:
            mean = {li: round(float(np.mean(v[-16:])), 4) for li, v in pol.losses.items()}
            print(f"[csa2] {j + 1}/{len(docs)} docs, KL (last 16 chunks) {mean} {time.time() - t0:.0f}s",
                  flush=True)
    pol.train_csa2 = False
    pol.csa2_state = {li: {k: v.detach().clone() for k, v in m.state_dict().items()}
                      for (name, li), m in pol.modules.items() if name == policy["name"]}
    pol.modules = {}  # rebuilt per policy from csa2_state
    torch.save({li: {k: v.cpu() for k, v in sd.items()} for li, sd in pol.csa2_state.items()}, out / "csa2.pt")
    (out / "csa2_losses.json").write_text(json.dumps({str(k): v for k, v in pol.losses.items()}))


def run_nll(model, pol, policies, args, out):
    res = {}
    for L in args.lengths:
        docs = long_docs(args.docs, L, args.nll_docs, args.bos)
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
        t0 = time.time()
        rows = retrieval_rows(tok, args, L)
        print(f"[ret] L={L}: {len(rows)} questions prepared in {time.time() - t0:.0f}s", flush=True)
        for p in policies:
            pol.policy = p
            t0 = time.time()
            acc = defaultdict(list)
            for j, r in enumerate(rows):
                ctx = torch.tensor([r["items"][0]["ctx"]])
                tq = time.time()
                pick, _ = score_choice(model, ctx, [it["cont"] for it in r["items"]], args.chunk, pol)
                acc[r["task"].split("_")[0]].append(pick == r["gold"])
                print(f"[ret.q] {p['name']} {j + 1}/{len(rows)} {r['task']} ctx {ctx.shape[1]} "
                      f"conts {sum(len(it['cont']) for it in r['items'])} {pick == r['gold']} "
                      f"{time.time() - tq:.1f}s", flush=True)
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
    p.add_argument("--policies", default="dense/logn/window/page_mean", help="'/'-separated policies")
    p.add_argument("--lengths", default="65536,131072,262144")
    p.add_argument("--chunk", type=int, default=16384)
    p.add_argument("--docs", default="", help="indexed dataset prefix of long held-out documents")
    p.add_argument("--nll-docs", type=int, default=4)
    p.add_argument("--raw", default="", help="raw.jsonl (natural filler for the retrieval tasks)")
    p.add_argument("--n", type=int, default=6, help="questions per retrieval task and length")
    p.add_argument("--bos", type=int, default=1)
    p.add_argument("--parity", type=int, default=32768, help="parity check length (0: off)")
    p.add_argument("--train-indexer", type=int, default=0, help="documents for the indexer warm-up (0: off)")
    p.add_argument("--indexer-len", type=int, default=65536)
    p.add_argument("--indexer-heads", type=int, default=4)
    p.add_argument("--indexer-dim", type=int, default=64)
    p.add_argument("--indexer-lr", type=float, default=1e-3)
    p.add_argument("--indexers", default="", help="indexers.pt of an earlier run (instead of --train-indexer)")
    p.add_argument("--train-csa2", type=int, default=0, help="documents for the CSA2 indexer warm-up (0: off)")
    p.add_argument("--csa2", default="", help="csa2.pt of an earlier run (instead of --train-csa2)")
    args = p.parse_args()
    args.lengths = [int(x) for x in args.lengths.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "args.json").write_text(json.dumps(vars(args), default=str, indent=1))
    t0 = time.time()
    model = rp.load_model(args.model, args.tiny)
    pol = AttentionPolicy(model)
    print(f"[probe] model loaded in {time.time() - t0:.0f}s; {pol.n} attention layers", flush=True)
    policies = [parse_policy(x) for x in args.policies.split("/")]
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
    if args.train_indexer and (args.docs or args.tiny):
        train_indexers(model, pol, args, args.out)
    elif args.indexers:  # trained indexers of an earlier run
        pol.add_indexers(model, args.indexer_heads, args.indexer_dim, 64, args.indexer_lr)
        state = torch.load(args.indexers)
        for li, sd in state.items():
            pol.indexers[int(li)].load_state_dict(sd)
    csa2 = [x for x in policies if x["kind"] == "csa2"]
    if csa2 and args.train_csa2 and (args.docs or args.tiny):
        train_csa2(model, pol, csa2[0], args, args.out)
    elif csa2 and args.csa2:
        pol.csa2_state = {int(li): sd for li, sd in torch.load(args.csa2).items()}
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
        pol.policy = policies[-1]  # the choice path under the last policy (decode / tail selection)
        pick, sc = score_choice(model, torch.randint(3, 50_000, (1, 3000)), [[5, 6, 7], [8, 9], [10]], args.chunk,
                                pol)
        print(f"[probe] tiny choice scoring ok: {pick} {[round(x, 3) for x in sc]}", flush=True)
    write_summary(args.out, nll, ret)
    print(f"[probe] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

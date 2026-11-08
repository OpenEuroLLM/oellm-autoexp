"""CED retrofit probe (docs/campaign_3b_hybrid_midtraining.md §7, 2026-11-05) on an HF export of oellm_ambitious.

DeepSeek-V4.1's Causal Encoder-Decoder projects the upper half's global KV from the middle layer's hidden state
H_mid, so prefill only runs the lower half. For our hybrid, every cross-token path of the upper half would have
to come from H_mid: the attention layers' k / v and the CKDA layers' k / v / beta / forget gate (their state
writes). Queries, output gates, MoE, mHC stay per layer. This probe measures how far a per-layer LINEAR map from
H_mid (all mHC streams, n x hidden) reproduces each upper mixer's input on its state-writing path -- the CED
initialization of a retrofit -- and what substituting it costs:

  fit   ridge regression x_l ~ W_l [H_mid; 1] for every upper attention / CKDA layer l (its normalized mixer
        input), on documents from the front of each validation set; ridge strength per layer on held-out data
  r2    held-out R^2 of x_l and of the projections that write the state (attention k, v; CKDA k, v, beta, f)
  nll   held-out NLL with the predicted input on the state-writing path only (queries and gates keep the true
        input): all upper mixers / attention only / CKDA only, and each layer alone on one domain

    python ced_probe.py --model <export> --data web_en=<prefix>,...,long_64k=<prefix> --long long_64k --out <dir>
    python ced_probe.py --tiny --out /tmp/x        # local check: reduced random model
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import types
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import router_probe as rp  # noqa: E402

os.environ.setdefault("OELLM_HF_PIPELINE", "0")
ALPHAS = (1e-5, 1e-4, 1e-3, 1e-2)


class Proxy(torch.nn.Module):
    """Stands in for a projection of the FLA layer: applies it to the probe's predicted input when one is set."""

    def __init__(self, inner, probe):
        super().__init__()
        self.inner = inner
        self.probe = [probe]

    def forward(self, x):
        xhat = self.probe[0].xhat
        return self.inner(x if xhat is None else xhat)


class CEDProbe:
    def __init__(self, model, mid: int | None):
        self.model = model
        inner = model.model
        kinds = [layer.kind for layer in inner.layers]
        n = len(kinds)
        # the middle: the first layer of the upper half of the sublayers (98 -> 49: the second Engram layer)
        self.mid = mid if mid is not None else n // 2
        self.upper = [i for i in range(self.mid, n) if kinds[i] in ("attention", "ckda")]
        self.kinds = kinds
        self.capture = False
        self.h_mid = None
        self.x = {}
        self.sub = set()  # upper layers whose state-writing path gets the prediction
        self.W = {}  # layer -> [D + 1, h] fp32 (on the H_mid device)
        self.xhat = None
        layer = inner.layers[self.mid]
        orig = layer.forward

        def mid_forward(this, streams, *a, **k):
            self.h_mid = streams
            return orig(streams, *a, **k)

        layer.forward = types.MethodType(mid_forward, layer)
        for i in self.upper:
            mixer = inner.layers[i].mixer
            if kinds[i] == "ckda":
                fla = mixer.layer
                for name in ("k_proj", "v_proj", "f_proj", "b_proj"):
                    setattr(fla, name, Proxy(getattr(fla, name), self))
                mixer.forward = types.MethodType(self._ckda_forward(i), mixer)
            else:
                mixer._probe_orig = mixer.forward
                mixer.forward = types.MethodType(self._attn_forward(i), mixer)

    def predict(self, i, like):
        h = self.h_mid
        flat = h.reshape(-1, h.shape[-2] * h.shape[-1]).float()
        w = self.W[i]
        out = flat @ w[:-1] + w[-1]
        return out.to(like.dtype).view(like.shape).to(like.device)

    def _ckda_forward(self, i):
        probe = self

        def forward(mixer, hidden_states, state=None, attention_mask=None, use_cache=True):
            if probe.capture:
                probe.x[i] = hidden_states
            layer = mixer.layer
            fused = layer.fuse_projections
            layer.fuse_projections = False  # the proxies are module calls: the unfused projection path
            probe.xhat = probe.predict(i, hidden_states) if i in probe.sub else None
            try:
                ctx = torch.cuda.device(hidden_states.device) if hidden_states.is_cuda else contextlib.nullcontext()
                with ctx:
                    out, _, _ = layer(hidden_states=hidden_states, attention_mask=attention_mask,
                                      past_key_values=None, use_cache=False)
            finally:
                layer.fuse_projections = fused
                probe.xhat = None
            return out, None

        return forward

    def _attn_forward(self, i):
        probe = self

        def forward(att, x, state=None, keep_state=True):
            if probe.capture:
                probe.x[i] = x
            if i not in probe.sub:
                return att._probe_orig(x, state, keep_state)
            xk = probe.predict(i, x)
            b, s, _ = x.shape
            nh, nkv, d = att.nh, att.nkv, att.d
            q = att.q_proj(x).view(b, s, nh, d)
            k = att.k_proj(xk).view(b, s, nkv, d)
            v = att.v_proj(xk).view(b, s, nkv, d)
            if att.q_norm is not None:
                q, k = att.q_norm(q), att.k_norm(k)
            o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                               is_causal=True, scale=d**-0.5, enable_gqa=nh != nkv).transpose(1, 2)
            rep = nh // nkv
            if att.xsa:
                vf = v.float().unsqueeze(-2)
                of = o.float().reshape(b, s, nkv, rep, d)
                c = (of * vf).sum(-1, keepdim=True) / (vf.pow(2).sum(-1, keepdim=True) + att.xsa_eps)
                of = (of - c * vf).reshape(b, s, nh, d)
                if att.gate_proj is not None:
                    of = of * torch.sigmoid(att.gate_proj(x).view(b, s, nh, d).float())
                o = of.to(x.dtype)
            elif att.gate_proj is not None:
                o = (o.float() * torch.sigmoid(att.gate_proj(x).view(b, s, nh, d).float())).to(x.dtype)
            return att.o_proj(o.reshape(b, s, nh * d)), None

        return forward

    def write_projections(self, i):
        """The projections that write layer i's state, as (name, callable on the mixer input)."""
        mixer = self.model.model.layers[i].mixer
        if self.kinds[i] == "attention":
            return [("k", mixer.k_proj), ("v", mixer.v_proj)]
        fla = mixer.layer
        return [("k", fla.k_proj.inner), ("v", fla.v_proj.inner), ("beta", fla.b_proj.inner), ("f", fla.f_proj.inner)]


def forward(model, ids):
    dev = next(model.parameters()).device
    with torch.no_grad():
        return model.model(input_ids=ids.to(dev)).last_hidden_state


def fit(probe, model, iters, args, out):
    xtx, xty, n = None, {}, 0
    t0 = time.time()
    for b in range(args.fit_batches):
        for dom, it in list(iters.items()):
            ids = next(it, None)
            if ids is None:
                iters.pop(dom)
                continue
            probe.capture, probe.x = True, {}
            forward(model, ids)
            probe.capture = False
            h = probe.h_mid
            dev = h.device
            feat = torch.cat([h.reshape(-1, h.shape[-2] * h.shape[-1]).double(),
                              torch.ones(h.shape[0] * h.shape[1], 1, device=dev, dtype=torch.float64)], 1)
            xtx = feat.T @ feat if xtx is None else xtx + feat.T @ feat
            for i in probe.upper:
                y = probe.x[i].reshape(-1, probe.x[i].shape[-1]).to(dev).double()
                xty[i] = feat.T @ y if i not in xty else xty[i] + feat.T @ y
            n += feat.shape[0]
            del feat
            print(f"[fit] batch {b} {dom} tokens {n} {time.time() - t0:.0f}s", flush=True)
    d = xtx.shape[0]
    scale = torch.diagonal(xtx)[:-1].mean()
    sols = {}
    for a in ALPHAS:
        reg = xtx.clone()
        idx = torch.arange(d - 1, device=reg.device)
        reg[idx, idx] += a * scale  # no penalty on the intercept
        chol = torch.linalg.cholesky(reg)
        sols[a] = {i: torch.cholesky_solve(xty[i], chol).float() for i in probe.upper}
        del reg, chol
    return sols, n


def r2_eval(probe, model, batches, sols):
    acc = defaultdict(lambda: defaultdict(float))  # (layer, alpha, name) -> sums
    for dom, ids in batches:
        probe.capture, probe.x = True, {}
        forward(model, ids)
        probe.capture = False
        for i in probe.upper:
            x = probe.x[i]
            projs = probe.write_projections(i)
            for a, sol in sols.items():
                probe.W[i] = sol[i]
                xh = probe.predict(i, x)
                with torch.no_grad():
                    for name, fn in [("x", None)] + projs:
                        y = x if fn is None else fn(x)
                        y = y.float().reshape(-1, y.shape[-1])
                        yh = (xh if fn is None else fn(xh)).float().reshape(y.shape)
                        k = (i, a, name)
                        acc[k]["sse"] += float((y - yh).pow(2).sum())
                        acc[k]["sy"] += float(y.sum())
                        acc[k]["sy2"] += float(y.pow(2).sum())
                        acc[k]["n"] += y.numel()
                        acc[k]["cos"] += float(F.cosine_similarity(y, yh, dim=-1).sum())
                        acc[k]["rows"] += y.shape[0]
    r2 = {}
    for (i, a, name), s in acc.items():
        sst = s["sy2"] - s["sy"] ** 2 / s["n"]
        r2[(i, a, name)] = {"r2": 1 - s["sse"] / max(sst, 1e-12), "cos": s["cos"] / s["rows"]}
    return r2


def nll_eval(probe, model, batches, sub):
    probe.sub = set(sub)
    tot = n = 0.0
    for _, ids in batches:
        a, c = rp.nll(model, ids, next(model.parameters()).device)
        tot += a
        n += c
    probe.sub = set()
    return tot / n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model")
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--data", default="random=random")
    p.add_argument("--long", default="long_64k")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--long-seq-len", type=int, default=65536)
    p.add_argument("--rows", type=int, default=16)
    p.add_argument("--fit-batches", type=int, default=8)
    p.add_argument("--eval-batches", type=int, default=2)
    p.add_argument("--single-domain", default="web_en", help="per-layer substitution on this domain")
    p.add_argument("--mid", type=int, default=None, help="first sublayer of the upper half (default n // 2)")
    p.add_argument("--bos", type=int, default=1)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "args.json").write_text(json.dumps(vars(args), default=str, indent=1))
    t0 = time.time()
    model = rp.load_model(args.model, args.tiny)
    print(f"[ced] model loaded in {time.time() - t0:.0f}s", flush=True)
    if args.tiny:
        args.mid = args.mid if args.mid is not None else 5
    probe = CEDProbe(model, args.mid)
    print(f"[ced] mid {probe.mid}; upper mixers {probe.upper}", flush=True)
    a = argparse.Namespace(**{**vars(args), "batches": args.fit_batches, "r2_batches": args.eval_batches})
    sols, n_fit = fit(probe, model, rp.domain_iters(args.data, a, "head"), args, args.out)
    held = [(dom, ids) for dom, it in rp.domain_iters(args.data, a, "tail").items() for ids in it]
    r2 = r2_eval(probe, model, held, sols)
    best = {i: max(ALPHAS, key=lambda al: r2[(i, al, "x")]["r2"]) for i in probe.upper}
    for i in probe.upper:
        probe.W[i] = sols[best[i]][i]
    torch.save({i: probe.W[i].to(torch.bfloat16).cpu() for i in probe.upper}, args.out / "ced_maps.pt")
    res = {"fit_tokens": n_fit, "mid": probe.mid, "best_alpha": best,
           "r2": {f"{i}|{al}|{nm}": v for (i, al, nm), v in r2.items()}, "nll": {}}
    by_dom = defaultdict(list)
    for dom, ids in held:
        by_dom[dom].append((dom, ids))
    attn = [i for i in probe.upper if probe.kinds[i] == "attention"]
    ckda = [i for i in probe.upper if probe.kinds[i] == "ckda"]
    configs = {"none": [], "all": probe.upper, "attention": attn, "ckda": ckda,
               "first_half": probe.upper[: len(probe.upper) // 2], "second_half": probe.upper[len(probe.upper) // 2:]}
    for dom, bl in by_dom.items():
        for name, sub in configs.items():
            res["nll"][f"{dom}|{name}"] = nll_eval(probe, model, bl, sub)
            print(f"[ced] {dom} {name}: nll {res['nll'][f'{dom}|{name}']:.5f}", flush=True)
    single = by_dom.get(args.single_domain) or next(iter(by_dom.values()))
    for i in probe.upper:
        res["nll"][f"single|{i}"] = nll_eval(probe, model, single, [i])
    (args.out / "ced_results.json").write_text(json.dumps(res, indent=1))
    write_summary(res, probe, args.out / "ced_summary.md", args.single_domain)
    print(f"[ced] done in {time.time() - t0:.0f}s", flush=True)


def write_summary(res, probe, path, single_dom):
    lines = [f"# CED retrofit probe: upper mixers from H_mid (sublayer {probe.mid}) by a linear map",
             "", f"Fit tokens: {res['fit_tokens']}. Held-out R^2 (cosine) at each layer's best ridge strength.", "",
             "| layer | kind | alpha | x | k | v | beta | f | ΔNLL alone |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    base_single = res["nll"].get(f"{single_dom}|none")
    for i in probe.upper:
        al = res["best_alpha"][i]
        cells = []
        for nm in ("x", "k", "v", "beta", "f"):
            v = res["r2"].get(f"{i}|{al}|{nm}")
            cells.append(f"{v['r2']:.3f} ({v['cos']:.3f})" if v else "")
        d = res["nll"][f"single|{i}"] - base_single if base_single is not None else float("nan")
        lines.append(f"| {i} | {probe.kinds[i]} | {al:g} | " + " | ".join(cells) + f" | {d:+.4f} |")
    lines += ["", "| domain | NLL | Δ all | Δ attention | Δ CKDA | Δ first half | Δ second half |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    doms = sorted({k.split("|")[0] for k in res["nll"] if not k.startswith("single")})
    for dom in doms:
        b = res["nll"][f"{dom}|none"]
        cells = [f"{res['nll'][f'{dom}|{c}'] - b:+.4f}" for c in ("all", "attention", "ckda", "first_half", "second_half")]
        lines.append(f"| {dom} | {b:.4f} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

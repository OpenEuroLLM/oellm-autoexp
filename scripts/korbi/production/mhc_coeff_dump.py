#!/usr/bin/env python3
"""Per-token mHC coefficients and streams of an HF export (one node; the export's own model code), the input of
an LDM init that imitates mHC-4 (docs: the mHC -> LDM switch of the v4b main run).

The 62000 mixers showed that the bias-only coefficients are ~uniform (read entropy 0.997, write 0.95) while the
run's logged token means are not (0.71 / 0.61): mHC's stream use lives in the token-dependent term alpha phi x^.
So the imitation needs activations. Every mHC decoder layer's forward is wrapped (the model itself unchanged):

  all tokens      the coefficients in fp32 from the layer's input streams (Megatron's _mhc_coeffs; the V layers'
                  res gate lerp(I, H_res, rho) as the export applies it): sums and sums of squares of h_pre, h_post
                  and H_res; the cosine of the layer input u = h_pre . streams with the stream mean m and with the
                  collapsed input c = e + sum_j delta_j (the PR #82 conversion before any training)
  sampled tokens  (--samples per sequence, the same positions in every layer) the input streams [4, h], the
                  sublayer output delta [h] and h_pre / h_post / H_res, so any candidate input (static fold,
                  fitted LDM read, ...) can be rebuilt offline from e and the deltas; the last layer's output
                  streams (final_streams: the LM head reads their mean)

engram_branch_gates (the bg_learned fork): an Engram layer writes one output per stream; delta is their mean, the
value a single-stream layer writes when its n gates are averaged.

Sets: the run's fixed validation sets (BOS + document, as trained), --docs per set, cut to --seq-len.
Output <out>/: layer_<l>.npz (samples), stats.json (per layer all-token statistics), meta.json.

    python mhc_coeff_dump.py --model-dir <export> --out <dir> [--docs 16] [--seq-len 4096] [--samples 32]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SETS = ("all", "web_en", "multilingual", "code", "math", "reasoning", "long_64k")


def sinkhorn(logits: torch.Tensor, iters: int) -> torch.Tensor:
    """Megatron's _sinkhorn_knopp_impl (log space, columns then rows)."""
    x = logits.float()
    for _ in range(iters):
        x = x - torch.logsumexp(x, dim=-2, keepdim=True)
        x = x - torch.logsumexp(x, dim=-1, keepdim=True)
    return torch.exp(x)


def coefficients(mhc, streams: torch.Tensor, res_gate):
    """(h_pre [T, n], h_post [T, n], H_res [T, n, n]) in fp32 for streams [T, n, h] (Megatron _mhc_coeffs)."""
    t, n, h = streams.shape
    flat = streams.reshape(t, n * h)
    r = torch.linalg.vector_norm(flat.float(), dim=-1, keepdim=True) / (n * h) ** 0.5
    c = torch.matmul(flat, mhc.phi.to(flat.dtype)).float() / r.clamp(min=mhc.eps)
    pre, post, res = torch.split(c, [n, n, n * n], dim=-1)
    h_pre = torch.sigmoid(mhc.alpha_pre.float() * pre + mhc.b_pre.float())
    h_post = 2.0 * torch.sigmoid(mhc.alpha_post.float() * post + mhc.b_post.float())
    h_res = sinkhorn(mhc.alpha_res.float() * res.view(t, n, n) + mhc.b_res.float(), mhc.iters)
    if res_gate is not None:
        eye = torch.eye(n, device=h_res.device, dtype=h_res.dtype)
        h_res = torch.lerp(eye, h_res, res_gate)
    return h_pre, h_post, h_res


class Recorder:
    """State shared by the wrapped layers during one sequence's forward."""

    def __init__(self, n_layers: int, n: int):
        self.n_layers, self.n = n_layers, n
        self.reset()

    def reset(self) -> None:
        n_layers, n = self.n_layers, self.n
        self.pos = None          # sampled positions of the current sequence (LongTensor on CPU)
        self.collapsed = None    # c = e + sum_j delta_j, fp32 [T, h]
        self.samples = {l: {k: [] for k in ("streams", "delta", "h_pre", "h_post", "h_res", "final_streams")}
                        for l in range(n_layers)}
        z = lambda *s: torch.zeros(*s, dtype=torch.float64)  # noqa: E731
        self.stats = {l: {"tokens": 0, "h_pre": z(n), "h_pre_sq": z(n), "h_post": z(n), "h_post_sq": z(n),
                          "h_res": z(n, n), "h_res_sq": z(n, n), "cos_mean": 0.0, "cos_collapse": 0.0,
                          "share_entropy": 0.0}
                      for l in range(n_layers)}
        self.calls = 0


def instrument(model, rec: Recorder) -> list[int]:
    layers = []
    for idx, layer in enumerate(model.model.layers):
        if getattr(layer, "ldm", False) or getattr(layer, "mhc", None) is None:
            continue
        layers.append(idx)
        orig_forward, orig_mix = layer.forward, layer._mix
        if layer.kind == "engram" and getattr(layer.mixer, "branch_gates", False):
            # per-stream Engram gates: the layer bypasses _mix and writes one output per stream; delta is their
            # mean, the value a single-stream (LDM) layer writes when its n gates are averaged
            orig_mixer = layer.mixer.forward

            def mixer_forward(*args, _orig=orig_mixer, _layer=layer, **kw):
                outs, state = _orig(*args, **kw)
                _layer._mhcdump_delta = outs.mean(dim=-2) if outs.dim() == 4 else outs
                return outs, state

            layer.mixer.forward = mixer_forward

        def mix(self, *args, _orig=orig_mix, _l=idx, **kw):
            out, state = _orig(*args, **kw)
            self._mhcdump_delta = out
            return out, state

        def forward(self, streams, *args, _orig=orig_forward, _l=idx, **kw):
            b, s, n, h = streams.shape
            flat = streams.reshape(b * s, n, h)
            with torch.no_grad():
                h_pre, h_post, h_res = coefficients(self.mhc, flat, self.res_gate)
            new_streams, state = _orig(streams, *args, **kw)
            with torch.no_grad():
                delta = self._mhcdump_delta.reshape(b * s, h)
                self._mhcdump_delta = None
                st = rec.stats[_l]
                st["tokens"] += b * s
                for k, v in (("h_pre", h_pre), ("h_post", h_post), ("h_res", h_res)):
                    st[k] += v.double().sum(0).cpu()
                    st[k + "_sq"] += v.double().square().sum(0).cpu()
                share = h_pre / h_pre.sum(-1, keepdim=True)
                st["share_entropy"] += float(-(share * share.clamp_min(1e-30).log()).sum(-1).sum() / np.log(n))
                u = (h_pre.unsqueeze(-1) * flat.float()).sum(-2)
                m = flat.float().mean(-2)
                if _l == layers[0]:
                    rec.collapsed = flat[:, 0].float().clone()  # e: every stream is the embedding at layer 0
                c = rec.collapsed.to(u.device)
                st["cos_mean"] += float(F.cosine_similarity(u, m, dim=-1).sum())
                st["cos_collapse"] += float(F.cosine_similarity(u, c, dim=-1).sum())
                rec.collapsed = c + delta.float()
                p = rec.pos.to(flat.device)
                smp = rec.samples[_l]
                smp["streams"].append(flat[p].to(torch.bfloat16).cpu())
                smp["delta"].append(delta[p].to(torch.bfloat16).cpu())
                smp["h_pre"].append(h_pre[p].cpu())
                smp["h_post"].append(h_post[p].cpu())
                smp["h_res"].append(h_res[p].cpu())
                if _l == layers[-1]:  # the LM head reads the mean of the last layer's output streams
                    smp["final_streams"].append(new_streams.reshape(b * s, n, h)[p].to(torch.bfloat16).cpu())
                rec.calls += 1
            return new_streams, state

        layer._mix = types.MethodType(mix, layer)
        layer.forward = types.MethodType(forward, layer)
    return layers


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--docs", type=int, default=16, help="documents per validation set")
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--min-len", type=int, default=1024, help="shortest document taken")
    ap.add_argument("--samples", type=int, default=32, help="sampled token positions per sequence")
    ap.add_argument("--gpu-mem-gib", type=float, default=54)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "scripts" / "rluukkon"))
    from engram_gate_diag import token_nll, valid_docs
    from transformers import AutoConfig, AutoTokenizer

    from oellm_autoexp.hf_export.score_on_flag import layer_device_map, load_engram_tables_cpu

    t0 = time.time()
    cfg = AutoConfig.from_pretrained(a.model_dir, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    bos = tok.bos_token_id
    model = load_engram_tables_cpu(a.model_dir, layer_device_map(cfg, a.gpu_mem_gib, True)).eval()
    dev = next(model.parameters()).device
    print(f"loaded in {time.time() - t0:.0f}s; bos {bos}", flush=True)

    sets = [s for s in a.sets.split(",") if s]
    docs = [(name, d) for name in sets for d in valid_docs(name, a.docs, a.seq_len, a.min_len, a.seed)]
    probe = torch.tensor(np.concatenate([[bos], docs[0][1][:1023]]), device=dev).unsqueeze(0)
    with torch.no_grad():
        ref = token_nll(model, probe)
    n = cfg.mhc_expansion_rate
    rec = Recorder(len(model.model.layers), n)
    layers = instrument(model, rec)
    rec.pos = torch.arange(0, probe.shape[1], max(1, probe.shape[1] // a.samples))[: a.samples]
    with torch.no_grad():
        mine = token_nll(model, probe)
    check = {"mhc_layers": len(layers), "calls": rec.calls, "ref_nll": float(ref.mean()),
             "instrumented_nll": float(mine.mean()), "max_abs_token_diff": float(np.abs(ref - mine).max())}
    print(json.dumps(check), flush=True)
    if rec.calls != len(layers):
        raise SystemExit("the wrapped layer forwards are not in use")
    if abs(check["instrumented_nll"] - check["ref_nll"]) > 2e-3:
        raise SystemExit("the instrumentation changed the model's output")
    rec.reset()  # drop the probe's records; the wrappers keep this Recorder

    gen = torch.Generator().manual_seed(a.seed)
    meta = {"model_dir": str(a.model_dir), "sets": [], "seq_lens": [], "positions": [], "check": check,
            "layers": layers, "kinds": [model.model.layers[l].kind for l in layers],
            "res_gate": {l: model.model.layers[l].res_gate for l in layers if model.model.layers[l].res_gate is not None}}
    nll = []
    for i, (name, d) in enumerate(docs):
        ids = torch.tensor(np.concatenate([[bos], d]), device=dev).unsqueeze(0)
        T = ids.shape[1]
        pos = torch.sort(torch.randperm(T - 1, generator=gen)[: a.samples] + 1).values  # never the BOS
        rec.pos = pos
        with torch.no_grad():
            nll.append(float(token_nll(model, ids).mean()))
        meta["sets"].append(name)
        meta["seq_lens"].append(T)
        meta["positions"].append(pos.tolist())
        if i % 16 == 0:
            print(f"[{time.time() - t0:6.0f}s] {i + 1}/{len(docs)} {name} T={T} nll={nll[-1]:.4f}", flush=True)
    meta["nll"] = nll

    a.out.mkdir(parents=True, exist_ok=True)
    stats = {}
    for l in layers:
        st = rec.stats[l]
        T = st["tokens"]
        mean = {k: (st[k] / T) for k in ("h_pre", "h_post", "h_res")}
        std = {k: (st[k + "_sq"] / T - mean[k].square()).clamp_min(0).sqrt() for k in ("h_pre", "h_post", "h_res")}
        stats[l] = {"kind": model.model.layers[l].kind, "tokens": T,
                    **{f"{k}_mean": mean[k].tolist() for k in mean}, **{f"{k}_std": std[k].tolist() for k in std},
                    "share_entropy": st["share_entropy"] / T, "cos_u_mean": st["cos_mean"] / T,
                    "cos_u_collapse": st["cos_collapse"] / T}
        smp = rec.samples[l]
        np.savez(a.out / f"layer_{l:03d}.npz",
                 streams=torch.cat(smp["streams"]).view(torch.int16).numpy(),  # bf16 bits
                 delta=torch.cat(smp["delta"]).view(torch.int16).numpy(),
                 h_pre=torch.cat(smp["h_pre"]).numpy(), h_post=torch.cat(smp["h_post"]).numpy(),
                 h_res=torch.cat(smp["h_res"]).numpy(),
                 **({"final_streams": torch.cat(smp["final_streams"]).view(torch.int16).numpy()}
                    if smp["final_streams"] else {}))
    (a.out / "stats.json").write_text(json.dumps(stats, indent=1))
    (a.out / "meta.json").write_text(json.dumps(meta))
    print(f"[{time.time() - t0:6.0f}s] wrote {a.out}: {len(layers)} layers, "
          f"{sum(len(p) for p in meta['positions'])} sampled tokens, mean nll {np.mean(nll):.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

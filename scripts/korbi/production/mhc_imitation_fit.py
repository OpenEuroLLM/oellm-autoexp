#!/usr/bin/env python3
"""How well can a residual without mHC reproduce mHC-4's layer inputs? Offline, from mhc_coeff_dump.py's samples.

For a sampled token the dump holds the embedding e (layer 0's streams), every sublayer output delta_j and every
layer's input streams + coefficients, so the layer input mHC actually feeds the sublayer, u_l = h_pre . X_l, and
any candidate built from {e, delta_0 .. delta_{l-1}} can be compared (the cosine: a norm follows every read):

  collapse      e + sum_j delta_j                      PR #82's conversion (every write weight 1)
  fold          e + sum_j wbar_j delta_j               the same with each layer's all-token mean write folded in
  mean_stream   mean_m X_l (per token)                  the per-token write fold (what the LM head reads)
  static_best   e*b_e + sum_j b_{l,j} delta_j          the best fixed coefficients per layer (least squares in
                                                        activation space, fitted on half the sequences, scored
                                                        on the other half): the ceiling of any LDM init with
                                                        token-independent addresses (an LDM read is such a sum)

All first order: the deltas are mHC's own; after a switch they change with their inputs.

    python mhc_imitation_fit.py --dump <mhc_coeff_dump out dir> [--out <json>] [--device cuda]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def bf16(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(a).view(torch.bfloat16)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ridge", type=float, default=1e-4, help="relative ridge on the least-squares fit")
    a = ap.parse_args()
    dev = a.device
    meta = json.loads((a.dump / "meta.json").read_text())
    stats = json.loads((a.dump / "stats.json").read_text())
    layers = meta["layers"]
    seq_of = np.concatenate([[i] * len(p) for i, p in enumerate(meta["positions"])])
    test = torch.from_numpy(seq_of % 2 == 1).to(dev)  # odd sequences held out
    sets = np.array(meta["sets"])[seq_of]

    D, U, M, H = [], [], [], []
    for l in layers:
        z = np.load(a.dump / f"layer_{l:03d}.npz")
        X = bf16(z["streams"]).to(dev).float()            # [T, n, h]
        if l == layers[0]:
            D.append(X[:, 0])                              # e
        h_pre = torch.from_numpy(z["h_pre"]).to(dev)
        U.append((h_pre.unsqueeze(-1) * X).sum(-2))        # u_l
        M.append(X.mean(-2))
        D.append(bf16(z["delta"]).to(dev).float())         # delta_l
        H.append(float(np.mean(stats[str(l)]["h_post_mean"])))
    D = torch.stack(D, 1)                                  # [T, L + 1, h]: e, delta_0 ..
    T, B, h = D.shape
    K = torch.einsum("tjh,tkh->tjk", D, D)                 # per-token Gram of the basis
    print(f"{T} tokens, {len(layers)} layers, hidden {h}", flush=True)

    def cos_of(beta, Du, uu, KK):
        """cosine of u with sum_j beta_j D_j per token; beta [T, k] or [k]."""
        beta = beta.expand(Du.shape[0], -1) if beta.dim() == 1 else beta
        num = (beta * Du).sum(-1)
        den = uu.sqrt() * torch.einsum("tj,tjk,tk->t", beta, KK, beta).clamp_min(1e-30).sqrt()
        return num / den

    rows = []
    wbar = torch.tensor([1.0] + H, device=dev)
    for i, l in enumerate(layers):
        k = i + 1                                          # e, delta_0 .. delta_{i-1}
        u = U[i]
        Du = torch.einsum("tjh,th->tj", D[:, :k], u)
        uu = (u * u).sum(-1)
        KK = K[:, :k, :k]
        res = {"layer": l, "kind": stats[str(l)]["kind"],
               "collapse": cos_of(torch.ones(k, device=dev), Du, uu, KK),
               "fold": cos_of(wbar[:k], Du, uu, KK),
               "mean_stream": torch.nn.functional.cosine_similarity(u, M[i], dim=-1)}
        # static_best: min_b sum_t |u_t - D_t b|^2 / |u_t|^2 on the fit half
        w = (~test).float() / uu
        G = torch.einsum("t,tjk->jk", w, KK).double()
        bvec = torch.einsum("t,tj->j", w, Du).double()
        G = G + a.ridge * G.diagonal().mean() * torch.eye(k, device=dev, dtype=G.dtype)
        beta = torch.linalg.solve(G, bvec).float()
        res["static_best"] = cos_of(beta, Du, uu, KK)
        res["beta"] = beta.tolist()
        rows.append(res)

    def summary(sel):
        out = []
        for r in rows:
            out.append({"layer": r["layer"], "kind": r["kind"],
                        **{c: float(r[c][sel].mean()) for c in ("collapse", "fold", "mean_stream", "static_best")}})
        return out

    held = summary(test)
    print(f"{'layer':>5s} {'kind':>10s} {'collapse':>9s} {'fold':>9s} {'mean_str':>9s} {'static':>9s}  (held-out mean cosine)")
    for r in held:
        if r["layer"] % 8 == 0 or r["layer"] == layers[-1]:
            print(f"{r['layer']:5d} {r['kind']:>10s} {r['collapse']:9.4f} {r['fold']:9.4f} {r['mean_stream']:9.4f} "
                  f"{r['static_best']:9.4f}")
    avg = {c: float(np.mean([r[c] for r in held])) for c in ("collapse", "fold", "mean_stream", "static_best")}
    worst = {c: min(held, key=lambda r: r[c]) for c in avg}
    print("mean over layers:", {c: round(v, 4) for c, v in avg.items()})
    print("worst layer:", {c: (w["layer"], round(w[c], 4)) for c, w in worst.items()})
    per_set = {s: summary(test & torch.from_numpy(sets == s).to(dev)) for s in sorted(set(meta["sets"]))}
    per_set_avg = {s: {c: float(np.mean([r[c] for r in v])) for c in avg} for s, v in per_set.items()}
    for s, v in per_set_avg.items():
        print(f"  {s:13s}", {c: round(x, 4) for c, x in v.items()})
    if a.out:
        a.out.write_text(json.dumps({"held_out": held, "mean": avg, "per_set": per_set_avg,
                                     "beta": {r["layer"]: r["beta"] for r in rows}, "tokens": T}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fit a Linear Depth Memory with token-independent addresses to mHC-4's layer inputs (mhc_imitation_fit --gram).

With fixed addresses an LDM read is a fixed linear combination of the earlier values, so in the basis
{e, delta_0 .. delta_{L-1}} the whole memory is a K x (L + 1) coefficient matrix and the recursion is exact
(depth_memory.py, paper Eq. 6-8; one group, since mHC's coefficients are the same for every hidden dimension):

  write (key k, |k| = 1, weight w in (0, 2), forgetting lambda, value basis vector e_b):
      kb = sqrt(w) k, vb = sqrt(w) e_b, kappa = P kb / (lambda + kb' P kb), M += kappa (vb - M' kb)', P = (P - kappa kb' P) / lambda
  read before layer l (query q, |q| = 1, gain s = gamma * gate):   coefficients c_l = s M' q
  layer input  x_l + r_l  =  e + sum_{j<l} delta_j + c_l           (x: the plain residual, every write weight 1)

M starts at 0, P at alpha I (depth_memory_alpha 1); the embedding is written first. The objective per layer is
the share of mHC's input u_l (weighted by 1 / |u_l|^2) that the candidate explains at its best scale (a norm follows
every read): R_l = (beta' b_l)^2 / (beta' G_l beta) / N, from the normal equations of the fit half; reported on the
held-out half next to collapse (beta = 1) and the full-rank fixed-coefficient ceiling (beta = G^-1 b).

    python ldm_imitation_init.py --gram <gram.npz> [--keys 8,16] [--steps 3000] [--out <json>]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

torch.set_default_dtype(torch.float64)


def load(path: Path):
    z = np.load(path)
    layers = [int(x) for x in z["layers"]]
    g = lambda k: [torch.from_numpy(z[f"{k}_{l}"]) for l in layers]  # noqa: E731
    n = [z[f"n_fit_{l}"] for l in layers]
    return layers, g("G_fit"), g("b_fit"), g("G_test"), g("b_test"), float(n[0][0]), float(n[0][1])


def explained(beta, G, b, N):
    return float((beta @ b) ** 2 / (beta @ G @ beta).clamp_min(1e-30) / N)


class LDM(torch.nn.Module):
    def __init__(self, L: int, K: int, seed: int, alpha: float = 1.0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.L, self.K, self.alpha = L, K, alpha
        self.key = torch.nn.Parameter(torch.randn(L + 1, K, generator=g))     # e's write, then each layer's
        self.w = torch.nn.Parameter(torch.zeros(L + 1))                         # write weight 2 sigmoid(.)
        self.f = torch.nn.Parameter(torch.full((L + 1,), math.log(0.99 / 0.01)))  # forgetting sigmoid(.)
        self.q = torch.nn.Parameter(torch.randn(L, K, generator=g))
        self.s = torch.nn.Parameter(torch.zeros(L))                             # read gain: 0 = collapse

    def forward(self, production: bool = False):
        """Input coefficients [L, L + 1] of every layer (row l: e, delta_0 .. delta_{L-1}). ``production``: P in
        fp32 and M rounded to fp16 after every write (depth_memory_covariance_dtype float32, storage float16)."""
        L, K = self.L, self.K
        B = L + 1
        M = torch.zeros(K, B)
        P = self.alpha * torch.eye(K)
        cast_P = (lambda t: t.float().double()) if production else (lambda t: t)  # noqa: E731
        cast_M = (lambda t: t.half().double()) if production else (lambda t: t)  # noqa: E731
        keys = torch.nn.functional.normalize(self.key, dim=-1)
        qs = torch.nn.functional.normalize(self.q, dim=-1)
        w, lam = 2 * torch.sigmoid(self.w), torch.sigmoid(self.f)
        rows = []
        eye = torch.eye(B)
        for t in range(L + 1):                       # write t: value e (t = 0) or delta_{t-1}
            if t > 0:                                # the read before layer t - 1 sees writes 0 .. t - 1
                l = t - 1
                x = torch.zeros(B)
                x[: t] = 1.0
                rows.append(x + self.s[l] * (M.t() @ qs[l]))
            if t == L:
                break
            kb = w[t].sqrt() * keys[t]
            kappa = P @ kb / (lam[t] + kb @ P @ kb)
            M = cast_M(M + torch.outer(kappa, w[t].sqrt() * eye[t] - M.t() @ kb))
            P = cast_P((P - torch.outer(kappa, kb @ P)) / lam[t])
        return torch.stack(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gram", type=Path, required=True)
    ap.add_argument("--keys", default="8,16")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tokens", type=Path, default=None,
                    help="mhc_imitation_fit --tokens output: also report the held-out per-token cosine (a norm follows "
                         "every read, so each token's scale is free) and the rows at production precision")
    a = ap.parse_args()
    layers, Gf, bf, Gt, bt, Nf, Nt = load(a.gram)
    L = len(layers)

    def score(rows, G, b, N):
        return [explained(rows[i][: i + 1], G[i], b[i], N) for i in range(L)]

    tok = None
    if a.tokens:
        z = np.load(a.tokens)
        tok = {k: torch.from_numpy(z[k]) for k in ("K", "Du", "uu", "test")}
        held_tok = tok["test"].bool()

    def token_cos(rows):
        """held-out mean per-token cosine of u_l with sum_j rows[l, j] D_j, per layer."""
        out = []
        for i in range(L):
            k = i + 1
            beta = rows[i][:k].to(torch.float32)
            Du, uu, KK = tok["Du"][i, held_tok, :k], tok["uu"][i, held_tok], tok["K"][held_tok, :k, :k]
            num = Du @ beta
            den = uu.sqrt() * torch.einsum("j,tjk,k->t", beta, KK, beta).clamp_min(1e-30).sqrt()
            out.append(float((num / den).mean()))
        return out

    coll = score([torch.ones(L + 1) for _ in range(L)], Gt, bt, Nt)
    ceil = []
    for i in range(L):
        G = Gf[i] + 1e-4 * Gf[i].diagonal().mean() * torch.eye(i + 1)
        ceil.append(explained(torch.linalg.solve(G, bf[i]), Gt[i], bt[i], Nt))
    result = {"layers": layers, "collapse": coll, "ceiling": ceil}
    if tok is not None:
        ceil_rows = []
        for i in range(L):
            G = Gf[i] + 1e-4 * Gf[i].diagonal().mean() * torch.eye(i + 1)
            ceil_rows.append(torch.cat([torch.linalg.solve(G, bf[i]), torch.zeros(L - i)]))
        result["collapse_cos"] = token_cos([torch.ones(L + 1) for _ in range(L)])
        result["ceiling_cos"] = token_cos(ceil_rows)
        print(f"held-out per-token cosine, mean over layers: collapse {np.mean(result['collapse_cos']):.4f}  "
              f"fixed-coefficient ceiling {np.mean(result['ceiling_cos']):.4f}", flush=True)
    print(f"held-out explained share, mean over layers: collapse {np.mean(coll):.4f}  fixed-coefficient ceiling "
          f"{np.mean(ceil):.4f}", flush=True)
    for K in [int(k) for k in a.keys.split(",")]:
        best = None
        for seed in range(a.seeds):
            m = LDM(L, K, seed)
            opt = torch.optim.Adam(m.parameters(), lr=a.lr)
            for step in range(a.steps):
                rows = m()
                loss = -sum((rows[i][: i + 1] @ bf[i]) ** 2 / (rows[i][: i + 1] @ Gf[i] @ rows[i][: i + 1]) / Nf
                            for i in range(L)) / L
                opt.zero_grad()
                loss.backward()
                opt.step()
                if step % 500 == 0 or step == a.steps - 1:
                    print(f"  K {K} seed {seed} step {step:5d} fit-half explained {-loss.item():.4f}", flush=True)
            with torch.no_grad():
                held = score(m(), Gt, bt, Nt)
            if best is None or np.mean(held) > np.mean(best[0]):
                best = (held, {k: v.detach().tolist() for k, v in m.state_dict().items()})
        print(f"K {K}: held-out explained {np.mean(best[0]):.4f} (min {min(best[0]):.4f}, layer "
              f"{layers[int(np.argmin(best[0]))]})", flush=True)
        if tok is not None:
            m = LDM(L, K, 0)
            m.load_state_dict({k: torch.tensor(v) for k, v in best[1].items()})
            with torch.no_grad():
                exact, prod = m(), m(production=True)
            cos_exact, cos_prod = token_cos(exact), token_cos(prod)
            result[f"ldm_K{K}_cos"], result[f"ldm_K{K}_cos_production"] = cos_exact, cos_prod
            print(f"K {K}: held-out per-token cosine {np.mean(cos_exact):.4f} (min {min(cos_exact):.4f}, layer "
                  f"{layers[int(np.argmin(cos_exact))]}); at production precision {np.mean(cos_prod):.4f}, "
                  f"max row change {float((exact - prod).abs().max()):.2e}", flush=True)
        result[f"ldm_K{K}"] = best[0]
        result[f"ldm_K{K}_params"] = best[1]
    if a.out:
        a.out.write_text(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

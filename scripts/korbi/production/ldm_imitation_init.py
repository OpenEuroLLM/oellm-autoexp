#!/usr/bin/env python3
"""Fit a Linear Depth Memory with token-independent addresses to mHC-4's layer inputs (mhc_imitation_fit --gram).

With fixed addresses an LDM read is a fixed linear combination of the earlier values, so in the basis
{e, delta_0 .. delta_{L-1}} the whole memory is a K x (L + 1) coefficient matrix and the recursion is exact
(depth_memory.py, paper Eq. 6-8; one group, since mHC's coefficients are the same for every hidden dimension):

  write t = 0 .. L (value e, then delta_0 .. delta_{L-1}; key k, |k| = 1, weight w in (0, 2), forgetting lambda):
      kb = sqrt(w) k, vb = sqrt(w) e_t, kappa = P kb / (lambda + kb' P kb), M += kappa (vb - M' kb)', P = (P - kappa kb' P) / lambda
  read l after write l (query q, |q| = 1, gain s = gamma * gate):   coefficients c_l = s M' q
  layer input  x_l + r_l  =  e + sum_{j<l} delta_j + c_l           (x: the plain residual, every write weight 1)
  final read after write L: the LM head's input (when the gram has the 'final' target)

M starts at 0, P at alpha I (depth_memory_alpha 1); the embedding is written first. The objective per target is
the share of mHC's input u_l (weighted by 1 / |u_l|^2) that the candidate explains at its best scale (a norm follows
every read): R_l = (beta' b_l)^2 / (beta' G_l beta) / N, from the normal equations of the fit half.

Robustness (a duplicated / perturbed key meets the covariance of directions no key excited, grown by
1 / prod(lambda)): --runs takes K:<lambda_min>:<key noise>, lambda_min bounds the forgetting from below, the key
noise is fresh at every fit step; every fit is also scored with a 0.01 key perturbation. With --tokens the report is
the held-out per-token cosine (each token's scale is free), also at production precision (fp32 P, fp16 M).

    python ldm_imitation_init.py --gram <gram.npz> --tokens <tokens.npz> \\
        --runs K8:0:0,K8:0.9:0,K8:0.9:0.02 [--steps 3000] [--out <json>]
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
    n = z[f"n_fit_{layers[0]}"]
    final = bool(z["final"]) if "final" in z.files else False
    return layers, final, g("G_fit"), g("b_fit"), g("G_test"), g("b_test"), float(n[0]), float(n[1])


def explained(beta, G, b, N):
    return float((beta @ b) ** 2 / (beta @ G @ beta).clamp_min(1e-30) / N)


class LDM(torch.nn.Module):
    """Fixed-address LDM over ``writes`` values (e, delta_0 ..) with ``reads`` reads (read t after write t)."""

    def __init__(self, writes: int, reads: int, K: int, seed: int, lambda_min: float = 0.0, alpha: float = 1.0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.writes, self.reads, self.K, self.alpha, self.lambda_min = writes, reads, K, alpha, lambda_min
        self.key = torch.nn.Parameter(torch.randn(writes, K, generator=g))
        self.w = torch.nn.Parameter(torch.zeros(writes))                                # write weight 2 sigmoid(.)
        lam0 = max(0.99, lambda_min + 0.5 * (1 - lambda_min))
        self.f = torch.nn.Parameter(torch.full((writes,), self._logit((lam0 - lambda_min) / (1 - lambda_min))))
        self.q = torch.nn.Parameter(torch.randn(reads, K, generator=g))
        self.s = torch.nn.Parameter(torch.zeros(reads))                                 # read gain: 0 = collapse

    @staticmethod
    def _logit(p: float) -> float:
        p = min(max(p, 1e-6), 1 - 1e-6)
        return math.log(p / (1 - p))

    def lam(self):
        return self.lambda_min + (1 - self.lambda_min) * torch.sigmoid(self.f)

    def forward(self, production: bool = False, key_noise: float = 0.0, keys=None, queries=None, gen=None):
        """Input coefficients [reads, writes] (row t: e, delta_0 ..). ``production``: P in fp32 and M rounded to
        fp16 after every write. ``keys`` / ``queries`` override the (normalized) addresses."""
        B, K = self.writes, self.K
        M = torch.zeros(K, B)
        P = self.alpha * torch.eye(K)
        cast_P = (lambda t: t.float().double()) if production else (lambda t: t)  # noqa: E731
        cast_M = (lambda t: t.half().double()) if production else (lambda t: t)  # noqa: E731
        if keys is None:
            key = self.key if key_noise == 0 else self.key + key_noise * torch.randn(self.key.shape, generator=gen)
            keys = torch.nn.functional.normalize(key, dim=-1)
        qs = torch.nn.functional.normalize(self.q, dim=-1) if queries is None else queries
        w, lam = 2 * torch.sigmoid(self.w), self.lam()
        eye = torch.eye(B)
        rows = []
        for t in range(B):
            kb = w[t].sqrt() * keys[t]
            kappa = P @ kb / (lam[t] + kb @ P @ kb)
            M = cast_M(M + torch.outer(kappa, w[t].sqrt() * eye[t] - M.t() @ kb))
            P = cast_P((P - torch.outer(kappa, kb @ P)) / lam[t])
            if t < self.reads:                      # read t sees e, delta_0 .. delta_{t-1}
                x = torch.zeros(B)
                x[: t + 1] = 1.0
                rows.append(x + self.s[t] * (M.t() @ qs[t]))
        return torch.stack(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gram", type=Path, required=True)
    ap.add_argument("--tokens", type=Path, default=None,
                    help="mhc_imitation_fit --tokens output: the held-out per-token cosine")
    ap.add_argument("--runs", default="K8:0:0,K8:0.9:0,K8:0.9:0.02",
                    help="comma list of K<key dim>:<lambda_min>:<key noise>")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    layers, final, Gf, bf, Gt, bt, Nf, Nt = load(a.gram)
    R = len(layers)                      # read targets (incl. the final one)
    W = R if final else R + 1            # writes: e + one per layer
    runs = [(int(r.split(":")[0].lstrip("K")), float(r.split(":")[1]), float(r.split(":")[2]))
            for r in a.runs.split(",")]

    tok = None
    if a.tokens:
        z = np.load(a.tokens)
        held = torch.from_numpy(z["test"]).bool()
        tok = {"K": torch.from_numpy(z["K"])[held], "Du": torch.from_numpy(z["Du"])[:, held],
               "uu": torch.from_numpy(z["uu"])[:, held]}

    def token_cos(rows):
        out = []
        for i in range(R):
            k = i + 1
            beta = rows[i][:k].to(torch.float32)
            num = tok["Du"][i, :, :k] @ beta
            den = tok["uu"][i].sqrt() * torch.einsum("j,tjk,k->t", beta, tok["K"][:, :k, :k], beta).clamp_min(1e-30).sqrt()
            out.append(float((num / den).mean()))
        return out

    def held_score(rows):
        return token_cos(rows) if tok is not None else [explained(rows[i][: i + 1], Gt[i], bt[i], Nt) for i in range(R)]

    name = "per-token cosine" if tok is not None else "explained share"
    ceil_rows = []
    for i in range(R):
        G = Gf[i] + 1e-4 * Gf[i].diagonal().mean() * torch.eye(i + 1)
        ceil_rows.append(torch.cat([torch.linalg.solve(G, bf[i]), torch.zeros(W - i - 1)]))
    result = {"layers": layers, "final": final, "metric": name,
              "collapse": held_score([torch.ones(W) for _ in range(R)]), "ceiling": held_score(ceil_rows)}
    print(f"held-out {name}, mean over {R} targets: collapse {np.mean(result['collapse']):.4f}  fixed-coefficient "
          f"ceiling {np.mean(result['ceiling']):.4f}", flush=True)
    for K, lam_min, noise in runs:
        tag = f"K{K}_lam{lam_min:g}_noise{noise:g}"
        best = None
        for seed in range(a.seeds):
            m = LDM(W, R, K, seed, lam_min)
            opt = torch.optim.Adam(m.parameters(), lr=a.lr)
            gen = torch.Generator().manual_seed(1000 + seed)
            for step in range(a.steps):
                rows = m(key_noise=noise, gen=gen)
                loss = -sum((rows[i][: i + 1] @ bf[i]) ** 2 / (rows[i][: i + 1] @ Gf[i] @ rows[i][: i + 1]) / Nf
                            for i in range(R)) / R
                opt.zero_grad()
                loss.backward()
                opt.step()
                if step % 1000 == 0 or step == a.steps - 1:
                    print(f"  {tag} seed {seed} step {step:5d} fit-half explained {-loss.item():.4f}", flush=True)
            with torch.no_grad():
                score = held_score(m())
            if best is None or np.mean(score) > np.mean(best[0]):
                best = (score, m)
        score, m = best
        with torch.no_grad():
            prod = held_score(m(production=True))
            perturbed = held_score(m(key_noise=0.01, gen=torch.Generator().manual_seed(7)))
            lam = m.lam()
        entry = {"score": score, "production": prod, "key_noise_0.01": perturbed,
                 "lambda_prod": float(lam.prod()), "lambda_min_fit": float(lam.min()),
                 "gain_absmax": float(m.s.detach().abs().max()), "K": K, "lambda_min": lam_min, "key_noise": noise,
                 "params": {k: v.detach().tolist() for k, v in m.state_dict().items()}}
        result[tag] = entry
        print(f"{tag}: held-out {name} {np.mean(score):.4f} (min {min(score):.4f}); production {np.mean(prod):.4f}; "
              f"with 0.01 key noise {np.mean(perturbed):.4f}; prod(lambda) {entry['lambda_prod']:.2e}, "
              f"|gain| max {entry['gain_absmax']:.1f}", flush=True)
    if a.out:
        a.out.write_text(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

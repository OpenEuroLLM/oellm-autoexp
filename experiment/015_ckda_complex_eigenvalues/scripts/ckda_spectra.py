"""Per-step transition spectra of the CKDA layers of an exported ambitious model, on the fixed
validation sets, plus loss ablations of the signed / complex part.

THE TRANSITION. Each CKDA step applies, per head, M_t = (I - beta_t k_t k_t^T) Diag(alpha_t) to the
state: k_t L2-normalised (the kernel's qk l2norm), alpha_t in [-1, 1]^128 per channel (signed_sigmoid2,
|alpha| >= exp(lower_bound)), beta_t = 2 sigmoid(b_t) in [0, 2] per head (allow_neg_eigval). Its
characteristic function is

    det(M - lam I) = prod_i (alpha_i - lam) * f(lam),   f(lam) = 1 - beta sum_i alpha_i k_i^2 / (alpha_i - lam).

Between two consecutive alphas of the same sign f runs from +-inf to -+inf, so it has a real root there.
Hence at most ONE pair of eigenvalues can leave the real axis -- the two roots not bracketed by
same-sign poles -- and only if beta > 1 and alpha has both signs (beta <= 1: M is similar to the
symmetric P^1/2 D P^1/2 with P = I - beta k k^T >= 0; alpha of one sign: similar to a symmetric matrix
too). The pair is found exactly: bisection for the d - 2 bracketed real roots, then its sum from the
trace (sum_i alpha_i (1 - beta k_i^2)) and its product from the determinant ((1 - beta) prod_i alpha_i).
--self-check compares against torch.linalg.eigvals.

HOOK. FLA's `chunk_kda` as called by ComplexKimiDeltaAttention (fla.layers.complex_kda_layer) is wrapped,
so the analysis sees exactly the q / k / g / beta / sign the kernel gets; a forward pre-hook on each CKDA
module tells the wrapper which layer runs. The same wrapper applies the ablations:
    abs_alpha     alpha -> |alpha| everywhere (sign = None: the unsigned kernel path)
    beta_le1      beta -> min(beta, 1) everywhere (b -> min(b, 0))
    cplx_abs      alpha -> |alpha| only at the steps (t, head) whose pair is complex
    cplx_beta1    beta -> min(beta, 1) only at those steps
    ctrl_abs      alpha -> |alpha| at as many random mixed-sign REAL steps per layer (control for cplx_abs)
optionally restricted to a group of CKDA layers (--group-size: cplx_abs per group).

    python experiment/015_ckda_complex_eigenvalues/scripts/ckda_spectra.py --model-dir <hf export> \
        --valid-dir <v8ml-valid> --sets web_en,code,math,multilingual,reasoning,long_64k --docs-per-set 16 \
        --seq-len 8192 --gpu-mem-gib 54 --engram-tables-cpu --out <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------- the spectrum


def _bisect_chunk(a, w, b, valid_sign, iters):
    """Roots of f in every interval (a_i, a_{i+1}) of the sorted alphas a [n, d] (only meaningful where
    both ends have the same sign); valid_sign [n, d-1] = the sign of f just right of a_i."""
    lo, hi = a[:, :-1].clone(), a[:, 1:].clone()
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f = 1.0 - b * (w[:, None, :] / (a[:, None, :] - mid[:, :, None])).sum(-1)
        right = torch.nan_to_num(f * valid_sign, nan=1.0) > 0  # still on a_i's side: the root is right of mid
        lo = torch.where(right, mid, lo)
        hi = torch.where(right, hi, mid)
    return 0.5 * (lo + hi)


def free_pair(alpha: torch.Tensor, k2: torch.Tensor, beta: torch.Tensor, iters: int = 44, chunk: int = 1024):
    """alpha [n, d] (nonzero), k2 [n, d] (squared unit key), beta [n] -> (re, im) [n] of the eigenvalue of
    M = (I - beta k k^T) Diag(alpha) that is not bracketed by two same-sign alphas, with im >= 0 (im > 0:
    a complex pair re +- i im); for alpha of one sign there is one free root (returned as re, im = 0)."""
    out_re = torch.empty(alpha.shape[0], dtype=torch.float64, device=alpha.device)
    out_im = torch.empty_like(out_re)
    for s in range(0, alpha.shape[0], chunk):
        a, order = alpha[s : s + chunk].double().sort(dim=-1)
        kk = k2[s : s + chunk].double().gather(-1, order)
        b = beta[s : s + chunk].double()[:, None]
        w = a * kk
        lo, hi = a[:, :-1], a[:, 1:]
        valid = (lo > 0) == (hi > 0)
        vsign = torch.where(lo > 0, 1.0, -1.0).double()
        roots = _bisect_chunk(a, w, b, vsign, iters)
        tr = (a * (1.0 - b * kk)).sum(-1)
        n_valid = valid.sum(-1)
        ssum = tr - torch.where(valid, roots, 0.0).sum(-1)
        # product of the free pair = det / prod(bracketed roots), in logs (128 factors of >= e^-5)
        logdet = torch.log(torch.abs(1.0 - b[:, 0])) + a.abs().log().sum(-1)
        sgn = torch.sign(1.0 - b[:, 0]) * torch.where(a < 0, -1.0, 1.0).prod(-1)
        logr = torch.where(valid, roots.abs().log(), 0.0).sum(-1)
        sgn = sgn * torch.where(valid & (roots < 0), -1.0, 1.0).prod(-1)
        prod = sgn * torch.exp(logdet - logr)
        disc = ssum * ssum - 4.0 * prod
        two = n_valid == a.shape[1] - 2
        cplx = two & (disc < 0)
        re = torch.where(two, 0.5 * ssum, ssum)
        im = torch.where(cplx, 0.5 * torch.sqrt((-disc).clamp_min(0.0)), torch.zeros_like(ssum))
        out_re[s : s + chunk] = re
        out_im[s : s + chunk] = im
    return out_re, out_im


def self_check(n: int = 512, d: int = 128, device="cpu", seed: int = 0) -> dict:
    """free_pair vs torch.linalg.eigvals on random signed transitions (beta in (1, 2], mixed signs)."""
    g = torch.Generator().manual_seed(seed)
    mag = torch.exp(-5 * torch.rand(n, d, generator=g, dtype=torch.float64))
    sgn = torch.where(torch.rand(n, d, generator=g) < torch.rand(n, 1, generator=g), -1.0, 1.0).double()
    alpha = mag * sgn
    k = F.normalize(torch.randn(n, d, generator=g, dtype=torch.float64) * torch.rand(n, d, generator=g) ** 3, dim=-1)
    beta = 1 + torch.rand(n, generator=g, dtype=torch.float64)
    re, im = free_pair(alpha.to(device), (k * k).to(device), beta.to(device))
    re, im = re.cpu(), im.cpu()
    M = torch.diag_embed(alpha) - beta[:, None, None] * k[:, :, None] * (alpha * k)[:, None, :]
    ev = torch.linalg.eigvals(M)
    npairs = (ev.imag > 1e-9).sum(-1)
    top = torch.gather(ev, 1, ev.imag.argmax(-1, keepdim=True))[:, 0]
    has = im > 1e-9
    err = (torch.complex(re, im) - top).abs()[has & (npairs == 1)]
    return {"n": n, "max_pairs_eigvals": int(npairs.max()), "frac_complex_eigvals": float((npairs > 0).double().mean()),
            "frac_complex_ours": float(has.double().mean()), "disagree": int(((npairs > 0) != has).sum()),
            "max_abs_err": float(err.max()) if err.numel() else 0.0}


# --------------------------------------------------------------------------- statistics

# per-(t, head) metrics, summed into the accumulators (means = sum / count)
METRICS = ("count", "cplx", "cplx_r50", "imag", "theta", "radius", "beta", "beta_gt1", "neg_frac", "k2_neg",
           "mixed", "alpha_abs", "cand")
POS_METRICS = ("count", "cplx", "cplx_r50", "imag", "beta_gt1", "neg_frac", "k2_neg")
NBINS_RE, NBINS_IM = 80, 40


def log2_bin(pos: torch.Tensor) -> torch.Tensor:
    return torch.where(pos < 1, 0, torch.floor(torch.log2(pos.clamp_min(1).float())).long() + 1)


class Stats:
    def __init__(self, n_layers: int, n_heads: int, seq_len: int, vocab: int):
        self.L, self.H, self.T = n_layers, n_heads, seq_len
        self.nlb = int(math.log2(seq_len)) + 2
        z = lambda *s: np.zeros(s, dtype=np.float64)  # noqa: E731
        self.lh = {m: z(n_layers, n_heads) for m in METRICS}  # layer x head
        self.lhp = {m: z(n_layers, n_heads, self.nlb) for m in POS_METRICS}  # layer x head x log2(pos) bin
        self.lp = {m: z(n_layers, seq_len) for m in POS_METRICS}  # layer x position (heads summed)
        self.hist = z(n_layers, NBINS_RE, NBINS_IM)  # the complex pair's upper eigenvalue, re in [-1,1], im in [0,1]
        self.theta_hist = z(n_layers, 36)  # arg of the pair in (0, pi), weighted by count
        self.beta_hist = z(n_layers, 40)
        self.tok = {m: np.zeros((vocab,), dtype=np.float64) for m in ("count", "cplx", "imag")}  # heads x layers summed
        self.tok_layer_cplx = np.zeros((n_layers, vocab), dtype=np.float32)

    def add(self, layer: int, ids: torch.Tensor, per: dict[str, torch.Tensor]):
        """per[m]: [T, H] float64 on the GPU; ids [T]."""
        T = ids.shape[0]
        for m in METRICS:
            self.lh[m][layer] += per[m].sum(0).cpu().numpy()
        pos = torch.arange(T, device=per["count"].device)  # the layer's GPU (ids sit on the embedding's)
        lb = log2_bin(pos)
        for m in POS_METRICS:
            v = per[m]
            acc = torch.zeros(self.H, self.nlb, dtype=torch.float64, device=v.device)
            acc.index_add_(1, lb, v.T)
            self.lhp[m][layer] += acc.cpu().numpy()
            self.lp[m][layer, :T] += v.sum(1).cpu().numpy()
        c = per["cplx"] > 0
        if bool(c.any()):
            re, im = per["re"][c], per["imag"][c]
            h, _, _ = np.histogram2d(re.cpu().numpy(), im.cpu().numpy(), bins=(NBINS_RE, NBINS_IM),
                                     range=((-1, 1), (0, 1)))
            self.hist[layer] += h
            th = per["theta"][c].cpu().numpy()
            self.theta_hist[layer] += np.histogram(th, bins=36, range=(0, math.pi))[0]
        self.beta_hist[layer] += np.histogram(per["beta"].flatten().cpu().numpy(), bins=40, range=(0, 2))[0]
        idc = ids.cpu().numpy()
        for m, key in (("count", "count"), ("cplx", "cplx"), ("imag", "imag")):
            np.add.at(self.tok[m], idc, per[key].sum(1).cpu().numpy())
        np.add.at(self.tok_layer_cplx[layer], idc, per["cplx"].sum(1).float().cpu().numpy())

    def save(self, path: Path):
        arrs = {f"lh_{m}": v for m, v in self.lh.items()}
        arrs |= {f"lhp_{m}": v for m, v in self.lhp.items()}
        arrs |= {f"lp_{m}": v for m, v in self.lp.items()}
        arrs |= {f"tok_{m}": v for m, v in self.tok.items()}
        arrs |= {"hist": self.hist, "theta_hist": self.theta_hist, "beta_hist": self.beta_hist,
                 "tok_layer_cplx": self.tok_layer_cplx}
        np.savez_compressed(path, **arrs)


# --------------------------------------------------------------------------- the hook


class State:
    layer = -1  # CKDA index of the running layer
    stats: Stats | None = None
    ids: torch.Tensor | None = None
    ablate: str | None = None
    ablate_layers: set | None = None
    dump: dict | None = None  # per-layer [T, H] arrays of one document
    rng: torch.Generator | None = None
    n_calls = 0
    solver_s = 0.0
    verify = False  # compare a few steps per layer against torch.linalg.eigvals (first document)
    verify_log: list = []


def step_spectra(k, g, beta, sign, want_pair: bool = True):
    """The kernel's inputs [1, T, H, K] / [1, T, H] -> per-(t, head) metrics [T, H] and the complex mask."""
    T, H, K = k.shape[1:]
    with torch.no_grad():
        kh = F.normalize(k[0].double(), dim=-1, eps=1e-6)  # [T, H, K]
        k2 = kh * kh
        mag = g[0].double().exp()
        neg = sign[0] < 0 if sign is not None else torch.zeros_like(mag, dtype=torch.bool)
        alpha = torch.where(neg, -mag, mag)
        bt = 2.0 * torch.sigmoid(beta[0].double())  # [T, H]
        nneg = neg.sum(-1)
        mixed = (nneg > 0) & (nneg < K)
        cand = mixed & (bt > 1.0)
        re = torch.zeros(T, H, dtype=torch.float64, device=k.device)
        im = torch.zeros_like(re)
        if want_pair and bool(cand.any()):
            t0 = time.time()
            idx = cand.flatten().nonzero()[:, 0]
            r, i = free_pair(alpha.reshape(T * H, K)[idx], k2.reshape(T * H, K)[idx], bt.flatten()[idx])
            re.view(-1)[idx] = r
            im.view(-1)[idx] = i
            State.solver_s += time.time() - t0
        cplx = im > 1e-9
        radius = torch.where(cplx, torch.sqrt(re * re + im * im), 0.0)
        per = {
            "count": torch.ones_like(re), "cplx": cplx.double(), "cplx_r50": (cplx & (radius > 0.5)).double(),
            "imag": im, "re": re, "theta": torch.where(cplx, torch.atan2(im, re), 0.0), "radius": radius,
            "beta": bt, "beta_gt1": (bt > 1).double(), "neg_frac": neg.double().mean(-1),
            "k2_neg": torch.where(neg, k2, 0.0).sum(-1), "mixed": mixed.double(), "alpha_abs": mag.mean(-1),
            "cand": cand.double(),
        }
    return per, cplx, mixed


def verify_steps(L, k, g, sign, beta, per, n: int = 6):
    """The pair of a few real candidate steps of this layer vs torch.linalg.eigvals of the full M."""
    cand = per["cand"].flatten().nonzero()[:, 0]
    if not cand.numel():
        return
    H = k.shape[2]
    for i in cand[torch.randperm(cand.numel(), generator=State.rng)[:n].to(cand.device)].tolist():
        t, h = divmod(i, H)
        kh = F.normalize(k[0, t, h].double(), dim=-1, eps=1e-6).cpu()
        alpha = (g[0, t, h].double().exp() * torch.where(sign[0, t, h] < 0, -1.0, 1.0).double()).cpu()
        b = float(2.0 * torch.sigmoid(beta[0, t, h].double()))
        M = torch.diag(alpha) - b * kh[:, None] * (alpha * kh)[None, :]
        ev = torch.linalg.eigvals(M)
        top = ev[ev.imag.argmax()]
        ours = complex(float(per["re"][t, h]), float(per["imag"][t, h]))
        State.verify_log.append({"layer": L, "t": t, "head": h, "beta": b, "pairs_eigvals": int((ev.imag > 1e-9).sum()),
                                 "ours": [ours.real, ours.imag], "eigvals": [float(top.real), float(top.imag)],
                                 "err": abs(ours - complex(top)) if ours.imag > 1e-9 else None})


def wrap_chunk_kda(mod):
    orig = mod.chunk_kda

    def chunk_kda(*args, **kw):
        q, k, v = kw["q"], kw["k"], kw["v"]
        g, beta, sign = kw["g"], kw["beta"], kw.get("sign")
        L = State.layer
        State.n_calls += 1
        active = State.ablate is not None and (State.ablate_layers is None or L in State.ablate_layers)
        need = State.stats is not None or (active and State.ablate in ("cplx_abs", "cplx_beta1", "ctrl_abs"))
        if need and q.shape[0] == 1 and kw.get("cu_seqlens") is None:
            per, cplx, mixed = step_spectra(k, g, beta, sign)
            if State.stats is not None:
                State.stats.add(L, State.ids[: k.shape[1]], per)
            if State.verify:
                verify_steps(L, k, g, sign, beta, per)
            if State.dump is not None:
                State.dump[L] = {m: per[m].float().cpu().numpy().astype(np.float16)
                                 for m in ("cplx", "radius", "theta", "beta", "neg_frac", "k2_neg", "alpha_abs")}
        if active:
            if State.ablate == "abs_alpha":
                kw["sign"] = None
            elif State.ablate == "beta_le1":
                kw["beta"] = beta.clamp(max=0)
            elif State.ablate in ("cplx_abs", "ctrl_abs"):
                mask = cplx
                if State.ablate == "ctrl_abs":  # as many random mixed-sign real steps
                    pool = (mixed & ~cplx).flatten().nonzero()[:, 0]
                    n = min(int(cplx.sum()), pool.numel())
                    mask = torch.zeros_like(cplx)
                    if n:
                        pick = pool[torch.randperm(pool.numel(), generator=State.rng, device="cpu")[:n].to(pool.device)]
                        mask.view(-1)[pick] = True
                kw["sign"] = torch.where(mask[None, :, :, None], torch.ones_like(sign), sign)
            elif State.ablate == "cplx_beta1":
                kw["beta"] = torch.where(cplx[None], beta.clamp(max=0), beta)
        return orig(*args, **kw)

    mod.chunk_kda = chunk_kda


def install(model) -> list[int]:
    """Wrap the CKDA layers' chunk_kda; returns the HF layer indices of the CKDA layers in order."""
    import importlib

    ckda = [i for i, t in enumerate(model.config.block_types) if t == "ckda"]
    mod = importlib.import_module(type(model.model.layers[ckda[0]].mixer.layer).__module__)
    wrap_chunk_kda(mod)
    for j, i in enumerate(ckda):
        def pre(_m, _a, j=j):
            State.layer = j
        model.model.layers[i].mixer.layer.register_forward_pre_hook(pre)
    return ckda


# --------------------------------------------------------------------------- data + loss


def read_index(prefix: str):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from oellm_autoexp.hf_export.score_on_flag import read_index as ri

    return ri(prefix)


def sample_docs(prefix: str, n: int, min_len: int, seed: int) -> list[tuple[int, np.ndarray]]:
    dtype, lens, ptrs = read_index(prefix)
    ok = np.nonzero(lens >= min_len)[0]
    rng = random.Random(seed)
    pick = sorted(rng.sample(list(ok), min(n, len(ok))))
    data = np.memmap(prefix + ".bin", dtype=dtype, mode="r")
    return [(int(j), np.asarray(data[ptrs[j] // dtype.itemsize : ptrs[j] // dtype.itemsize + lens[j]], dtype=np.int64))
            for j in pick]


def token_nll(model, ids: torch.Tensor, chunk: int = 1024) -> np.ndarray:
    hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[0, :-1]
    head = model.lm_head
    dev = head.weight.device
    hidden, target = hidden.to(dev), ids[0, 1:].to(dev)
    out = []
    for i in range(0, hidden.shape[0], chunk):
        logits = head(hidden[i : i + chunk]).float()
        out.append(F.cross_entropy(logits, target[i : i + chunk], reduction="none"))
    return torch.cat(out).cpu().numpy()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--valid-dir", type=Path)
    p.add_argument("--sets", default="web_en,code,math,multilingual,reasoning,long_64k")
    p.add_argument("--docs-per-set", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--min-len", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dump-docs", type=int, default=1, help="documents per set with per-token arrays saved")
    p.add_argument("--ablations", default="abs_alpha,beta_le1,cplx_abs,cplx_beta1,ctrl_abs")
    p.add_argument("--group-size", type=int, default=7, help="cplx_abs per group of CKDA layers (0: off)")
    p.add_argument("--gpu-mem-gib", type=float, default=None)
    p.add_argument("--engram-tables-cpu", action="store_true")
    p.add_argument("--self-check", action="store_true", help="only the solver check (no model)")
    p.add_argument("--no-stats", action="store_true", help="baseline loss + ablations only (no statistics pass)")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    chk = self_check(device=dev)
    print("self-check:", chk, flush=True)
    (a.out / "self_check.json").write_text(json.dumps(chk, indent=1))
    if chk["disagree"] > 2 or chk["max_abs_err"] > 1e-6 or chk["max_pairs_eigvals"] > 1:
        raise SystemExit("solver self-check failed")
    if a.self_check:
        return

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from transformers import AutoConfig, AutoModelForCausalLM

    from oellm_autoexp.hf_export.score_on_flag import layer_device_map, load_engram_tables_cpu

    cfg = AutoConfig.from_pretrained(a.model_dir, trust_remote_code=True)
    t0 = time.time()
    dmap = layer_device_map(cfg, a.gpu_mem_gib, a.engram_tables_cpu) if a.gpu_mem_gib else "auto"
    if a.engram_tables_cpu:
        model = load_engram_tables_cpu(a.model_dir, dmap).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(a.model_dir, trust_remote_code=True, dtype=torch.bfloat16,
                                                     device_map=dmap).eval()
    print(f"loaded {a.model_dir.name} in {time.time() - t0:.0f}s", flush=True)
    ckda = install(model)
    H = cfg.fla_layer["kwargs"]["num_heads"]
    bos = cfg.bos_token_id
    edev = next(model.parameters()).device
    meta = {"model": str(a.model_dir), "ckda_hf_layers": ckda, "block_types": cfg.block_types, "heads": H,
            "seq_len": a.seq_len, "sets": {}, "args": {k: str(v) for k, v in vars(a).items()}}

    docs = {}
    for s in a.sets.split(","):
        docs[s] = sample_docs(str(a.valid_dir / f"{s}_text_document"), a.docs_per_set, a.min_len, a.seed)
        meta["sets"][s] = [{"doc": j, "len": int(min(len(d) + 1, a.seq_len))} for j, d in docs[s]]

    def ids_of(d):
        return torch.tensor(np.concatenate([[bos], d[: a.seq_len - 1]]), device=edev).unsqueeze(0)

    # ---- pass 0: statistics + baseline loss
    State.rng = torch.Generator().manual_seed(a.seed)
    nll = {}
    for s, items in docs.items():
        State.stats = None if a.no_stats else Stats(len(ckda), H, a.seq_len, cfg.vocab_size)
        nll[s] = []
        for n, (j, d) in enumerate(items):
            ids = ids_of(d)
            State.ids = ids[0].cpu()  # host copy: the CKDA layers run on other GPUs than the embedding
            State.dump = {} if n < a.dump_docs and not a.no_stats else None
            State.verify = n == 0 and not a.no_stats
            t1 = time.time()
            with torch.no_grad():
                nll[s].append(token_nll(model, ids))
            if State.dump is not None:
                np.savez_compressed(a.out / f"dump_{s}_{n}.npz", ids=ids[0].cpu().numpy(), nll=nll[s][-1],
                                    **{f"{m}": np.stack([State.dump[L][m] for L in range(len(ckda))])
                                       for m in State.dump[0]})
            print(f"[stats] {s} doc {n} ({ids.shape[1]} tok) nll {nll[s][-1].mean():.4f} "
                  f"{time.time() - t1:.1f}s solver {State.solver_s:.0f}s", flush=True)
        if State.stats is not None:
            State.stats.save(a.out / f"stats_{s}.npz")
    State.stats, State.dump, State.verify = None, None, False
    errs = [v["err"] for v in State.verify_log if v["err"] is not None]
    flips = sum((v["pairs_eigvals"] > 0) != (v["ours"][1] > 1e-9) for v in State.verify_log)
    print(f"verify: {len(State.verify_log)} steps, existence disagreements {flips}, max err "
          f"{max(errs) if errs else 0:.2e}", flush=True)
    (a.out / "verify.json").write_text(json.dumps(State.verify_log, indent=1))
    np.savez_compressed(a.out / "nll_base.npz", **{f"{s}_{n}": v for s in nll for n, v in enumerate(nll[s])})
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1))

    # ---- ablations
    runs = [(ab, None) for ab in a.ablations.split(",") if ab]
    if a.group_size:
        runs += [("cplx_abs", set(range(g, min(g + a.group_size, len(ckda)))))
                 for g in range(0, len(ckda), a.group_size)]
    summary = {"base": {s: float(np.concatenate(v).mean()) for s, v in nll.items()}}
    for ab, layers in runs:
        tag = ab if layers is None else f"{ab}_L{min(layers)}-{max(layers)}"
        State.ablate, State.ablate_layers = ab, layers
        State.rng = torch.Generator().manual_seed(a.seed)
        res = {}
        t1 = time.time()
        for s, items in docs.items():
            res[s] = []
            for j, d in items:
                with torch.no_grad():
                    res[s].append(token_nll(model, ids_of(d)))
        np.savez_compressed(a.out / f"nll_{tag}.npz", **{f"{s}_{n}": v for s in res for n, v in enumerate(res[s])})
        summary[tag] = {s: float(np.concatenate(v).mean()) for s, v in res.items()}
        print(f"[ablate] {tag}: " + " ".join(f"{s} {summary[tag][s] - summary['base'][s]:+.4f}" for s in res)
              + f" ({time.time() - t1:.0f}s)", flush=True)
        (a.out / "ablations.json").write_text(json.dumps(summary, indent=1))
    State.ablate = None
    print("done", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""The step-0 loss of an mHC-4 -> LDM switch on one node: the HF export as trained (mHC), then the same loaded model
converted in place to Linear Depth Memory with the fitted fixed addresses of ldm_imitation_init.py.

Conversion (the export's own LDM classes, the model's weights untouched): every layer gets an LDM mixer (its write
and the following read's query), the model the boundary (the embedding write, the first query); the residual
becomes x + r. A site's address network is set to give the fitted constant key / query / gate for every token:
w1 at its init (std --init-std), the address and gate heads at --head-std (tiny, nonzero: MuOwn needs a row
direction) with the fitted values in their biases, the forgetting logit and read gain as fitted, all groups alike
apart from the read side's symmetry breaking (variant suffixes):

  +q<e>     every group's query + e * N(0, 1 / K) (the read is linear: harmless; keys stay exact)
  +dup      K -> 2K, every slot twice (keys and queries [k, k] / sqrt 2: function-preserving)

engram_branch_gates (the bg_learned fork): without mHC every stream's gate query is the layer input, and the layer
writes the mean of its n gated outputs (the averaged gates).

Variants (--variants): mhc (as trained), collapse (every read gain 0: PR #82's conversion before training), and
<run tag>[+dup][+q<noise>] from the --init json (e.g. K8_lam0.9_noise0.02+dup+q0.05).

    python ldm_imitation_eval.py --model-dir <export> --init <ldm_init.json> --out <json> \\
        [--variants mhc,collapse,K8_lam0.9_noise0.02,K8_lam0.9_noise0.02+dup+q0.05] [--docs 8] [--seq-len 4096]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

SETS = ("all", "web_en", "multilingual", "code", "math", "reasoning", "long_64k")


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


class Sites:
    """The LDM modules attached to a loaded mHC model, and the switch between mHC and LDM."""

    def __init__(self, model, hidden_dim: int, groups: int, alpha: float):
        self.model, self.cfg = model, model.config
        self.mod = sys.modules[type(model.model.layers[0]).__module__]
        self.layers = list(model.model.layers)
        self.hidden_dim, self.groups, self.alpha = hidden_dim, groups, alpha
        self.K = None
        self.n = self.cfg.mhc_expansion_rate
        self.branch = [l for l in self.layers if l.kind == "engram" and getattr(l.mixer, "branch_gates", False)]
        for layer in self.branch:   # averaged gates: one query (the input) for every stream, the mean output
            orig = layer.mixer.forward

            def forward(hidden_states, input_ids, token_mask=None, state=None, streams=None, _orig=orig,
                        _layer=layer):
                if streams is not None or not _layer.ldm:
                    return _orig(hidden_states, input_ids, token_mask, state, streams=streams)
                q = hidden_states.unsqueeze(-2).expand(*hidden_states.shape[:-1], self.n, hidden_states.shape[-1])
                outs, new_state = _orig(hidden_states, input_ids, token_mask, state, streams=q)
                return outs.mean(dim=-2), new_state

            layer.mixer.forward = forward

    def build(self, K: int) -> None:
        """(Re)create the LDM modules for key width K on the layers' devices (bf16 like the model)."""
        cfg = self.cfg
        cfg.ldm_groups, cfg.ldm_key_dim, cfg.ldm_hidden_dim, cfg.ldm_alpha = self.groups, K, self.hidden_dim, self.alpha
        dtype = next(self.model.parameters()).dtype
        for layer in self.layers:
            dev = next(layer.parameters()).device
            layer.ldm_mixer = self.mod.OellmLDMMixer(cfg).to(dev, dtype)
        dev0 = self.model.model.embed_tokens.weight.device
        self.boundary = self.mod.OellmLDMBoundary(cfg).to(dev0, dtype)
        self.K = K

    def use(self, ldm: bool) -> None:
        for layer in self.layers:
            layer.ldm = ldm
        self.model.model.ldm_boundary = self.boundary if ldm else None

    def write_sites(self):
        return [self.boundary.embed_write] + [l.ldm_mixer.write_site for l in self.layers]

    def read_sites(self):
        return [self.boundary.first_query] + [l.ldm_mixer.next_query for l in self.layers]


@torch.no_grad()
def set_net(net, address: torch.Tensor, gate_bias: torch.Tensor, init_std: float, head_std: float, gen) -> None:
    """Address network giving ``address`` [G, K] (normalized per group) and gate 2 sigmoid(gate_bias) [G]."""
    dev = net.w1.weight.device
    rnd = lambda shape, std: (torch.randn(shape, generator=gen) * std).to(dev, net.w1.weight.dtype)  # noqa: E731
    net.norm_weight.fill_(1.0)
    net.w1.weight.copy_(rnd(net.w1.weight.shape, init_std))
    net.w1.bias.zero_()
    net.address.weight.copy_(rnd(net.address.weight.shape, head_std))
    net.address.bias.copy_(address.reshape(-1).to(dev, net.address.bias.dtype))
    net.gate.weight.copy_(rnd(net.gate.weight.shape, head_std))
    net.gate.bias.copy_(gate_bias.to(dev, net.gate.bias.dtype))


@torch.no_grad()
def configure(sites: Sites, params: dict | None, duplicate: bool, query_noise: float, init_std: float,
              head_std: float, seed: int) -> None:
    """Set every site from the fitted parameters (None: collapse, every read gain 0)."""
    gen = torch.Generator().manual_seed(seed)
    G = sites.groups
    writes, reads = sites.write_sites(), sites.read_sites()
    if params is None:
        K = sites.K
        for site in writes:
            set_net(site.net, torch.ones(G, K) / math.sqrt(K), torch.zeros(G), init_std, head_std, gen)
            site.forget_logit.fill_(logit(0.99))
        for site in reads:
            set_net(site.net, torch.ones(G, K) / math.sqrt(K), torch.zeros(G), init_std, head_std, gen)
            site.read_gain.zero_()
        return
    key = torch.nn.functional.normalize(torch.tensor(params["key"]), dim=-1)
    q = torch.nn.functional.normalize(torch.tensor(params["q"]), dim=-1)
    w = 2 * torch.sigmoid(torch.tensor(params["w"]))
    lam = params["lambda_min"] + (1 - params["lambda_min"]) * torch.sigmoid(torch.tensor(params["f"]))
    s = torch.tensor(params["s"])
    if duplicate:
        key = torch.cat([key, key], -1) / math.sqrt(2)
        q = torch.cat([q, q], -1) / math.sqrt(2)
    K = key.shape[-1]
    if len(writes) != key.shape[0]:
        raise SystemExit(f"{key.shape[0]} fitted writes for {len(writes)} sites")
    for t, site in enumerate(writes):
        set_net(site.net, key[t].expand(G, K), torch.full((G,), logit(float(w[t]) / 2)), init_std, head_std, gen)
        site.forget_logit.fill_(logit(float(lam[t])))
    for t, site in enumerate(reads):
        if t >= q.shape[0]:            # no fitted final read: the LM head reads x
            set_net(site.net, torch.ones(G, K) / math.sqrt(K), torch.zeros(G), init_std, head_std, gen)
            site.read_gain.zero_()
            continue
        qq = q[t].expand(G, K) + query_noise * torch.randn(G, K, generator=gen) / math.sqrt(K)
        set_net(site.net, torch.nn.functional.normalize(qq, dim=-1), torch.zeros(G), init_std, head_std, gen)
        site.read_gain.fill_(float(s[t]))        # gate 2 sigmoid(0) = 1: gamma * gate = s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--init", type=Path, required=True, help="ldm_imitation_init.py --out json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--variants", default="mhc,collapse")
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--docs", type=int, default=8, help="documents per validation set")
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--min-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=4321, help="document seed (the dump used 1234: other documents)")
    ap.add_argument("--hidden-dim", type=int, default=128, help="depth_memory_hidden_dim")
    ap.add_argument("--groups", type=int, default=16)
    ap.add_argument("--init-std", type=float, default=0.02)
    # the keys must stay constant per token: at 1e-4 the token-dependent part is ~0.2 % of the key, where a fitted
    # key's perturbation already moves the read coefficients (ldm_imitation_init: 0.001 key noise -> O(0.1))
    ap.add_argument("--head-std", type=float, default=1e-6)
    ap.add_argument("--gpu-mem-gib", type=float, default=50)
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "scripts" / "rluukkon"))
    from engram_gate_diag import token_nll, valid_docs
    from transformers import AutoConfig, AutoTokenizer

    from oellm_autoexp.hf_export.score_on_flag import layer_device_map, load_engram_tables_cpu

    t0 = time.time()
    init = json.loads(a.init.read_text())
    cfg = AutoConfig.from_pretrained(a.model_dir, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    bos = tok.bos_token_id
    model = load_engram_tables_cpu(a.model_dir, layer_device_map(cfg, a.gpu_mem_gib, True)).eval()
    dev = next(model.parameters()).device
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)
    docs = [(name, d) for name in a.sets.split(",") if name
            for d in valid_docs(name, a.docs, a.seq_len, a.min_len, a.seed)]
    sites = Sites(model, a.hidden_dim, a.groups, alpha=1.0)

    def evaluate(label):
        per = []
        for name, d in docs:
            ids = torch.tensor(np.concatenate([[bos], d]), device=dev).unsqueeze(0)
            with torch.no_grad():
                per.append((name, float(token_nll(model, ids).mean())))
        sets = {s: float(np.mean([v for n, v in per if n == s])) for s in dict.fromkeys(n for n, _ in per)}
        mean = float(np.mean([v for _, v in per]))
        print(f"[{time.time() - t0:6.0f}s] {label:42s} nll {mean:.4f}  " +
              " ".join(f"{k}={v:.3f}" for k, v in sets.items()), flush=True)
        return {"nll": mean, "sets": sets, "docs": [v for _, v in per]}

    results = {"model_dir": str(a.model_dir), "init": str(a.init), "docs": len(docs), "variants": {}}
    for variant in a.variants.split(","):
        if variant == "mhc":
            if sites.K is not None:
                sites.use(False)
            results["variants"][variant] = evaluate(variant)
            continue
        parts = variant.split("+")
        dup = "dup" in parts
        noise = next((float(p[1:]) for p in parts if p.startswith("q")), 0.0)
        params = None
        K = 8
        if parts[0] != "collapse":
            entry = init[parts[0]]
            params = dict(entry["params"], lambda_min=entry.get("lambda_min", 0.0))
            K = entry["K"] * (2 if dup else 1)
        if sites.K != K:
            sites.build(K)
        configure(sites, params, dup, noise, a.init_std, a.head_std, seed=0)
        sites.use(True)
        results["variants"][variant] = evaluate(variant)
        sites.use(False)
    base = results["variants"].get("mhc", {}).get("nll")
    if base is not None:
        for v, r in results["variants"].items():
            r["delta_vs_mhc"] = r["nll"] - base
        print("delta vs mhc:", {v: round(r["delta_vs_mhc"], 4) for v, r in results["variants"].items()}, flush=True)
    a.out.write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

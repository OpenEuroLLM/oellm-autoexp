"""Tables + figures from a ckda_spectra.py output directory.

    python experiment/015_ckda_complex_eigenvalues/scripts/summarize.py <out dir> [--tokenizer <hf dir>]

Writes <out dir>/summary.json (per layer / head / set / position bin / token numbers, ablations) and
<out dir>/fig_*.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def ratio(a, b):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(b > 0, a / b, np.nan)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--tokenizer", type=Path, default=None)
    p.add_argument("--top-tokens", type=int, default=40)
    p.add_argument("--min-token-count", type=int, default=20)
    a = p.parse_args()
    meta = json.loads((a.out / "meta.json").read_text())
    sets = list(meta["sets"])
    st = {s: dict(np.load(a.out / f"stats_{s}.npz")) for s in sets}
    tot = {k: sum(st[s][k] for s in sets) for k in st[sets[0]]}
    L, H = tot["lh_count"].shape
    blk = [i for i in range(len(meta["block_types"])) if meta["block_types"][i] == "ckda"]
    # the Megatron block (mixer index) of each CKDA layer: count non-MoE symbols before it
    mixers = [i for i, t in enumerate(meta["block_types"]) if t in ("attention", "ckda")]
    block_of = [mixers.index(h) for h in blk]
    S = {"meta": {"model": meta["model"], "layers": L, "heads": H, "sets": {s: len(v) for s, v in meta["sets"].items()},
                  "tokens": {s: int(st[s]["lh_count"][0].sum() / 1) // H for s in sets}, "ckda_block": block_of}}

    def per_layer(d):
        c = d["lh_count"].sum(1)
        row = {m: ratio(d[f"lh_{m}"].sum(1), c) for m in
               ("cplx", "cplx_r50", "beta_gt1", "mixed", "cand", "neg_frac", "k2_neg", "alpha_abs", "beta", "imag")}
        row["radius_c"] = ratio(d["lh_radius"].sum(1), d["lh_cplx"].sum(1))
        row["theta_c"] = ratio(d["lh_theta"].sum(1), d["lh_cplx"].sum(1))
        row["imag_c"] = ratio(d["lh_imag"].sum(1), d["lh_cplx"].sum(1))
        return {k: [float(x) for x in v] for k, v in row.items()}

    S["layer"] = per_layer(tot)
    S["layer_by_set"] = {s: per_layer(st[s]) for s in sets}
    S["overall"] = {m: float(tot[f"lh_{m}"].sum() / tot["lh_count"].sum()) for m in
                    ("cplx", "cplx_r50", "beta_gt1", "mixed", "cand", "neg_frac", "k2_neg", "alpha_abs", "beta")}
    S["overall"]["radius_c"] = float(tot["lh_radius"].sum() / max(tot["lh_cplx"].sum(), 1))
    S["overall"]["theta_c"] = float(tot["lh_theta"].sum() / max(tot["lh_cplx"].sum(), 1))
    S["overall_by_set"] = {s: {m: float(st[s][f"lh_{m}"].sum() / st[s]["lh_count"].sum())
                               for m in ("cplx", "cplx_r50", "beta_gt1", "neg_frac", "mixed")} for s in sets}
    S["layer_head_cplx"] = ratio(tot["lh_cplx"], tot["lh_count"]).tolist()
    S["layer_head_cplx_r50"] = ratio(tot["lh_cplx_r50"], tot["lh_count"]).tolist()
    S["layer_head_neg_frac"] = ratio(tot["lh_neg_frac"], tot["lh_count"]).tolist()
    S["layer_head_beta_gt1"] = ratio(tot["lh_beta_gt1"], tot["lh_count"]).tolist()
    # positions: log2 bins (bin 0 = position 0 = BOS, bin b = [2^(b-1), 2^b))
    S["posbin_cplx"] = ratio(tot["lhp_cplx"].sum(1), tot["lhp_count"].sum(1)).tolist()  # [L, bins]
    S["posbin_cplx_r50"] = ratio(tot["lhp_cplx_r50"].sum(1), tot["lhp_count"].sum(1)).tolist()
    S["posbin_count"] = (tot["lhp_count"].sum(1)[0] / H).tolist()
    # tokens
    tc, tx = tot["tok_count"], tot["tok_cplx"]
    ok = tc >= a.min_token_count * L * H
    frac = ratio(tx, tc)
    order = np.argsort(-np.where(ok, frac, -1))
    tok = None
    if a.tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(a.tokenizer)

    def name(i):
        return tok.decode([int(i)]) if tok else str(int(i))

    S["tokens_top"] = [{"id": int(i), "tok": name(i), "n": int(tc[i] / (L * H)), "cplx": float(frac[i])}
                       for i in order[: a.top_tokens] if ok[i]]
    S["tokens_bottom"] = [{"id": int(i), "tok": name(i), "n": int(tc[i] / (L * H)), "cplx": float(frac[i])}
                          for i in order[::-1] if ok[i]][: a.top_tokens]
    S["tokens_frequent"] = [{"id": int(i), "tok": name(i), "n": int(tc[i] / (L * H)), "cplx": float(frac[i])}
                            for i in np.argsort(-tc)[: a.top_tokens]]
    # ablations
    abl = a.out / "ablations.json"
    if abl.exists():
        A = json.loads(abl.read_text())
        S["ablations"] = {k: {s: v[s] - A["base"][s] for s in v} for k, v in A.items() if k != "base"}
        S["base_nll"] = A["base"]
    vf = a.out / "verify.json"
    if vf.exists():
        V = json.loads(vf.read_text())
        errs = [v["err"] for v in V if v["err"] is not None]
        S["verify"] = {"steps": len(V), "disagree": sum((v["pairs_eigvals"] > 0) != (v["ours"][1] > 1e-9) for v in V),
                       "max_err": max(errs) if errs else 0.0}
    (a.out / "summary.json").write_text(json.dumps(S, indent=1))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(L)
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    ax[0].plot(x, S["layer"]["cplx"], "o-", label="complex pair")
    ax[0].plot(x, S["layer"]["cplx_r50"], "s-", label="complex, |lam| > 0.5")
    ax[0].plot(x, S["layer"]["cand"], "--", label="beta > 1 & mixed signs")
    ax[0].set_ylabel("fraction of (token, head)")
    ax[0].legend()
    ax[1].plot(x, S["layer"]["neg_frac"], "o-", label="alpha < 0 (channels)")
    ax[1].plot(x, S["layer"]["k2_neg"], "s-", label="key mass on alpha < 0")
    ax[1].plot(x, S["layer"]["beta_gt1"], "^-", label="beta > 1")
    ax[1].legend()
    ax[2].plot(x, S["layer"]["radius_c"], "o-", label="|lam| of the pair")
    ax[2].plot(x, np.array(S["layer"]["theta_c"]) / np.pi, "s-", label="arg / pi")
    ax[2].set_xlabel("CKDA layer")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(a.out / "fig_layers.png", dpi=120)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    for k, (key, ttl) in enumerate((("layer_head_cplx", "complex fraction"), ("layer_head_neg_frac", "alpha<0 fraction"))):
        im = ax[k].imshow(np.array(S[key]).T, aspect="auto", origin="lower", cmap="viridis")
        ax[k].set_xlabel("CKDA layer")
        ax[k].set_ylabel("head")
        ax[k].set_title(ttl)
        fig.colorbar(im, ax=ax[k])
    fig.tight_layout()
    fig.savefig(a.out / "fig_layer_head.png", dpi=120)
    fig, ax = plt.subplots(figsize=(10, 5))
    pb = np.array(S["posbin_cplx"])
    for g in range(0, L, 7):
        ax.plot(np.nanmean(pb[g : g + 7], 0), "o-", label=f"layers {g}-{min(g + 6, L - 1)}")
    ax.set_xlabel("log2 position bin (0 = BOS)")
    ax.set_ylabel("complex fraction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(a.out / "fig_positions.png", dpi=120)
    fig, axs = plt.subplots(6, 7, figsize=(21, 14))
    for l in range(L):
        axx = axs.flat[l]
        h = tot["hist"][l]
        axx.imshow(np.log1p(h.T), origin="lower", extent=(-1, 1, 0, 1), aspect="auto", cmap="magma")
        axx.set_title(f"L{l}: {S['layer']['cplx'][l]:.3f}", fontsize=8)
        axx.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(a.out / "fig_complex_plane.png", dpi=100)
    print(json.dumps(S["overall"], indent=1))


if __name__ == "__main__":
    main()

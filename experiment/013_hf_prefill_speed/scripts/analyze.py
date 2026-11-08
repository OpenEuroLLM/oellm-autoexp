"""Tables of an MN5 run: full-model speed per case / workload (+ speed-up over hf008) and the
logprob parity of fast and hf008 against each other and against the reference subset; the reduced
model's per-kind layer estimates.

    python analyze.py results/mn5_run1 > results/mn5_run1/summary.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


def parity(a, b):
    """Two lists of [b, s-1] logprob tensors -> mean NLL of each, delta, mean / max |dlogp|."""
    a, b = torch.cat([x.flatten() for x in a]), torch.cat([x.flatten() for x in b])
    n = min(a.numel(), b.numel())
    a, b = a[:n].double(), b[:n].double()
    d = (a - b).abs()
    return {"tokens": n, "nll_a": -a.mean().item(), "nll_b": -b.mean().item(), "nll_delta": (b - a).mean().item() * -1,
            "mean_abs_dlogp": d.mean().item(), "max_abs_dlogp": d.max().item()}


def main():
    run = Path(sys.argv[1])
    full = run / "full"
    res = {v: json.loads((full / f"{v}.json").read_text()) for v in ("fast", "hf008") if (full / f"{v}.json").exists()}
    print(f"# {run.name}\n")
    if res:
        print("## Full model (v4_2ph fp8_6000, 4 x H100)\n")
        for v, r in res.items():
            print(f"- {v}: load {r['load_seconds']:.0f} s, GiB per GPU {r['gib_per_gpu']}")
        base = res.get("hf008", {}).get("cases", {}).get("hf008", {})
        print("\n| case | work | tokens | seconds | tok/s | vs hf008 |\n|---|---|---:|---:|---:|---:|")
        for v, r in res.items():
            for case, works in r["cases"].items():
                for w, x in works.items():
                    sp = f"{base[w]['seconds'] / x['seconds']:.2f}x" if w in base and case != "reference" else ""
                    print(f"| {case} | {w} | {x['tokens']} | {x['seconds']:.3f} | {x['tok_s']:.0f} | {sp} |")
        print("\n## Logprob parity\n\n| a | b | work | tokens | NLL a | NLL b | NLL b - a | mean abs dlogp | max abs dlogp |\n|---|---|---|---:|---:|---:|---:|---:|---:|")
        pairs = [("hf008", "fast"), ("reference", "fast"), ("reference", "hf008")]
        for a, b in pairs:
            for w in ("w1", "w2", "w3", "w4"):
                fa, fb = full / f"{a}.{w}.pt", full / f"{b}.{w}.pt"
                if fa.exists() and fb.exists():
                    p = parity(torch.load(fa), torch.load(fb))
                    p["nll_delta"] = p["nll_b"] - p["nll_a"]
                    print(f"| {a} | {b} | {w} | {p['tokens']} | {p['nll_a']:.5f} | {p['nll_b']:.5f} | {p['nll_delta']:+.5f} | "
                          f"{p['mean_abs_dlogp']:.4f} | {p['max_abs_dlogp']:.3f} |")
    for name in ("layers_hf008", "layers_fast"):
        f = run / f"{name}.json"
        if not f.exists():
            continue
        print(f"\n## Reduced model, full-model estimate per kind: {name} (ms)\n")
        rows = json.loads(f.read_text())["results"]
        keys = list(rows[0]["full_model_breakdown_ms"])
        print("| shape | est. ms | est. tok/s | " + " | ".join(keys) + " |\n|---|---:|---:|" + "---:|" * len(keys))
        for r in rows:
            print(f"| {r['shape']} | {r['full_model_ms_est']:.0f} | {r['full_model_tok_s_est']:.0f} | "
                  + " | ".join(f"{r['full_model_breakdown_ms'][k]:.1f}" for k in keys) + " |")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Per-position NLL on long held-out documents with the HF export as it is (its sparse attention,
config.sparse_attention): chunked prefill through the model's cache, the NLL of every next token, averaged in position
bins (0-4k, 4k-16k, 16k-32k, 32k-64k, ...). The documents are those long_attention_probe.py's NLL task reads (the
first --docs-n of a seeded order of documents >= the length, BOS prepended).

    nll_by_position.py --model <hf export> --docs <prefix of documents >= L> --out <json> [--lengths 65536] [--docs-n 8]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--docs", required=True, help="indexed dataset prefix (v8ml-valid/long_64k_text_document)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=[65536])
    ap.add_argument("--docs-n", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import numpy as np
    import router_probe as rp
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, dtype=torch.bfloat16,
                                                 device_map="auto").eval()
    sa = getattr(model.config, "sparse_attention", None)
    bos = tok.bos_token_id
    docs = rp.read_indexed(a.docs)
    res = {"model": a.model, "sparse_attention": sa, "lengths": {}}
    for L in a.lengths:
        order = [i for i in np.random.default_rng(a.seed).permutation(len(docs)) if docs.lengths[i] >= L - 1]
        sel = [torch.tensor([[bos] + docs[int(i)][: L - 1].tolist()]) for i in order[: a.docs_n]]
        t0 = time.time()
        nlls = []
        for ids in sel:
            dev = next(model.parameters()).device
            cache, out = None, []
            with torch.no_grad():
                for c in range(0, L, a.chunk):
                    part = ids[:, c:c + a.chunk].to(dev)
                    o = model.model(input_ids=part, past_key_values=cache, use_cache=True)
                    cache, h = o.past_key_values, o.last_hidden_state
                    tgt = ids[0, c + 1:c + 1 + part.shape[1]].to(h.device)
                    for c0 in range(0, tgt.numel(), 4096):
                        lp = torch.log_softmax(model.lm_head(h[0, c0:c0 + 4096][: tgt.numel() - c0]).float(), -1)
                        out.append(-lp.gather(-1, tgt[c0:c0 + 4096, None])[:, 0].cpu())
            nlls.append(torch.cat(out))
        nll = torch.stack(nlls)  # [n, L - 1]
        edges = sorted({e for e in [0, 4096, 16384, 32768, 65536, 131072, 262144, 524288] if e < L}) + [L - 1]
        bins = {f"{lo // 1024}k-{hi // 1024}k": float(nll[:, lo:hi].mean()) for lo, hi in zip(edges[:-1], edges[1:])}
        res["lengths"][str(L)] = {"docs": len(sel), "mean": float(nll.mean()), "bins": bins,
                                  "seconds": round(time.time() - t0, 1)}
        print(f"[nll_by_position] L={L}: {float(nll.mean()):.4f} {bins} {time.time() - t0:.0f}s", flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()

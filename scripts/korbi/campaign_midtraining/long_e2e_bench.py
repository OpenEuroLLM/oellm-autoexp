#!/usr/bin/env python3
"""End-to-end long-context inference cost of an HF export as it is (its sparse attention, OELLM_SPARSE_* env): a
prompt of L tokens prefilled in chunks through the cache (long_generate.py's path), then greedy decode of
--decode tokens. Reports per length: prefill seconds (and the last chunk's, which grows with the context for dense
attention), decode ms per token at that context, peak GPU memory. One GPU, batch 1. The prompt is natural filler
(the downstream raw.jsonl passages, repeated) so the selection sees real keys.

    long_e2e_bench.py --model <hf export> --raw <raw.jsonl> --out <json> [--lengths 131072 ... 1048576]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "downstream"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=[131072, 262144, 524288, 1048576])
    ap.add_argument("--chunk", type=int, default=32768)
    ap.add_argument("--decode", type=int, default=64)
    a = ap.parse_args()

    import prepare_tasks as pt
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, dtype=torch.bfloat16,
                                                 device_map="cuda:0").eval()
    dev = torch.device("cuda:0")
    filler = pt._filler(tok, a.raw)
    rows = []
    for L in a.lengths:
        ids = ([tok.bos_token_id] + filler * (L // len(filler) + 1))[:L]
        x = torch.tensor([ids], device=dev)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        row = {"length": L, "sparse": getattr(model.config, "sparse_attention", None) is not None,
               "select": os.environ.get("OELLM_SPARSE_SELECT", "block")}
        try:
            with torch.no_grad():
                cache, logits, chunks = None, None, []
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for s in range(0, L, a.chunk):
                    c0 = time.perf_counter()
                    out = model(input_ids=x[:, s:s + a.chunk], past_key_values=cache, use_cache=True,
                                logits_to_keep=1)
                    cache, logits = out.past_key_values, out.logits[:, -1]
                    torch.cuda.synchronize()
                    chunks.append(time.perf_counter() - c0)
                row["prefill_s"] = time.perf_counter() - t0
                row["last_chunk_s"] = chunks[-1]
                row["prefill_peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
                times = []
                for _ in range(a.decode):
                    t = logits.argmax(-1, keepdim=True)
                    c0 = time.perf_counter()
                    out = model(input_ids=t, past_key_values=cache, use_cache=True, logits_to_keep=1)
                    cache, logits = out.past_key_values, out.logits[:, -1]
                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - c0)
                times = sorted(times[4:])  # past the first steps' warm-up
                row["decode_ms_median"] = 1e3 * times[len(times) // 2]
                row["peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
            del cache, out, logits
        except torch.OutOfMemoryError:
            row["oom"] = True
        rows.append(row)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"model": a.model, "gpu": torch.cuda.get_device_name(), "rows": rows},
                                    indent=2) + "\n")


if __name__ == "__main__":
    main()

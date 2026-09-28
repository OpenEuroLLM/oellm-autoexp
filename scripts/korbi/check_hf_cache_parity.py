#!/usr/bin/env python3
"""Assert cached and uncached decoding agree for a converted hybrid model.

The KV/recurrent-state cache is the one part of the port with no Megatron
counterpart to diff against, so it is checked against the model's own
cache-free path: greedy-decode the same prompt twice, once with `use_cache=True`
(prefill + single-token steps through the step kernels) and once with
`use_cache=False` (full recompute every step), and require identical tokens.

Run per variant — each mixer family has its own state and its own step kernel:

    python scripts/korbi/check_hf_cache_parity.py --model-dir <hf_dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--prompt-len", type=int, default=37, help="deliberately not a chunk multiple")
    p.add_argument("--new-tokens", type=int, default=24)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, trust_remote_code=True, dtype=torch.bfloat16
    ).to(args.device).eval()

    torch.manual_seed(0)
    ids = torch.randint(0, 50000, (1, args.prompt_len), device=args.device)

    def greedy(use_cache: bool) -> torch.Tensor:
        out = model.generate(
            ids,
            max_new_tokens=args.new_tokens,
            do_sample=False,
            use_cache=use_cache,
            pad_token_id=cfg.vocab_size - 1,
        )
        return out[0, args.prompt_len :]

    with torch.no_grad():
        cached = greedy(True)
        uncached = greedy(False)

    same = int((cached == uncached).sum())
    print(f"model            : {args.model_dir.name}")
    print(f"mixers           : {sorted(set(cfg.mixer_types))}")
    print(f"prompt/new       : {args.prompt_len}/{args.new_tokens}")
    print(f"cached  tokens   : {cached.tolist()}")
    print(f"uncached tokens  : {uncached.tolist()}")
    print(f"agreement        : {same}/{len(cached)}")
    ok = same == len(cached)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""Check an HF export without loading weights: the config loads, the model builds on the meta
device, and its parameter / buffer names equal the shard index's (missing or unexpected names are
what ``from_pretrained`` would otherwise only warn about).

    python -m oellm_autoexp.hf_export.check_export <export dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


def main() -> None:
    from transformers import AutoConfig, AutoModelForCausalLM

    d = Path(sys.argv[1])
    cfg = AutoConfig.from_pretrained(d, trust_remote_code=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    have = set(model.state_dict())
    stored = set(json.loads((d / "model.safetensors.index.json").read_text())["weight_map"])
    missing, unexpected = sorted(have - stored), sorted(stored - have)
    tied = {"lm_head.weight"} if getattr(cfg, "tie_word_embeddings", False) else set()
    missing = [n for n in missing if n not in tied]
    print(f"{d.name}: {type(model).__name__}, {len(have)} model tensors, {len(stored)} stored")
    print("missing:", missing[:20], len(missing))
    print("unexpected:", unexpected[:20], len(unexpected))
    sys.exit(1 if missing or unexpected else 0)


if __name__ == "__main__":
    main()

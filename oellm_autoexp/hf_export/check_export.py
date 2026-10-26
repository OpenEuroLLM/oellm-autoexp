"""Check an HF export without loading weights: the config loads, the model builds on the meta
device, and its parameter / buffer names and shapes equal the shards' (read from the safetensors
headers; ``from_pretrained`` only warns about missing names and raises on shapes after the load).

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
    from safetensors import safe_open

    weight_map = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
    stored = set(weight_map)
    shapes = {}
    for shard in sorted(set(weight_map.values())):
        with safe_open(str(d / shard), framework="pt") as f:
            shapes.update({k: tuple(f.get_slice(k).get_shape()) for k in f.keys()})
    sd = model.state_dict()
    wrong = sorted(f"{k}: stored {shapes[k]} vs model {tuple(sd[k].shape)}"
                   for k in have & stored if shapes[k] != tuple(sd[k].shape))
    missing, unexpected = sorted(have - stored), sorted(stored - have)
    tied = {"lm_head.weight"} if getattr(cfg, "tie_word_embeddings", False) else set()
    missing = [n for n in missing if n not in tied]
    print(f"{d.name}: {type(model).__name__}, {len(have)} model tensors, {len(stored)} stored")
    print("missing:", missing[:20], len(missing))
    print("unexpected:", unexpected[:20], len(unexpected))
    print("shape mismatches:", wrong[:20], len(wrong))
    sys.exit(1 if missing or unexpected or wrong else 0)


if __name__ == "__main__":
    main()

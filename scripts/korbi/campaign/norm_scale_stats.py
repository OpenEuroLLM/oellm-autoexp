"""The 1-D scales of a hybrid model across checkpoints, per layer: the pre-sublayer norm gains
(input_layernorm, pre_mlp_layernorm, ...), the final norm, and LDM's scalars (read_gain,
forget_logit -> lambda, the address networks' norm_weight). Reads only these small tensors (the
fp32 model weights) from torch_dist / fsdp_dtensor checkpoints; per tensor mean, rms, min, max.

    python norm_scale_stats.py <out.json> <tag>=<checkpoint dir> ...
"""

import json
import re
import sys

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader

KEY = re.compile(
    r"^(?:model\.)?(?:module\.)*(decoder\.(?:final_norm|final_layernorm)\.weight|"
    r"decoder\.layers\.\d+\.(?:[a-z_]*layernorm|[a-z_]*_norm|post_[a-z]+_norm)\.weight|"
    r"decoder\.layers\.\d+\.[a-z_]*mixer\.(?:next_query\.read_gain|write_site\.forget_logit|"
    r"next_query\.net\.norm_weight|write_site\.net\.norm_weight))$"
)


def scan(path: str) -> dict:
    md = FileSystemReader(path).read_metadata().state_dict_metadata
    want = {}
    for k in md:
        if "optimizer" in k:
            continue
        m = KEY.match(k)
        if m:
            want[k] = m.group(1)
    sd = {k: torch.empty(md[k].size, dtype=md[k].properties.dtype) for k in want}
    dcp.load(sd, storage_reader=FileSystemReader(path), no_dist=True)
    out = {}
    for k, name in want.items():
        t = sd[k].float()
        if name.endswith("forget_logit"):
            t, name = torch.sigmoid(t), name.replace("forget_logit", "forget")
        out[name] = [float(t.mean()), float(t.square().mean().sqrt()), float(t.min()), float(t.max())]
    print(f"{path}: {len(out)} tensors", flush=True)
    return out


def main():
    res = {}
    for spec in sys.argv[2:]:
        tag, path = spec.split("=", 1)
        res[tag] = scan(path)
        json.dump(res, open(sys.argv[1], "w"))
    print("wrote", sys.argv[1])


if __name__ == "__main__":
    main()

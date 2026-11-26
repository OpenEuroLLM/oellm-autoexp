"""e2e helper: the mHC checkpoint with every mixer at its init read / write (phi = 0, b_pre = logit(1/n),
b_post = 0), where all streams stay equal and mHC is exactly the plain residual (with the V layers' g_v).
Converting this checkpoint must therefore not change the loss.
    identity_mixers.py <src iter dir> <dst iter dir>
"""
import math
import os
import re
import sys

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dcp_surgery import write_surgery  # noqa: E402

MIXER = re.compile(r"model\.module\.decoder\.layers\.\d+\.(mhc_attn_mixer|mhc_mlp_mixer|residual_scheme\.mhc_mixer)\.(phi|b_pre|b_post)")

src, dst = sys.argv[1], sys.argv[2]
md = FileSystemReader(src).read_metadata()
smd = md.state_dict_metadata
replace = {}
for key, meta in smd.items():
    m = MIXER.fullmatch(key)
    if not m:
        continue
    size, dtype = tuple(meta.size), meta.properties.dtype
    if m.group(2) == "b_pre":
        n = size[0]
        replace[key] = torch.full(size, math.log((1.0 / n) / (1.0 - 1.0 / n)), dtype=dtype)
    else:
        replace[key] = torch.zeros(size, dtype=dtype)
print(f"{len(replace)} mixer tensors reset")
write_surgery(src, dst, md, replace, tag="identity")
it = int(re.search(r"iter_(\d+)", dst).group(1))
open(os.path.join(os.path.dirname(dst), "latest_checkpointed_iteration.txt"), "w").write(f"{it}\n")
back = {k: torch.empty(v.shape, dtype=v.dtype) for k, v in replace.items()}
dcp.load(back, storage_reader=FileSystemReader(dst), no_dist=True)
assert all(torch.equal(back[k], v) for k, v in replace.items())
print(f"wrote {dst}")

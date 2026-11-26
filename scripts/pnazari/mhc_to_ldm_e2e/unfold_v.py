"""e2e negative control: a converted tree with the V layers' linear_proj (and its MuOwn row states) put back unfolded
from the source, i.e. the conversion without the g_v fold.   unfold_v.py <source iter dir> <converted iter dir> <dst>"""
import os
import re
import sys

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dcp_surgery import write_surgery  # noqa: E402

src, conv, dst = sys.argv[1:4]
smd = FileSystemReader(src).read_metadata().state_dict_metadata
keys = [k for k in smd if re.search(r"layers\.\d+\.self_attention\.linear_proj\.weight(\.(g|m_g|v_g))?$", k)
        and re.sub(r"self_attention\.linear_proj.*", "v_gate.out_gate", k.replace(
            "optimizer.0.state.module.module.module.", "model.module.")) in smd]
want = {k: torch.empty(tuple(smd[k].size), dtype=smd[k].properties.dtype) for k in keys}
dcp.load(want, storage_reader=FileSystemReader(src), no_dist=True)
print(f"{len(want)} V tensors put back unfolded: {sorted(want)}")
write_surgery(conv, dst, FileSystemReader(conv).read_metadata(), want, tag="unfold")
open(os.path.join(os.path.dirname(dst), "latest_checkpointed_iteration.txt"), "w").write("20\n")

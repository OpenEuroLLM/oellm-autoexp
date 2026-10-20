"""Per-kernel CUDA time of a 4-sublayer mHC chain (fwd + bwd), fused read with the write backward
separate vs inside the read backward (MHC_FUSED_WRITE_BWD), 8k tokens x 4 x 2048; plus the
registers / spills and the autotuned config of the backward kernels.

    python scripts/korbi/hybrid_speed/prof_mhc_chain.py
"""

import torch
from torch.profiler import ProfilerActivity, profile

from megatron.core.transformer import mhc_fused
from megatron.core.transformer.mhc import MHCMixer

N, C, T = 4, 2048, 8192
torch.manual_seed(0)
mixers = [MHCMixer(C, N).cuda() for _ in range(4)]
for m in mixers:
    with torch.no_grad():
        m.phi.normal_(0, 0.02)
streams = torch.randn(T, 1, N, C, device="cuda", dtype=torch.bfloat16, requires_grad=True)
f = torch.randn(T, 1, C, device="cuda", dtype=torch.bfloat16, requires_grad=True)
g = torch.randn(T, 1, N, C, device="cuda", dtype=torch.bfloat16)
mhc_fused.ENABLED = True


def step():
    s = streams
    for m in mixers:
        link = mhc_fused.ReadWriteLink()
        x, _, hp, hr = mhc_fused.coeffs_and_read(m, s, link)
        s = m.write(s, x + f, hp, hr, link=link)
    s.backward(g)


for fuse in (False, True):
    mhc_fused.FUSE_WRITE_BWD = fuse
    for _ in range(5):
        step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            step()
        torch.cuda.synchronize()
    print(f"===== FUSE_WRITE_BWD={fuse} (per 10 steps of 4 sublayers)", flush=True)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=14, max_name_column_width=60),
          flush=True)

for name in ("_fused_bwd_kernel", "_dstreams_kernel"):
    k = getattr(mhc_fused, name)
    print(name, "best", k.best_config, flush=True)
    for dev_cache in k.fn.device_caches.values() if hasattr(k.fn, "device_caches") else []:
        for key, ck in dev_cache[0].items():
            print("  regs", getattr(ck, "n_regs", "?"), "spills", getattr(ck, "n_spills", "?"),
                  str(key)[:80], flush=True)

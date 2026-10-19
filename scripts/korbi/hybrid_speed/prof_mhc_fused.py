"""Per-kernel CUDA time of one mHC sublayer (read -> write, fwd + bwd), compiled vs MHC_FUSED_READ,
at 8k tokens x 4 streams x 2048 (torch.profiler, 10 steps after warmup).

    python scripts/korbi/hybrid_speed/prof_mhc_fused.py
"""

import torch
from torch.profiler import ProfilerActivity, profile

from megatron.core.transformer import mhc_fused
from megatron.core.transformer.mhc import MHCMixer

N, C, T = 4, 2048, 8192
torch.manual_seed(0)
mixer = MHCMixer(C, N).cuda()
with torch.no_grad():
    mixer.phi.normal_(0, 0.02)
streams = torch.randn(T, 1, N, C, device="cuda", dtype=torch.bfloat16, requires_grad=True)
f = torch.randn(T, 1, C, device="cuda", dtype=torch.bfloat16, requires_grad=True)
g = torch.randn(T, 1, N, C, device="cuda", dtype=torch.bfloat16)


def step(fused):
    if fused:
        link = mhc_fused.ReadWriteLink()
        x, _, hp, hr = mhc_fused.coeffs_and_read(mixer, streams, link)
    else:
        link = None
        x, _, hp, hr = mixer.coeffs_and_read(streams)
    mixer.write(streams, x + f, hp, hr, link=link).backward(g)


for fused in (False, True):
    mhc_fused.ENABLED = fused
    for _ in range(5):
        step(fused)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            step(fused)
        torch.cuda.synchronize()
    print(f"===== fused={fused}", flush=True)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=70),
          flush=True)

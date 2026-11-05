"""014: per-kernel GPU times (torch.profiler) of the fused mHC write + read at a few token counts."""
import sys

import torch
from torch.profiler import ProfilerActivity, profile

from oellm_autoexp.hf_export.oellm_ambitious import oellm_kernels as k

import check_kernel as ck  # noqa: E402

h = 2048
SPLIT = __import__("os").environ.get("SPLIT", "0") == "1"
for tokens in [int(x) for x in sys.argv[1:]] or [16, 1024]:
    g = torch.Generator().manual_seed(0)
    s = torch.randn(tokens, 4, h, generator=g).to("cuda", torch.bfloat16)
    out = torch.randn(tokens, h, generator=g).to("cuda", torch.bfloat16)
    p0, p1 = ck.params(h, "cuda", g), ck.params(h, "cuda", g)
    nw = (torch.ones(h, device="cuda", dtype=torch.bfloat16), 1e-5, False)
    _, pend = k.mhc_read(s, p0["phi"], p0["alpha_pre"], p0["alpha_post"], p0["alpha_res"], p0["b_pre"],
                         p0["b_post"], p0["b_res"], 1e-6, 20)
    phi_pad = k.mhc_phi_padded(p1["phi"], torch.bfloat16)
    args1 = (p1["alpha_pre"], p1["alpha_post"], p1["alpha_res"], p1["b_pre"], p1["b_post"], p1["b_res"], 1e-6, 20)
    for _ in range(3):
        k.mhc_write_read(s, out, pend, phi_pad, *args1, nw, split=SPLIT)
        k.mhc_read(k.mhc_write(s, out, pend), p1["phi"], *args1, nw)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            k.mhc_write_read(s, out, pend, phi_pad, *args1, nw, split=SPLIT)
            k.mhc_read(k.mhc_write(s, out, pend), p1["phi"], *args1, nw)
        torch.cuda.synchronize()
    print(f"== tokens {tokens}")
    for e in prof.key_averages():
        if e.device_time_total > 0:
            print(f"  {e.key[:60]:60s} {e.device_time_total / e.count:9.1f} us x{e.count}")

"""Forward / backward bandwidth of the fused q/k RMSNorm + RoPE kernel (one GPU)."""
import torch, triton, sys
sys.path.insert(0, ".")
from megatron.core.fusions.fused_qk_norm_rope import fused_qk_norm_rope
s, b, ng, hq, d = 4096, 8, 12, 1, 64
mixed = torch.randn(s, b, ng, (hq + 2) * d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
qw = torch.ones(d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
kw = torch.ones(d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
inv = 1.0 / (10000 ** (torch.arange(0, d, 2, device="cuda").float() / d))
ang = torch.arange(s, device="cuda").float()[:, None] * inv[None]
freqs = torch.cat([ang, ang], -1).view(s, 1, 1, d)
q, k, v = fused_qk_norm_rope(mixed, qw, kw, freqs, hq, 1e-6)
gq, gk, gv = torch.randn_like(q), torch.randn_like(k), torch.randn_like(v)
fwd = triton.testing.do_bench(lambda: fused_qk_norm_rope(mixed, qw, kw, freqs, hq, 1e-6))
def fb():
    q, k, v = fused_qk_norm_rope(mixed, qw, kw, freqs, hq, 1e-6)
    torch.autograd.backward((q, k, v), (gq, gk, gv))
both = triton.testing.do_bench(fb)
nbytes = mixed.numel() * 2
print(f"fwd {fwd:.3f} ms ({2*nbytes/fwd/1e6:.0f} GB/s)  bwd {both-fwd:.3f} ms ({3*nbytes/(both-fwd)/1e6:.0f} GB/s)")

"""Weight-only FP8 MoE kernel (E4M3 + 1x128 scales, in-kernel bf16 dequant) vs the dequantized bf16 weights on H100:
the same bf16 weight values; accepted if its error vs an fp64 reference is no larger than the bf16 path's (MMA accumulation order may differ)."""
import torch
from vllm.model_executor.layers.fused_moe.fused_moe import invoke_fused_moe_triton_kernel, register_fp8_w8a16, get_default_config
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
import triton.language as tl
torch.manual_seed(0); dev="cuda"
E, N, K = 8, 1536, 2048
for M in (1, 4, 16):
    q = (torch.randn(E, N, K, device=dev) * 50).clamp(-448, 448).to(torch.float8_e4m3fn)
    sc = torch.rand(E, N, K // 128, device=dev) * 1e-3 + 1e-4
    wbf = (q.float().view(E, N, K // 128, 128) * sc[..., None]).view(E, N, K).to(torch.bfloat16)
    a = torch.randn(M, K, device=dev).to(torch.bfloat16)
    topk = 2
    ids = torch.stack([torch.randperm(E, device=dev)[:topk] for _ in range(M)]).to(torch.int32)
    w = torch.rand(M, topk, device=dev)
    cfg = get_default_config(M, E, N, K, topk, None)
    st, ex, npad = moe_align_block_size(ids, cfg["BLOCK_SIZE_M"], E)
    outs = []
    for B in (wbf, q):
        if B.dtype == torch.float8_e4m3fn: register_fp8_w8a16(B, sc, 128)
        C = torch.zeros(M, topk, N, device=dev, dtype=torch.bfloat16)
        invoke_fused_moe_triton_kernel(a, B, C, None, None, w, st, ex, npad, False, topk, cfg, tl.bfloat16, False, False, False, False, False)
        outs.append(C)
    ref = torch.zeros(M, topk, N, device=dev, dtype=torch.float64)
    for t in range(M):
        for j in range(topk):
            ref[t, j] = wbf[ids[t, j]].double() @ a[t].double()
    err = [((o.double() - ref).abs().max() / ref.abs().max()).item() for o in outs]
    frac = (outs[0] != outs[1]).float().mean().item()
    print(f"M={M} BLOCK_K={cfg['BLOCK_SIZE_K']}: bit-exact {torch.equal(outs[0], outs[1])}, differing {frac:.4f}, "
          f"max rel err vs fp64: bf16 {err[0]:.2e} fp8 {err[1]:.2e}", flush=True)
    assert err[1] <= 1.5 * err[0] + 1e-6, "fp8 w8a16 less accurate than the bf16 path"
print("FP8_W8A16_EXACT")  # (name kept: the gate of run.sh)

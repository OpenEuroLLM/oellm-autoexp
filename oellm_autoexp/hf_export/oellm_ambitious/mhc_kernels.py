"""Inference-only fused 4-stream Sinkhorn; the reference keeps autograd support."""
import torch
import triton
import triton.language as tl


@triton.jit
def _sinkhorn4_kernel(X, Y, TOKENS: tl.constexpr, ITERS: tl.constexpr,
                      BLOCK: tl.constexpr):
    token = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = tl.arange(0, 4)
    col = tl.arange(0, 4)
    offset = token[:, None, None] * 16 + row[None, :, None] * 4 + col[None, None, :]
    x = tl.load(X + offset, mask=token[:, None, None] < TOKENS, other=0).to(tl.float32)
    for _ in range(ITERS):
        maximum = tl.max(x, axis=1)
        lse = maximum + tl.log(tl.sum(tl.exp(x - maximum[:, None, :]), axis=1))
        x = x - lse[:, None, :]
        maximum = tl.max(x, axis=2)
        lse = maximum + tl.log(tl.sum(tl.exp(x - maximum[:, :, None]), axis=2))
        x = x - lse[:, :, None]
    tl.store(Y + offset, tl.exp(x), mask=token[:, None, None] < TOKENS)


def sinkhorn4(logits: torch.Tensor, iters: int) -> torch.Tensor:
    """Apply column/row log normalization in exactly the reference order."""
    if logits.shape[-2:] != (4, 4) or not logits.is_cuda:
        raise ValueError("sinkhorn4 requires CUDA tensors ending in [4, 4]")
    x = logits.float().contiguous()
    out = torch.empty_like(x)
    tokens = x.numel() // 16
    if tokens:
        with torch.cuda.device(x.device):
            _sinkhorn4_kernel[(triton.cdiv(tokens, 32),)](
                x, out, tokens, iters, 32, num_warps=4, enable_fp_fusion=False
            )
    return out

"""bf16 vs FP8 (1 x 128 blockwise) grouped GEMM at the MoonEP expert shapes, one H100.

The question for "FP8 compute after fsdp_fp8_param_gather": with the expert weights already in
FP8 (they arrive that way from the all-gather), is the forward grouped GEMM faster in FP8 than in
bf16 -- counting the activation quantization (1 x 128 blocks along K), which the FP8 path still
needs every call? Weights are quantized once outside the timing (the gather delivers them).

Shapes: one EP rank at 216 experts / EP 4: 54 local experts; MoonEP gives every rank S x K =
65,536 x 5 = 327,680 token rows, here in 54 groups padded to 128 rows (MoonEP's token padding).
fc1: [1536, 2048] (gate + up) on x [rows, 2048]; fc2: [2048, 768] on a [rows, 768].

    python scripts/korbi/hybrid_speed/bench_fp8_grouped.py [--experts 54] [--rows 327680]
"""
import argparse

import torch
import transformer_engine_torch as tex
from transformer_engine.pytorch.cpp_extensions import general_grouped_gemm_for_grouped_tensor
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
from transformer_engine.pytorch.tensor.storage.grouped_tensor_storage import GroupedTensorStorage


def grouped(data, splits, offsets, last_dim):
    return GroupedTensorStorage(
        shape=(data.size(0), last_dim), dtype=data.dtype, num_tensors=splits.numel(),
        quantizer=None, data=data.reshape(-1), first_dims=splits,
        tensor_offsets=offsets * last_dim,
    )


def timed(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=54)
    ap.add_argument("--rows", type=int, default=65536 * 5)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--ffn", type=int, default=768)
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = "cuda"
    E, H, F = args.experts, args.hidden, args.ffn
    # uneven groups (MoonEP balances ranks, not experts), each padded to 128 rows
    w = torch.rand(E) + 0.5
    raw = (w / w.sum() * args.rows).floor().long()
    splits = ((raw + 127) // 128 * 128).to(dev)
    rows = int(splits.sum())
    offsets = torch.cat([splits.new_zeros(1), splits.cumsum(0)])
    print(f"experts {E}, rows {rows} (mean {rows // E} per group), H {H}, F {F}")

    results = []
    for name, out_dim, in_dim in (("fc1", 2 * F, H), ("fc2", H, F)):
        W = [(torch.randn(out_dim, in_dim, device=dev) * in_dim ** -0.5).to(torch.bfloat16)
             for _ in range(E)]
        x = torch.randn(rows, in_dim, device=dev, dtype=torch.bfloat16)
        y_bf16 = torch.empty(rows, out_dim, device=dev, dtype=torch.bfloat16)
        y_fp8 = torch.empty_like(y_bf16)

        def run_bf16():
            general_grouped_gemm_for_grouped_tensor(
                W, grouped(x, splits, offsets, in_dim), grouped(y_bf16, splits, offsets, out_dim),
                layout="TN")

        # weights: FP8 E4M3, 1 x 128 blocks along K (what the gather delivers), once
        wq = Float8BlockQuantizer(tex.DType.kFloat8E4M3, rowwise=True, columnwise=False,
                                  block_scaling_dim=1)
        W8 = [wq(t) for t in W]
        xq = Float8BlockQuantizer(tex.DType.kFloat8E4M3, rowwise=True, columnwise=False,
                                  block_scaling_dim=1)
        xq.optimize_for_gemm = True

        def quant_x():
            return tex.group_quantize(x, xq, E, splits.to(torch.int64),
                                      tensor_offsets=offsets * in_dim)

        gx = quant_x()

        def run_fp8_gemm_only():
            general_grouped_gemm_for_grouped_tensor(
                W8, gx, grouped(y_fp8, splits, offsets, out_dim), layout="TN")

        def run_fp8():
            g = quant_x()
            general_grouped_gemm_for_grouped_tensor(
                W8, g, grouped(y_fp8, splits, offsets, out_dim), layout="TN")

        t_bf16 = timed(run_bf16)
        t_q = timed(quant_x)
        t_g = timed(run_fp8_gemm_only)
        t_f = timed(run_fp8)
        run_bf16(); run_fp8(); torch.cuda.synchronize()
        rel = ((y_fp8.float() - y_bf16.float()).norm() / y_bf16.float().norm()).item()
        flops = 2 * rows * in_dim * out_dim
        print(f"{name}: bf16 {t_bf16:7.3f} ms ({flops / t_bf16 / 1e9:6.1f} TFLOP/s) | fp8 total "
              f"{t_f:7.3f} ms ({flops / t_f / 1e9:6.1f}) = quantize x {t_q:6.3f} + GEMM {t_g:6.3f} "
              f"({flops / t_g / 1e9:6.1f}) | speedup {t_bf16 / t_f:5.2f}x | rel err {rel:.4f}")
        results.append((name, t_bf16, t_f))
    tb, tf = sum(r[1] for r in results), sum(r[2] for r in results)
    print(f"fc1 + fc2 forward: bf16 {tb:.3f} ms, fp8 {tf:.3f} ms, speedup {tb / tf:.2f}x")


if __name__ == "__main__":
    main()

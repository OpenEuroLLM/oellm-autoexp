## Notes (hand-written; the tables below are regenerated)

Setup: hidden 768, 12 blocks, seq 4096, mbs 8, gbs 32 (DP 4), mock data, bf16, CUDA graphs
`local` unless an arm says otherwise. TFLOP/s is Megatron's analytic count (per-architecture
formula); tok/s/GPU is the formula-free comparison -- the ratios below are tok/s vs `dense`.
Image: MegatronTraining-hybrid-ep (NGC 26.09, torch 2.14 / TE 2.19) since v22 (`aux.image=ep`);
earlier suites ran on the 25.12 base image. Goal: every feature arm >= 0.7 of dense.

### Baseline (optimized reference, v2+)
- `dense` 186 -> ~310-330 TFLOP/s (~320k tok/s/GPU): Liger fused LM-head CE, no dataloader
  mask, fused q/k RMSNorm + RoPE (Triton), TE fused residual/RMSNorm, fused short conv.
- No gain: TE CE, flash backend (0.92x), FP8 (0.97x), full-iteration graph (1.03x),
  no distributed optimizer (0.98x).
- torchtitan twin (titan_dense_mn5.yaml, same image, FineWeb data): t1 suite in flight.

### Feature arms (tok/s vs dense, latest runs as of 2026-10-16)
| >= 0.7 | < 0.7 |
|---|---|
| ngpt 0.83, canon 0.75, tri_gdn 0.73 (input-GEMM padding) | complex_kda 0.67, kda 0.66 (padded FLA input GEMM in v24) |
| mlstm 0.91, routed_k2 0.73, attn_res 0.76, gdn 0.76, MoE muown 0.81, muown_offload 0.80 | routed_k2_mix 0.63, pkm 0.60 (mbs 4), mamba2 0.61 (off) |
| | mhc4 0.58, tri_kda 0.54, tri_ckda 0.53, MoE adam_offload 0.62 |
| | power 0.41 (retention Triton kernel; backends x chunk sizes: kt_pow) |

- signed_kda == complex_kda (dropped from the suites).
- Triadic: CuTe kernels need head_dim 128 (other dims fall back to the 2-4x slower
  expanded-key KDA path). Triadic KDA's channel gate costs ~2.7 ms/layer over tri_gdn
  (Triton gram bwd 1.6, glue bwd 0.8, gram fwd 0.6).
- Power: deg-2 state is d(d+1)/2 wide -- hd128 is 2.3x slower than hd64; hd >= 256 fails
  in the retention kernel.
- PKM: bag kernels are near memory bandwidth (200M-row-param table per layer); the rest is
  the dp-sharded dedup gather / index_add.
- Residual schemes (12 layers, mbs 8, ms / peak GB): residual 96 / 14.0, attn_res4 130 / 15.2,
  routed_k2_rev 144 / 15.6, routed_k2 134 / 18.8, routed_k2_mix 154 / 19.0, mhc4 173 / 20.0
  (mHC saves the 4x stream per sublayer: recompute is the lever).
- Head dims (bench_block, H100): gdn / mlstm / kda / complex_kda all run 64-512; attention
  hd512 is 2.2x slower.

### Parallelism (par3, 130M)
TP2+SP and PP2 run for all arms; CP2 now implemented for engram, kda / complex_kda (FLA
KCP) and triadic / power (full-sequence fallback, compute not split): v23cp2 / kt_cp in
flight. Known gaps: Liger needs TP1, CUDA graphs under PP, EP a2a overlap for HybridModel.

### MoE (old 50M active / 500M; re-measured at 120M / 1.2B in v25)
CUTLASS grouped GEMM 1.06x (TE 2.10, segfaults for noshared / latent; fine on TE 2.18+),
NeMo-style config + CUTLASS 1.00x, latent 1.09x on the ep image (v22); HybridEP ~1.00x at EP4 intra-node; router fusion
and conn8 no gain.

# Hybrid speed state

updated 2026-10-17 00:04 from `/gpfs/scratch/ehpc533/users/korbip/hybrid_v019_runs/autoexp/hybrid_speed` (median step after iteration 30; one node, 4x H100 64GB; H100 bf16 dense peak ~989 TFLOP/s)

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
- torchtitan main (titan_main_mn5.yaml, TitanTraining-main image, Qwen3 130M): best 319.5k tok/s/GPU
  (GraphTrainer, full Inductor, FA3 varlen, mbs 32; 38.6 % MFU) vs Megatron 343.8k at mbs 32:
  Megatron 1.08x faster -> Megatron stays. FA4 / flex_flash hangs on H100.

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

## dense130M: latest per arm

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status | suite |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| dense | 123.6M | 102 | 312.2 | 1.00 | 322,520 | 1.00 | 127 | 13.4 | ok | v26_dense130M |
| pkm (!) | 219.8M | 170 | 181.8 | 0.58 | 192,923 | 0.60 | 124 | 9.7 | ok | v22_dense130M |
| routed_k2 (!) | 123.7M | 139 | 227.6 | 0.73 | 235,065 | 0.73 | 127 | 18.3 | ok | v22_dense130M |
| routed_k2_mix (!) | 124.0M | 161 | 197.1 | 0.63 | 203,718 | 0.63 | 127 | 18.5 | ok | v22_dense130M |
| routed_k2_rev (!) | 123.7M | 152 | 208.7 | 0.67 | 215,579 | 0.67 | 127 | 15.1 | ok | v22_dense130M |
| routed_k2_cg (!) | 123.7M | 144 | 220.0 | 0.70 | 227,240 | 0.70 | 127 | 18.1 | ok | v12routed_dense130M |
| routed_k2_mix_cg (!) | 124.0M | 162 | 196.3 | 0.63 | 202,835 | 0.63 | 127 | 18.3 | ok | v12routed_dense130M |
| routed_k2_rev_cg | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  | v22_dense130M |
| gated_delta_net (!) | 129.1M | 134 | 206.4 | 0.66 | 245,178 | 0.76 | 110 | 16.0 | ok | v22_dense130M |
| kda (!) | 125.5M | 154 | 174.1 | 0.56 | 212,366 | 0.66 | 107 | 15.6 | ok | v22_dense130M |
| complex_kda (!) | 130.0M | 152 | 182.9 | 0.59 | 216,005 | 0.67 | 111 | 17.5 | ok | v22_dense130M |
| signed_kda (!) | 130.0M | 150 | 184.9 | 0.59 | 218,453 | 0.68 | 111 | 17.5 | ok | v21_dense130M |
| kda_xsa (!) | 125.5M | 161 | 166.6 | 0.53 | 203,212 | 0.63 | 107 | 16.0 | ok | v22_dense130M |
| kda_hd128 (!) | 127.3M | 156 | 176.1 | 0.56 | 209,380 | 0.65 | 110 | 15.3 | ok | v22_dense130M |
| dense_gpt | 123.6M | 102 | 312.2 | 1.00 | 322,520 | 1.00 | 127 | 13.4 | ok | v22_dense130M |
| dense_nocg | 123.6M | 102 | 309.6 | 0.99 | 319,844 | 0.99 | 127 | 13.6 | ok | v22_dense130M |
| base_liger | 123.6M | 103 | 308.8 | 0.99 | 319,065 | 0.99 | 127 | 13.4 | ok | v22_dense130M |
| base_ce_te | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  | v22_dense130M |
| base_flash | 123.6M | 119 | 267.5 | 0.86 | 276,290 | 0.86 | 127 | 13.9 | ok | v17_dense130M |
| base_graph_full | 123.6M | 101 | 314.6 | 1.01 | 325,079 | 1.01 | 127 | 13.6 | ok | v22_dense130M |
| base_liger_mbs16 | 123.6M | 198 | 320.9 | 1.03 | 331,492 | 1.03 | 254 | 24.3 | ok | v26_dense130M |
| base_liger_nocg | 123.6M | 102 | 309.5 | 0.99 | 319,688 | 0.99 | 127 | 13.6 | ok | v22_dense130M |
| base_liger_nodistopt | 123.6M | 105 | 300.8 | 0.96 | 310,744 | 0.96 | 127 | 13.7 | ok | v22_dense130M |
| base_liger_nomask | 123.6M | 102 | 309.9 | 0.99 | 320,156 | 0.99 | 127 | 13.4 | ok | v22_dense130M |
| base_liger_nomask_mbs16_nocg | 123.6M | 196 | 323.8 | 1.04 | 334,538 | 1.04 | 254 | 24.5 | ok | v22_dense130M |
| base_liger_noqkln | 123.6M | 102 | 310.4 | 0.99 | 320,626 | 0.99 | 127 | 11.6 | ok | v22_dense130M |
| base_liger_mbs4 | 123.6M | 106 | 299.0 | 0.96 | 308,986 | 0.96 | 127 | 7.9 | ok | v22_dense130M |
| base_qknr | 123.6M | 103 | 309.1 | 0.99 | 319,376 | 0.99 | 127 | 13.4 | ok | v22_dense130M |
| base_resnorm | 123.6M | 102 | 310.7 | 1.00 | 320,940 | 1.00 | 127 | 13.4 | ok | v22_dense130M |
| base_opfuser | 123.6M | 103 | 308.5 | 0.99 | 318,755 | 0.99 | 127 | 13.4 | ok | v22_dense130M |
| base_qknr_resnorm | 123.6M | 102 | 310.1 | 0.99 | 320,313 | 0.99 | 127 | 13.4 | ok | v22_dense130M |
| base_best | 123.6M | 102 | 310.0 | 0.99 | 320,313 | 0.99 | 127 | 13.4 | ok | v22_dense130M |
| base_best_swiglu | 123.6M | 103 | 309.1 | 0.99 | 319,376 | 0.99 | 127 | 13.4 | ok | v22_dense130M |
| base_best_mbs16_nocg | 123.6M | 198 | 321.1 | 1.03 | 331,744 | 1.03 | 254 | 24.5 | ok | v22_dense130M |
| base_best_hd128 | 123.6M | 98 | 322.9 | 1.03 | 333,517 | 1.03 | 127 | 13.4 | ok | v22_dense130M |
| base_fp8 | 123.6M | 107 | 297.4 | 0.95 | 307,248 | 0.95 | 127 | 12.5 | ok | v22_dense130M |
| nope | 123.6M | 107 | 295.8 | 0.95 | 305,672 | 0.95 | 127 | 12.8 | ok | v22_dense130M |
| gqa4 | 123.6M | 102 | 310.2 | 0.99 | 320,626 | 0.99 | 127 | 13.4 | ok | v22_dense130M |
| xsa | 123.6M | 112 | 283.4 | 0.91 | 292,833 | 0.91 | 127 | 13.9 | ok | v22_dense130M |
| gate | 130.7M | 110 | 289.0 | 0.93 | 298,706 | 0.93 | 127 | 15.1 | ok | v22_dense130M |
| xsa_gate | 130.7M | 118 | 268.1 | 0.86 | 276,991 | 0.86 | 127 | 15.1 | ok | v22_dense130M |
| swa3 | 123.6M | 96 | 287.6 | 0.92 | 342,225 | 1.06 | 110 | 13.4 | ok | v22_dense130M |
| mla (!) | 116.5M | 106 | 177.2 | 0.57 | 310,009 | 0.96 | 75 | 12.5 | ok | v22_dense130M |
| mla_gqa_xsa_gate (!) | 116.9M | 114 | 163.8 | 0.52 | 286,559 | 0.89 | 75 | 13.2 | ok | v22_dense130M |
| mamba2 (!) | 148.7M | 166 | 195.0 | 0.62 | 197,994 | 0.61 | 129 | 16.7 | ok | v22_dense130M |
| mamba2_ck64 (!) | 148.7M | 176 | 183.4 | 0.59 | 186,288 | 0.58 | 129 | 17.1 | ok | v22_dense130M |
| mamba2_ck256 (!) | 148.7M | 168 | 192.0 | 0.61 | 194,932 | 0.60 | 129 | 17.1 | ok | v22_dense130M |
| gdn_native (!) | 129.1M | 134 | 205.7 | 0.66 | 244,355 | 0.76 | 110 | 16.0 | ok | v22_dense130M |
| mlstm (!) | 129.1M | 112 | 246.7 | 0.79 | 293,095 | 0.91 | 110 | 14.3 | ok | v22_dense130M |
| mlstm_ck64 (!) | 129.1M | 114 | 240.2 | 0.77 | 286,308 | 0.89 | 110 | 14.3 | ok | v22_dense130M |
| mlstm_ck256 | 129.1M | 110 | 251.9 | 0.81 | 297,485 | 0.92 | 111 | 14.3 | ok | v22_dense130M |
| tri_gdn_e4_cute (!) | 126.0M | 138 | 212.7 | 0.68 | 237,106 | 0.74 | 118 | 21.5 | ok | v22_dense130M |
| tri_kda_e4 (!) | 127.8M | 185 | 160.4 | 0.51 | 176,742 | 0.55 | 119 | 25.4 | ok | v22_dense130M |
| tri_ckda_e4 (!) | 127.8M | 187 | 158.9 | 0.51 | 175,043 | 0.54 | 119 | 25.6 | ok | v22_dense130M |
| power (!) | 123.7M | 200 | 188.2 | 0.60 | 163,840 | 0.51 | 151 | 12.2 | ok | v27_dense130M |
| power_ck64 (!) | 123.7M | 220 | 171.4 | 0.55 | 149,149 | 0.46 | 151 | 7.4 | ok | v22_dense130M |
| power_ck128 (!) | 123.7M | 219 | 171.9 | 0.55 | 149,626 | 0.46 | 151 | 7.4 | ok | v22_dense130M |
| shortconv | 151.9M | 123 | 302.6 | 0.97 | 265,866 | 0.82 | 149 | 16.9 | ok | v22_dense130M |
| canon (!) | 124.0M | 136 | 234.1 | 0.75 | 241,830 | 0.75 | 127 | 15.6 | ok | v22_dense130M |
| engram (!) | 192.3M | 128 | 249.2 | 0.80 | 255,004 | 0.79 | 128 | 15.4 | ok | v22_dense130M |
| mhc4 (!) | 125.4M | 173 | 183.4 | 0.59 | 189,520 | 0.59 | 127 | 19.2 | ok | v22_dense130M |
| attn_res (!) | 123.6M | 130 | 244.0 | 0.78 | 252,062 | 0.78 | 127 | 14.8 | ok | v22_dense130M |
| postnorm | 123.6M | 107 | 297.6 | 0.95 | 307,392 | 0.95 | 127 | 14.5 | ok | v22_dense130M |
| ngpt | 123.6M | 120 | 263.3 | 0.84 | 272,159 | 0.84 | 127 | 13.9 | ok | v22_dense130M |
| ple128 | 203.2M | 118 | 273.4 | 0.88 | 278,403 | 0.86 | 129 | 15.4 | ok | v22_dense130M |
| value_emb (!) | 587.2M | 133 | 239.0 | 0.77 | 246,933 | 0.77 | 127 | 17.5 | ok | v22_dense130M |
| muon | 123.6M | 108 | 294.8 | 0.94 | 304,535 | 0.94 | 127 | 13.3 | ok | v22_dense130M |
| muown | 123.6M | 111 | 286.0 | 0.92 | 295,473 | 0.92 | 127 | 13.3 | ok | v22_dense130M |
| adam_precaware | 123.6M | 102 | 310.4 | 0.99 | 320,626 | 0.99 | 127 | 13.3 | ok | v22_dense130M |
| adam_offload | 123.6M | 115 | 276.6 | 0.89 | 285,809 | 0.89 | 127 | 13.3 | ok | v22_dense130M |
| muown_offload | 123.6M | 112 | 282.4 | 0.90 | 291,790 | 0.90 | 127 | 13.3 | ok | v22_dense130M |
| dense_fineweb | 123.6M | 103 | 307.8 | 0.99 | 318,136 | 0.99 | 127 | 13.4 | ok | v26_dense130M |
| dense_fineweb_mbs16 | 123.6M | 196 | 323.5 | 1.04 | 334,197 | 1.04 | 254 | 24.3 | ok | v26_dense130M |
| power_hd128 | 123.6M | 167 | 427.3 | 1.37 | 196,098 | 0.61 | 286 | 12.2 | ok | v27_dense130M |
| power_chunked (!) | 123.7M | 241 | 156.4 | 0.50 | 136,080 | 0.42 | 151 | 12.5 | ok | v27_dense130M |
| power_ck256 (!) | 123.7M | 240 | 157.2 | 0.50 | 136,790 | 0.42 | 151 | 12.5 | ok | v2ck_dense130M |
| base_liger_teact | 123.6M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  | v4base_dense130M |

## moe50M500M: latest per arm

reference: `moe`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status | suite |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| nemo_flex_tuned | 166.1M | 163 | 127.9 | 1.01 | 201,154 | 1.01 | 83 | 14.7 | ok | v13moe_moe50M500M |
| conn8 | 166.1M | 165 | 126.1 | 0.99 | 198,293 | 0.99 | 83 | 14.7 | ok | v22_moe50M500M |
| moe | 166.1M | 164 | 127.0 | 1.00 | 199,683 | 1.00 | 83 | 14.7 | ok | v22_moe50M500M |
| cutlass_gg | 166.1M | 163 | 127.5 | 1.00 | 200,477 | 1.00 | 83 | 14.7 | ok | v22_moe50M500M |
| nemo | 166.1M | 176 | 118.5 | 0.93 | 186,394 | 0.93 | 83 | 14.7 | ok | v22_moe50M500M |
| router_fusion | 166.1M | 162 | 128.8 | 1.01 | 202,584 | 1.01 | 83 | 14.8 | ok | v22_moe50M500M |
| tuned | 166.1M | 168 | 124.0 | 0.98 | 194,990 | 0.98 | 83 | 14.8 | ok | v13moe_moe50M500M |
| nemo_flex | 166.1M | 169 | 123.1 | 0.97 | 193,550 | 0.97 | 83 | 14.6 | ok | v22_moe50M500M |
| a2a_overlap | 166.1M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  | v22_moe50M500M |
| nemo_flex_nocg | 166.1M | 181 | 115.2 | 0.91 | 181,139 | 0.91 | 83 | 15.7 | ok | v22_moe50M500M |
| gc10 | 166.1M | 204 | 105.8 | 0.83 | 160,431 | 0.80 | 86 | 14.7 | ok | v22_moe50M500M |
| cublas_gg | 166.1M | 174 | 119.5 | 0.94 | 187,998 | 0.94 | 83 | 14.7 | ok | v14moe_moe50M500M |
| noshared (!) | 152.0M | 207 | 100.8 | 0.79 | 158,491 | 0.79 | 83 | 15.4 | ok | v22_moe50M500M |
| nemo_flex_rf | 166.1M | 168 | 124.5 | 0.98 | 195,630 | 0.98 | 83 | 14.7 | ok | v22_moe50M500M |
| muown | 166.1M | 202 | 103.0 | 0.81 | 162,057 | 0.81 | 83 | 14.7 | ok | v22_moe50M500M |
| nemo_flex_a2a | 166.1M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  | v22_moe50M500M |
| muown_offload (!) | 166.1M | 206 | 101.2 | 0.80 | 159,223 | 0.80 | 83 | 14.7 | ok | v22_moe50M500M |
| nemo_a2a | 166.1M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  | v22_moe50M500M |
| adam_offload (!) | 166.1M | 263 | 79.2 | 0.62 | 124,475 | 0.62 | 83 | 14.2 | ok | v22_moe50M500M |
| gdn3 (!) | 168.6M | 184 | 97.2 | 0.77 | 178,572 | 0.89 | 71 | 16.2 | ok | v22_moe50M500M |
| noshared_cublas (!) | 152.0M | 222 | 93.7 | 0.74 | 147,305 | 0.74 | 83 | 15.4 | ok | v15a_moe50M500M |
| auxfree | 166.1M | 165 | 126.5 | 1.00 | 198,895 | 1.00 | 83 | 14.7 | ok | v22_moe50M500M |
| latent | 112.7M | 174 | 107.5 | 0.85 | 188,647 | 0.94 | 75 | 13.6 | ok | v22_moe50M500M |
| nemo_cutlass | 166.1M | 164 | 127.2 | 1.00 | 200,049 | 1.00 | 83 | 14.7 | ok | v22_moe50M500M |
| nemo_cutlass_noshared (!) | 152.0M | 211 | 98.8 | 0.78 | 155,372 | 0.78 | 83 | 15.4 | ok | v22_moe50M500M |
| nemo_cutlass_latent | 112.7M | 150 | 124.2 | 0.98 | 218,017 | 1.09 | 75 | 13.6 | ok | v22_moe50M500M |
| noshared_cutlass (!) | 152.0M | 216 | 96.5 | 0.76 | 151,739 | 0.76 | 83 | 15.4 | ok | v22_moe50M500M |
| moe_nocg | 166.1M | 174 | 119.5 | 0.94 | 188,052 | 0.94 | 83 | 15.8 | ok | v22_moe50M500M |
| moe_ep1 | 505.9M | 175 | 119.3 | 0.94 | 187,728 | 0.94 | 83 | 16.3 | ok | v22_moe50M500M |
| dense50M | 80.8M | 79 | 263.0 | 2.07 | 413,737 | 2.07 | 83 | 11.5 | ok | v22_moe50M500M |
| zero16 | 166.2M | 178 | 116.9 | 0.92 | 183,832 | 0.92 | 83 | 14.7 | ok | v22_moe50M500M |

## moe120M1200M: latest per arm

reference: `moe`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status | suite |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| nemo | 369.0M | 256 | 121.3 | 0.90 | 128,100 | 0.90 | 124 | 18.7 | ok | v25_moe120M1200M |
| nemo_cutlass | 369.0M | 229 | 135.2 | 1.00 | 142,842 | 1.00 | 124 | 18.7 | ok | v25_moe120M1200M |
| moe | 369.0M | 230 | 134.6 | 1.00 | 142,253 | 1.00 | 124 | 18.7 | ok | v25_moe120M1200M |
| noshared_cutlass (!) | 351.3M | 325 | 95.5 | 0.71 | 100,856 | 0.71 | 124 | 19.8 | ok | v25_moe120M1200M |
| conn8 | 369.0M | 232 | 133.9 | 0.99 | 141,424 | 0.99 | 124 | 18.7 | ok | v25_moe120M1200M |
| nemo_flex | 369.0M | 235 | 131.9 | 0.98 | 139,349 | 0.98 | 124 | 18.6 | ok | v25_moe120M1200M |
| nemo_cutlass_noshared (!) | 351.3M | 344 | 90.2 | 0.67 | 95,297 | 0.67 | 124 | 19.8 | ok | v25_moe120M1200M |
| moe_nocg | 369.0M | 236 | 131.3 | 0.98 | 138,730 | 0.98 | 124 | 21.7 | ok | v25_moe120M1200M |
| cutlass_gg | 369.0M | 227 | 136.6 | 1.01 | 144,289 | 1.01 | 124 | 18.7 | ok | v25_moe120M1200M |
| nemo_flex_nocg | 369.0M | 249 | 124.8 | 0.93 | 131,784 | 0.93 | 124 | 21.6 | ok | v25_moe120M1200M |
| nemo_cutlass_latent | 234.5M | 240 | 120.5 | 0.90 | 136,562 | 0.96 | 116 | 16.6 | ok | v25_moe120M1200M |
| moe_ep1 | 1218.3M | 282 | 110.1 | 0.82 | 116,322 | 0.82 | 124 | 23.4 | ok | v25_moe120M1200M |
| nemo_flex_rf | 369.0M | 236 | 131.6 | 0.98 | 139,024 | 0.98 | 124 | 18.8 | ok | v25_moe120M1200M |
| dense120M | 120.0M | 100 | 309.0 | 2.29 | 326,375 | 2.29 | 124 | 13.1 | ok | v25_moe120M1200M |
| nemo_flex_a2a | 369.0M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  | v25_moe120M1200M |
| noshared (!) | 351.3M | 315 | 98.6 | 0.73 | 104,158 | 0.73 | 124 | 19.8 | ok | v25_moe120M1200M |
| nemo_a2a | 369.0M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  | v25_moe120M1200M |
| auxfree | 369.0M | 258 | 120.3 | 0.89 | 127,106 | 0.89 | 124 | 18.7 | ok | v25_moe120M1200M |
| latent | 234.5M | 233 | 124.2 | 0.92 | 140,665 | 0.99 | 116 | 16.6 | ok | v25_moe120M1200M |
| zero16 (!) | 369.1M | 288 | 107.6 | 0.80 | 113,699 | 0.80 | 124 | 21.4 | ok | v25_moe120M1200M |
| muown (!) | 369.0M | 316 | 98.1 | 0.73 | 103,581 | 0.73 | 124 | 18.7 | ok | v25_moe120M1200M |
| muown_offload (!) | 369.0M | 328 | 94.5 | 0.70 | 99,826 | 0.70 | 124 | 18.7 | ok | v25_moe120M1200M |
| adam_offload (!) | 369.0M | 349 | 88.9 | 0.66 | 93,931 | 0.66 | 124 | 17.6 | ok | v25_moe120M1200M |
| gdn3 (!) | 374.5M | 262 | 102.8 | 0.76 | 125,284 | 0.88 | 108 | 21.3 | ok | v25_moe120M1200M |
| router_fusion | 369.0M | 226 | 137.3 | 1.02 | 145,055 | 1.02 | 124 | 18.9 | ok | v25_moe120M1200M |
| a2a_overlap | 369.0M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  | v25_moe120M1200M |
| gc10 | 369.0M | 275 | 115.0 | 0.85 | 119,156 | 0.84 | 127 | 18.7 | ok | v25_moe120M1200M |

## Per suite

### ck2_resume_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| muown | 123.6M | 2610 | 12.1 | - | 12,553 | - | 126 | 13.5 | ok |
| adam_offload | 123.6M | 2711 | 11.7 | - | 12,085 | - | 127 | 13.3 | ok |
| muown_offload | 123.6M | 2620 | 12.1 | - | 12,508 | - | 127 | 13.5 | ok |

### ck2_save_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| muown | 123.6M | 111 | 285.6 | - | 295,074 | - | 127 | 13.3 | ok |
| adam_offload | 123.6M | 125 | 253.7 | - | 262,144 | - | 127 | 13.3 | ok |
| muown_offload | 123.6M | 113 | 280.8 | - | 289,982 | - | 127 | 13.3 | ok |

### ck_resume_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| muown | 123.6M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| adam_offload | 123.6M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| muown_offload | 123.6M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |

### ck_save_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| muown | 123.6M | 111 | 285.1 | - | 294,412 | - | 127 | 13.3 | ok |
| adam_offload | 123.6M | 124 | 256.2 | - | 264,792 | - | 127 | 13.3 | ok |
| muown_offload | 123.6M | 113 | 281.4 | - | 290,754 | - | 127 | 13.3 | ok |

### par2_cp2_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dense | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| xsa | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| gate | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| mla | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| gated_delta_net | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| mlstm | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| kda | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| complex_kda | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| signed_kda | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| tri_gdn_e4_cute | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| shortconv | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| canon | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| engram | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| pkm | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| mhc4 | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| routed_k2 | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| attn_res | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| ngpt | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| value_emb | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |

### par2_pp2_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dense | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| xsa | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| gate | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| mla | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| gated_delta_net | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| mlstm | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| kda | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| complex_kda | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| signed_kda | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| tri_gdn_e4_cute | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| shortconv | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| canon | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| engram | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| pkm | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| mhc4 | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| routed_k2 | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| attn_res | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| ngpt | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| value_emb | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |

### par2_ref_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dense | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| xsa | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| gate | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| mla | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| gated_delta_net | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| mlstm | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| kda | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| complex_kda | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| signed_kda | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| tri_gdn_e4_cute | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| shortconv | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| canon | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| engram | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| pkm | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| mhc4 | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| routed_k2 | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| attn_res | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| ngpt | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| value_emb | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |

### par2_tp2_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dense | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| xsa | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| gate | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| mla | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| gated_delta_net | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| mlstm | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| kda | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| complex_kda | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| signed_kda | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| tri_gdn_e4_cute | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| shortconv | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| canon | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| engram | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| pkm | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| mhc4 | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| routed_k2 | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| attn_res | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| ngpt | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| value_emb | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |

### par3_cp2_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dense | 123.6M | 210 | 151.3 | 1.00 | 156,261 | 1.00 | 127 | 14.0 | ok |
| xsa | 123.6M | 219 | 144.6 | 0.96 | 149,489 | 0.96 | 127 | 14.6 | ok |
| gate | 130.7M | 216 | 146.8 | 0.97 | 151,669 | 0.97 | 127 | 14.7 | ok |
| mla (!) | 116.5M | 206 | 90.8 | 0.60 | 158,875 | 1.02 | 75 | 13.9 | ok |
| gated_delta_net (!) | 129.1M | 253 | 109.0 | 0.72 | 129,364 | 0.83 | 110 | 15.6 | ok |
| mlstm (!) | 129.1M | 234 | 118.0 | 0.78 | 140,244 | 0.90 | 110 | 14.7 | ok |
| kda | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| complex_kda | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| signed_kda | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| tri_gdn_e4_cute | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| shortconv | 151.9M | 273 | 136.6 | 0.90 | 120,029 | 0.77 | 149 | 16.5 | ok |
| canon (!) | 124.0M | 467 | 67.9 | 0.45 | 70,137 | 0.45 | 127 | 20.0 | ok |
| engram | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| pkm (!) | 219.8M | 353 | 87.5 | 0.58 | 92,880 | 0.59 | 123 | 9.9 | ok |
| mhc4 (!) | 125.4M | 301 | 105.3 | 0.70 | 108,773 | 0.70 | 127 | 17.0 | ok |
| routed_k2 | 123.7M | 258 | 122.7 | 0.81 | 126,762 | 0.81 | 127 | 16.4 | ok |
| attn_res | 123.6M | 246 | 128.7 | 0.85 | 132,987 | 0.85 | 127 | 14.6 | ok |
| ngpt | 123.6M | 228 | 139.2 | 0.92 | 143,845 | 0.92 | 127 | 14.6 | ok |
| value_emb | 587.2M | 243 | 130.4 | 0.86 | 134,709 | 0.86 | 127 | 19.1 | ok |

### par3_pp2_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dense | 81.1M | 281 | 113.0 | 1.00 | 116,737 | 1.00 | 127 | 21.7 | ok |
| xsa | 81.1M | 297 | 106.9 | 0.95 | 110,442 | 0.95 | 127 | 22.3 | ok |
| gate | 84.6M | 292 | 108.8 | 0.96 | 112,412 | 0.96 | 127 | 22.3 | ok |
| mla (!) | 77.6M | 277 | 67.7 | 0.60 | 118,382 | 1.01 | 75 | 21.6 | ok |
| gated_delta_net (!) | 84.2M | 320 | 86.2 | 0.76 | 102,352 | 0.88 | 110 | 23.1 | ok |
| mlstm | 84.2M | 288 | 95.9 | 0.85 | 113,976 | 0.98 | 110 | 22.3 | ok |
| kda (!) | 82.2M | 349 | 77.0 | 0.68 | 93,905 | 0.80 | 107 | 22.9 | ok |
| complex_kda (!) | 84.7M | 344 | 80.6 | 0.71 | 95,200 | 0.82 | 111 | 23.8 | ok |
| signed_kda (!) | 84.7M | 350 | 79.1 | 0.70 | 93,516 | 0.80 | 111 | 23.8 | ok |
| tri_gdn_e4_cute (!) | 82.4M | 326 | 90.1 | 0.80 | 100,485 | 0.86 | 118 | 25.5 | ok |
| shortconv | 95.3M | 311 | 119.9 | 1.06 | 105,329 | 0.90 | 149 | 23.6 | ok |
| canon | 81.3M | 318 | 99.8 | 0.88 | 103,044 | 0.88 | 127 | 23.1 | ok |
| engram | 149.0M | 319 | 100.5 | 0.89 | 102,785 | 0.88 | 128 | 23.5 | ok |
| pkm (!) | 179.5M | 361 | 85.5 | 0.76 | 90,758 | 0.78 | 124 | 13.3 | ok |
| mhc4 (!) | 82.0M | 403 | 78.8 | 0.70 | 81,391 | 0.70 | 127 | 24.9 | ok |
| routed_k2 | 81.2M | 343 | 92.4 | 0.82 | 95,450 | 0.82 | 127 | 24.5 | ok |
| attn_res | 81.1M | 348 | 91.2 | 0.81 | 94,215 | 0.81 | 127 | 22.9 | ok |
| ngpt | 81.1M | 296 | 107.2 | 0.95 | 110,815 | 0.95 | 127 | 22.3 | ok |
| value_emb | 312.9M | 316 | 100.2 | 0.89 | 103,565 | 0.89 | 127 | 25.4 | ok |

### par3_ref_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dense | 123.6M | 163 | 194.4 | 1.00 | 200,907 | 1.00 | 127 | 26.9 | ok |
| xsa | 123.6M | 174 | 181.9 | 0.94 | 187,998 | 0.94 | 127 | 28.0 | ok |
| gate | 130.7M | 171 | 185.8 | 0.96 | 191,963 | 0.96 | 127 | 28.0 | ok |
| mla (!) | 116.5M | 159 | 117.5 | 0.60 | 205,571 | 1.02 | 75 | 26.6 | ok |
| gated_delta_net (!) | 129.1M | 191 | 144.5 | 0.74 | 171,605 | 0.85 | 110 | 29.9 | ok |
| mlstm | 129.1M | 167 | 164.7 | 0.85 | 195,747 | 0.97 | 110 | 28.2 | ok |
| kda (!) | 125.5M | 210 | 128.0 | 0.66 | 156,187 | 0.78 | 107 | 29.5 | ok |
| complex_kda (!) | 130.0M | 207 | 134.4 | 0.69 | 158,644 | 0.79 | 111 | 31.4 | ok |
| signed_kda (!) | 130.0M | 210 | 131.9 | 0.68 | 155,778 | 0.78 | 111 | 31.4 | ok |
| tri_gdn_e4_cute (!) | 126.0M | 195 | 151.1 | 0.78 | 168,386 | 0.84 | 118 | 35.4 | ok |
| shortconv | 151.9M | 184 | 202.4 | 1.04 | 177,845 | 0.89 | 149 | 30.5 | ok |
| canon | 124.0M | 189 | 168.1 | 0.86 | 173,605 | 0.86 | 127 | 29.7 | ok |
| engram | 192.3M | 190 | 168.8 | 0.87 | 172,645 | 0.86 | 128 | 29.3 | ok |
| pkm (!) | 219.8M | 237 | 130.1 | 0.67 | 138,000 | 0.69 | 124 | 16.5 | ok |
| mhc4 (!) | 125.4M | 242 | 131.1 | 0.67 | 135,461 | 0.67 | 127 | 32.7 | ok |
| routed_k2 | 123.7M | 201 | 157.5 | 0.81 | 162,701 | 0.81 | 127 | 31.5 | ok |
| attn_res | 123.6M | 197 | 160.9 | 0.83 | 166,208 | 0.83 | 127 | 28.0 | ok |
| ngpt | 123.6M | 176 | 180.2 | 0.93 | 186,235 | 0.93 | 127 | 28.0 | ok |
| value_emb | 587.2M | 192 | 164.9 | 0.85 | 170,268 | 0.85 | 127 | 33.0 | ok |

### par3_tp2_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dense | 61.9M | 229 | 138.4 | 1.00 | 142,904 | 1.00 | 127 | 13.7 | ok |
| xsa | 61.9M | 240 | 132.0 | 0.95 | 136,306 | 0.95 | 127 | 14.3 | ok |
| gate | 65.4M | 235 | 134.9 | 0.97 | 139,261 | 0.97 | 127 | 14.3 | ok |
| mla (!) | 61.3M | 225 | 83.2 | 0.60 | 145,442 | 1.02 | 75 | 13.6 | ok |
| gated_delta_net (!) | 64.6M | 265 | 104.1 | 0.75 | 123,560 | 0.86 | 110 | 15.2 | ok |
| mlstm | 64.6M | 247 | 111.9 | 0.81 | 132,879 | 0.93 | 110 | 14.4 | ok |
| kda (!) | 74.4M | 344 | 78.0 | 0.56 | 95,118 | 0.67 | 107 | 18.0 | ok |
| complex_kda (!) | 78.8M | 338 | 82.3 | 0.59 | 97,090 | 0.68 | 111 | 19.9 | ok |
| signed_kda (!) | 78.8M | 340 | 81.7 | 0.59 | 96,490 | 0.68 | 111 | 19.9 | ok |
| tri_gdn_e4_cute (!) | 74.9M | 314 | 93.6 | 0.68 | 104,274 | 0.73 | 118 | 23.9 | ok |
| shortconv | 76.0M | 282 | 132.4 | 0.96 | 116,281 | 0.81 | 149 | 15.6 | ok |
| canon | 62.1M | 263 | 120.6 | 0.87 | 124,546 | 0.87 | 127 | 15.1 | ok |
| engram | 130.6M | 278 | 115.1 | 0.83 | 117,743 | 0.82 | 128 | 16.2 | ok |
| pkm (!) | 162.8M | 373 | 83.0 | 0.60 | 87,932 | 0.62 | 124 | 10.0 | ok |
| mhc4 (!) | 63.6M | 323 | 98.3 | 0.71 | 101,512 | 0.71 | 127 | 16.7 | ok |
| routed_k2 | 62.0M | 275 | 115.2 | 0.83 | 119,005 | 0.83 | 127 | 16.1 | ok |
| attn_res | 61.9M | 267 | 118.9 | 0.86 | 122,796 | 0.86 | 127 | 14.3 | ok |
| ngpt | 61.9M | 245 | 129.6 | 0.94 | 133,802 | 0.94 | 127 | 14.3 | ok |
| value_emb | 294.2M | 256 | 123.8 | 0.89 | 127,850 | 0.89 | 127 | 17.4 | ok |

### par_cp2_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dense | 123.6M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| xsa | 123.6M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| gate | 130.7M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| mla | 116.5M | 138 | 135.8 | - | 237,621 | - | 75 | 7.5 | ok |
| gated_delta_net | 129.1M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| mlstm | 129.1M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| kda | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| complex_kda | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| signed_kda | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| tri_gdn_e4_cute | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| shortconv | 151.9M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| canon | 124.0M | 411 | 77.1 | - | 79,689 | - | 127 | 13.6 | ok |
| engram | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| pkm | 219.8M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| mhc4 | 125.4M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| routed_k2 | 123.7M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| attn_res | 123.6M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| ngpt | 123.6M | 158 | 200.9 | - | 207,655 | - | 127 | 8.2 | ok |
| value_emb | 587.2M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |

### par_pp2_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dense | 81.1M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| xsa | 81.1M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| gate | 84.6M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| mla | 77.6M | - | - | - | - | - | - | 11.1 | FAIL: ChildFailedError:  |
| gated_delta_net | 84.2M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| mlstm | 84.2M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| kda | 82.2M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| complex_kda | 84.7M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| signed_kda | 84.7M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| tri_gdn_e4_cute | 82.4M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| shortconv | 95.3M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| canon | 81.3M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| engram | 149.0M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| pkm | 179.5M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| mhc4 | 82.0M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| routed_k2 | 81.2M | - | - | - | - | - | - | 16.4 | FAIL: ChildFailedError:  |
| attn_res | 81.1M | - | - | - | - | - | - | 13.4 | FAIL: ChildFailedError:  |
| ngpt | 81.1M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| value_emb | 312.9M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |

### par_ref_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dense | 123.6M | 102 | 310.8 | 1.00 | 321,098 | 1.00 | 127 | 13.4 | ok |
| xsa | 123.6M | 112 | 283.9 | 0.91 | 293,357 | 0.91 | 127 | 13.9 | ok |
| gate | 130.7M | 109 | 290.1 | 0.93 | 299,799 | 0.93 | 127 | 15.1 | ok |
| mla (!) | 116.5M | 106 | 176.6 | 0.57 | 308,841 | 0.96 | 75 | 12.5 | ok |
| gated_delta_net (!) | 129.1M | 134 | 205.2 | 0.66 | 243,628 | 0.76 | 110 | 16.0 | ok |
| mlstm (!) | 129.1M | 113 | 244.8 | 0.79 | 290,883 | 0.91 | 110 | 14.3 | ok |
| kda (!) | 125.5M | 154 | 174.4 | 0.56 | 212,779 | 0.66 | 107 | 15.6 | ok |
| complex_kda | 130.0M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| signed_kda | 130.0M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| tri_gdn_e4_cute (!) | 126.0M | 148 | 197.9 | 0.64 | 220,660 | 0.69 | 118 | 21.5 | ok |
| shortconv | 151.9M | 123 | 304.0 | 0.98 | 267,058 | 0.83 | 149 | 16.9 | ok |
| canon (!) | 124.0M | 136 | 234.1 | 0.75 | 241,830 | 0.75 | 127 | 15.6 | ok |
| engram | 192.3M | 129 | 248.9 | 0.80 | 254,608 | 0.79 | 128 | 15.4 | ok |
| pkm (!) | 219.8M | 167 | 185.4 | 0.60 | 196,687 | 0.61 | 124 | 9.7 | ok |
| mhc4 (!) | 125.4M | 176 | 179.7 | 0.58 | 185,654 | 0.58 | 127 | 19.2 | ok |
| routed_k2 (!) | 123.7M | 140 | 226.5 | 0.73 | 234,057 | 0.73 | 127 | 18.3 | ok |
| attn_res (!) | 123.6M | 136 | 232.9 | 0.75 | 240,587 | 0.75 | 127 | 14.8 | ok |
| ngpt | 123.6M | 122 | 259.9 | 0.84 | 268,480 | 0.84 | 127 | 13.9 | ok |
| value_emb (!) | 587.2M | 133 | 237.9 | 0.77 | 245,821 | 0.77 | 127 | 17.5 | ok |

### par_tp2_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dense | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| xsa | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| gate | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| mla | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| gated_delta_net | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| mlstm | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| kda | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| complex_kda | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| signed_kda | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| tri_gdn_e4_cute | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| shortconv | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| canon | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| engram | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| pkm | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| mhc4 | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| routed_k2 | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| attn_res | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| ngpt | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| value_emb | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |

### v10_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dense | 123.6M | 102 | 311.2 | 1.00 | 321,570 | 1.00 | 127 | 13.4 | ok |
| pkm (!) | 219.8M | 225 | 137.0 | 0.44 | 145,345 | 0.45 | 124 | 10.4 | ok |

### v11routed_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| routed_k2 | 123.7M | 141 | 225.6 | - | 233,141 | - | 127 | 18.3 | ok |
| routed_k2_mix | 124.0M | 162 | 195.2 | - | 201,711 | - | 127 | 18.5 | ok |
| routed_k2_rev | 123.7M | 150 | 210.9 | - | 217,872 | - | 127 | 15.1 | ok |
| routed_k2_cg | 123.7M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| routed_k2_mix_cg | 124.0M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| routed_k2_rev_cg | 123.7M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |

### v12kda_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dense | 123.6M | 102 | 309.5 | 1.00 | 319,688 | 1.00 | 127 | 13.4 | ok |
| gated_delta_net (!) | 129.1M | 134 | 205.3 | 0.66 | 243,810 | 0.76 | 110 | 16.0 | ok |
| kda (!) | 125.5M | 152 | 176.9 | 0.57 | 215,721 | 0.67 | 107 | 15.6 | ok |
| complex_kda (!) | 125.5M | 180 | 149.4 | 0.48 | 182,146 | 0.57 | 108 | 19.6 | ok |
| signed_kda (!) | 130.0M | 154 | 179.8 | 0.58 | 212,366 | 0.66 | 111 | 18.2 | ok |
| kda_xsa (!) | 125.5M | 160 | 167.4 | 0.54 | 204,162 | 0.64 | 107 | 16.5 | ok |
| kda_hd128 (!) | 127.3M | 154 | 178.4 | 0.58 | 212,228 | 0.66 | 110 | 15.2 | ok |

### v12pkm_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pkm | 219.8M | 211 | 146.7 | - | 155,594 | - | 124 | 9.8 | ok |

### v12routed_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| routed_k2_cg | 123.7M | 144 | 220.0 | - | 227,240 | - | 127 | 18.1 | ok |
| routed_k2_mix_cg | 124.0M | 162 | 196.3 | - | 202,835 | - | 127 | 18.3 | ok |
| routed_k2_rev_cg | 123.7M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |

### v17_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dense | 123.6M | 101 | 312.9 | 1.00 | 323,315 | 1.00 | 127 | 13.4 | ok |
| dense_gpt | 123.6M | 103 | 309.1 | 0.99 | 319,376 | 0.99 | 127 | 13.4 | ok |
| dense_nocg | 123.6M | 103 | 308.8 | 0.99 | 319,065 | 0.99 | 127 | 13.6 | ok |
| base_liger | 123.6M | 101 | 312.8 | 1.00 | 323,156 | 1.00 | 127 | 13.4 | ok |
| base_ce_te | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| base_flash | 123.6M | 119 | 267.5 | 0.85 | 276,290 | 0.85 | 127 | 13.9 | ok |
| base_graph_full | 123.6M | 101 | 313.1 | 1.00 | 323,475 | 1.00 | 127 | 13.6 | ok |
| base_liger_mbs16 | 123.6M | 197 | 322.4 | 1.03 | 333,093 | 1.03 | 254 | 24.3 | ok |
| base_liger_nocg | 123.6M | 102 | 310.8 | 0.99 | 321,098 | 0.99 | 127 | 13.6 | ok |
| base_liger_nodistopt | 123.6M | 105 | 301.0 | 0.96 | 310,892 | 0.96 | 127 | 13.7 | ok |
| base_liger_nomask | 123.6M | 102 | 310.1 | 0.99 | 320,469 | 0.99 | 127 | 13.4 | ok |
| base_liger_nomask_mbs16_nocg | 123.6M | 195 | 325.4 | 1.04 | 336,254 | 1.04 | 254 | 24.5 | ok |
| base_liger_noqkln | 123.6M | 101 | 312.9 | 1.00 | 323,315 | 1.00 | 127 | 11.6 | ok |
| base_liger_mbs4 | 123.6M | 105 | 301.8 | 0.96 | 311,779 | 0.96 | 127 | 7.9 | ok |
| base_qknr | 123.6M | 101 | 312.8 | 1.00 | 323,156 | 1.00 | 127 | 13.4 | ok |
| base_resnorm | 123.6M | 102 | 312.6 | 1.00 | 322,837 | 1.00 | 127 | 13.4 | ok |
| base_opfuser | 123.6M | 102 | 310.1 | 0.99 | 320,313 | 0.99 | 127 | 13.4 | ok |
| base_qknr_resnorm | 123.6M | 102 | 311.1 | 0.99 | 321,412 | 0.99 | 127 | 13.4 | ok |
| base_best | 123.6M | 102 | 310.4 | 0.99 | 320,626 | 0.99 | 127 | 13.4 | ok |
| base_best_swiglu | 123.6M | 102 | 310.5 | 0.99 | 320,783 | 0.99 | 127 | 13.4 | ok |
| base_best_mbs16_nocg | 123.6M | 196 | 322.9 | 1.03 | 333,601 | 1.03 | 254 | 24.5 | ok |
| base_best_hd128 | 123.6M | 98 | 324.8 | 1.04 | 335,394 | 1.04 | 127 | 13.4 | ok |
| base_fp8 | 123.6M | 107 | 296.1 | 0.95 | 305,957 | 0.95 | 127 | 12.5 | ok |
| nope | 123.6M | 107 | 296.6 | 0.95 | 306,529 | 0.95 | 127 | 12.8 | ok |
| gqa4 | 123.6M | 102 | 310.1 | 0.99 | 320,313 | 0.99 | 127 | 13.4 | ok |
| xsa | 123.6M | 112 | 282.9 | 0.90 | 292,310 | 0.90 | 127 | 13.9 | ok |
| gate | 130.7M | 110 | 288.8 | 0.92 | 298,298 | 0.92 | 127 | 15.1 | ok |
| xsa_gate | 130.7M | 118 | 268.8 | 0.86 | 277,695 | 0.86 | 127 | 15.1 | ok |
| swa3 | 123.6M | 95 | 289.9 | 0.93 | 344,926 | 1.07 | 110 | 13.4 | ok |
| mla (!) | 116.5M | 106 | 177.0 | 0.57 | 309,716 | 0.96 | 75 | 12.5 | ok |
| mla_gqa_xsa_gate (!) | 116.9M | 115 | 162.9 | 0.52 | 284,939 | 0.88 | 75 | 13.2 | ok |
| mamba2 (!) | 148.7M | 165 | 195.5 | 0.62 | 198,534 | 0.61 | 129 | 16.7 | ok |
| mamba2_ck64 (!) | 148.7M | 175 | 184.1 | 0.59 | 186,925 | 0.58 | 129 | 17.1 | ok |
| mamba2_ck256 (!) | 148.7M | 168 | 192.2 | 0.61 | 195,164 | 0.60 | 129 | 17.1 | ok |
| gdn_native (!) | 129.1M | 134 | 205.1 | 0.66 | 243,628 | 0.75 | 110 | 16.0 | ok |
| gated_delta_net (!) | 129.1M | 134 | 205.9 | 0.66 | 244,537 | 0.76 | 110 | 16.0 | ok |
| mlstm (!) | 129.1M | 113 | 244.8 | 0.78 | 291,012 | 0.90 | 110 | 14.3 | ok |
| kda (!) | 125.5M | 154 | 174.2 | 0.56 | 212,503 | 0.66 | 107 | 15.6 | ok |
| complex_kda (!) | 125.5M | 182 | 147.9 | 0.47 | 180,341 | 0.56 | 107 | 19.6 | ok |
| signed_kda (!) | 130.0M | 154 | 179.6 | 0.57 | 212,159 | 0.66 | 111 | 17.7 | ok |
| kda_xsa (!) | 125.5M | 161 | 166.6 | 0.53 | 203,275 | 0.63 | 107 | 16.0 | ok |
| mlstm_ck64 (!) | 129.1M | 116 | 237.4 | 0.76 | 282,971 | 0.88 | 110 | 14.3 | ok |
| mlstm_ck256 (!) | 129.1M | 111 | 249.8 | 0.80 | 294,941 | 0.91 | 111 | 14.3 | ok |
| kda_hd128 (!) | 127.3M | 156 | 177.0 | 0.57 | 210,456 | 0.65 | 110 | 15.2 | ok |
| tri_gdn_e4_cute (!) | 126.0M | 148 | 198.0 | 0.63 | 220,734 | 0.68 | 118 | 21.5 | ok |
| tri_kda_e4 (!) | 127.8M | 207 | 143.6 | 0.46 | 158,223 | 0.49 | 119 | 25.6 | ok |
| tri_ckda_e4 (!) | 127.8M | 228 | 130.5 | 0.42 | 143,782 | 0.44 | 119 | 26.7 | ok |
| power (!) | 123.7M | 259 | 145.2 | 0.46 | 126,322 | 0.39 | 151 | 12.5 | ok |
| power_ck64 (!) | 123.7M | 406 | 92.8 | 0.30 | 80,749 | 0.25 | 151 | 25.6 | ok |
| power_ck128 (!) | 123.7M | 295 | 127.4 | 0.41 | 110,909 | 0.34 | 151 | 16.9 | ok |
| shortconv | 151.9M | 123 | 302.9 | 0.97 | 266,190 | 0.82 | 149 | 16.9 | ok |
| canon (!) | 124.0M | 135 | 234.5 | 0.75 | 242,277 | 0.75 | 127 | 15.6 | ok |
| engram (!) | 192.3M | 129 | 249.1 | 0.80 | 254,806 | 0.79 | 128 | 15.4 | ok |
| pkm (!) | 219.8M | 207 | 149.1 | 0.48 | 158,223 | 0.49 | 124 | 9.8 | ok |
| mhc4 (!) | 125.4M | 176 | 180.0 | 0.58 | 185,970 | 0.58 | 127 | 19.2 | ok |
| routed_k2 (!) | 123.7M | 140 | 226.7 | 0.72 | 234,224 | 0.72 | 127 | 18.3 | ok |
| routed_k2_mix (!) | 124.0M | 162 | 196.2 | 0.63 | 202,710 | 0.63 | 127 | 18.5 | ok |
| routed_k2_rev (!) | 123.7M | 152 | 209.4 | 0.67 | 216,290 | 0.67 | 127 | 15.1 | ok |
| routed_k2_cg | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| routed_k2_mix_cg | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| routed_k2_rev_cg | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| attn_res (!) | 123.6M | 135 | 235.4 | 0.75 | 243,267 | 0.75 | 127 | 14.8 | ok |
| postnorm | 123.6M | 107 | 297.5 | 0.95 | 307,392 | 0.95 | 127 | 14.5 | ok |
| ngpt | 123.6M | 122 | 259.6 | 0.83 | 268,151 | 0.83 | 127 | 13.9 | ok |
| ple128 | 203.2M | 117 | 275.2 | 0.88 | 280,308 | 0.87 | 129 | 15.4 | ok |
| value_emb (!) | 587.2M | 134 | 237.2 | 0.76 | 245,178 | 0.76 | 127 | 17.5 | ok |
| muon | 123.6M | 109 | 290.1 | 0.93 | 299,799 | 0.93 | 127 | 13.3 | ok |
| muown | 123.6M | 111 | 285.8 | 0.91 | 295,207 | 0.91 | 127 | 13.3 | ok |
| adam_precaware | 123.6M | 102 | 310.4 | 0.99 | 320,626 | 0.99 | 127 | 13.3 | ok |
| adam_offload | 123.6M | 127 | 250.5 | 0.80 | 258,831 | 0.80 | 127 | 13.3 | ok |
| muown_offload | 123.6M | 113 | 281.3 | 0.90 | 290,496 | 0.90 | 127 | 13.3 | ok |

### v18ckda_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| complex_kda | 125.5M | 155 | 173.4 | - | 211,406 | - | 108 | 17.5 | ok |
| kda | 125.5M | 154 | 174.0 | - | 212,228 | - | 107 | 15.6 | ok |
| signed_kda | 130.0M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |

### v19tri2_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| complex_kda | 130.0M | 151 | 183.6 | - | 216,791 | - | 111 | 17.5 | ok |
| kda | 125.5M | 154 | 174.0 | - | 212,228 | - | 107 | 15.6 | ok |
| pkm | 219.8M | 168 | 183.6 | - | 194,874 | - | 124 | 9.7 | ok |
| signed_kda | 130.0M | 151 | 183.8 | - | 217,222 | - | 111 | 17.5 | ok |
| tri_ckda_e4 | 127.8M | 201 | 147.9 | - | 162,944 | - | 119 | 24.6 | ok |
| tri_gdn_e4_cute | 126.0M | 149 | 197.2 | - | 219,919 | - | 118 | 21.5 | ok |
| tri_kda_e4 | 127.8M | 198 | 150.4 | - | 165,746 | - | 119 | 24.3 | ok |

### v19tri_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| complex_kda | - | - | - | - | - | - | - | - | no log |
| kda | - | - | - | - | - | - | - | - | no log |
| signed_kda | - | - | - | - | - | - | - | - | no log |
| tri_ckda_e4 | - | - | - | - | - | - | - | - | no log |
| tri_gdn_e4_cute | - | - | - | - | - | - | - | - | no log |
| tri_kda_e4 | - | - | - | - | - | - | - | - | no log |

### v1_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dense | 123.6M | 170 | 187.1 | 1.00 | 193,208 | 1.00 | 127 | 26.6 | ok |
| dense_gpt | 123.6M | 167 | 189.6 | 1.01 | 195,922 | 1.01 | 127 | 26.6 | ok |
| dense_nocg | 123.6M | 165 | 192.5 | 1.03 | 198,895 | 1.03 | 127 | 27.0 | ok |
| nope | 123.6M | 166 | 191.6 | 1.02 | 197,874 | 1.02 | 127 | 26.6 | ok |
| gqa4 | 123.6M | 170 | 187.1 | 1.00 | 193,208 | 1.00 | 127 | 26.6 | ok |
| xsa | 123.6M | 180 | 176.1 | 0.94 | 181,842 | 0.94 | 127 | 27.8 | ok |
| gate | 130.7M | 191 | 166.4 | 0.89 | 171,920 | 0.89 | 127 | 28.4 | ok |
| xsa_gate | 130.7M | 186 | 170.4 | 0.91 | 176,077 | 0.91 | 127 | 28.4 | ok |
| swa3 | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| mla | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| mla_gqa_xsa_gate | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| mamba2 (!) | 148.7M | 235 | 137.4 | 0.73 | 139,557 | 0.72 | 129 | 29.6 | ok |
| gdn_native | 129.1M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| gated_delta_net | 129.1M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| mlstm | 129.1M | 174 | 158.9 | 0.85 | 188,810 | 0.98 | 110 | 28.0 | ok |
| kda (!) | 125.5M | 215 | 124.8 | 0.67 | 152,162 | 0.79 | 107 | 29.3 | ok |
| signed_kda (!) | 130.0M | 239 | 116.1 | 0.62 | 137,219 | 0.71 | 111 | 33.3 | ok |
| complex_kda (!) | 125.5M | 243 | 110.7 | 0.59 | 134,987 | 0.70 | 107 | 33.3 | ok |
| kda_xsa (!) | 125.5M | 222 | 120.7 | 0.65 | 147,272 | 0.76 | 107 | 29.7 | ok |
| kda_hd128 (!) | 127.3M | 217 | 127.2 | 0.68 | 151,214 | 0.78 | 110 | 28.9 | ok |
| tri_gdn_e4_cute (!) | 126.0M | 244 | 120.6 | 0.64 | 134,405 | 0.70 | 118 | 34.4 | ok |
| tri_kda_e4 (!) | 127.8M | 464 | 64.1 | 0.34 | 70,644 | 0.37 | 119 | 43.1 | ok |
| tri_ckda_e4 | 127.8M | - | - | - | - | - | - | 44.8 | FAIL: ChildFailedError:  |
| power | 123.7M | - | - | - | - | - | - | 44.8 | FAIL: ChildFailedError:  |
| shortconv | 151.9M | 223 | 167.2 | 0.89 | 146,942 | 0.76 | 149 | 31.3 | ok |
| canon (!) | 124.0M | 213 | 148.8 | 0.80 | 153,660 | 0.80 | 127 | 29.4 | ok |
| engram | 192.3M | 196 | 163.6 | 0.87 | 167,354 | 0.87 | 128 | 28.6 | ok |
| pkm | 219.8M | - | - | - | - | - | - | 40.3 | FAIL: ChildFailedError:  |
| mhc4 (!) | 125.4M | 256 | 123.8 | 0.66 | 127,850 | 0.66 | 127 | 32.5 | ok |
| routed_k2 | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| routed_k2_mix | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| routed_k2_rev | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| attn_res | 123.6M | - | - | - | - | - | - | 27.8 | FAIL: ChildFailedError:  |
| postnorm | 123.6M | 174 | 182.2 | 0.97 | 188,322 | 0.97 | 127 | 27.8 | ok |
| ngpt (!) | 123.6M | 220 | 144.1 | 0.77 | 148,878 | 0.77 | 127 | 30.0 | ok |
| ple128 | 203.2M | 185 | 173.6 | 0.93 | 176,742 | 0.91 | 129 | 28.7 | ok |
| value_emb | 587.2M | 200 | 158.8 | 0.85 | 164,045 | 0.85 | 127 | 31.9 | ok |
| muon | 123.6M | 178 | 178.6 | 0.95 | 184,505 | 0.95 | 127 | 26.6 | ok |
| muown | 123.6M | 180 | 175.8 | 0.94 | 181,641 | 0.94 | 127 | 26.6 | ok |

### v1b_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dense | 123.6M | 170 | 186.3 | 1.00 | 192,413 | 1.00 | 127 | 26.6 | ok |
| power (!) | 123.7M | 338 | 111.5 | 0.60 | 97,033 | 0.50 | 151 | 23.0 | ok |
| muown | 123.6M | 181 | 175.4 | 0.94 | 181,239 | 0.94 | 127 | 26.6 | ok |
| adam_precaware | 123.6M | 170 | 186.6 | 1.00 | 192,753 | 1.00 | 127 | 26.5 | ok |
| adam_offload | 123.6M | 201 | 157.6 | 0.85 | 162,782 | 0.85 | 127 | 26.5 | ok |
| muown_offload | 123.6M | 183 | 173.3 | 0.93 | 179,060 | 0.93 | 127 | 26.6 | ok |

### v1c_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| swa3 | 123.6M | 164 | 168.4 | - | 200,293 | - | 110 | 26.6 | ok |
| mla | 116.5M | 165 | 113.6 | - | 198,835 | - | 75 | 26.4 | ok |
| mla_gqa_xsa_gate | 116.9M | 174 | 107.6 | - | 188,214 | - | 75 | 27.0 | ok |
| gdn_native | 129.1M | 198 | 139.0 | - | 165,078 | - | 110 | 30.1 | ok |
| gated_delta_net | 129.1M | 198 | 139.0 | - | 165,078 | - | 110 | 30.1 | ok |

### v1dbg_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dense | 123.6M | 171 | 185.8 | 1.00 | 191,963 | 1.00 | 127 | 26.6 | ok |
| dense_nocg | 123.6M | 166 | 191.4 | 1.03 | 197,755 | 1.03 | 127 | 27.0 | ok |
| kda (!) | 125.5M | 216 | 124.5 | 0.67 | 151,844 | 0.79 | 107 | 29.3 | ok |
| mamba2 (!) | 148.7M | 236 | 136.9 | 0.74 | 139,024 | 0.72 | 129 | 29.6 | ok |
| power | 123.7M | - | - | - | - | - | - | 44.8 | FAIL: ChildFailedError:  |
| tri_gdn_e4_cute (!) | 126.0M | 245 | 120.2 | 0.65 | 133,966 | 0.70 | 118 | 34.4 | ok |

### v20tri_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| tri_ckda_e4 | 127.8M | 192 | 155.3 | - | 171,112 | - | 119 | 24.6 | ok |
| tri_gdn_e4_cute | 126.0M | 139 | 211.0 | - | 235,233 | - | 118 | 21.5 | ok |
| tri_kda_e4 | 127.8M | 188 | 158.3 | - | 174,391 | - | 119 | 24.4 | ok |

### v21_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| kda | 125.5M | 154 | 173.9 | - | 212,091 | - | 107 | 15.6 | ok |
| complex_kda | 130.0M | 151 | 183.4 | - | 216,576 | - | 111 | 17.5 | ok |
| signed_kda | 130.0M | 150 | 184.9 | - | 218,453 | - | 111 | 17.5 | ok |
| power | 123.7M | 252 | 149.6 | - | 130,238 | - | 151 | 12.5 | ok |

### v22_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dense | 123.6M | 102 | 310.6 | 1.00 | 320,940 | 1.00 | 127 | 13.4 | ok |
| dense_gpt | 123.6M | 102 | 312.2 | 1.01 | 322,520 | 1.00 | 127 | 13.4 | ok |
| dense_nocg | 123.6M | 102 | 309.6 | 1.00 | 319,844 | 1.00 | 127 | 13.6 | ok |
| base_liger | 123.6M | 103 | 308.8 | 0.99 | 319,065 | 0.99 | 127 | 13.4 | ok |
| base_ce_te | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| base_flash | 123.6M | - | - | - | - | - | - | - | FAIL: no measured iteration |
| base_graph_full | 123.6M | 101 | 314.6 | 1.01 | 325,079 | 1.01 | 127 | 13.6 | ok |
| base_liger_mbs16 | 123.6M | 197 | 322.4 | 1.04 | 333,093 | 1.04 | 254 | 24.3 | ok |
| base_liger_nocg | 123.6M | 102 | 309.5 | 1.00 | 319,688 | 1.00 | 127 | 13.6 | ok |
| base_liger_nodistopt | 123.6M | 105 | 300.8 | 0.97 | 310,744 | 0.97 | 127 | 13.7 | ok |
| base_liger_nomask | 123.6M | 102 | 309.9 | 1.00 | 320,156 | 1.00 | 127 | 13.4 | ok |
| base_liger_nomask_mbs16_nocg | 123.6M | 196 | 323.8 | 1.04 | 334,538 | 1.04 | 254 | 24.5 | ok |
| base_liger_noqkln | 123.6M | 102 | 310.4 | 1.00 | 320,626 | 1.00 | 127 | 11.6 | ok |
| base_liger_mbs4 | 123.6M | 106 | 299.0 | 0.96 | 308,986 | 0.96 | 127 | 7.9 | ok |
| base_qknr | 123.6M | 103 | 309.1 | 1.00 | 319,376 | 1.00 | 127 | 13.4 | ok |
| base_resnorm | 123.6M | 102 | 310.7 | 1.00 | 320,940 | 1.00 | 127 | 13.4 | ok |
| base_opfuser | 123.6M | 103 | 308.5 | 0.99 | 318,755 | 0.99 | 127 | 13.4 | ok |
| base_qknr_resnorm | 123.6M | 102 | 310.1 | 1.00 | 320,313 | 1.00 | 127 | 13.4 | ok |
| base_best | 123.6M | 102 | 310.0 | 1.00 | 320,313 | 1.00 | 127 | 13.4 | ok |
| base_best_swiglu | 123.6M | 103 | 309.1 | 1.00 | 319,376 | 1.00 | 127 | 13.4 | ok |
| base_best_mbs16_nocg | 123.6M | 198 | 321.1 | 1.03 | 331,744 | 1.03 | 254 | 24.5 | ok |
| base_best_hd128 | 123.6M | 98 | 322.9 | 1.04 | 333,517 | 1.04 | 127 | 13.4 | ok |
| base_fp8 | 123.6M | 107 | 297.4 | 0.96 | 307,248 | 0.96 | 127 | 12.5 | ok |
| nope | 123.6M | 107 | 295.8 | 0.95 | 305,672 | 0.95 | 127 | 12.8 | ok |
| gqa4 | 123.6M | 102 | 310.2 | 1.00 | 320,626 | 1.00 | 127 | 13.4 | ok |
| xsa | 123.6M | 112 | 283.4 | 0.91 | 292,833 | 0.91 | 127 | 13.9 | ok |
| gate | 130.7M | 110 | 289.0 | 0.93 | 298,706 | 0.93 | 127 | 15.1 | ok |
| xsa_gate | 130.7M | 118 | 268.1 | 0.86 | 276,991 | 0.86 | 127 | 15.1 | ok |
| swa3 | 123.6M | 96 | 287.6 | 0.93 | 342,225 | 1.07 | 110 | 13.4 | ok |
| mla (!) | 116.5M | 106 | 177.2 | 0.57 | 310,009 | 0.97 | 75 | 12.5 | ok |
| mla_gqa_xsa_gate (!) | 116.9M | 114 | 163.8 | 0.53 | 286,559 | 0.89 | 75 | 13.2 | ok |
| mamba2 (!) | 148.7M | 166 | 195.0 | 0.63 | 197,994 | 0.62 | 129 | 16.7 | ok |
| mamba2_ck64 (!) | 148.7M | 176 | 183.4 | 0.59 | 186,288 | 0.58 | 129 | 17.1 | ok |
| mamba2_ck256 (!) | 148.7M | 168 | 192.0 | 0.62 | 194,932 | 0.61 | 129 | 17.1 | ok |
| gdn_native (!) | 129.1M | 134 | 205.7 | 0.66 | 244,355 | 0.76 | 110 | 16.0 | ok |
| gated_delta_net (!) | 129.1M | 134 | 206.4 | 0.66 | 245,178 | 0.76 | 110 | 16.0 | ok |
| mlstm (!) | 129.1M | 112 | 246.7 | 0.79 | 293,095 | 0.91 | 110 | 14.3 | ok |
| kda (!) | 125.5M | 154 | 174.1 | 0.56 | 212,366 | 0.66 | 107 | 15.6 | ok |
| complex_kda (!) | 130.0M | 152 | 182.9 | 0.59 | 216,005 | 0.67 | 111 | 17.5 | ok |
| kda_xsa (!) | 125.5M | 161 | 166.6 | 0.54 | 203,212 | 0.63 | 107 | 16.0 | ok |
| mlstm_ck64 (!) | 129.1M | 114 | 240.2 | 0.77 | 286,308 | 0.89 | 110 | 14.3 | ok |
| mlstm_ck256 | 129.1M | 110 | 251.9 | 0.81 | 297,485 | 0.93 | 111 | 14.3 | ok |
| kda_hd128 (!) | 127.3M | 156 | 176.1 | 0.57 | 209,380 | 0.65 | 110 | 15.3 | ok |
| tri_gdn_e4_cute (!) | 126.0M | 138 | 212.7 | 0.68 | 237,106 | 0.74 | 118 | 21.5 | ok |
| tri_kda_e4 (!) | 127.8M | 185 | 160.4 | 0.52 | 176,742 | 0.55 | 119 | 25.4 | ok |
| tri_ckda_e4 (!) | 127.8M | 187 | 158.9 | 0.51 | 175,043 | 0.55 | 119 | 25.6 | ok |
| power (!) | 123.7M | 241 | 156.5 | 0.50 | 136,193 | 0.42 | 151 | 12.5 | ok |
| power_ck64 (!) | 123.7M | 220 | 171.4 | 0.55 | 149,149 | 0.46 | 151 | 7.4 | ok |
| power_ck128 (!) | 123.7M | 219 | 171.9 | 0.55 | 149,626 | 0.47 | 151 | 7.4 | ok |
| shortconv | 151.9M | 123 | 302.6 | 0.97 | 265,866 | 0.83 | 149 | 16.9 | ok |
| canon (!) | 124.0M | 136 | 234.1 | 0.75 | 241,830 | 0.75 | 127 | 15.6 | ok |
| engram | 192.3M | 128 | 249.2 | 0.80 | 255,004 | 0.79 | 128 | 15.4 | ok |
| pkm (!) | 219.8M | 170 | 181.8 | 0.59 | 192,923 | 0.60 | 124 | 9.7 | ok |
| mhc4 (!) | 125.4M | 173 | 183.4 | 0.59 | 189,520 | 0.59 | 127 | 19.2 | ok |
| routed_k2 (!) | 123.7M | 139 | 227.6 | 0.73 | 235,065 | 0.73 | 127 | 18.3 | ok |
| routed_k2_mix (!) | 124.0M | 161 | 197.1 | 0.63 | 203,718 | 0.63 | 127 | 18.5 | ok |
| routed_k2_rev (!) | 123.7M | 152 | 208.7 | 0.67 | 215,579 | 0.67 | 127 | 15.1 | ok |
| routed_k2_cg | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| routed_k2_mix_cg | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| routed_k2_rev_cg | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| attn_res (!) | 123.6M | 130 | 244.0 | 0.79 | 252,062 | 0.79 | 127 | 14.8 | ok |
| postnorm | 123.6M | 107 | 297.6 | 0.96 | 307,392 | 0.96 | 127 | 14.5 | ok |
| ngpt | 123.6M | 120 | 263.3 | 0.85 | 272,159 | 0.85 | 127 | 13.9 | ok |
| ple128 | 203.2M | 118 | 273.4 | 0.88 | 278,403 | 0.87 | 129 | 15.4 | ok |
| value_emb (!) | 587.2M | 133 | 239.0 | 0.77 | 246,933 | 0.77 | 127 | 17.5 | ok |
| muon | 123.6M | 108 | 294.8 | 0.95 | 304,535 | 0.95 | 127 | 13.3 | ok |
| muown | 123.6M | 111 | 286.0 | 0.92 | 295,473 | 0.92 | 127 | 13.3 | ok |
| adam_precaware | 123.6M | 102 | 310.4 | 1.00 | 320,626 | 1.00 | 127 | 13.3 | ok |
| adam_offload | 123.6M | 115 | 276.6 | 0.89 | 285,809 | 0.89 | 127 | 13.3 | ok |
| muown_offload | 123.6M | 112 | 282.4 | 0.91 | 291,790 | 0.91 | 127 | 13.3 | ok |

### v23cp2_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dense | 123.6M | 134 | 236.0 | 1.00 | 243,810 | 1.00 | 127 | 7.9 | ok |
| xsa | 123.6M | 144 | 220.5 | 0.93 | 227,872 | 0.93 | 127 | 8.5 | ok |
| gate | 130.7M | 142 | 224.1 | 0.95 | 231,494 | 0.95 | 127 | 8.8 | ok |
| mla (!) | 116.5M | 138 | 135.6 | 0.57 | 237,363 | 0.97 | 75 | 7.5 | ok |
| mamba2 (!) | 148.7M | 279 | 115.8 | 0.49 | 117,617 | 0.48 | 129 | 10.4 | ok |
| gated_delta_net (!) | 129.1M | 190 | 145.4 | 0.62 | 172,645 | 0.71 | 110 | 9.3 | ok |
| mlstm (!) | 129.1M | 173 | 159.5 | 0.68 | 189,520 | 0.78 | 110 | 8.4 | ok |
| kda | 125.5M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| complex_kda | 130.0M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| tri_gdn_e4_cute | 126.0M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| tri_kda_e4 | 127.8M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| tri_ckda_e4 | 127.8M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| power | 123.7M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| shortconv (!) | 151.9M | 199 | 187.7 | 0.80 | 164,912 | 0.68 | 149 | 10.4 | ok |
| canon (!) | 124.0M | 412 | 77.1 | 0.33 | 79,611 | 0.33 | 127 | 13.6 | ok |
| engram (!) | 192.3M | 190 | 168.8 | 0.72 | 172,690 | 0.71 | 128 | 10.0 | ok |
| pkm (!) | 219.8M | 203 | 152.3 | 0.65 | 161,698 | 0.66 | 123 | 6.2 | ok |
| mhc4 (!) | 125.4M | 214 | 148.2 | 0.63 | 153,050 | 0.63 | 127 | 10.9 | ok |
| routed_k2 (!) | 123.7M | 201 | 157.9 | 0.67 | 163,187 | 0.67 | 127 | 10.5 | ok |
| attn_res (!) | 123.6M | 178 | 178.4 | 0.76 | 184,297 | 0.76 | 127 | 8.8 | ok |

### v24_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dense | 123.6M | 102 | 310.5 | 1.00 | 320,626 | 1.00 | 127 | 13.4 | ok |
| kda (!) | 125.5M | 154 | 174.2 | 0.56 | 212,572 | 0.66 | 107 | 15.6 | ok |
| complex_kda (!) | 130.0M | 152 | 182.8 | 0.59 | 215,792 | 0.67 | 111 | 17.5 | ok |

### v26_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dense | 123.6M | 102 | 312.2 | 1.00 | 322,520 | 1.00 | 127 | 13.4 | ok |
| dense_fineweb | 123.6M | 103 | 307.8 | 0.99 | 318,136 | 0.99 | 127 | 13.4 | ok |
| dense_fineweb_mbs16 | 123.6M | 196 | 323.5 | 1.04 | 334,197 | 1.04 | 254 | 24.3 | ok |
| base_liger_mbs16 | 123.6M | 198 | 320.9 | 1.03 | 331,492 | 1.03 | 254 | 24.3 | ok |

### v27_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dense | 123.6M | 102 | 310.2 | 1.00 | 320,626 | 1.00 | 127 | 13.4 | ok |
| power (!) | 123.7M | 200 | 188.2 | 0.61 | 163,840 | 0.51 | 151 | 12.2 | ok |
| power_hd128 | 123.6M | 167 | 427.3 | 1.38 | 196,098 | 0.61 | 286 | 12.2 | ok |
| power_chunked (!) | 123.7M | 241 | 156.4 | 0.50 | 136,080 | 0.42 | 151 | 12.5 | ok |

### v28cp2_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dense | 123.6M | 134 | 236.5 | 1.00 | 244,355 | 1.00 | 127 | 7.9 | ok |
| kda (!) | 125.5M | 218 | 123.3 | 0.52 | 150,450 | 0.62 | 107 | 9.3 | ok |
| complex_kda (!) | 130.0M | 217 | 127.9 | 0.54 | 151,005 | 0.62 | 111 | 10.3 | ok |
| tri_gdn_e4_cute (!) | 126.0M | 246 | 119.8 | 0.51 | 133,475 | 0.55 | 118 | 18.9 | ok |
| tri_kda_e4 (!) | 127.8M | 353 | 84.3 | 0.36 | 92,827 | 0.38 | 119 | 22.9 | ok |
| tri_ckda_e4 (!) | 127.8M | 346 | 86.1 | 0.36 | 94,842 | 0.39 | 119 | 23.2 | ok |
| power (!) | 123.7M | 372 | 101.1 | 0.43 | 88,015 | 0.36 | 151 | 8.5 | ok |

### v2_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dense | 123.6M | 101 | 312.9 | 1.00 | 323,156 | 1.00 | 127 | 13.4 | ok |
| dense_gpt | 123.6M | 102 | 310.9 | 0.99 | 321,255 | 0.99 | 127 | 13.4 | ok |
| dense_nocg | 123.6M | 102 | 310.1 | 0.99 | 320,313 | 0.99 | 127 | 13.6 | ok |
| base_liger | 123.6M | 102 | 309.9 | 0.99 | 320,156 | 0.99 | 127 | 13.4 | ok |
| base_ce_te | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| base_flash | 123.6M | 118 | 269.1 | 0.86 | 278,166 | 0.86 | 127 | 13.9 | ok |
| base_graph_full | 123.6M | 100 | 315.9 | 1.01 | 326,375 | 1.01 | 127 | 13.6 | ok |
| base_liger_mbs16 | 123.6M | 197 | 322.6 | 1.03 | 333,262 | 1.03 | 254 | 24.3 | ok |
| base_liger_nocg | 123.6M | 103 | 307.9 | 0.98 | 318,136 | 0.98 | 127 | 13.6 | ok |
| base_liger_nodistopt | 123.6M | 105 | 301.1 | 0.96 | 311,039 | 0.96 | 127 | 13.7 | ok |
| base_liger_nomask | 123.6M | 102 | 311.0 | 0.99 | 321,255 | 0.99 | 127 | 13.4 | ok |
| base_liger_nomask_mbs16_nocg | 123.6M | 197 | 321.4 | 1.03 | 332,080 | 1.03 | 254 | 24.5 | ok |
| base_liger_noqkln | 123.6M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| base_liger_mbs4 | 123.6M | 106 | 299.1 | 0.96 | 308,986 | 0.96 | 127 | 7.9 | ok |
| base_qknr | 123.6M | 101 | 313.1 | 1.00 | 323,475 | 1.00 | 127 | 13.4 | ok |
| base_resnorm | 123.6M | 102 | 310.2 | 0.99 | 320,626 | 0.99 | 127 | 13.4 | ok |
| base_opfuser | 123.6M | 103 | 308.9 | 0.99 | 319,065 | 0.99 | 127 | 13.4 | ok |
| base_qknr_resnorm | 123.6M | 103 | 309.2 | 0.99 | 319,532 | 0.99 | 127 | 13.4 | ok |
| base_best | 123.6M | 102 | 310.1 | 0.99 | 320,313 | 0.99 | 127 | 13.4 | ok |
| base_best_swiglu | 123.6M | 102 | 309.9 | 0.99 | 320,313 | 0.99 | 127 | 13.4 | ok |
| base_best_mbs16_nocg | 123.6M | 196 | 323.9 | 1.04 | 334,623 | 1.04 | 254 | 24.5 | ok |
| base_best_hd128 | 123.6M | 96 | 329.2 | 1.05 | 340,270 | 1.05 | 127 | 13.4 | ok |
| base_fp8 | 123.6M | 107 | 296.3 | 0.95 | 306,243 | 0.95 | 127 | 12.5 | ok |
| nope | 123.6M | 107 | 296.0 | 0.95 | 305,957 | 0.95 | 127 | 12.8 | ok |
| gqa4 | 123.6M | 102 | 311.2 | 0.99 | 321,570 | 1.00 | 127 | 13.4 | ok |
| xsa | 123.6M | 111 | 285.5 | 0.91 | 294,941 | 0.91 | 127 | 13.9 | ok |
| gate (!) | 130.7M | 132 | 240.5 | 0.77 | 248,431 | 0.77 | 127 | 14.5 | ok |
| xsa_gate (!) | 130.7M | 128 | 248.4 | 0.79 | 256,601 | 0.79 | 127 | 14.5 | ok |
| swa3 | 123.6M | 95 | 290.2 | 0.93 | 345,290 | 1.07 | 110 | 13.4 | ok |
| mla (!) | 116.5M | 106 | 176.6 | 0.56 | 308,841 | 0.96 | 75 | 12.5 | ok |
| mla_gqa_xsa_gate (!) | 116.9M | 115 | 162.9 | 0.52 | 285,063 | 0.88 | 75 | 13.2 | ok |
| mamba2 (!) | 148.7M | 174 | 185.3 | 0.59 | 188,214 | 0.58 | 129 | 16.7 | ok |
| gdn_native (!) | 129.1M | 136 | 202.7 | 0.65 | 240,764 | 0.75 | 110 | 16.4 | ok |
| gated_delta_net (!) | 129.1M | 137 | 201.1 | 0.64 | 238,834 | 0.74 | 110 | 16.4 | ok |
| mlstm (!) | 129.1M | 113 | 244.8 | 0.78 | 291,012 | 0.90 | 110 | 14.3 | ok |
| kda (!) | 125.5M | 154 | 174.4 | 0.56 | 212,641 | 0.66 | 107 | 15.6 | ok |
| signed_kda (!) | 130.0M | 178 | 155.9 | 0.50 | 184,297 | 0.57 | 111 | 19.6 | ok |
| complex_kda (!) | 125.5M | 182 | 147.9 | 0.47 | 180,341 | 0.56 | 107 | 19.6 | ok |
| kda_xsa (!) | 125.5M | 161 | 166.6 | 0.53 | 203,275 | 0.63 | 107 | 16.0 | ok |
| kda_hd128 (!) | 127.3M | 156 | 177.2 | 0.57 | 210,727 | 0.65 | 110 | 15.2 | ok |
| tri_gdn_e4_cute (!) | 126.0M | 183 | 161.0 | 0.51 | 179,452 | 0.56 | 118 | 20.7 | ok |
| tri_kda_e4 (!) | 127.8M | 259 | 114.8 | 0.37 | 126,469 | 0.39 | 119 | 24.8 | ok |
| tri_ckda_e4 (!) | 127.8M | 280 | 106.3 | 0.34 | 117,175 | 0.36 | 119 | 25.9 | ok |
| power (!) | 123.7M | 275 | 136.9 | 0.44 | 119,200 | 0.37 | 151 | 16.9 | ok |
| shortconv (!) | 151.9M | 154 | 242.6 | 0.78 | 213,195 | 0.66 | 149 | 18.1 | ok |
| canon | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| engram (!) | 192.3M | 130 | 246.7 | 0.79 | 252,450 | 0.78 | 128 | 15.4 | ok |
| pkm | 219.8M | - | - | - | - | - | - | 35.3 | FAIL: ChildFailedError:  |
| mhc4 (!) | 125.4M | 188 | 168.4 | 0.54 | 173,928 | 0.54 | 127 | 19.2 | ok |
| routed_k2 (!) | 123.7M | 141 | 225.4 | 0.72 | 232,893 | 0.72 | 127 | 18.3 | ok |
| routed_k2_mix (!) | 124.0M | 227 | 139.5 | 0.45 | 144,130 | 0.45 | 127 | 18.5 | ok |
| routed_k2_rev (!) | 123.7M | 152 | 208.9 | 0.67 | 215,863 | 0.67 | 127 | 15.1 | ok |
| attn_res (!) | 123.6M | 136 | 233.3 | 0.75 | 241,118 | 0.75 | 127 | 14.8 | ok |
| postnorm | 123.6M | 107 | 296.2 | 0.95 | 305,957 | 0.95 | 127 | 14.5 | ok |
| ngpt (!) | 123.6M | 161 | 196.9 | 0.63 | 203,402 | 0.63 | 127 | 16.2 | ok |
| ple128 | 203.2M | 118 | 272.9 | 0.87 | 277,930 | 0.86 | 129 | 15.4 | ok |
| value_emb (!) | 587.2M | 134 | 237.5 | 0.76 | 245,361 | 0.76 | 127 | 17.5 | ok |
| muon | 123.6M | 109 | 290.4 | 0.93 | 300,073 | 0.93 | 127 | 13.3 | ok |
| muown | 123.6M | 113 | 281.2 | 0.90 | 290,496 | 0.90 | 127 | 13.3 | ok |
| adam_precaware | 123.6M | 102 | 309.6 | 0.99 | 320,000 | 0.99 | 127 | 13.3 | ok |
| adam_offload (!) | 123.6M | 132 | 239.7 | 0.77 | 247,586 | 0.77 | 127 | 13.3 | ok |
| muown_offload | 123.6M | 115 | 276.4 | 0.88 | 285,684 | 0.88 | 127 | 13.3 | ok |

### v2base_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dense | 123.6M | 170 | 186.3 | 1.00 | 192,470 | 1.00 | 127 | 26.6 | ok |
| base_liger | 123.6M | 118 | 268.3 | 1.44 | 277,225 | 1.44 | 127 | 12.9 | ok |
| base_ce_te | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| base_flash | 123.6M | 185 | 171.8 | 0.92 | 177,412 | 0.92 | 127 | 27.2 | ok |
| base_graph_full | 123.6M | 165 | 192.6 | 1.03 | 198,956 | 1.03 | 127 | 27.0 | ok |
| base_liger_mbs16 | 123.6M | 226 | 281.2 | 1.51 | 290,561 | 1.51 | 254 | 23.4 | ok |
| base_liger_nocg | 123.6M | 114 | 278.5 | 1.49 | 287,691 | 1.49 | 127 | 13.2 | ok |
| base_fp8 | 123.6M | 176 | 180.3 | 0.97 | 186,288 | 0.97 | 127 | 25.8 | ok |

### v2ck_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| mamba2_ck64 | 148.7M | 185 | 174.6 | - | 177,220 | - | 129 | 17.1 | ok |
| mamba2_ck256 | 148.7M | 177 | 182.3 | - | 185,130 | - | 129 | 17.1 | ok |
| mlstm_ck64 | 129.1M | 116 | 236.9 | - | 282,361 | - | 110 | 14.3 | ok |
| mlstm_ck256 | 129.1M | 111 | 249.3 | - | 294,412 | - | 111 | 14.3 | ok |
| power_ck64 | 123.7M | 386 | 97.6 | - | 84,913 | - | 151 | 25.6 | ok |
| power_ck256 | 123.7M | 240 | 157.2 | - | 136,790 | - | 151 | 12.5 | ok |

### v2fix_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| base_liger_noqkln | 123.6M | 102 | 310.1 | - | 320,313 | - | 127 | 11.6 | ok |
| canon | 124.0M | 154 | 205.6 | - | 212,503 | - | 127 | 15.6 | ok |

### v3base_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dense | 123.6M | 170 | 186.4 | 1.00 | 192,526 | 1.00 | 127 | 26.6 | ok |
| base_liger | 123.6M | 118 | 268.8 | 1.44 | 277,695 | 1.44 | 127 | 12.9 | ok |
| base_liger_nodistopt | 123.6M | 121 | 262.9 | 1.41 | 271,483 | 1.41 | 127 | 13.3 | ok |
| base_liger_teact | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| base_liger_noqkln | 123.6M | 110 | 289.4 | 1.55 | 298,978 | 1.55 | 127 | 11.8 | ok |
| base_liger_mbs4 | 123.6M | 124 | 255.9 | 1.37 | 264,365 | 1.37 | 127 | 7.7 | ok |

### v3gate_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dense | 123.6M | 102 | 310.6 | 1.00 | 320,940 | 1.00 | 127 | 13.4 | ok |
| xsa | 123.6M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| gate | 130.7M | 110 | 289.1 | 0.93 | 298,706 | 0.93 | 127 | 15.1 | ok |
| xsa_gate | 130.7M | 118 | 267.6 | 0.86 | 276,523 | 0.86 | 127 | 15.1 | ok |

### v3sconv_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| shortconv | 151.9M | 123 | 302.8 | - | 266,082 | - | 149 | 16.9 | ok |

### v3tri_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| tri_gdn_e4_cute | 126.0M | 148 | 198.2 | - | 220,958 | - | 118 | 21.5 | ok |
| tri_kda_e4 | 127.8M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| tri_ckda_e4 | 127.8M | 245 | 121.4 | - | 133,692 | - | 119 | 26.7 | ok |

### v3trikda_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| tri_kda_e4 | 127.8M | 221 | 134.8 | - | 148,507 | - | 119 | 25.6 | ok |

### v3xsa_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| xsa | 123.6M | 112 | 283.8 | - | 293,095 | - | 127 | 13.9 | ok |

### v4base_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| base_liger | 123.6M | 118 | 269.0 | - | 277,930 | - | 127 | 12.9 | ok |
| base_liger_teact | 123.6M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| base_liger_nomask | 123.6M | 111 | 286.1 | - | 295,740 | - | 127 | 12.8 | ok |
| base_liger_nomask_mbs16_nocg | 123.6M | 210 | 301.6 | - | 311,631 | - | 254 | 23.4 | ok |

### v4mix_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dense | 123.6M | 102 | 309.8 | 1.00 | 320,000 | 1.00 | 127 | 13.4 | ok |
| gdn_native (!) | 129.1M | 137 | 201.5 | 0.65 | 239,357 | 0.75 | 110 | 16.4 | ok |
| gated_delta_net (!) | 129.1M | 137 | 201.5 | 0.65 | 239,357 | 0.75 | 110 | 16.4 | ok |
| kda (!) | 125.5M | 168 | 159.6 | 0.52 | 194,700 | 0.61 | 107 | 16.9 | ok |
| signed_kda (!) | 130.0M | 198 | 139.9 | 0.45 | 165,328 | 0.52 | 111 | 21.3 | ok |
| complex_kda (!) | 125.5M | 196 | 137.4 | 0.44 | 167,440 | 0.52 | 108 | 20.9 | ok |
| kda_xsa (!) | 125.5M | 176 | 152.4 | 0.49 | 185,970 | 0.58 | 107 | 17.7 | ok |
| power (!) | 123.7M | 259 | 145.3 | 0.47 | 126,469 | 0.40 | 151 | 12.5 | ok |

### v5base_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| base_liger_nomask | 123.6M | 111 | 286.2 | - | 295,740 | - | 127 | 12.8 | ok |
| base_qknr | 123.6M | 102 | 309.9 | - | 320,313 | - | 127 | 13.4 | ok |
| base_resnorm | 123.6M | 110 | 287.6 | - | 297,081 | - | 127 | 12.8 | ok |
| base_opfuser | 123.6M | 111 | 286.0 | - | 295,473 | - | 127 | 12.8 | ok |
| base_qknr_resnorm | 123.6M | 102 | 311.4 | - | 321,728 | - | 127 | 13.4 | ok |

### v5mix_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| gated_delta_net | 129.1M | 136 | 202.9 | - | 241,030 | - | 110 | 16.4 | ok |
| kda | 125.5M | 155 | 173.6 | - | 211,816 | - | 107 | 15.6 | ok |
| signed_kda | 130.0M | 178 | 156.3 | - | 184,608 | - | 111 | 19.6 | ok |
| complex_kda | 125.5M | 182 | 147.6 | - | 179,945 | - | 108 | 19.6 | ok |
| pkm | 219.8M | 325 | 95.1 | - | 100,918 | - | 124 | 18.7 | ok |

### v5routed_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| routed_k2 | 123.7M | 140 | 226.4 | - | 233,890 | - | 127 | 18.3 | ok |
| routed_k2_mix | 124.0M | 228 | 139.4 | - | 144,035 | - | 127 | 18.5 | ok |
| routed_k2_rev | 123.7M | 151 | 209.4 | - | 216,362 | - | 127 | 15.1 | ok |
| routed_k2_cg | 123.7M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| routed_k2_mix_cg | 124.0M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| routed_k2_rev_cg | 123.7M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |

### v5tri_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| kda_hd128 | 127.3M | 217 | 126.8 | - | 150,727 | - | 110 | 28.9 | ok |
| tri_gdn_e4_cute | 126.0M | 244 | 120.3 | - | 134,130 | - | 118 | 34.4 | ok |
| tri_kda_e4 | 127.8M | 322 | 92.5 | - | 101,875 | - | 119 | 38.5 | ok |
| tri_ckda_e4 | 127.8M | 342 | 86.9 | - | 95,687 | - | 119 | 39.6 | ok |

### v6base_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| base_best | 123.6M | 102 | 310.7 | - | 320,940 | - | 127 | 13.4 | ok |
| base_best_mbs16_nocg | 123.6M | 197 | 321.5 | - | 332,080 | - | 254 | 24.5 | ok |
| base_best_hd128 | 123.6M | 98 | 324.4 | - | 335,051 | - | 127 | 13.3 | ok |

### v7base_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| base_best | 123.6M | 102 | 310.2 | - | 320,469 | - | 127 | 13.4 | ok |
| base_best_swiglu | 123.6M | 103 | 308.5 | - | 318,755 | - | 127 | 13.4 | ok |

### v7kdac_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| kda | 125.5M | 154 | 174.8 | - | 213,195 | - | 107 | 15.6 | ok |
| signed_kda | 130.0M | 155 | 179.4 | - | 211,953 | - | 111 | 17.7 | ok |
| complex_kda | 125.5M | 182 | 147.7 | - | 180,143 | - | 107 | 19.6 | ok |
| kda_xsa | 125.5M | 161 | 167.0 | - | 203,781 | - | 107 | 16.0 | ok |
| kda_hd128 | 127.3M | 155 | 177.4 | - | 210,862 | - | 110 | 15.2 | ok |

### v7mamba_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| mamba2 | 148.7M | 165 | 195.8 | - | 198,775 | - | 129 | 16.7 | ok |
| mamba2_ck256 | 148.7M | 168 | 192.4 | - | 195,338 | - | 129 | 17.1 | ok |

### v7muown_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| muown | 123.6M | 110 | 288.5 | - | 298,026 | - | 127 | 13.3 | ok |
| adam_offload | 123.6M | 124 | 256.5 | - | 265,113 | - | 127 | 13.3 | ok |
| muown_offload | 123.6M | 113 | 281.5 | - | 290,754 | - | 127 | 13.3 | ok |

### v8mhc_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| signed_kda | 130.0M | 154 | 179.6 | - | 212,228 | - | 111 | 17.7 | ok |
| mhc4 | 125.4M | 177 | 179.4 | - | 185,339 | - | 127 | 19.2 | ok |

### v9_dense130M

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dense | 123.6M | 102 | 310.2 | 1.00 | 320,469 | 1.00 | 127 | 13.4 | ok |
| canon (!) | 124.0M | 136 | 234.1 | 0.75 | 241,830 | 0.75 | 127 | 15.6 | ok |
| engram | 192.3M | 129 | 248.5 | 0.80 | 254,311 | 0.79 | 128 | 15.4 | ok |
| pkm | 219.8M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |

### ck2_resume_moe50M500M

reference: `moe`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| muown | 166.1M | 21300 | 1.0 | - | 1,538 | - | 85 | 15.3 | ok |
| muown_offload | 166.1M | 20993 | 1.0 | - | 1,561 | - | 84 | 15.3 | ok |
| adam_offload | 166.1M | 5763 | 3.6 | - | 5,686 | - | 83 | 14.0 | ok |

### ck2_save_moe50M500M

reference: `moe`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| muown | 166.1M | 211 | 98.8 | - | 155,262 | - | 83 | 14.7 | ok |
| muown_offload | 166.1M | 219 | 95.1 | - | 149,591 | - | 83 | 14.7 | ok |
| adam_offload | 166.1M | 288 | 72.2 | - | 113,620 | - | 83 | 14.2 | ok |

### ck_resume_moe50M500M

reference: `moe`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| muown | 166.1M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| muown_offload | 166.1M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| adam_offload | 166.1M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |

### ck_save_moe50M500M

reference: `moe`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| muown | 166.1M | 211 | 98.7 | - | 155,152 | - | 83 | 14.7 | ok |
| muown_offload | 166.1M | 218 | 95.3 | - | 149,968 | - | 83 | 14.7 | ok |
| adam_offload | 166.1M | 286 | 72.9 | - | 114,694 | - | 83 | 14.2 | ok |

### par2_pp2_moe50M500M

reference: `moe`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| moe | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| latent | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| zero16 | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |

### par2_ref_moe50M500M

reference: `moe`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| moe | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| latent | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| zero16 | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |

### par2_tp2_moe50M500M

reference: `moe`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| moe | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| latent | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| zero16 | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |

### par3_pp2_moe50M500M

reference: `moe`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| moe | 152.6M | 402 | 51.8 | 1.00 | 81,502 | 1.00 | 83 | 23.8 | ok |
| latent (!) | 97.5M | 460 | 40.5 | 0.78 | 71,181 | 0.87 | 75 | 22.3 | ok |
| zero16 | 152.6M | 392 | 53.0 | 1.02 | 83,496 | 1.02 | 83 | 23.1 | ok |

### par3_ref_moe50M500M

reference: `moe`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| moe | 166.1M | 245 | 85.2 | 1.00 | 133,993 | 1.00 | 83 | 29.2 | ok |
| latent | 112.7M | 253 | 73.9 | 0.87 | 129,697 | 0.97 | 75 | 28.1 | ok |
| zero16 | 166.2M | 242 | 86.1 | 1.01 | 135,349 | 1.01 | 83 | 28.1 | ok |

### par3_tp2_moe50M500M

reference: `moe`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| moe | 139.9M | 462 | 45.1 | 1.00 | 70,980 | 1.00 | 83 | 16.7 | ok |
| latent (!) | 86.5M | 603 | 31.0 | 0.69 | 54,382 | 0.77 | 75 | 15.0 | ok |
| zero16 | 140.0M | 421 | 49.5 | 1.10 | 77,852 | 1.10 | 83 | 15.9 | ok |

### par_pp2_moe50M500M

reference: `moe`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| moe | 152.6M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| latent | 97.5M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| zero16 | 152.6M | - | - | - | - | - | - | 12.2 | FAIL: ChildFailedError:  |

### par_ref_moe50M500M

reference: `moe`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| moe | 166.1M | 174 | 119.4 | 1.00 | 187,782 | 1.00 | 83 | 14.7 | ok |
| latent | 112.7M | 180 | 103.6 | 0.87 | 181,691 | 0.97 | 75 | 13.6 | ok |
| zero16 | 166.2M | 181 | 114.8 | 0.96 | 180,590 | 0.96 | 83 | 14.7 | ok |

### par_tp2_moe50M500M

reference: `moe`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| moe | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| latent | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| zero16 | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |

### v13moe_moe50M500M

reference: `moe`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| nemo_flex_tuned | 166.1M | 163 | 127.9 | 1.07 | 201,154 | 1.07 | 83 | 14.7 | ok |
| conn8 | 166.1M | 175 | 118.9 | 0.99 | 186,925 | 0.99 | 83 | 14.7 | ok |
| moe | 166.1M | 174 | 119.5 | 1.00 | 187,944 | 1.00 | 83 | 14.7 | ok |
| cutlass_gg | 166.1M | 164 | 126.7 | 1.06 | 199,258 | 1.06 | 83 | 14.7 | ok |
| nemo | 166.1M | 178 | 117.1 | 0.98 | 184,090 | 0.98 | 83 | 14.7 | ok |
| router_fusion | 166.1M | 176 | 118.5 | 0.99 | 186,394 | 0.99 | 83 | 14.8 | ok |
| tuned | 166.1M | 168 | 124.0 | 1.04 | 194,990 | 1.04 | 83 | 14.8 | ok |
| nemo_flex | 166.1M | 170 | 122.2 | 1.02 | 192,188 | 1.02 | 83 | 14.6 | ok |
| a2a_overlap | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| nemo_flex_nocg | 166.1M | 181 | 115.2 | 0.96 | 181,089 | 0.96 | 83 | 15.7 | ok |
| gc10 | 166.1M | 212 | 99.2 | 0.83 | 154,931 | 0.82 | 84 | 14.7 | ok |

### v14moe_moe50M500M

reference: `moe`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| nemo | 166.1M | 163 | 127.6 | 1.01 | 200,723 | 1.01 | 83 | 14.7 | ok |
| cublas_gg | 166.1M | 174 | 119.5 | 0.95 | 187,998 | 0.95 | 83 | 14.7 | ok |
| moe | 166.1M | 165 | 126.5 | 1.00 | 198,895 | 1.00 | 83 | 14.7 | ok |
| nemo_flex | 166.1M | 165 | 126.3 | 1.00 | 198,654 | 1.00 | 83 | 14.6 | ok |
| noshared | 152.0M | - | - | - | - | - | - | 15.4 | FAIL: ChildFailedError:  |
| nemo_flex_rf | 166.1M | 163 | 128.1 | 1.01 | 201,401 | 1.01 | 83 | 14.7 | ok |
| muown | 166.1M | 200 | 104.0 | 0.82 | 163,595 | 0.82 | 83 | 14.7 | ok |
| nemo_flex_a2a | 166.1M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| muown_offload (!) | 166.1M | 207 | 100.5 | 0.80 | 158,147 | 0.80 | 83 | 14.7 | ok |
| nemo_a2a | 166.1M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| adam_offload (!) | 166.1M | 260 | 80.0 | 0.63 | 125,958 | 0.63 | 83 | 14.2 | ok |
| gdn3 (!) | 168.6M | 184 | 96.8 | 0.77 | 177,652 | 0.89 | 71 | 16.2 | ok |
| a2a_overlap | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |

### v15a_moe50M500M

reference: `moe`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| moe | 166.1M | 166 | 125.7 | 1.00 | 197,695 | 1.00 | 83 | 14.7 | ok |
| noshared_cublas (!) | 152.0M | 222 | 93.7 | 0.75 | 147,305 | 0.75 | 83 | 15.4 | ok |
| noshared | 152.0M | - | - | - | - | - | - | 15.4 | FAIL: ChildFailedError:  |
| auxfree | 166.1M | 167 | 125.0 | 0.99 | 196,687 | 0.99 | 83 | 14.7 | ok |
| latent | 112.7M | - | - | - | - | - | - | 13.6 | FAIL: ChildFailedError:  |

### v15b_moe50M500M

reference: `moe`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| noshared | 152.0M | - | - | - | - | - | - | 15.4 | FAIL: ChildFailedError:  |

### v16moe_moe50M500M

reference: `moe`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| nemo_cutlass | 166.1M | 164 | 127.0 | 1.08 | 199,744 | 1.08 | 83 | 14.7 | ok |
| nemo | 166.1M | 174 | 119.8 | 1.02 | 188,484 | 1.02 | 83 | 14.7 | ok |
| moe | 166.1M | 177 | 117.5 | 1.00 | 184,817 | 1.00 | 83 | 14.7 | ok |
| nemo_cutlass_noshared | 152.0M | 205 | 101.5 | 0.86 | 159,649 | 0.86 | 83 | 15.4 | ok |
| noshared (!) | 152.0M | 226 | 92.3 | 0.79 | 145,152 | 0.79 | 83 | 15.4 | ok |
| nemo_cutlass_latent | 112.7M | 157 | 119.0 | 1.01 | 208,846 | 1.13 | 75 | 13.6 | ok |
| latent | 112.7M | 181 | 102.9 | 0.88 | 180,590 | 0.98 | 75 | 13.6 | ok |

### v17_moe50M500M

reference: `moe`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| nemo | 166.1M | 176 | 118.0 | 0.98 | 185,654 | 0.98 | 83 | 14.7 | ok |
| conn8 | 166.1M | 173 | 120.2 | 1.00 | 189,083 | 1.00 | 83 | 14.7 | ok |
| noshared_cutlass | 152.0M | - | - | - | - | - | - | 15.4 | FAIL: ChildFailedError:  |
| nemo_cutlass | 166.1M | 164 | 127.4 | 1.06 | 200,293 | 1.06 | 83 | 14.7 | ok |
| moe | 166.1M | 173 | 120.2 | 1.00 | 188,919 | 1.00 | 83 | 14.7 | ok |
| nemo_flex | 166.1M | 170 | 122.9 | 1.02 | 193,265 | 1.02 | 83 | 14.6 | ok |
| cutlass_gg | 166.1M | 165 | 126.2 | 1.05 | 198,474 | 1.05 | 83 | 14.7 | ok |
| nemo_cutlass_noshared | 152.0M | 214 | 97.5 | 0.81 | 153,265 | 0.81 | 83 | 15.4 | ok |
| moe_nocg | 166.1M | 182 | 114.8 | 0.96 | 180,441 | 0.96 | 83 | 15.8 | ok |
| nemo_flex_nocg | 166.1M | 180 | 115.5 | 0.96 | 181,540 | 0.96 | 83 | 15.7 | ok |
| nemo_cutlass_latent | 112.7M | 153 | 121.8 | 1.01 | 213,681 | 1.13 | 75 | 13.6 | ok |
| moe_ep1 | 505.9M | 174 | 119.5 | 0.99 | 187,944 | 0.99 | 83 | 16.3 | ok |
| nemo_flex_rf | 166.1M | 167 | 124.9 | 1.04 | 196,510 | 1.04 | 83 | 14.7 | ok |
| dense50M | 80.8M | 79 | 263.6 | 2.19 | 414,785 | 2.20 | 83 | 11.5 | ok |
| nemo_flex_a2a | 166.1M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| noshared (!) | 152.0M | 223 | 93.6 | 0.78 | 147,107 | 0.78 | 83 | 15.4 | ok |
| nemo_a2a | 166.1M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| auxfree | 166.1M | 186 | 112.2 | 0.93 | 176,552 | 0.93 | 83 | 14.7 | ok |
| latent | 112.7M | 179 | 104.0 | 0.87 | 182,653 | 0.97 | 75 | 13.6 | ok |
| zero16 | 166.2M | 181 | 115.3 | 0.96 | 181,389 | 0.96 | 83 | 14.7 | ok |
| muown | 166.1M | 203 | 102.5 | 0.85 | 161,260 | 0.85 | 83 | 14.7 | ok |
| muown_offload | 166.1M | 212 | 98.1 | 0.82 | 154,311 | 0.82 | 83 | 14.7 | ok |
| adam_offload (!) | 166.1M | 269 | 77.6 | 0.65 | 121,905 | 0.65 | 83 | 14.2 | ok |
| gdn3 | 168.6M | 185 | 96.6 | 0.80 | 177,364 | 0.94 | 71 | 16.2 | ok |
| router_fusion | 166.1M | 177 | 117.8 | 0.98 | 185,287 | 0.98 | 83 | 14.8 | ok |
| a2a_overlap | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| gc10 | 166.1M | 214 | 98.5 | 0.82 | 153,408 | 0.81 | 84 | 14.7 | ok |

### v1_moe50M500M

reference: `moe`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| moe | 166.1M | 240 | 86.8 | 1.00 | 136,590 | 1.00 | 83 | 28.2 | ok |
| moe_nocg | 166.1M | 255 | 81.8 | 0.94 | 128,603 | 0.94 | 83 | 29.3 | ok |
| moe_ep1 | 505.9M | 241 | 86.5 | 1.00 | 135,995 | 1.00 | 83 | 29.8 | ok |
| dense50M | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| noshared (!) | 152.0M | 305 | 68.3 | 0.79 | 107,524 | 0.79 | 83 | 28.9 | ok |
| auxfree | 166.1M | 259 | 80.6 | 0.93 | 126,738 | 0.93 | 83 | 28.2 | ok |
| latent | 112.7M | 249 | 74.9 | 0.86 | 131,519 | 0.96 | 75 | 27.1 | ok |
| zero16 | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| muown (!) | 166.1M | 438 | 47.6 | 0.55 | 74,873 | 0.55 | 83 | 28.1 | ok |
| gdn3 | 168.6M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |

### v22_moe50M500M

reference: `moe`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| nemo_cutlass | 166.1M | 164 | 127.2 | 1.00 | 200,049 | 1.00 | 83 | 14.7 | ok |
| moe | 166.1M | 164 | 127.0 | 1.00 | 199,683 | 1.00 | 83 | 14.7 | ok |
| nemo | 166.1M | 176 | 118.5 | 0.93 | 186,394 | 0.93 | 83 | 14.7 | ok |
| noshared_cutlass (!) | 152.0M | 216 | 96.5 | 0.76 | 151,739 | 0.76 | 83 | 15.4 | ok |
| conn8 | 166.1M | 165 | 126.1 | 0.99 | 198,293 | 0.99 | 83 | 14.7 | ok |
| nemo_cutlass_noshared (!) | 152.0M | 211 | 98.8 | 0.78 | 155,372 | 0.78 | 83 | 15.4 | ok |
| moe_nocg | 166.1M | 174 | 119.5 | 0.94 | 188,052 | 0.94 | 83 | 15.8 | ok |
| nemo_flex | 166.1M | 169 | 123.1 | 0.97 | 193,550 | 0.97 | 83 | 14.6 | ok |
| cutlass_gg | 166.1M | 163 | 127.5 | 1.00 | 200,477 | 1.00 | 83 | 14.7 | ok |
| nemo_cutlass_latent | 112.7M | 150 | 124.2 | 0.98 | 218,017 | 1.09 | 75 | 13.6 | ok |
| moe_ep1 | 505.9M | 175 | 119.3 | 0.94 | 187,728 | 0.94 | 83 | 16.3 | ok |
| nemo_flex_nocg | 166.1M | 181 | 115.2 | 0.91 | 181,139 | 0.91 | 83 | 15.7 | ok |
| dense50M | 80.8M | 79 | 263.0 | 2.07 | 413,737 | 2.07 | 83 | 11.5 | ok |
| nemo_flex_rf | 166.1M | 168 | 124.5 | 0.98 | 195,630 | 0.98 | 83 | 14.7 | ok |
| noshared (!) | 152.0M | 207 | 100.8 | 0.79 | 158,491 | 0.79 | 83 | 15.4 | ok |
| nemo_flex_a2a | 166.1M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| auxfree | 166.1M | 165 | 126.5 | 1.00 | 198,895 | 1.00 | 83 | 14.7 | ok |
| nemo_a2a | 166.1M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| latent | 112.7M | 174 | 107.5 | 0.85 | 188,647 | 0.94 | 75 | 13.6 | ok |
| zero16 | 166.2M | 178 | 116.9 | 0.92 | 183,832 | 0.92 | 83 | 14.7 | ok |
| muown | 166.1M | 202 | 103.0 | 0.81 | 162,057 | 0.81 | 83 | 14.7 | ok |
| muown_offload (!) | 166.1M | 206 | 101.2 | 0.80 | 159,223 | 0.80 | 83 | 14.7 | ok |
| adam_offload (!) | 166.1M | 263 | 79.2 | 0.62 | 124,475 | 0.62 | 83 | 14.2 | ok |
| gdn3 (!) | 168.6M | 184 | 97.2 | 0.77 | 178,572 | 0.89 | 71 | 16.2 | ok |
| router_fusion | 166.1M | 162 | 128.8 | 1.01 | 202,584 | 1.01 | 83 | 14.8 | ok |
| a2a_overlap | 166.1M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| gc10 | 166.1M | 204 | 105.8 | 0.83 | 160,431 | 0.80 | 86 | 14.7 | ok |

### v2_moe50M500M

reference: `moe`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| moe | 166.1M | 174 | 119.9 | 1.00 | 188,539 | 1.00 | 83 | 14.7 | ok |
| moe_nocg | 166.1M | 182 | 114.6 | 0.96 | 180,242 | 0.96 | 83 | 15.8 | ok |
| moe_ep1 | 505.9M | 174 | 119.8 | 1.00 | 188,376 | 1.00 | 83 | 16.3 | ok |
| dense50M | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| noshared (!) | 152.0M | 245 | 85.0 | 0.71 | 133,720 | 0.71 | 83 | 15.4 | ok |
| auxfree | 166.1M | 185 | 112.3 | 0.94 | 176,695 | 0.94 | 83 | 14.7 | ok |
| latent | 112.7M | 187 | 99.8 | 0.83 | 175,136 | 0.93 | 75 | 13.6 | ok |
| zero16 | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| muown (!) | 166.1M | 372 | 56.0 | 0.47 | 88,169 | 0.47 | 83 | 14.7 | ok |
| muown_offload (!) | 166.1M | 448 | 46.5 | 0.39 | 73,086 | 0.39 | 83 | 14.7 | ok |
| adam_offload (!) | 166.1M | 416 | 50.1 | 0.42 | 78,836 | 0.42 | 83 | 14.2 | ok |
| gdn3 (!) | 168.6M | 187 | 95.4 | 0.80 | 175,230 | 0.93 | 71 | 16.5 | ok |

### v2fix_moe50M500M

reference: `moe`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dense50M | 80.8M | 79 | 263.1 | - | 413,737 | - | 83 | 11.5 | ok |
| zero16 | 166.2M | 180 | 115.4 | - | 181,540 | - | 83 | 14.7 | ok |

### v7muown_moe50M500M

reference: `moe`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| moe | 166.1M | 173 | 120.7 | 1.00 | 189,794 | 1.00 | 83 | 14.7 | ok |
| muown | 166.1M | 203 | 102.7 | 0.85 | 161,458 | 0.85 | 83 | 14.7 | ok |
| muown_offload | 166.1M | 210 | 99.1 | 0.82 | 155,816 | 0.82 | 83 | 14.7 | ok |
| adam_offload (!) | 166.1M | 271 | 76.8 | 0.64 | 120,893 | 0.64 | 83 | 14.2 | ok |

### v25_moe120M1200M

reference: `moe`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| nemo | 369.0M | 256 | 121.3 | 0.90 | 128,100 | 0.90 | 124 | 18.7 | ok |
| nemo_cutlass | 369.0M | 229 | 135.2 | 1.00 | 142,842 | 1.00 | 124 | 18.7 | ok |
| moe | 369.0M | 230 | 134.6 | 1.00 | 142,253 | 1.00 | 124 | 18.7 | ok |
| noshared_cutlass (!) | 351.3M | 325 | 95.5 | 0.71 | 100,856 | 0.71 | 124 | 19.8 | ok |
| conn8 | 369.0M | 232 | 133.9 | 0.99 | 141,424 | 0.99 | 124 | 18.7 | ok |
| nemo_flex | 369.0M | 235 | 131.9 | 0.98 | 139,349 | 0.98 | 124 | 18.6 | ok |
| nemo_cutlass_noshared (!) | 351.3M | 344 | 90.2 | 0.67 | 95,297 | 0.67 | 124 | 19.8 | ok |
| moe_nocg | 369.0M | 236 | 131.3 | 0.98 | 138,730 | 0.98 | 124 | 21.7 | ok |
| cutlass_gg | 369.0M | 227 | 136.6 | 1.01 | 144,289 | 1.01 | 124 | 18.7 | ok |
| nemo_flex_nocg | 369.0M | 249 | 124.8 | 0.93 | 131,784 | 0.93 | 124 | 21.6 | ok |
| nemo_cutlass_latent | 234.5M | 240 | 120.5 | 0.90 | 136,562 | 0.96 | 116 | 16.6 | ok |
| moe_ep1 | 1218.3M | 282 | 110.1 | 0.82 | 116,322 | 0.82 | 124 | 23.4 | ok |
| nemo_flex_rf | 369.0M | 236 | 131.6 | 0.98 | 139,024 | 0.98 | 124 | 18.8 | ok |
| dense120M | 120.0M | 100 | 309.0 | 2.29 | 326,375 | 2.29 | 124 | 13.1 | ok |
| nemo_flex_a2a | 369.0M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| noshared (!) | 351.3M | 315 | 98.6 | 0.73 | 104,158 | 0.73 | 124 | 19.8 | ok |
| nemo_a2a | 369.0M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| auxfree | 369.0M | 258 | 120.3 | 0.89 | 127,106 | 0.89 | 124 | 18.7 | ok |
| latent | 234.5M | 233 | 124.2 | 0.92 | 140,665 | 0.99 | 116 | 16.6 | ok |
| zero16 (!) | 369.1M | 288 | 107.6 | 0.80 | 113,699 | 0.80 | 124 | 21.4 | ok |
| muown (!) | 369.0M | 316 | 98.1 | 0.73 | 103,581 | 0.73 | 124 | 18.7 | ok |
| muown_offload (!) | 369.0M | 328 | 94.5 | 0.70 | 99,826 | 0.70 | 124 | 18.7 | ok |
| adam_offload (!) | 369.0M | 349 | 88.9 | 0.66 | 93,931 | 0.66 | 124 | 17.6 | ok |
| gdn3 (!) | 374.5M | 262 | 102.8 | 0.76 | 125,284 | 0.88 | 108 | 21.3 | ok |
| router_fusion | 369.0M | 226 | 137.3 | 1.02 | 145,055 | 1.02 | 124 | 18.9 | ok |
| a2a_overlap | 369.0M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  |
| gc10 | 369.0M | 275 | 115.0 | 0.85 | 119,156 | 0.84 | 127 | 18.7 | ok |

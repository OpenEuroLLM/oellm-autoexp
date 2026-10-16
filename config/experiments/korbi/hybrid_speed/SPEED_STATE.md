# Hybrid speed state

updated 2026-10-16 14:30 from `/gpfs/scratch/ehpc533/users/korbip/hybrid_v019_runs/autoexp/hybrid_speed` (median step after iteration 30; one node, 4x H100 64GB; H100 bf16 dense peak ~989 TFLOP/s)

## Notes (hand-written; the tables below are regenerated)

Setup: hidden 768, 12 blocks, seq 4096, mbs 8, gbs 32 (DP 4), mock data, bf16, CUDA graphs
`local` unless an arm says otherwise. TFLOP/s is Megatron's analytic count (per-architecture
formula); tok/s/GPU is the formula-free comparison. torchtitan's MFU count gives nearly the
same FLOPs/token for this model (~965M), so a "400 TFLOP/s" target means the same in both.

### Baseline (target >= 400 TFLOP/s)
- `dense` 186 -> `base_liger` 269 (Liger fused LM-head CE: no 50k-vocab logits; memory 26.6 -> 12.9 GB)
- Profile of `base_liger` (steady steps): GPU busy 98% -- kernel-bound, not host-bound
  (an earlier "55% busy" was a double-counting bug in kernel_summary.py, fixed).
  Per step: GEMM 37 ms (~65% of peak), elementwise 27 (SwiGLU 8.6, residual adds 5,
  RoPE 3.8, q/k/v concat 2.6), attention 21 (cuDNN bwd ~355 TFLOP/s), norms 11,
  comm 8, copies 7 (2.4 ms H2D = the dense attention mask), loss 4.6.
- No gain: TE CE (disabled in this Megatron), flash backend (0.92x), FP8 (0.97x),
  full-iteration graph (1.03x), no distributed optimizer (0.98x); qk-norm costs 7%.
- In flight: no dataloader mask, fused q/k norm + RoPE (new Triton kernel, Megatron
  `fused_qk_norm_rope`), TE fused residual/RMSNorm, TE op fuser, TE SwiGLU, mbs 16.
- Next: attention (FA3 / newer cuDNN in the 26.09 container), SwiGLU fusion, the titan
  comparison (needs the 26.09 container: this torchtitan needs torch > 2.10).

### Features (vs the old 186 TFLOP/s reference; to be re-run on the optimized baseline)
Below 80%: mamba2, GDN, KDA family, triadic, power (Triton baseline), mHC, nGPT; MLA's
low TFLOP/s is Megatron's FLOP formula (its tok/s matches dense).

Open failures:
- `routed_k2*`: routed residual refuses CUDA graphs (needs graph support, or run without)
- `attn_res`: CUDA-graph runner mismatch at capture
- `pkm`, `tri_ckda_e4`: out of memory at mbs 8 (rerun at mbs 4)

Fixed: GDN under CUDA graphs, NoPE MLA (RoPE fusion flag), SWA window value, power at mbs 4.

## dense130M: latest per arm

reference: `dense`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status | suite |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| dense | 123.6M | 102 | 309.8 | 1.00 | 320,000 | 1.00 | 127 | 13.4 | ok | v4mix_dense130M |
| dense_gpt | 123.6M | 102 | 310.9 | 1.00 | 321,255 | 1.00 | 127 | 13.4 | ok | v2_dense130M |
| dense_nocg | 123.6M | 102 | 310.1 | 1.00 | 320,313 | 1.00 | 127 | 13.6 | ok | v2_dense130M |
| nope | 123.6M | 107 | 296.0 | 0.96 | 305,957 | 0.96 | 127 | 12.8 | ok | v2_dense130M |
| gqa4 | 123.6M | 102 | 311.2 | 1.00 | 321,570 | 1.00 | 127 | 13.4 | ok | v2_dense130M |
| xsa | 123.6M | 112 | 283.8 | 0.92 | 293,095 | 0.92 | 127 | 13.9 | ok | v3xsa_dense130M |
| gate | 130.7M | 110 | 289.1 | 0.93 | 298,706 | 0.93 | 127 | 15.1 | ok | v3gate_dense130M |
| xsa_gate | 130.7M | 118 | 267.6 | 0.86 | 276,523 | 0.86 | 127 | 15.1 | ok | v3gate_dense130M |
| swa3 | 123.6M | 95 | 290.2 | 0.94 | 345,290 | 1.08 | 110 | 13.4 | ok | v2_dense130M |
| mla (!) | 116.5M | 106 | 176.6 | 0.57 | 308,841 | 0.97 | 75 | 12.5 | ok | v2_dense130M |
| mla_gqa_xsa_gate (!) | 116.9M | 115 | 162.9 | 0.53 | 285,063 | 0.89 | 75 | 13.2 | ok | v2_dense130M |
| mamba2 (!) | 148.7M | 174 | 185.3 | 0.60 | 188,214 | 0.59 | 129 | 16.7 | ok | v2_dense130M |
| gdn_native (!) | 129.1M | 137 | 201.5 | 0.65 | 239,357 | 0.75 | 110 | 16.4 | ok | v4mix_dense130M |
| gated_delta_net (!) | 129.1M | 137 | 201.5 | 0.65 | 239,357 | 0.75 | 110 | 16.4 | ok | v4mix_dense130M |
| mlstm (!) | 129.1M | 113 | 244.8 | 0.79 | 291,012 | 0.91 | 110 | 14.3 | ok | v2_dense130M |
| kda (!) | 125.5M | 168 | 159.6 | 0.52 | 194,700 | 0.61 | 107 | 16.9 | ok | v4mix_dense130M |
| signed_kda (!) | 130.0M | 198 | 139.9 | 0.45 | 165,328 | 0.52 | 111 | 21.3 | ok | v4mix_dense130M |
| complex_kda (!) | 125.5M | 196 | 137.4 | 0.44 | 167,440 | 0.52 | 108 | 20.9 | ok | v4mix_dense130M |
| kda_xsa (!) | 125.5M | 176 | 152.4 | 0.49 | 185,970 | 0.58 | 107 | 17.7 | ok | v4mix_dense130M |
| kda_hd128 (!) | 127.3M | 156 | 177.2 | 0.57 | 210,727 | 0.66 | 110 | 15.2 | ok | v2_dense130M |
| tri_gdn_e4_cute (!) | 126.0M | 148 | 198.2 | 0.64 | 220,958 | 0.69 | 118 | 21.5 | ok | v3tri_dense130M |
| tri_kda_e4 (!) | 127.8M | 221 | 134.8 | 0.44 | 148,507 | 0.46 | 119 | 25.6 | ok | v3trikda_dense130M |
| tri_ckda_e4 (!) | 127.8M | 245 | 121.4 | 0.39 | 133,692 | 0.42 | 119 | 26.7 | ok | v3tri_dense130M |
| power (!) | 123.7M | 259 | 145.3 | 0.47 | 126,469 | 0.40 | 151 | 12.5 | ok | v4mix_dense130M |
| shortconv | 151.9M | 123 | 302.8 | 0.98 | 266,082 | 0.83 | 149 | 16.9 | ok | v3sconv_dense130M |
| canon (!) | 124.0M | 154 | 205.6 | 0.66 | 212,503 | 0.66 | 127 | 15.6 | ok | v2fix_dense130M |
| engram (!) | 192.3M | 130 | 246.7 | 0.80 | 252,450 | 0.79 | 128 | 15.4 | ok | v2_dense130M |
| pkm | 219.8M | - | - | - | - | - | - | 35.3 | FAIL: ChildFailedError:  | v2_dense130M |
| mhc4 (!) | 125.4M | 188 | 168.4 | 0.54 | 173,928 | 0.54 | 127 | 19.2 | ok | v2_dense130M |
| routed_k2 (!) | 123.7M | 141 | 225.4 | 0.73 | 232,893 | 0.73 | 127 | 18.3 | ok | v2_dense130M |
| routed_k2_mix (!) | 124.0M | 227 | 139.5 | 0.45 | 144,130 | 0.45 | 127 | 18.5 | ok | v2_dense130M |
| routed_k2_rev (!) | 123.7M | 152 | 208.9 | 0.67 | 215,863 | 0.67 | 127 | 15.1 | ok | v2_dense130M |
| attn_res (!) | 123.6M | 136 | 233.3 | 0.75 | 241,118 | 0.75 | 127 | 14.8 | ok | v2_dense130M |
| postnorm | 123.6M | 107 | 296.2 | 0.96 | 305,957 | 0.96 | 127 | 14.5 | ok | v2_dense130M |
| ngpt (!) | 123.6M | 161 | 196.9 | 0.64 | 203,402 | 0.64 | 127 | 16.2 | ok | v2_dense130M |
| ple128 | 203.2M | 118 | 272.9 | 0.88 | 277,930 | 0.87 | 129 | 15.4 | ok | v2_dense130M |
| value_emb (!) | 587.2M | 134 | 237.5 | 0.77 | 245,361 | 0.77 | 127 | 17.5 | ok | v2_dense130M |
| muon | 123.6M | 109 | 290.4 | 0.94 | 300,073 | 0.94 | 127 | 13.3 | ok | v2_dense130M |
| muown | 123.6M | 113 | 281.2 | 0.91 | 290,496 | 0.91 | 127 | 13.3 | ok | v2_dense130M |
| adam_precaware | 123.6M | 102 | 309.6 | 1.00 | 320,000 | 1.00 | 127 | 13.3 | ok | v2_dense130M |
| adam_offload (!) | 123.6M | 132 | 239.7 | 0.77 | 247,586 | 0.77 | 127 | 13.3 | ok | v2_dense130M |
| muown_offload | 123.6M | 115 | 276.4 | 0.89 | 285,684 | 0.89 | 127 | 13.3 | ok | v2_dense130M |
| base_liger | 123.6M | 102 | 309.9 | 1.00 | 320,156 | 1.00 | 127 | 13.4 | ok | v2_dense130M |
| base_ce_te | - | - | - | - | - | - | - | - | FAIL: ChildFailedError:  | v2_dense130M |
| base_flash | 123.6M | 118 | 269.1 | 0.87 | 278,166 | 0.87 | 127 | 13.9 | ok | v2_dense130M |
| base_graph_full | 123.6M | 100 | 315.9 | 1.02 | 326,375 | 1.02 | 127 | 13.6 | ok | v2_dense130M |
| base_liger_mbs16 | 123.6M | 197 | 322.6 | 1.04 | 333,262 | 1.04 | 254 | 24.3 | ok | v2_dense130M |
| base_liger_nocg | 123.6M | 103 | 307.9 | 0.99 | 318,136 | 0.99 | 127 | 13.6 | ok | v2_dense130M |
| base_liger_nodistopt | 123.6M | 105 | 301.1 | 0.97 | 311,039 | 0.97 | 127 | 13.7 | ok | v2_dense130M |
| base_liger_nomask | 123.6M | 102 | 311.0 | 1.00 | 321,255 | 1.00 | 127 | 13.4 | ok | v2_dense130M |
| base_liger_nomask_mbs16_nocg | 123.6M | 197 | 321.4 | 1.04 | 332,080 | 1.04 | 254 | 24.5 | ok | v2_dense130M |
| base_liger_noqkln | 123.6M | 102 | 310.1 | 1.00 | 320,313 | 1.00 | 127 | 11.6 | ok | v2fix_dense130M |
| base_liger_mbs4 | 123.6M | 106 | 299.1 | 0.97 | 308,986 | 0.97 | 127 | 7.9 | ok | v2_dense130M |
| base_qknr | 123.6M | 101 | 313.1 | 1.01 | 323,475 | 1.01 | 127 | 13.4 | ok | v2_dense130M |
| base_resnorm | 123.6M | 102 | 310.2 | 1.00 | 320,626 | 1.00 | 127 | 13.4 | ok | v2_dense130M |
| base_opfuser | 123.6M | 103 | 308.9 | 1.00 | 319,065 | 1.00 | 127 | 13.4 | ok | v2_dense130M |
| base_qknr_resnorm | 123.6M | 103 | 309.2 | 1.00 | 319,532 | 1.00 | 127 | 13.4 | ok | v2_dense130M |
| base_best | 123.6M | 102 | 310.1 | 1.00 | 320,313 | 1.00 | 127 | 13.4 | ok | v2_dense130M |
| base_best_swiglu | 123.6M | 102 | 309.9 | 1.00 | 320,313 | 1.00 | 127 | 13.4 | ok | v2_dense130M |
| base_best_mbs16_nocg | 123.6M | 196 | 323.9 | 1.05 | 334,623 | 1.05 | 254 | 24.5 | ok | v2_dense130M |
| base_best_hd128 | 123.6M | 96 | 329.2 | 1.06 | 340,270 | 1.06 | 127 | 13.4 | ok | v2_dense130M |
| base_fp8 | 123.6M | 107 | 296.3 | 0.96 | 306,243 | 0.96 | 127 | 12.5 | ok | v2_dense130M |
| mamba2_ck64 (!) | 148.7M | 185 | 174.6 | 0.56 | 177,220 | 0.55 | 129 | 17.1 | ok | v2ck_dense130M |
| mamba2_ck256 (!) | 148.7M | 177 | 182.3 | 0.59 | 185,130 | 0.58 | 129 | 17.1 | ok | v2ck_dense130M |
| mlstm_ck64 (!) | 129.1M | 116 | 236.9 | 0.76 | 282,361 | 0.88 | 110 | 14.3 | ok | v2ck_dense130M |
| mlstm_ck256 | 129.1M | 111 | 249.3 | 0.80 | 294,412 | 0.92 | 111 | 14.3 | ok | v2ck_dense130M |
| power_ck64 (!) | 123.7M | 386 | 97.6 | 0.32 | 84,913 | 0.27 | 151 | 25.6 | ok | v2ck_dense130M |
| power_ck256 (!) | 123.7M | 240 | 157.2 | 0.51 | 136,790 | 0.43 | 151 | 12.5 | ok | v2ck_dense130M |
| base_liger_teact | 123.6M | - | - | - | - | - | - | - | FAIL: ChildFailedError:  | v4base_dense130M |

## moe50M500M: latest per arm

reference: `moe`; flagged (!) below 80% of its TFLOP/s

| arm | params | step ms | TFLOP/s | vs ref | tok/s/GPU | vs ref | TF/step | mem GB | status | suite |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| moe | 166.1M | 174 | 119.9 | 1.00 | 188,539 | 1.00 | 83 | 14.7 | ok | v2_moe50M500M |
| moe_nocg | 166.1M | 182 | 114.6 | 0.96 | 180,242 | 0.96 | 83 | 15.8 | ok | v2_moe50M500M |
| moe_ep1 | 505.9M | 174 | 119.8 | 1.00 | 188,376 | 1.00 | 83 | 16.3 | ok | v2_moe50M500M |
| dense50M | 80.8M | 79 | 263.1 | 2.19 | 413,737 | 2.19 | 83 | 11.5 | ok | v2fix_moe50M500M |
| noshared (!) | 152.0M | 245 | 85.0 | 0.71 | 133,720 | 0.71 | 83 | 15.4 | ok | v2_moe50M500M |
| auxfree | 166.1M | 185 | 112.3 | 0.94 | 176,695 | 0.94 | 83 | 14.7 | ok | v2_moe50M500M |
| latent | 112.7M | 187 | 99.8 | 0.83 | 175,136 | 0.93 | 75 | 13.6 | ok | v2_moe50M500M |
| zero16 | 166.2M | 180 | 115.4 | 0.96 | 181,540 | 0.96 | 83 | 14.7 | ok | v2fix_moe50M500M |
| muown (!) | 166.1M | 372 | 56.0 | 0.47 | 88,169 | 0.47 | 83 | 14.7 | ok | v2_moe50M500M |
| gdn3 (!) | 168.6M | 187 | 95.4 | 0.80 | 175,230 | 0.93 | 71 | 16.5 | ok | v2_moe50M500M |
| muown_offload (!) | 166.1M | 448 | 46.5 | 0.39 | 73,086 | 0.39 | 83 | 14.7 | ok | v2_moe50M500M |
| adam_offload (!) | 166.1M | 416 | 50.1 | 0.42 | 78,836 | 0.42 | 83 | 14.2 | ok | v2_moe50M500M |

## Per suite

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

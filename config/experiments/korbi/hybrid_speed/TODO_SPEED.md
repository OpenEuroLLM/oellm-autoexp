# Speed TODO: what is left on the table, feature by feature

Ratios are tok/s/GPU vs the family reference (SPEED_STATE.md, 2026-10-16). "Bound" is what the
feature should cost if implemented well: its extra FLOPs / bytes, or the overhead its paper
or other frameworks report. **Left**: L = large (>= 15% of the arm's speed), M = 5-15%,
S = < 5% / nothing. Goal: every arm >= 0.7 of dense, and no L left before scaling up.

## Correctness first

- adam_offload resume is not exact (ck2: dense loss at iteration 55 7.351 straight vs 7.482
  resumed; MoE 0.0111 vs 0.0105 at iteration 60); muown / muown_offload resume match to ~4 digits.
  Fixed so far in HybridDeviceOptimizer: the GPU part's FusedAdam step was not saved
  (bias correction restarted), fp32 GPU grads raced the GPU step, fp32 CPU masters were
  pageable. Neither explains the 100%-offload runs: kt_hdo runs a dist-checkpoint
  resume test (tests/unit_tests/dist_checkpointing/test_optimizer_cpu_offload_resume.py)
  on MN5; then re-run ck2 for adam_offload.

## 0. Baseline first (blocks everything below)

| item | now | left | action |
|---|---|---|---|
| dense vs torchtitan | **decided: Megatron.** Same 130M / FineWeb / node, best of each: Megatron mbs 32 343.8k tok/s/GPU (342.9k without CUDA graphs; mbs 16 331.9k) vs titan main GraphTrainer (whole-step graph, full Inductor) + FA3 varlen, mbs 32: 319.5k (38.6 % MFU; mbs 16 305.2k; flex instead of FA3 292.4k; FSDP2 + regional compile 252.2k). flex_flash (FA4) hangs on H100 (upstream: FA4 hangs under compile on sm90) | Megatron 1.08x ahead at 130M; at A3B-30B (8 nodes, attention-only twin) titan HybridEP ~3.63k vs Megatron attn_tuned ~6.6k tok/s/GPU: **0.55x** | none; revisit only if titan gains whole-step graphs for MoE / our mixers |
| dense MFU | ~310 TFLOP/s analytic (~31% of 989) at 130M; GEMM ~65% of peak, elementwise ~20% of step | M | the remaining elementwise (SwiGLU, residual adds, copies): torch.compile'd blocks or TE fusions; mbs 16 = +4% |
| nope | 0.96 | S | DONE 2026-10-17: the fused q/k-norm path covers NoPE (freqs=None: norm only); zero-centred gamma was already handled at the call site (1 + w). Re-measure nope with the fusion |
| gqa4 / mla / swa3 / dense_gpt | 1.00 / 0.97 / 1.07 / 1.00 | S | - |
| base_fp8 | 0.95 | S | not worth it at 130M (re-check at >= 1B) |
| base_ce_te | FAIL | S | Liger covers it; drop the arm |

## 0b. Data loading on MN5 (GPFS stalls)

Real-data runs can stall on the dataloader: GPFS read latency spikes, the same failure as the
JUPITER main sweep (fixed there with 16 workers + prefetch_factor 8). Speed numbers so far are
mostly mock data or FineWeb on one node, so a data stall would not show up in them.

| item | now | left | action |
|---|---|---|---|
| node-local prefetch + windowed shuffle (branch `data_prefetch_shuffle`) | **merged** (df970e9) and **wired in as an option, off by default**: `+data_staging=mirror_mn5` (config/data_staging/mirror_mn5.yaml) runs pretrain_hybrid through `scripts/pretrain_hybrid_prefetch.py`. That wrapper applies the patches (node-local mirror read-through, K-lane windowed shuffle, consumption cursor, and new: the `cyclic` sampler reads the windowed order sequentially and repeats at the end instead of Megatron's per-epoch randperm). Local rank 0 per node also starts the prefetcher in the container, with arguments from the run's own Megatron arguments, so it works with arrays: per-split JSON via `--prefix json:<path>` (156 datamix4 prefixes), data-parallel ranks per node from TP/CP/PP. v0.19 fix: `IndexedDataset.initialize` pass-through. Unit tests: tests/unit/data_staging/test_megatron_patch_samplers.py (5 pass). Not run on MN5 yet | M for many concurrent small runs (shared GPFS), likely S for a single large run. Observed 2026-10-17: a campaign run's per-run index build for the 156-prefix datamix4 blend took > 10 min on rank 0 and the other ranks' barrier hit the NCCL timeout (hp_s1_3bt gbs 256). Timeout raised to 45 min in base_train_mn5.yaml; a shared, prebuilt index cache (built once, read-only for the array) would remove the rebuild entirely | when needed: smoke on MN5 (1 node, datamix4, s1): mirror active on all ranks, hit rate, the 156-file blend-weight discovery, bounded /scratch use; then the A/B (tok/s p50 / p99 step time vs stock) on a many-small-runs array before turning it on for the screen / HP sweeps |

## 1. Attention variants

| item | now | bound | left | action |
|---|---|---|---|---|
| xsa | 0.91 | ~0.97 (one projection per head) | M | fuse XSA into the attention epilogue / output-norm kernel |
| gate | 0.93 | ~0.95 (+1 GEMM of d x d) | S | gate GEMM into the QKV GEMM (one wider GEMM) |
| xsa_gate | 0.87 | ~0.93 | M | both of the above |
| mla_gqa_xsa_gate | 0.89 | ~0.93 | S/M | follows from xsa / gate |

## 2. Linear / recurrent mixers (3:1 hybrids)

| item | now | bound | left | action |
|---|---|---|---|---|
| mlstm | 0.91 | - | S | - |
| gated_delta_net / gdn_native | 0.76 | ~0.82 | M | GDN in_proj is 3096 wide (odd: sm80 GEMMs, as FLA's) -- pad inside the module; check FLA chunk kernels vs CuTe GDN (tri_gdn E=1) |
| kda / complex_kda | 0.66 / 0.67 (v30) | ~0.75 (GDN + per-channel gate) | M | FLA backends all within 2% (hybrid_optimized best, -1%); padded input GEMM ~0. Left: profile vs GDN: chunk_kda bwd self-time 3.7 ms/layer (vs GDN), causal conv bwd, gate |
| kda_xsa / kda_hd128 | 0.63 / 0.66 | follows kda | M | follows kda + xsa |
| tri_gdn_e4_cute | 0.73 | ~0.78 | M | fuse the second-key / slice-decay prep; hd != 128 runs the 2-4x slower expanded-key path (pad hd 64 -> 128 is exact for GDN) |
| tri_kda / tri_ckda | 0.57 / 0.56 (v30, Triton gram bwd: 1.41 vs 2.20 ms) | ~0.68 | L -> M | DONE: Triton gram bwd default. Left: prep fused into gram fwd (0.4 ms x2), glue bwd (0.8 ms), CuTe-side channel gate |
| power | 0.65 (hd64), 0.69 (hd128) with the attention form (v31: own Triton kernel v1; tuned v2 in v32) | flash 2.3 ms vs ours 7.2 ms / retention 8.2 ms per layer (hd64) | M | DONE: attention form + own flash-style kernel (no softmax rescaling). Next: v32 numbers; FA3-style (wgmma / TMA) kernel if still short |
| mamba2 | 0.62 | ~0.75 | M | parked (off by decision); Nemotron-3 setup = standard mamba-ssm |

## 3. Extra layers / memories

| item | now | bound | left | action |
|---|---|---|---|---|
| shortconv | 0.83 | - | S | - |
| canon | 0.75 | ~0.85 | M | canon needs the unfused input norm: fuse canon conv into the residual-norm kernel |
| engram | 0.79 | ~0.85 | S/M | table lookups / sparse all-to-all; profile once |
| pkm | 0.61 (mbs 4) | ~0.68 (bag kernels near bandwidth) | M | fuse the dp-shard gather into the bag fwd and the index_add into bag bwd_v; mbs 8 (memory) |
| ple128 | 0.87 | ~0.92 | S/M | profile |
| value_emb | 0.76 | ~0.9 | M/L | 190M extra embedding params with dense grads + all-reduce: sharded / sparse table like engram |

## 4. Residual schemes

| item | now | bound | left | action |
|---|---|---|---|---|
| mhc4 | 0.58 | ~0.8 at hidden 768 (stream traffic: ~6-7 passes over the 4x stream per sublayer, ~1 ms; the paper's ~7% is at 27B where GEMMs dominate) | L | write fwd is already fused near bandwidth; left: write bwd at ~3x its traffic bound (0.66 ms/sublayer), 870 small stream-grad adds (~10 ms/step), cat copies (~4 ms/step); bf16 streams if they are fp32; recompute for memory |
| routed_k2 | 0.73 | ~0.88 | M/L | same treatment as mhc: fused gather / scatter of the 2 routed streams |
| routed_k2_mix / _rev | 0.63 / 0.67 | ~0.85 | L | mix: fuse the balance + mix into one kernel; rev_cg FAILs under CUDA graphs |
| attn_res | 0.76 (no CUDA graphs) | ~0.9 (Block AttnRes reports small overhead) | M/L | make it CUDA-graphable (capture mismatch); fused depth-attention read |
| ngpt | 0.84 | ~0.92 | M | fused normalize + lerp kernels for all sublayers (only post-norm lerp compiled so far) |
| postnorm | 0.96 | - | S | - |

## 5. Optimizers

| item | now | left | action |
|---|---|---|---|
| muon / muown | 0.93 / 0.92 | S | batched Newton-Schulz done; amortized at larger gbs |
| adam_offload | 0.81 dense, 0.62 MoE | M | CPU Adam bound; keep for the large runs only ("in the run") |
| muown_offload | 0.91 dense, 0.80 MoE | S | - |

## 6. MoE (re-measure everything at 120M active / 1.2B total: v25)

| item | old (50M/500M) | left | action |
|---|---|---|---|
| moe vs dense of the same active width | **0.44** (v25: 142k vs 326k) | L | v25prof (one steady step, 276 ms vs dense 101): GPU covered 91% (not idle -- kernel_summary's step count is wrong for MoE). Main stream 182 ms vs dense 97; per step: param all-gather 34 ms + EP a2a SendRecv 28 ms (incl. imbalance waits), elementwise 21 ms (1530 launches), dispatcher chunk sort 16 ms, small expert GEMMs ~13 ms, permute/unpermute 9 ms, Adam over 370M params/GPU (expert DP = 1) 9 ms, router topk 5 ms. Drop-and-pad + MoE graph 1.09x (v29). Levers: EP a2a overlap (porting #4942/#4944), HybridEP without the chunk sort, fused router / permute elementwise |
| CP2 (dense suite) | kda/ckda/tri/power FAILed: CPU index copy in graph capture (fixed, v28cp2) | - | engram, pkm, residual schemes OK under CP2 |
| a2a_overlap / nemo_*a2a | FAIL (HybridModel has no build_schedule_plan) | L? | upstream: PR #4942 (grouped HybridStack + EP overlap) / #4944 (training wire-up), open; 19 commits, 14 conflicting files on our branch (hybrid_model / hybrid_block / allocation) -- port once the profile shows the a2a share |
| noshared (top-6) | 0.79 | M | top-6 dispatch cost; check with HybridEP |
| latent | 0.94 (1.09 on NeMo + CUTLASS) | - | CUTLASS GG on the ep image (TE 2.19) for all arms if stable |
| zero16 | 0.92 (no CUDA graphs) | M | make zero-compute experts CUDA-graphable |
| gc10 | 0.80 | - | GC every 10 steps is just expensive: keep 50+ |
| router_fusion / conn8 / auxfree / cutlass_gg / nemo_flex | ~1.00 | S | - |
| gdn3 | 0.89 | follows gdn | - |

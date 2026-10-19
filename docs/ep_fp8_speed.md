# EP backends + fp8 for the 3B-active hybrid MoE: integration and speed (2026-10-18)

**Goal (user, 2026-10-18):** test, integrate and optimise every expert-parallel backend of the
hybrid-ep image in Megatron-LM (HybridEP, DeepEP, MoonEP; TE's NCCL-EP comes with TE), including
fp8 with fused quantization, towards **> 250 TFLOP/s/GPU at the large scale**, after integration
and smoke tests at small scale. Budget: <= 1000 GPU-h on MN5 (ehpc533). Separate worktree
(`../oellm-autoexp-hybrid-ep`, branches `feat/ep-fp8-speed` / Megatron `ep-fp8-speed`), so the
round-4/5 work in the main checkout is untouched. Snapshot on MN5: `$R/autoexp_ep*`.

Companion: docs/hybrid_moe_3b_campaign.md §7.2, §11.1-§11.5 (the state this starts from).

## 0. Starting point

| model | layout | tok/s/GPU | TFLOP/s | note |
|---|---|---:|---:|---|
| A3B-30B attention-only, forced balancing | 8 nodes PP4 x VPP2 EP4 HybridEP, full recompute, 4k | 8,801 | 187.9 | exp_irene's 185.8 |
| A3B-30B CKDA 5:1 recipe | same | 6,908 | 137 | |
| production ~39B-A2.7B (160 x 768 top-5+1), routed 2/8, 8k | 16 nodes PP4 EP4 HybridEP, full recompute, mbs 1 | ~4,800 | ~82 | prodw, 2026-10-18 |

TFLOP/s is Megatron's model-FLOP count (`hybrid_flops`, x3 for fwd + bwd, **recompute not
counted**). At the production shape one token costs ~17.7 GFLOP (MoE 46 %, LM head 18 %, CKDA
27 %, attention 8 %), so 250 TFLOP/s = ~14.1k tok/s/GPU, 2.9x the current production number.

Ceilings that frame it: full recompute costs a whole extra forward (1/3 of the hardware FLOPs
are not counted), so the bf16 stack ceiling with full recompute was ~190 (188 x 4/3 = 250 without
recompute). The levers, roughly by size: (1) drop full recompute (memory: EP / PP layout,
selective recompute, fp8 activations, offload), (2) routing balance (forced balance gave +33 %
attention-only), (3) fp8 GEMMs with fused quantization, (4) EP comm overlap, (5) pipeline bubbles.

## 1. What the image has (MegatronTraining-hybrid-ep 202610161948)

- TE 2.19 + cuBLASLt 13.8, torch 2.14, DeepEP 1.2.1+de0dd11 (with HybridEP), MoonEP 0.0.1
  (d86fda4), NVSHMEM 3.7.2. **UCCL-EP is not installed** (best-effort build failed). Overlays can
  post-install / patch EP libraries (user, 2026-10-18).
- **HybridEP has an fp8 dispatch** (`HybridEPBuffer(use_fp8=True)`, uint8 tokens + scaling
  factor), contrary to §11.2 of the campaign doc; Megatron hard-codes it off (fused_a2a.py).
  DeepEP normal dispatch takes `(fp8 [N, H], scales [N, H/128])` (1x128 blockwise). Neither is
  wired in Megatron; the experts have no pre-quantized input path.
- **TE 2.19's graph-safe grouped path:** `tex.group_quantize` (one kernel for all experts) + one
  cuBLASLt grouped GEMM with device-side splits, for bf16, fp8 current scaling (cuBLASLt >= 13.5)
  and fp8 block scaling on Hopper (>= 13.6). The §11.1 fp8 microbenchmark measured the legacy
  split-quantize path (per-expert cast-transposes, host-bound). Megatron reaches the new path only
  through the op fuser, which on SwiGLU requires `NVTE_CUTEDSL_FUSED_GROUPED_MLP=1` (the sm100
  kernel's switch; on H100 TE then runs the unfused grouped ops). Megatron's unfused
  `TEGroupedMLP` passes host `m_splits` (`tokens_per_expert.tolist()`: one sync per MoE layer).
- **MoonEP**: NVLink-only (EP <= 4 on MN5), perfectly balanced dispatch through online-planned
  redundant experts: each rank computes exactly S x K tokens over 2 x epn expert slots (its epn
  experts + epn prefetched copies of hot remote experts), zero-copy static shapes. Needs the
  experts' weights as contiguous [epn, ...] tensors + NVLink prefetch pools, and a grad reduce of
  the prefetched copies: an expert-module integration, not only a dispatcher.

## 2. Monitor: hangs and memory

`templates/marenostrum_watchdog.sbatch` + `scripts/korbi/hybrid_speed/job_watchdog.py`
(`config/slurm/marenostrum_ehpc533_watchdog.yaml`; every ep_* suite uses it): one host-side
process per node samples nvidia-smi every 15 s into `<log dir>/watchdog/<jobid>/mem_<host>.csv`;
node 0 follows the Megatron log and, when no iteration appears within WATCHDOG_STARTUP_S or the
iteration does not advance for WATCHDOG_STALL_S, has every node dump py-spy stacks, writes a
`WATCHDOG:` verdict (memory-saturated or not, GPUs idle or not) plus per-GPU peak memory into the
job log, and scancels. Short wall times on top.

## 3. Phase A: one node, every backend x bf16 / fp8 (`ep_slice_mn5.yaml`, suite e1)

One 5:1 period (6 blocks) at the production layer shapes (hidden 2048, 160 x 768 top-5 + shared,
EP4 = 40 experts per GPU), 8k, mbs 1, gbs 32, no recompute, PP1, vocabulary 32k.

**Round 1 (e1) + 2 (e2), 30 steps, median of steps >= 8-10** (TFLOP/s = Megatron's count; the
slice has no recompute, so it is not comparable with the full-recompute A3B numbers):

| arm | tok/s/GPU | TFLOP/s | vs hep | mem | note |
|---|---:|---:|---:|---:|---|
| **hep** (flex / HybridEP) | 74,224 | 162.6 | 1.00 | 62 % | reference |
| hep_fuse (permute fused into the dispatch) | 74,644 | 163.6 | 1.01 | | |
| hep_force (forced balanced routing) | 74,403 | 163.0 | 1.00 | | balance is not the limit here (random-init router, EP4) |
| **hep_opfx** (TE grouped-tensor path, `NVTE_CUTEDSL_FUSED_GROUPED_MLP=1` + op fuser) | **77,073** | **168.9** | **1.04** | 62 % | one group_quantize + one cuBLASLt grouped GEMM, CUDA m_splits |
| hep_opf (op fuser alone) | fail | | | | Megatron asserts: SwiGLU needs the CuTe switch |
| hep_ovl (EP a2a overlap) | 73,534 | 161.1 | 0.99 | 69 % | nothing to overlap at PP1 / one micro-batch in flight |
| dep (flex / DeepEP) | 65,940 | 144.5 | 0.89 | | dep_ovl 0.90 |
| nep (flex / ncclep) | 69,137 | 151.5 | 0.93 | | after the TE 2.19 `ep_bootstrap(num_topk)` fix |
| a2a (alltoall) | 67,504 | 147.9 | 0.91 | | |
| hep_fp8t / hep_fp8b (fp8 tensorwise / blockwise, legacy path) | 59,820 / 66,845 | 131.1 / 146.5 | 0.81 / 0.90 | | |
| hep_fp8t_opfx / hep_fp8b_opfx (fp8 on the grouped path) | 66,831 / 70,996 | 146.4 / 155.6 | 0.90 / 0.96 | | still below bf16 (0.92 of hep_opfx) |
| dep_fp8b / a2a_fp8b | 58,593 / 58,884 | 128.4 / 129.0 | 0.79 / 0.79 | | a2a_fp8b after the pad_routing_map fix |
| hep_routed / hep_opfx_routed (routed residual 2 of 8) | 66,255 / 69,319 | 145.2 / 151.9 | 0.89 / 0.93 | | the routed residual costs 10 % |
| dep_routed | 59,235 | 129.8 | 0.80 | | |

**EP8 across two nodes (e3, DP 8):** ncclep 55,675 tok/s (122.0) = 0.87 of HybridEP EP4 within the
node (63,774, 139.7; same 2 nodes, expert DP 2) and 1.31x alltoall EP8 (42,529). **HybridEP and
DeepEP do not run internode on MN5**: HybridEP's DOCA path asserts in `cuMemHostRegister(...,
IOMEMORY)` of the NIC doorbell (`internode_doca.cu:124`: GPU-initiated RDMA needs a GPU mapping of
NIC MMIO, a driver setting MN5 nodes lack); DeepEP's internode kernels assume 8 GPUs per NVLink
domain (2 x 4 GPUs are taken for one 8-GPU node: "No NVLink connection between GPU 0 and GPU 0").
**ncclep is the internode EP backend on MN5.**

**MoonEP** needs NVLink multicast (NVSwitch) for its plan broadcast (`cuMulticastGetGranularity`
fails: "operation not supported"); MN5's 4-GPU boards have direct NVLink only. Patched
(`container/patches/moonep-no-multicast.patch`: unicast stores into every rank's chunk, then the
existing cross-rank barrier; `MOONEP_MULTICAST=auto|0|1`), installed as
`$HYBRID_ENV/pylibs_moonep` (the image's build + the patched Python), first on PYTHONPATH.

**Where the time goes (rank-0 profiles, one step = 8 micro-batches):** bf16 grouped path 911 ms
window, GPU busy 87 %: GEMM 286 ms, of which the grouped expert **wgrad** GEMMs 75 ms vs 17 ms for
the forward ones (fp32 main-grad accumulation at ~1k tokens per expert: memory-bound), CKDA
kernels 150 ms, elementwise 111 ms (fp32 grad adds 32 ms, bf16 adds 29 ms), HybridEP dispatch /
combine 60 ms + permute / unpermute 42 ms, optimizer 31 ms (expert DP 1). fp8 blockwise: GEMM
-45 ms, but +81 ms of quantization: 3,854 per-call `block_scaled_cast_transpose` launches (34 ms)
in the dense TE linears (attention, shared expert: too small for fp8) and 41 ms of grouped
quantization in the experts; GPU busy 81 %. -> experts-only fp8 (`fp8_experts_only`, new),
sync-free dispatch, more tokens per expert GEMM (round 3).

**Round 3 (e4 / e4b), reference = the grouped path (opfx, 168.5 TFLOP/s):**

| arm | tok/s/GPU | TFLOP/s | vs opfx | mem |
|---|---:|---:|---:|---:|
| opfx_cap (sync-free HybridEP, rank capacity 1.25) | 77,898 | 170.7 | 1.01 | 62 % |
| opfx_te (TE projections in CKDA) | 75,438 | 165.3 | 0.98 | 62 % |
| opfx_cap_te | 77,766 | 170.4 | 1.01 | 61 % |
| **opfx_mbs2** (micro batch 2) | 85,198 | 186.7 | **1.11** | 87 % |
| **opfx_cap_mbs2** | 85,313 | **186.9** | **1.11** | **73 %** |
| opfx_fp8eb / opfx_fp8et (fp8 in the routed experts only) | 72,266 / 68,251 | 158.3 / 149.5 | 0.94 / 0.89 | 63 % |
| opfx_mbs2_fp8eb | 81,908 | 179.5 | 0.96 of mbs2 | 83 % |
| opfx_cap_te_fp8eb | 72,032 | 157.8 | 0.94 | 63 % |

- **More tokens per expert GEMM is the per-layer lever** (+11 % at mbs 2): the grouped wgrad is
  memory-bound at ~1k tokens per expert. The static receive budget of the sync-free dispatch also
  cuts memory (73 vs 87 %: fewer allocator size classes).
- **fp8 does not pay on H100 for this model, even experts-only on the grouped path** (0.94 of
  bf16 blockwise, 0.89 tensorwise; 0.96 at mbs 2). The expert GEMMs are fwd ~17 ms, dgrad ~11 ms,
  wgrad ~75 ms per step: fp8 speeds up the first two, the memory-bound wgrad (fp32 main-grad
  accumulation) not at all, and the grouped quantization (41 ms/step) eats what is left.
- **fp8 dispatch (quantize before the all-to-all) does not change that**: HybridEP can send
  uint8 + per-128 fp32 scales (DeepSeek-style 1x128) and TE 2.19's op-fuser GroupedLinear takes a
  pre-quantized GroupedTensor, but rebuilding the columnwise copy the wgrad needs
  (`group_requantize_inplace`) is MXFP8-only (Blackwell); for 1x128 blockwise on Hopper the
  receiver would dequantize and requantize itself. The only saving left is half the dispatch
  bytes (~15 ms of a 900 ms step, < 2 %). Not built; worth revisiting on Blackwell (MXFP8: TE's
  NCCL-EP `dispatch_fwd_quant_recipe` does exactly this).

**MoonEP:** 4-GPU test against a dense fp32 reference passes (balanced and skewed routing, prefetched
copies 3 per step: output / dx / dprobs / dW1 / dW2 within 0.5 %, bf16 level;
tests/unit_tests/transformer/moe/test_moonep.py, job 47506054).

## 4. Phase B: the production model at full depth (8 nodes, gbs 128 = 16 micro-batches / pipeline)

~39B-A2.7B (48 blocks CKDA 5:1, 160 x 768 top-5 + shared, 8k doc-masked, 262k vocab), mbs 1,
full recompute, EP4 HybridEP, routed residual 2 of 8 unless "plain". 12-15 steps.

| arm | layout | TFLOP/s | tok/s/GPU | mem | note |
|---|---|---:|---:|---:|---|
| p_hep | PP4 even | 75.9 | 4,475 | 86 % | prodw-equivalent reference |
| p_opfx | PP4 even, grouped path | 76.9 | 4,534 | 86 % | |
| p_opfx_nomask | PP4, no document masking | 79.2 | 4,516 | 85 % | |
| p_opfx_vpp | VPP2 | OOM | | | 47.9 GiB PyTorch + ~14 GiB not PyTorch's |
| q_un | uneven PP4 15/15/14/4 | OOM | | | |
| **q_un_mem** | uneven PP4 + NCCL 4 ch + expandable segments | 76.6 | 4,516 | | memory settings make it fit |
| **q_vpp_mem / s_vpp_prof** | VPP2 + memory settings | 77.0 / 75.7 | 4,539 | 64 % | routed + VPP2 runs now (OOM / hang before) |
| s_vpp_cap15 | VPP2 + sync-free (cap 1.5) | 80.2 | 4,728 | 64 % | after the doc-masking rewind fix |
| q_un_plain | uneven PP4, plain residual | 86.2 | 5,082 | | |
| **s_vpp_plain** | VPP2, plain residual | **98.0** | 5,777 | 53 % | |
| s_vpp_mbs2 | VPP2, mbs 2 | OOM | | | |
| s_vpp_moon | VPP2, MoonEP | 71.3 | 4,207 | 66 % | |
| p_nep8 / q_nep8_pp2u | ncclep EP8, PP4 / PP2 | 62.8 / 69.7 | | | receive-capacity overflow (traps) at 1.5 and 2.0 |

- Neither the head-balanced uneven layout nor VPP2 beats even PP4 with the routed residual: the
  full-depth profile (VPP2, rank 0 / rank 24) spends 2.1 s / 4.45 s of a 7.5 s step in PP
  `SendRecv` kernels. The routed residual sends all 8 streams across every stage boundary
  (`depth_payload_slots`: 8 x [8192, 2048] bf16 = 268 MB per micro-batch, over InfiniBand
  because EP fills the node): routed costs 21 % at full depth (98.0 -> 77.0) vs 10 % in the slice.
- Fixes along the way: `pad_routing_map` in place (fused router), ncclep `num_topk` (TE 2.19),
  document masking + paged-stash retry (the cu_seqlens stash is replayed: `rewind()`).

**Round 4 (b4): real text** (nemotron-cc, 256k, 8k with document masking), VPP2 + memory settings:

| arm | routed 2 of 8 | plain residual |
|---|---:|---:|
| HybridEP grouped path | 73.4 | 86.2 |
| MoonEP (64 SMs) | 60.5 | **88.8** |
| HybridEP sync-free capacity 2.0 | 55.0 (re-runs) | |

MoonEP helps the plain hybrid (+3 %) but loses with the routed residual (to be profiled: the
routed sublayer recompute re-runs the MoE forward, i.e. MoonEP's planning / prefetch / dispatch).

**Why the hybrid is at half of Qwen3's TFLOP/s: exposed time** (`trace_breakdown.py`: union
of compute kernels over all streams; exposed = a comm kernel runs and no compute kernel does):

| per step | Qwen3 + MoonEP (rank 0) | hybrid routed, VPP2 (rank 0 / rank 24) |
|---|---:|---:|
| compute busy | 76 % | 55 % / 43 % |
| exposed PP p2p | 7 % | 20 % / 40 % |
| EP comm (not overlapped) | 15 % | 5 % / 3 % |
| idle (no kernel at all: host-bound) | 2 % | 16 % / 7 % |

- The last stage computes less (1.58 s vs 2.05 s per step on stage 0) and waits more: the
  pipeline is **stage-imbalanced**, the 262k head with the Liger fused CE costs ~4 blocks (not the
  10.5 its FLOPs suggest). Rebalanced VPP2 layout next: 7/7/7/5 | 6/6/6/4 + head.
- The hybrid is **host-bound** 16 % of the time at stage 0 (Qwen3 + MoonEP: 2 %).

**The ladder (h1): Qwen3-30B-A3B -> the production hybrid, one change per rung** (8 nodes, real
text, MoonEP 64 SMs, VPP2, full recompute, 4M tokens per step):

| rung | TFLOP/s | tok/s/GPU |
|---|---:|---:|
| Qwen3, 4k, mbs 4 (q5) | 195.1 | 6,823 |
| l1: 8k, mbs 2 | 209.6 | 6,273 |
| l2: 8k, mbs 1 | 196.1 | 5,867 |
| l5: + document masking + the hybrid MoE + the hybrid attention (all-attention) | 87.4 | 4,898 |
| l6: + CKDA 5:1 = the production hybrid (plain residual) | 77.3 | 4,551 |
| l6 at gbs 128 / mbs 2 | 90.1 / 86.0 | 5,295 / 5,067 |

- **CKDA is not the main cost** (l5 -> l6: -7 % tok/s).
- **Most of the TFLOP/s gap is FLOP accounting, not speed**: l2 -> l5 loses 17 % of the tok/s
  but 55 % of the TFLOP/s. The hybrid recipe does far fewer model FLOPs per token (top-5 + 1
  shared instead of top-8, 16 instead of 32 query heads, and document masking removes most of
  the 8k attention FLOPs). TFLOP/s is not comparable across these architectures; tok/s and the
  compute-busy share are. Single-feature rungs (masking, MoE, attention) in h1b.

**Single-feature rungs (h1b) -- document masking is the culprit:**

| rung (Qwen3, 8k, mbs 1) | tok/s/GPU | TFLOP/s |
|---|---:|---:|
| reference (l2) | 5,867 | 196.1 |
| + the hybrid's MoE only | 6,210 | 190.6 |
| + the hybrid's attention only | 6,653 | 158.1 |
| **+ document masking only (l3)** | **4,242** | **114.0** |
| + masking + MoE (l4) | 4,475 | 108.3 |

The hybrid's MoE and attention are *faster* per token (fewer FLOPs, hence lower TFLOP/s);
**document masking alone costs 28 % of the throughput**. The GPU-idle gaps of the doc-masked
hybrid (20 us - 0.5 ms: 1.9 s of a 13.7 s window, vs 0.9 % for Qwen3) start in the checkpointed
layer forward / recompute and in cuDNN `FusedAttnFunc` (host-heavy on packed THD batches),
the kernel rate is 11.7k/s (Qwen3: 5.3k/s): launch-bound. Next: FlashAttention varlen (h3), the
hybrid without masking as the bound. The rebalanced VPP2 layout (h2) does not help (79.2 vs 77.3).

**Qwen3.5-35B-A3B:** one-node slices run at 153-177 TFLOP/s for every variant (FLA / native GDN,
attention hd 256 / 128): nothing pathological. On 8 nodes its first two steps are Triton
autotuning of the GDN kernels (368 s, 194 s); step 3: 21.8 s = ~6,000 tok/s/GPU. Rerun with an
hour (h1c).

## 5. Phase Q: Qwen3-30B-A3B first (user, 2026-10-18: reach exp_irene's 170+ with standard attention)

exp_irene (branch `exp_irene`): Qwen3-30B-A3B (48 x attention + MoE, 32 q / 4 kv heads x 128,
128 x 768 top-8, ~50k vocab, 4k), PP4 x VPP2 balanced (7/7/7/6/7/7/6/1 + head), EP4 HybridEP,
full recompute, mbs 4, gbs 4096, **forced load balancing**: 185.8 bf16 / 182.1 fp8-delayed on 8 nodes.
Ours: the same model as HybridModel `*E` x 48 on the hybrid-ep image (`ep_qwen3_mn5.yaml`, q1,
8 nodes, gbs 1024, 15 steps):

| arm | TFLOP/s | tok/s/GPU | mem | note |
|---|---:|---:|---:|---|
| q_ref (exp_irene's layout, real routing) | 152.0 | 5,848 | 81 % | |
| q_ref_force (forced balancing) | 198.0 | 7,621 | 55 % | > exp_irene's 185.8 |
| q_opfx_force (grouped path, forced) | **201.4** | 7,750 | 55 % | the balanced ceiling |
| q_opfx (grouped path) | 162.5 | 6,254 | 99 % | OOM at step 6 (imbalance grows the buffers) |
| q_opfx_mem (+ NCCL 4 ch + expandable segments) | 160.2 | 6,163 | 51 % | |
| **q_moon (MoonEP, real routing)** | **171.5** | 6,602 | 56 % | balanced by construction: +13 % over q_ref |
| q_opfx_fp8b / q_opfx_fp8eb | 156.0 / 153.9 | | | fp8 still slower |
| q_opfx_cap (sync-free 1.5) | 84.6 | | | over-budget steps re-run (paged stash) |
| q_opfx_sel (selective recompute) | fail | | | TE: ScaledSwiGLU recompute needs the sm100 fused MLP |

- The imbalance of real (step-1..15) routing costs 25 % (152 vs 198); **MoonEP recovers 40 % of
  it** with real routing. MoonEP slice after zero-copy combine: 149.2 (+3 %), 172.2 at mbs 2.

**Round 2 (q2, mock data): MoonEP tuning** -- base 171.6 (reproducible), 48 comm SMs 174.1 (16:
156.3, 24: 165.4; default 32), mbs 8 174.4, gbs 4096 177.2 (HybridEP grouped path at gbs 4096:
170.4), slot padding 64 171.1, memory settings 171.4; **forced balancing 180.3 vs HybridEP's 201.4:
MoonEP's own overhead is ~10 %** (profiled in q4).

**Round 3 (q3): real text** (nemotron-cc mha 10p sample, OELLM 256k tokenizer = exp_irene's
data_mn5_256k, so the 262k LM head is in the FLOPs as in their 185.8):

| arm | TFLOP/s | tok/s/GPU | mem |
|---|---:|---:|---:|
| r_ref (exp_irene's settings, no memory settings) | OOM | | |
| **r_opfx_mem (HybridEP grouped path + NCCL 4 ch + expandable segments)** | **169.2** | 5,919 | 60 % |
| **r_opfx_mem_force (forced balancing, exp_irene's benchmark setting)** | **215.0** | 7,519 | 60 % |

- On exp_irene's own benchmark setting (forced balancing, 256k vocab) our stack gives **215 vs
  185.8** (+16 %: grouped-tensor experts, memory settings, the hybrid-ep image's TE 2.19 / torch 2.14).
- With real routing on real text, HybridEP reaches 169 (mock data's skew: 160).

**Round 3b / 4 (real text): MoonEP with real routing = 185.1 (32 comm SMs), 186.5 (48),
187.6 (64)** -- +11 % over HybridEP (167.7-169.2) and above exp_irene's forced-balance 185.8.
MoonEP forced 193.3 vs HybridEP forced 214.5: its overhead. Profile (c_moon_prof vs
c_hep_force_prof, per step): grouped GEMMs 4.66 s vs 3.58 s, of which wgrad 2.07 vs 1.07 s (the 32
prefetch slots are mostly empty, but an empty group with accumulate=True still reads and writes
its whole fp32 block), 1.06 s device-to-device copies (comm-buffer boundary), MoonEP comm kernels
~3.3 s vs HybridEP dispatch/combine/permute ~3.2 s. Fixed in round 5: prefetch-slot wgrad without
accumulation, zero-copy combine backward, the weighted-SwiGLU output saved instead of recomputed.
micro batch 8 OOMs with the 262k head.

**Round 5 (q5, after the MoonEP fixes):** MoonEP 64 SMs **195.1** (80 SMs 195.9), **gbs 4096: 203.0**,
forced 206.0 (vs HybridEP forced 214.5: overhead now ~4 %). HybridEP comm SMs 16 / 48: 169.9 /
170.0 (no effect).

**Document-masking fixes (h3b-h6), production hybrid plain residual, 8 nodes, real text, MoonEP:**

| setup | TFLOP/s | tok/s/GPU |
|---|---:|---:|
| 8k doc-masked, mbs 1 (l6) | 77.3 | 4,551 |
| + CKDA per-micro-batch cu_seqlens cache | 79.4 | 4,680 |
| + no Mamba-only total_tokens / seq_idx (two syncs per micro-batch and stage) | 83.5 | 4,910 |
| **+ 2 x 8k per micro-batch (`--dataloader-document-segment-length 8192`, seq 16k, gbs 256)** | **96.0** | **5,650** |
| 4 x 8k per micro-batch | OOM | |
| no document masking at all (the bound) | 110.2 | 6,288 |
| routed residual, HybridEP, mbs 1, fixes | 69.2 | 4,080 |
| routed residual + MoonEP, 2 x 8k | OOM | |

FlashAttention varlen (`attention_backend: flash`) hangs on packed batches (no iteration in 10 min).

**FLOP accounting across architectures:** Megatron counts the GDN / KDA delta rule in its recurrent
form (`heads x d^2 x 4` FMAs per token) -- the model-FLOP convention -- so a linear-attention layer
counts a few MFLOP per token where softmax attention at 4k counts ~67. Counted GFLOP per token:
Qwen3-30B-A3B 28.6, Qwen3.5-35B-A3B-like 19.5, the production hybrid 17.0. At equal tok/s the
hybrids therefore report 0.6-0.7x the TFLOP/s: Qwen3.5-like runs at 6,486 tok/s/GPU = 95 % of
Qwen3's 6,823, yet 126 vs 195 TFLOP/s.

## 6. Large scale (16 nodes, 64 GPUs; real text, 256k vocab, real routing, full recompute, PP4 x VPP2, EP4)

| model | setup | TFLOP/s | tok/s/GPU |
|---|---|---:|---:|
| Qwen3-30B-A3B | MoonEP 64 SMs + memory settings, gbs 2048 (8M tokens) | **193.6** | 6,773 |
| Qwen3-30B-A3B | same, gbs 1024 | 183.4 | 6,416 |
| Qwen3.5-35B-A3B-like (native GDN 3:1) | same, gbs 2048 | 124.7 | 6,399 |
| Qwen3.5-35B-A3B-like | same, gbs 4096 (16M) | 129.8 | 6,664 |
| production hybrid, plain residual | doc-masked 2 x 8k packs, sync fixes, gbs 512 (8M) | 87.1 | 5,126 |
| production hybrid, plain residual | same, gbs 1024 (16M): not bubble-limited | 87.0 | 5,115 |

8 nodes for reference: Qwen3 195.1 (gbs 1024) / 203.0 (gbs 4096); Qwen3.5-like 126.4; hybrid 2 x 8k
94.5-96.0; routed residual (mbs 1, HybridEP) 69-73.

**Context length (q9 / q9b, Qwen3-30B-A3B, 8 nodes, MoonEP 64 SMs + memory settings, real text +
routing, full recompute; medians of steps 3-8):**

| context | mbs | gbs (tokens) | TFLOP/s | tok/s/GPU |
|---|---:|---:|---:|---:|
| 4k (q5, reference) | 4 | 1024 (4M) | 195.1 | 6,823 |
| 8k | 2 | 2048 (16M) | 214.9 | 6,431 |
| 16k | 1 | 1024 (16M) | 236.2 | 5,484 |
| 16k | 1 | 2048 (32M) | 236.1 | 5,481 |
| **24k** | 1 | 640 (15.7M) | **252.8** | 4,791 |
| 32k | 1 | 512 | OOM | |

**> 250 model TFLOP/s is reached at 24k, but it is accounting, not speed**: the quadratic
attention FLOPs grow with the context, tok/s falls (-30 % vs 4k). The doubled batch at 16k does not
help (not bubble-limited). For the tok/s objective (user, 2026-10-19: optimize tok/s/GPU, not only
TFLOP/s) short context with a large batch stays best.

## 7. Conclusions and recommended settings (2026-10-19)

**> 250 TFLOP/s is not reached; ~195-203 is the practical ceiling on MN5 for Qwen3-30B-A3B with real
routing** (exp_irene's best: 185.8 with forced balancing; ours forced: 215). With full recompute --
needed: block recompute keeping even 2 of 14 layers per chunk OOMs at 64 GB -- 200 model TFLOP/s is
~267 TFLOP/s of executed bf16 work. The remaining exposed time is MoonEP's (serial) EP comm
(~15 % of the step) and pipeline p2p (~7 %); EP a2a overlap is excluded by full recompute.

Recompute less, last attempt (q7, Qwen3, MoonEP, memory settings): selective recompute of only the
MoE layers (`recompute_modules: [moe]`, also + layernorm, + core_attn) OOMs at the first step on
64 GB -- keeping the attention layers' activations for 48 layers x the in-flight VPP micro-batches
does not fit. Block recompute keeping 2 of 14 layers per chunk OOMed too (q6). Full recompute is
required on MN5, and with it ~200 model TFLOP/s is the ceiling we measured. Offloading the
attention-side activations to the host instead (`fine_grained_activation_offloading`, offload_modules
attn_norm / qkv_linear / core_attn / attn_proj / mlp_norm, with MoE-only recompute, q8) fails in
Megatron's offload handler after the first step under PP4 x VPP2 ("Chunk mismatch" in
`ChunkOffloadHandler`): not usable with the interleaved schedule as is.

What to use (all in Megatron `ep-fp8-speed` / the `ep_*` suites):
1. **Memory settings** (NCCL 4 channels per peer, expandable segments): required to fit / avoid OOMs.
2. **MoonEP** (`moe_flex_dispatcher_backend: moonep`, `moe_flex_dispatcher_num_sms: 64`,
   `$HYBRID_ENV/pylibs_moonep` first on PYTHONPATH) for standard-residual models: +15 % over HybridEP
   with real routing. HybridEP + the TE grouped-tensor experts (`use_transformer_engine_op_fuser` +
   `NVTE_CUTEDSL_FUSED_GROUPED_MLP=1`) otherwise, and for the routed residual (MoonEP + routed is
   slower / OOMs at 2 x 8k).
3. **Document masking:** the two host-sync fixes + `--dataloader-document-segment-length 8192` with
   2 x 8k packs (+24 % over the starting point for the hybrid, +16 % for doc-masked Qwen3).
4. Large global batches (16+ micro-batches per pipeline; 32 better).
5. Not: fp8 on H100 at these shapes (0.89-0.96x, also experts-only, also on the grouped path; fp8
   dispatch would save < 2 %), DeepEP / ncclep / alltoall (slower; ncclep is the only internode EP and
   traps on overflow), sync-free HybridEP at capacity <= 2 (step re-runs), FlashAttention varlen on
   packed batches (hangs), UCCL-EP (not in the image).

**Why the hybrids report fewer TFLOP/s:** FLOP accounting (linear attention counts a few MFLOP per
token) -- Qwen3.5-like runs at 95 % of Qwen3's tok/s -- plus, for the production recipe, document
masking (fixed in part above) and the routed residual (16 % step cost, 82 % of it exposed pipeline
p2p of the 8 streams; levers: fp8 / fewer streams at the boundary, fewer PP stages).

## GPU-h ledger

Total used: 678.5 GPU-h of the 1,000 (sacct over all ep suites: e1-e4 slices, b1-b5 production
model, q1-q6 Qwen3, h1-h6 ladder / document masking, l1-l2 16 nodes, MoonEP unit tests).

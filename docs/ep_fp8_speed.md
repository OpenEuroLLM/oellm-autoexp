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

## GPU-h ledger

| suite | what | GPU-h (est.) |
|---|---|---:|
| e1 | phase A, 20 arms x 1 node x <= 20 min | <= 27 |

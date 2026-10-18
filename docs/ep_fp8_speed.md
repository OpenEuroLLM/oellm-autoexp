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

(results pending)

## GPU-h ledger

| suite | what | GPU-h (est.) |
|---|---|---:|
| e1 | phase A, 20 arms x 1 node x <= 20 min | <= 27 |

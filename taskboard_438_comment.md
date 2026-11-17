## Throughput on MN5: our stack (MoonEP + memory / host settings), Qwen3.5-35B-A3B-like, Qwen3-30B-A3B and our hybrid

Short version: on the same model and benchmark setting as above (Qwen3.5-35B-A3B, 8 nodes, GBS 4096, forced balancing) our stack reaches **7,239 tokens/s/GPU (141.0 TFLOP/s/GPU, 14.3 % MFU)**, +4 % over 6,971. With **real routing** the gap is larger: our MoonEP backend keeps the per-rank load balanced by construction (redundant experts), so real routing costs little instead of −14 %. Our production hybrid (CKDA 5:1, 8k document-masked) went from 4,551 to **6,942 tokens/s/GPU**, now on par with Qwen3-30B-A3B per token.

MFU = TFLOP/s / 989 (H100 dense bf16), TFLOP/s = Megatron's model-FLOP count (recompute not counted). Medians over iterations ≥ 3.

### Setup (all runs unless noted)

- BF16, TP1, PP4 with the balanced VPP2 layouts (Qwen3.5: `6/6/6/3/6/6/6/1`, Qwen3: `7/7/7/6/7/7/6/1`, LM head on its own chunk), EP4 inside a node, ETP1, full recompute (uniform, 1 layer), distributed optimizer with grad / param overlap, P2P overlap, `CUDA_DEVICE_MAX_CONNECTIONS=8`.
- Real text (nemotron-cc sample) with the OpenEuroLLM 256k tokenizer (padded vocab 262,144), seq 4096 unless noted.
- **Memory settings (needed for VPP / uneven PP to fit):** `NCCL_MIN_NCHANNELS=4`, `NCCL_NCHANNELS_PER_PEER=4` (the MN5 slurm config forced 32 channels per peer: ~14 GiB per GPU were not PyTorch's), `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Host settings:** `manual_gc: true`, `manual_gc_interval: 10`, `mmap_bin_files: false` (from your configs) where marked "GC".
- EP backend: **MoonEP** (`moe_token_dispatcher_type: flex`, `moe_flex_dispatcher_backend: moonep`, 64 comm SMs) unless marked HybridEP.

### Qwen3.5-35B-A3B-like (HybridModel with Megatron's native GDN 3:1, gated attention 16 q / 2 kv × 256, partial RoPE 0.25, zero-centred norm, 256 × 512 top-8 + gated shared 512, global aux loss)

| Setup | Nodes | GBS | Routing | Tokens/s/GPU | TFLOP/s/GPU | MFU |
|---|---|---|---|---|---|---|
| above: HybridEP 16 SMs | 8 | 4096 | forced | 6,971 | 133.2 | 13.5 % |
| ours: HybridEP grouped path + memory settings + GC | 8 | 4096 | forced | **7,239** | **141.0** | **14.3 %** |
| above: HybridEP | 8 | 4096 | real | 6,016 | 115.0 | 11.6 % |
| ours: MoonEP + memory settings (no GC) | 8 | 1024 | real | 6,506 | 126.8 | 12.8 % |
| ours: MoonEP + memory settings (no GC) | 16 | 2048 | real | 6,399 | 124.7 | 12.6 % |
| ours: MoonEP + memory settings (no GC) | 16 | 4096 | real | 6,664 | 129.8 | 13.1 % |

The real-routing runs predate the GC setting; a rerun with GC and shared-expert overlap is queued.

Qwen3.5's TFLOP/s looks low next to Qwen3-30B-A3B mostly because of the FLOP count: Megatron counts a GDN layer at its recurrent cost (a few MFLOP per token, vs ~67 for softmax attention at 4k). Counted GFLOP per token: Qwen3-30B-A3B 28.6, Qwen3.5-like 19.5. In tokens/s/GPU Qwen3.5-like runs at 95–98 % of Qwen3-30B-A3B.

### Qwen3-30B-A3B (48 × attention + 128 × 768 top-8, same layout)

| Setup | Nodes | GBS | Routing | Tokens/s/GPU | TFLOP/s/GPU | MFU |
|---|---|---|---|---|---|---|
| HybridEP grouped path + memory settings | 8 | 1024 | forced | 7,519 | 215.0 | 21.7 % |
| HybridEP grouped path + memory settings | 8 | 1024 | real | 5,919 | 169.2 | 17.1 % |
| **MoonEP, 64 SMs** | 8 | 1024 | real | 6,823 | 195.1 | 19.7 % |
| MoonEP, 64 SMs | 8 | 4096 | real | ~7,100 | 203.0 | 20.5 % |
| MoonEP, 64 SMs | 16 | 2048 | real | 6,773 | 193.6 | 19.6 % |
| MoonEP, seq 8k, MBS 2 | 8 | 2048 | real | 6,431 | 214.9 | 21.7 % |
| MoonEP, seq 24k, MBS 1 | 8 | 640 | real | 4,791 | 252.8 | 25.6 % |

Longer context raises TFLOP/s through the quadratic attention FLOPs while tokens/s falls, so 4k with a large batch stays the best by tokens/s.

### Our production hybrid (~39B total / 2.7B active: CKDA 5:1, 160 × 768 top-5 + 1 shared, 262k vocab, 8k document-masked, plain residual)

| Step | Tokens/s/GPU | TFLOP/s/GPU |
|---|---|---|
| start: MBS 1, VPP2, MoonEP (8 nodes) | 4,551 | 77.3 |
| + CKDA per-micro-batch `cu_seqlens` cache + no Mamba-only syncs in `pretrain_hybrid` | 4,910 | 83.5 |
| + 2 × 8k documents packed per micro-batch (`--dataloader-document-segment-length 8192`: same samples, 16k tokens per micro-batch) | 5,650–5,965 | 96–101.5 |
| **+ GC** | **6,942** | **118.0** |
| (no document masking at all, for reference) | 6,980 | 122.4 |

Document masking now costs < 1 %. The routed residual (2 of 8) costs 30 % at this layout (4,824 tokens/s/GPU). It sends all 8 streams across every pipeline boundary. mHC-4 does not fit at 8k with this layout yet (out of memory on PP stage 0); coarser recompute checkpoints and PP4 without VPP are being tested.

### What made the difference

1. **MoonEP backend (new in Megatron).** MoonEP keeps every rank at exactly S × K token slots by pushing copies of hot experts to the other ranks. Real routing then costs ~4 % instead of 14–25 %. On Qwen3-30B-A3B with real routing: +15 % over HybridEP (195 vs 169).
   - We added the Megatron dispatcher and experts.
   - MoonEP needed a no-multicast patch, since MN5 has no NVSwitch.
   - Its own overhead was fixed with zero-copy combine and the activation saved instead of recomputed.
   - The prefetch slots' wgrad no longer accumulates, and the backward prefetch is skipped under full recompute.
2. **Memory settings (above).** exp_irene's settings on our image ran out of memory without them.
3. **Manual GC.** +16 % on the host-bound document-masked hybrid, nothing on Qwen3.
4. **Document-masking fixes for the hybrid.** Host syncs removed and multi-document packing per micro-batch.
5. TE 2.19 grouped-tensor experts (`use_transformer_engine_op_fuser` + `NVTE_CUTEDSL_FUSED_GROUPED_MLP=1`): +1–4 %.

### Tried, not better on MN5

- **FP8:** tensorwise, blockwise, delayed and experts-only were all 0.89–0.96× of bf16 at these shapes on H100.
- **DeepEP:** 0.89× of HybridEP within a node.
- **ncclep:** 0.93×. It is the only backend that works across nodes; HybridEP and DeepEP internode need GPU-initiated RDMA / 8-GPU NVLink domains.
- **Less recompute:** selective or block recompute runs out of memory. Activation offloading of the attention side works after a Megatron fix (port of NVIDIA/Megatron-LM#7908) but is 11–17 % slower than full recompute.
- **CUDA graphs:** per-layer graphs are rejected together with full recompute.
- **Sync-free HybridEP at capacity ≤ 1.5:** over-budget steps re-run.

### Environment

- Cluster: MareNostrum 5 (ehpc533)
- Code: [OpenEuroLLM/oellm-autoexp](https://github.com/OpenEuroLLM/oellm-autoexp/tree/feat/hybrid-v0.19), branch `feat/hybrid-v0.19` (`fe52c78`)
  - suites: `config/experiments/korbi/hybrid_speed/ep_*_mn5.yaml`, generated by `scripts/korbi/ep_speed/gen_ep_suites.py`
  - full write-up with all arms: `docs/ep_fp8_speed.md`
- Container: `MegatronTraining-hybrid-ep` (TE 2.19 with cuBLASLt grouped GEMM, DeepEP / HybridEP, MoonEP), MoonEP no-multicast patch from `container/patches/moonep-no-multicast.patch` on the PYTHONPATH
- W&B: none (TensorBoard logs on MN5)
- Megatron LM submodule: [OpenEuroLLM/NVIDIA-Megatron-LM](https://github.com/OpenEuroLLM/NVIDIA-Megatron-LM/tree/oellm/v0.19-hybrid), branch `oellm/v0.19-hybrid` at `56f2d4681`

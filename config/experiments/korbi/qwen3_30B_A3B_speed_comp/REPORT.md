# Qwen3 30B-A3B Speed Benchmark: torchtitan vs Megatron-LM

**Hardware**: JUPITER cluster, 2 nodes, 8× GH200 SXM (95 GiB each), NVLink intra-node, InfiniBand inter-node
**Model**: Qwen3 30B-A3B — 48 layers, hidden=2048, 128 experts, top-k=8, moe_inter_dim=768, seq_len=4096
**Peak FP16/BF16 throughput (GH200)**: 989.4 TFLOPS
**Full results table**: `qwen_30B_A3B_speed_results.txt`

---

## Summary

Both frameworks reach **~5,200–5,500 tok/s/GPU** at steady state — essentially at parity.
The apparent TFLOPS gap (145 Titan vs 114 Megatron) is a FLOPs accounting difference, not real compute.

| Backend | Config | tok/s/GPU | TFLOPS | MFU% | Mem | job |
|---------|--------|-----------|--------|------|-----|-----|
| **torchtitan** | compile, sel-AC opt=4, bs=64, EP4+PP2 | **5,446** | 145.2 | 14.7% | 84.9 GiB (89%) | 321069 |
| **Megatron-LM** | full-recompute, mbs=4, GBS=64, EP4+PP2 | **5,236** | 113.9 | 11.5% | ~80 GiB (84%) | 321801 |

Gap: ~4%, within run-to-run variability.

---

## Parallelism

Both backends use **EP=4 + PP=2** as the optimal layout:
- EP=4: expert shards stay intra-node (4 GPUs on NVLink → fast allgather)
- PP=2: pipeline stages cross nodes (IB P2P send/recv)
- TP=1: no tensor parallelism needed (hidden=2048 fits comfortably)

**EP=8+PP=1 ruled out**: EP allgather crosses nodes (IB), 37% slower than EP=4+PP=2.

---

## torchtitan Sweep Results

### Effect of activation checkpoint (AC) mode
`EP=4+PP=2, compile, bs=64`

| AC mode | tok/s/GPU | TFLOPS | Mem |
|---------|-----------|--------|-----|
| selective opt=4 (**optimal**) | 5,446 | 145.2 | 89% |
| selective opt=6 | 5,498 | 146.6 | 93% |
| selective opt=8 | 5,543 | 147.8 | 94% |
| full | ~3,294 | 87.8 | 59% |
| none | ~3,586 | 95.6 | 96% |

**selective-AC opt=4 is optimal**: recomputes 25% of layers (12/48), gives +65% vs full-AC
with safe memory (89%). opt=6/8 gain <2% more TFLOPS but push memory to 93-94%.
No-AC (none) runs out of memory headroom (96%) with only marginal gain.

### Effect of `torch.compile`
`EP=4+PP=2, selective-AC opt=4, bs=64`

| compile | tok/s/GPU | TFLOPS |
|---------|-----------|--------|
| yes | 5,446 | 145.2 |
| no | 4,884 | 130.2 |

compile gives **+11%** at steady state. Also accelerates warmup (converges ~40 steps earlier).

### Effect of batch size
`EP=4+PP=2, compile, selective-AC`

| BS | tok/s/GPU | TFLOPS | Mem |
|----|-----------|--------|-----|
| 64 | 5,446 | 145.2 | 89% |
| 128 | 5,348 | 142.6 | 87% |
| 16 | 4,554 | 121.4 | 79% |
| 8 | 4,315 | 115.1 | 79% |

bs=64 is optimal. bs=128 slightly slower (extra AllToAll/AllGather overhead > pipeline benefit).

### FP8 linear (quantize.linear.float8)
`EP=4+PP=2, compile, selective-AC opt=4, bs=64, recipe=rowwise`

| Config | tok/s/GPU | TFLOPS | vs BF16 |
|--------|-----------|--------|---------|
| BF16 compile (baseline) | 5,446 | 145 | — |
| FP8 compile bs=64 | ~5,000 | 133 | **-7%** |
| FP8 nocompile bs=64 | ~3,710 | 99 | -24% |

**FP8 linear is negative**: `quantize.linear.float8` only covers `nn.Linear` (attention layers),
not MoE expert GEMMs which use `grouped_mm`. FP8 cast overhead on attention > speedup.
Memory unchanged (Float8Linear stores BF16 master weights, casts to FP8 during forward).

### NCCL QPS sweep (NCCL_IB_QPS_PER_CONNECTION)
`EP=4+PP=2, compile, selective-AC opt=4, bs=64`

| QPS | tok/s/GPU at s=100 |
|-----|--------------------|
| 1 (default) | 137.2 (still warming up) |
| 4 | 141.4 (converged ~s70) |
| 8 | 141.9 (converged ~s30) |
| 16 | 139.3 (converged ~s30) |

No steady-state gain. QPS≥4 reduces compile warmup by 40-50 steps. Not recommended.

---

## Megatron-LM Sweep Results

### Parallelism
| Config | tok/s/GPU | TFLOPS | Mem |
|--------|-----------|--------|-----|
| EP=4+PP=2, mbs=4, full-recompute (**optimal**) | 5,236 | 113.9 | 84% |
| EP=4+PP=2, mbs=8, full-recompute | 4,373 | 95.2 | 96% |
| EP=8+PP=1, mbs=4, full-recompute | 3,211 | 69.8 | 95% |

### MoE dispatcher
`EP=4+PP=2, mbs=4, full-recompute`

| Dispatcher | tok/s/GPU | TFLOPS | Mem |
|------------|-----------|--------|-----|
| allgather (**optimal**) | 5,236 | 113.9 | 84% |
| alltoall | 4,654 | 101.2 | 89% |

**allgather beats alltoall by +13%** for EP=4 intra-node NVLink.
allgather exploits NVLink bandwidth more efficiently than alltoall for 4-GPU groups.

### No-recompute attempts (all OOM)
Extensive testing showed no-recompute is **not viable** for Megatron on 95 GiB GH200:

| Attempt | GBS | mbs | Memory at OOM | Why |
|---------|-----|-----|---------------|-----|
| no-recompute | 64 | 4 | 89.9 GiB | expert GEMM workspace |
| no-recompute | 64 | 2 | 94.5 GiB | same total (GBS/DP fixed) |
| no-recompute | 64 | 1 | OOM | same |
| no-recompute | 32 | 4 | 92.1 GiB | minimum valid GBS still OOMs |
| selective-recompute | 64 | 4 | 94.6 GiB | ≈ no-recompute (saves <1 GiB) |

**Root cause**: TransformerEngine's `grouped_linear` (expert FC1/FC2) allocates a fixed
workspace ∝ `mbs × seq_len × hidden` (~18 GiB for mbs=4) that cannot be reduced by
changing GBS or enabling gradient accumulation.

**Parallelism note**: Megatron computes `DP = world_size / (TP × PP) = 8 / (1×2) = 4`.
EP is orthogonal to DP. GBS must be divisible by `mbs × DP = 16`; minimum valid GBS = 32
(requires `num_microbatches = GBS/(mbs×DP) ≥ PP = 2`).

---

## Key Takeaways

1. **Frameworks at parity**: torchtitan and Megatron-LM both reach ~5,200–5,500 tok/s/GPU
   for Qwen3 30B-A3B on 2-node JUPITER. TFLOPS differ only due to FLOPs accounting.

2. **Titan advantages**: torch.compile (+11%), fine-grained selective-AC (64 opt levels),
   memory visibility (GiB reported), more flexible batch sizing.

3. **Megatron advantages**: Simpler config, no compile warmup (~50 extra steps), marginally
   better memory at full-recompute (84% vs 89%).

4. **Ceiling for both**: ~5,500 tok/s/GPU on 2×4×GH200 with this model.
   Both are memory-bound (activation checkpointing required).

5. **EP=4+PP=2 is critical**: EP=8+PP=1 is 37% slower due to inter-node expert routing.

6. **allgather > alltoall for EP=4 NVLink**: +13% on intra-node expert dispatch.

7. **FP8 not viable** without grouped_mm FP8 (PP=1 only, OOMs). Attention-only FP8 = -7%.

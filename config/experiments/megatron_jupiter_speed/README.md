# megatron_jupiter_speed — dense & MoE speed-test collection for JUPITER

One config per model size, each at the minimal node count that fits, carrying
the best-known JUPITER speed settings. Shared run settings (data, short run,
no ckpt/eval, logging, tokenizer) live in `common.yaml`, which every leaf
includes *after* its model config so it overrides production schedules.

Submit:

```bash
ssh jupiter "bash ~/work/Projects/oellm-autoexp/submit.sh \
  --config-name experiments/megatron_jupiter_speed/dense_1.7B --submit-and-exit"
```

| Config | Arch | Nodes | Parallelism | Status |
|---|---|---|---|---|
| `dense_1.7B` | Qwen3-1.7B (repository config) | 1 | TP1, mbs6, GBS48, no recompute, CUDA graphs | **validated: 367 TFLOPS / 37% MFU, 35,017 tok/s/GPU** (job 1051942) |
| `dense_3B` | ~3B Qwen3-style (PROPOSED: 32L, h2560, ffn8192) | 1 | TP1, mbs2, GBS32, no recompute, CUDA graphs | **validated: 355 TFLOPS / 36% MFU, 16,807 tok/s/GPU** (job 1051947) |
| `dense_9B_baby` | exact baby-9B (36L, h4096, ffn12288) | 1 | TP1, mbs1, GBS16, no recompute | **validated: 390 TFLOPS / 39% MFU, 8,383 tok/s/GPU** (job 1052011; 256k prod vocab needs TP4) |
| `dense_32B` | Qwen3-32B (64L, h5120, ffn25600) | 4 | TP4 + DP4, mbs1, GBS128, overlap, selective recompute | **validated: 317 TFLOPS / 32% MFU, 1,571 tok/s/GPU** (job 1034538) |
| `moe_1.7B` | ~1.7B/0.6B-active (PROPOSED: 20L, h1024, 32 experts) | 1 | EP4, mbs8, no recompute, allgather | **validated: 181 TFLOPS, 43,099 tok/s/GPU** — HybridEP is −11% here, allgather confirmed |
| `moe_3B` | ~3B/0.85B-active (PROPOSED: 24L, h1280, 40 experts) | 1 | EP4, mbs4, GBS16, no recompute, **HybridEP** | **validated: 195 TFLOPS, 32,770 tok/s/GPU** (job 1052295, +13% vs allgather) |
| `moe_9B` | ~8.6B/1.3B-active (PROPOSED: 32L, h1536, 72 experts) | 1 | EP4, mbs8, GBS32, full recompute, **HybridEP** | **validated: 166 TFLOPS, 18,397 tok/s/GPU** (job 1052298, +24% vs allgather) |
| `moe_30BA3B` | Qwen3-30B-A3B (repository config) | 2 | EP4+PP2+TP1, mbs4, GBS64, full recompute, **HybridEP + VPP(VP6)** | **validated: 163.5 TFLOPS, 7,515 tok/s/GPU** (job 1052257, +24% vs allgather GBS256) |

HybridEP configs (3B/9B/30B) require the
`MegatronTraining-JUPITER-deep-hybridep_aarch64.sif` container with libcuda
bind-mounts (in each config; Inductor asserts "libcuda.so cannot found"
without them). The 1.7B stays on the default pt2512 container/allgather —
the 16 reserved dispatch SMs cost more than dispatch saves at ~0.6B active.

Every size also has a `<config>_weakscale.yaml` companion: it includes the base
config by way of defaults and only adds GBS ∝ nodes plus a node sweep up to 256
(fixed per-GPU work — constant microbatch count per DP rank, EP/TP/PP layout
unchanged):

| Weak-scale config | Nodes | GBS |
|---|---|---|
| `dense_1.7B_weakscale` | 1→256 | 24·nodes |
| `dense_3B_weakscale` | 1→256 | 16·nodes |
| `dense_9B_baby_weakscale` | 1→256 | 16·nodes (TP4, DP=nodes) |
| `dense_32B_weakscale` | 4→256 | 32·nodes (TP4, DP=nodes) |
| `moe_1.7B_weakscale` / `moe_3B_weakscale` | 1→256 | 32·nodes |
| `moe_9B_weakscale` | 1→256 | 32·nodes |
| `moe_30BA3B_weakscale` | 2→256 | 32·nodes (GA 4, VP 6, 4% bubble const) |

Extract results (averages over the last N throughput lines, skips warmup;
works for Megatron and titan logs — compare backends by way of tok/s/GPU only):

```bash
ssh jupiter 'cd ~/work/Projects/oellm-autoexp && python scripts/extract_speed_results.py \
  /e/scratch/projectnucleus/poeppel1/output/megatron_jupiter_speed'
```

For the weak-scaling sweeps, flat tok/s/GPU across the world column = perfect
weak scaling.

All MoE configs share the Qwen3-MoE expert granularity (`moe_ffn 768`, `topk 8`)
and keep EP intra-node (EP=4 = GPUs/node) — inter-node EP measured ~37% slower.
The PROPOSED architectures are placeholders sized to the target param count
(math in each file header); swap in the real OELLM shapes once fixed.

Known JUPITER constraints baked in:
- `NCCL_NET_GDR_LEVEL=0` only where PP≥2 crosses nodes (`moe_30BA3B`).
- `CUDA_DEVICE_MAX_CONNECTIONS=1` where TP≥2 (`dense_9B_baby`, `dense_32B`).
- CUDA-graph configs replace `expandable_segments` with
  `garbage_collection_threshold:0.8` (graph capture conflicts).
- Data: shared `/backend/megatron/data_nemonicco_split_jupiter` include
  (neox-tokenized nemonicco, 99/1/0 split; eval disabled anyway). Vocab
  ~50k, so LM-head cost is understated vs a 152k/256k-vocab production run.

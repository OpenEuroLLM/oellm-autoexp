# MaxText MoE Speed Test Results — JUPITER (2 nodes, 8× GH200)

## Job: 342516 (2026-04-06)

### Configuration
- **Model**: Qwen3-30BA3B MoE architecture (mapped to Mixtral-style)
- **Layers**: 24 (reduced from 48 due to OOM — see below)
- **Params**: 20.258 billion (24 layers, 128 experts, topk=8)
- **Seq length**: 2048 (reduced from 4096 due to OOM)
- **Batch**: per_device_batch_size=1
- **Parallelism**: ICI EP=2 + FSDP_T=2 (intra-node), DCN FSDP=2 (inter-node)
- **Activation checkpointing**: full remat
- **MoE impl**: dense_matmul (megablox disabled — Pallas not supported on GPU)
- **scan_layers**: false (true causes OOM from stacked all-gather)
- **Container**: nvcr.io/nvidia/jax:24.10-py3 + MaxText (apptainer)
- **JAX**: 0.4.38.dev, XLA flags: --xla_gpu_enable_command_buffer=

### Results (steps 2-99, steady state)
| Metric | Value |
|--------|-------|
| **TFLOP/s/device** | **~2.73** |
| Tokens/s/device | ~224 |
| Step time | ~9.13 seconds |
| Total TFLOPs per step (per device) | 24.93 |
| Memory: model+optimizer | ~28.3 GiB/GPU |
| Memory: temp/activations | ~21.9 GiB/GPU |
| Memory: total | ~78.5 GiB/GPU (of 91.2 GiB available at 0.95 fraction) |

### TFLOP/s/device Formula (MaxText)
```
TFLOP/s/device = per_device_tflops_per_step / step_time_seconds
```
Where per_device_tflops includes 3× multiplier (fwd+bwd+grad) and is PER DEVICE (not total cluster).

### Comparison with Other Backends (Qwen3-30BA3B MoE, 2 nodes)

| Backend | TFLOP/s/device | Config | Notes |
|---------|---------------|--------|-------|
| **Megatron** | ~14.75 | 48L, EP=4, TP=2, allgather | tps/GPU metric |
| **Titan** | ~18.1 | 48L, EP=4, PP=2, compile | selective_ac opt=4 |
| **Automodel** | ~16.4 | 48L, EP=4, PP=2 | tps÷world_size normalized |
| **MaxText** | **~2.73** | 24L, EP=2, FSDP_T=2 | dense_matmul, involuntary remat |

**MaxText (dense_matmul) is ~5.4× slower than Megatron and ~6.6× slower than Titan per device.**

---

## Job: 343695 (2026-04-07) — ragged_dot (sparse_matmul=true)

### Configuration
Same as job 342516 except:
- **MoE impl**: `ragged_dot` by way of `jax.lax.ragged_dot` (`sparse_matmul=true, megablox=false`)
- ragged_dot only computes selected top-k expert rows per token (sparse), unlike dense_matmul which computes all 128 experts

### Results (steps 2-99, steady state)
| Metric | Value | vs dense_matmul |
|--------|-------|-----------------|
| **TFLOP/s/device** | **~5.30** | **1.94× faster** |
| Tokens/s/device | ~435 | 1.94× faster |
| Step time | ~4.71 seconds | 1.94× faster |
| Memory: argument (model+opt) | ~30.4 GiB/GPU | +2 GiB |
| Memory: temp/activations | ~48.3 GiB/GPU | +26 GiB |

### Updated Comparison (Qwen3-30BA3B MoE, 2 nodes)

| Backend | TFLOP/s/device | Config | Notes |
|---------|---------------|--------|-------|
| **Megatron** | ~14.75 | 48L, EP=4, TP=2, allgather | tps/GPU metric |
| **Titan** | ~18.1 | 48L, EP=4, PP=2, compile | selective_ac opt=4 |
| **Automodel** | ~16.4 | 48L, EP=4, PP=2 | tps÷world_size normalized |
| **MaxText (ragged_dot)** | **~5.30** | 24L, EP=2, FSDP_T=2 | sparse MoE by way of jax.lax.ragged_dot |
| MaxText (dense_matmul) | ~2.73 | 24L, EP=2, FSDP_T=2 | naive all-experts computation |

**MaxText (ragged_dot) is ~2.8× slower than Megatron and ~3.4× slower than Titan per device.**
Note: MaxText runs only 24 layers (vs 48 for others) due to memory constraints.

---

### Why MaxText Is Slow

1. **dense_matmul MoE implementation**: MaxText's GPU MoE uses `dense_matmul` (not megablox). The XLA SPMD partitioner can't find efficient resharding paths for EP+FSDP_transpose → causes "involuntary full rematerialization" (full un-sharding of expert weights during backward pass).

2. **No megablox on GPU**: MaxText's efficient MoE kernel (megablox) uses Pallas with dynamic grid bounds that are only supported on TPU. On GPU, falls back to dense_matmul.

3. **Memory-limited model**: Full 48-layer model doesn't fit on 8× GH200 due to involuntary rematerialization overhead. Had to reduce to 24 layers.

4. **TPU-first optimization**: MaxText's sharding strategies and axis rules are optimized for TPU ICI mesh topology, not GPU NVLink+IB.

### OOM History
- 48L, EP=4, FSDP=4: OOM at model init (mlp axis maps to fsdp_transpose, not fsdp)
- 48L, EP=2+FSDP_T=2+DCN_FSDP=2, bs=1, scan=false, fraction=0.95: OOM train step (37.45 GiB allocation)
- 48L, same, fraction=0.99: NCCL OOM (no CUDA memory left for NCCL after XLA took 99%)
- **24L, same, fraction=0.95: SUCCESS** ← final working config

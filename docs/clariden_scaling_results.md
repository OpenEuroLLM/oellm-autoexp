# Scaling results on Clariden

Mock-data throughput for three Megatron training layouts on CSCS Alps (Clariden). The loss is a plumbing check. Throughput is the `TFLOP/s/GPU` Megatron prints from its model forward-plus-backward FLOP count. Sequence length is 4096 everywhere. Data is mock (`NullTokenizer`, vocabulary 262144).

Numbers below are arithmetic means of the logged per-iteration fields over a steady window. Startup iterations are dropped, and the window is the same at every scale of a sweep. Efficiency uses step time. MFU divides the mean TFLOP/s/GPU by 989.4, the GH200 dense BF16 peak used in `scripts/korbi/scaling_curves.py`. Training is bf16. The 4-node dispatcher comparison that chose HybridEP and MoonEP is in `docs/qwen3.5_35B_A3B_clariden_throughput.md`.

Checked 3 Nov 2026. Three jobs are still pending on `QOSGrpNodeLimit`: Qwen3.5 HybridEP at 128 nodes (3829442), and Qwen3.5 MoonEP plus FSDP at 64 nodes (3837718) and 128 nodes (3837716).

## Cluster

Clariden nodes used here have 4 GH200 GPUs (about 96 GB HBM3 each) and 288 Grace cores. The account is `a0248`, the partition is `normal`. Jobs enter the container through the CSCS Container Engine: `srun --environment=` points at `container/alps/nemo-26.08.toml`, which loads the NeMo 26.08 image. Megatron-Core comes from `submodules/Megatron-LM` (`0.19.2+a2f7be60` on the smoke that validated the port). `torchrun` starts 4 processes per node with `--numa-binding node`. One Slurm task per node requests all 288 CPUs, so an expert-parallel group of 4 stays on one node.

Setup and the 1-node and 2-node validation of this port are in `docs/alps_clariden_setup.md`.

## Communication

Inside a node the GPUs talk over NVLink. Expert parallelism is 4 in every sweep, so the expert token exchange stays on that NVLink domain. The traffic that grows with the node count is the data-parallel exchange, and, on the pipeline-2 layout, the pipeline send and receive.

Cross-node NCCL goes through the `aws_ofi_nccl` hook in the EDF (`variant = "cuda-dl"`). The 2-node smoke (job 3720572, `NCCL_DEBUG=INFO`) selected the Slingshot provider: `NET/OFI Selected provider is cxi` with 4 NICs, `Using network AWS Libfabric`, and channels on `NET/AWS Libfabric/<n>/GDRDMA`. There were no `NET/Socket` lines. The scaling jobs use the same EDF with `NCCL_DEBUG=WARN`, so they do not repeat those lines. No fabric bandwidth is quoted here. The logs do not measure it.

## How to read a table

Strong scaling keeps the global batch fixed and adds GPUs, so the work per GPU falls and the ideal step time shrinks as `T_base × GPUs_base / GPUs`. Speedup is `T_base / T`, and efficiency is that speedup divided by `GPUs / GPUs_base`.

Weak scaling keeps the sequences per GPU fixed and grows the global batch with the GPU count. The ideal step time is flat at the smallest measured point. Efficiency is `T_base / T`.

The two Qwen3 sweeps use different 4-GPU baselines (global batch 2048 against global batch 64), so their absolute step times are not a comparison. Each efficiency is against that sweep's own ideal. The two Qwen3.5 sweeps are different models of parallelism as well: pipeline 2 with HybridEP against pipeline 1 with MoonEP and Megatron-FSDP, and different micro-batches. Their TFLOP/s numbers sit next to each other only as separate results.

On the Qwen3 runs the optimizer is Muon with 5 Newton–Schulz steps. That work is inside the step time and outside Megatron's FLOP counter, so those MFU figures are a lower bound on achieved utilization.

## Qwen3 10B-A0.9B, expert all-to-all

Configs: `config/experiments/swagatam/alps/qwen3_sparse_0.9B_wider_20L_smoke.yaml`, `qwen3_sparse_0.9B_wider_20L_strong_scaling.yaml`, `qwen3_sparse_0.9B_wider_20L_weak_scaling.yaml`. Launcher: `pretrain_gpt.py`. Logs from 22–24 Oct 2026.

### Model

20 layers, hidden 2048, 16 heads, 8 query groups, head dimension 128. Dense FFN width 6144, expert FFN 1280, SwiGLU, RMSNorm, QK-norm, RoPE base 10000, tied embeddings. 64 experts, top-4, router in fp32, aux loss `1e-2`, z-loss `1e-3`, grouped GEMM. The config labels this 10B total and about 0.9B active, non-embedding. Sequence 4096, bf16, mock data.

### Parallelism

Tensor, pipeline, and context parallelism are 1. Expert parallelism is 4 and stays inside the node, so the expert all-to-all (`moe_token_dispatcher_type: alltoall`, overlapped) does not cross Slingshot. Data-parallel size equals the GPU count. The distributed optimizer is on, with `overlap_grad_reduce` and `overlap_param_gather`. `CUDA_DEVICE_MAX_CONNECTIONS=32`. Sequence parallelism is requested and Megatron turns it off because tensor parallelism is 1. Optimizer is Muon (spectral, 5 Newton–Schulz steps) with Adam on the scalar parameters. Micro-batch is 1.

### Strong scaling

Global batch 2048 fixed. Gradient-accumulation steps fall as `2048 / GPUs`: 512, 256, 128, 64, 32, 16, 8, 4. Mean of iterations 11–40 (30 steps). The shortest early jobs reached iteration 41 and were then cancelled, so the window stops at 40 on every scale. Iterations 1–10 are startup. Ideal step at 4 GPUs is 144.6 s.

| GPUs | Nodes | GAS | Job | Mean step | Ideal | Speedup | Efficiency | TFLOP/s/GPU | MFU | Tok/s/GPU | Aggregate tok/s | Step CV |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 1 | 512 | 3721188 | 144.6 s | 144.6 s | 1.00× | 100% | 137.9 | 13.9% | 14,499 | 58.0k | 0.6% |
| 8 | 2 | 256 | 3721191 | 72.92 s | 72.32 s | 1.98× | 99.2% | 136.8 | 13.8% | 14,381 | 115k | 0.6% |
| 16 | 4 | 128 | 3721185 | 37.54 s | 36.16 s | 3.85× | 96.3% | 132.9 | 13.4% | 13,970 | 224k | 1.9% |
| 32 | 8 | 64 | 3721186 | 19.59 s | 18.08 s | 7.38× | 92.3% | 127.4 | 12.9% | 13,396 | 429k | 3.1% |
| 64 | 16 | 32 | 3721189 | 10.73 s | 9.04 s | 13.5× | 84.2% | 116.5 | 11.8% | 12,251 | 784k | 5.4% |
| 128 | 32 | 16 | 3721187 | 6.24 s | 4.52 s | 23.2× | 72.4% | 100.7 | 10.2% | 10,581 | 1.35M | 9.0% |
| 256 | 64 | 8 | 3721509 | 4.34 s | 2.26 s | 33.4× | 52.1% | 73.1 | 7.4% | 7,684 | 1.97M | 13% |
| 512 | 128 | 4 | 3721508 | 3.35 s | 1.13 s | 43.2× | 33.7% | 49.0 | 5.0% | 5,153 | 2.64M | 24% |

Efficiency stays above 92% through 32 GPUs. Past that the accumulation steps get short, and the on-node expert all-to-all plus the growing data-parallel exchange take a larger share of the step. At 512 GPUs the ideal step is 1.13 s and the measured mean is 3.35 s (43.2× against an ideal 128×).

The 512-GPU mean is the slow stretch inside iterations 11–40 (CV 24%). Faster steps in that job sit near 2.48 s and 63 TFLOP/s/GPU, about 46% of the ideal step. Averaging every iteration after 10 (90 steps; the job finished 100) gives 2.98 s, 54.4 TFLOP/s/GPU, and about 38% efficiency. The table keeps iterations 11–40 so the window matches the other rows.

### Weak scaling

16 sequences per GPU at every scale (GAS 16, micro-batch 1). Global batch is `64 × nodes`. Each GPU therefore sees `16 × 4096` tokens per step. Mean of iterations 11–60 (50 steps). Every job finished 60 iterations. Ideal step is the 4-GPU time, 5.04 s.

| GPUs | Nodes | GBS | Job | Mean step | Ideal | Step / ideal | Efficiency | TFLOP/s/GPU | MFU | Tok/s/GPU | Aggregate tok/s | Step CV |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 1 | 64 | 3725290 | 5.04 s | 5.04 s | 1.00× | 100% | 124.2 | 12.6% | 13,054 | 52.2k | 6.2% |
| 8 | 2 | 128 | 3725291 | 5.31 s | 5.04 s | 1.05× | 94.8% | 117.8 | 11.9% | 12,384 | 99.1k | 6.5% |
| 16 | 4 | 256 | 3725288 | 5.55 s | 5.04 s | 1.10× | 90.7% | 112.6 | 11.4% | 11,837 | 189k | 5.4% |
| 32 | 8 | 512 | 3725289 | 5.79 s | 5.04 s | 1.15× | 87.0% | 108.4 | 11.0% | 11,396 | 365k | 8.6% |
| 64 | 16 | 1,024 | 3725286 | 6.00 s | 5.04 s | 1.19× | 83.9% | 104.5 | 10.6% | 10,987 | 703k | 8.3% |
| 128 | 32 | 2,048 | 3725285 | 6.20 s | 5.04 s | 1.23× | 81.3% | 101.2 | 10.2% | 10,638 | 1.36M | 7.8% |
| 256 | 64 | 4,096 | 3725287 | 6.35 s | 5.04 s | 1.26× | 79.3% | 98.9 | 10.0% | 10,394 | 2.66M | 9.2% |
| 512 | 128 | 8,192 | 3725292 | 6.48 s | 5.04 s | 1.29× | 77.7% | 97.3 | 9.8% | 10,227 | 5.24M | 10.9% |

The step grows from 5.04 s to 6.48 s across a 128× increase in GPUs, and the last doubling (256 to 512 GPUs) adds 0.13 s. Expert all-to-all stays on the node. What grows is the data-parallel gradient and parameter exchange. Aggregate tokens/s rises from 52k to 5.24M.

## Qwen3.5-35B-A3B, HybridEP, pipeline 2

Config: `config/experiments/swagatam/alps/qwen3.5_moe_35B_A3B_gdn_weak_scaling.yaml` (model from `qwen3.5_moe_35B_A3B_gdn_smoke.yaml`). Launcher: `pretrain_gpt.py`. Logs from 1 Nov 2026. Only weak scaling was run.

### Model

40 layers. Each block of 4 is 3 Gated DeltaNet layers and 1 full attention layer (`linear_attention_freq: 4`): 30 Gated DeltaNet, 10 attention. Hidden 2048, 16 heads, 2 query groups, head dimension 256. Partial interleaved RoPE (64 of 256 channels, base 10⁷). Expert FFN 512, 256 routed experts, top-8, one shared expert of the same width. bf16, full activation recompute of 1 layer, Adam. The log reports 34.19B parameters total. Sequence 4096, mock data.

### Parallelism

Tensor parallelism 1, pipeline parallelism 2, expert parallelism 4. Expert parallelism fills one node, so HybridEP (`moe_token_dispatcher_type: flex`, `moe_flex_dispatcher_backend: hybridep`) exchanges tokens over NVLink. Pipeline parallelism is the cross-node traffic: the two stages send and receive over Slingshot. Data-parallel size is `2 × nodes`. Expert data-parallel size is `nodes / 2` (2 at 4 nodes, 32 at 64 nodes). `overlap_grad_reduce` and `overlap_param_gather` are on. Grouped GEMM uses the Cutlass kernel (`NVTE_USE_CUTLASS_GROUPED_GEMM=1`). Micro-batch 1, 32 micro-batches per GPU.

One node cannot run this layout: pipeline 2 and expert parallelism 4 need 8 GPUs. Two nodes was not run. Expert data-parallel size would be 1 there, and the expert Adam shard would sit on every GPU. The sweep starts at 4 nodes, the smallest point that has fit.

### Weak scaling

Global batch is `64 × nodes`. Each GPU sees 32 sequences per step (`32 × 4096` tokens). Mean of iterations 6–30 (25 steps). Iterations 1–2 are about 357 s and 189 s on every scale. Iterations 3–5 are still a bit high, so the window starts at 6. Every finished job ran 30/30 iterations. Ideal step is the 16-GPU time, 17.39 s.

| GPUs | Nodes | GBS | DP | Expert DP | Job | Mean step | Ideal | Step / ideal | Efficiency | TFLOP/s/GPU | MFU | Tok/s/GPU | Aggregate tok/s | Step CV |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 4 | 256 | 8 | 2 | 3829444 | 17.39 s | 17.39 s | 1.00× | 100% | 72.2 | 7.3% | 3,781 | 60.5k | 5.8% |
| 32 | 8 | 512 | 16 | 4 | 3829443 | 19.15 s | 17.39 s | 1.10× | 90.8% | 66.0 | 6.7% | 3,453 | 110k | 9.3% |
| 64 | 16 | 1,024 | 32 | 8 | 3829440 | 19.88 s | 17.39 s | 1.14× | 87.5% | 63.8 | 6.4% | 3,340 | 214k | 11.5% |
| 128 | 32 | 2,048 | 64 | 16 | 3829441 | 21.24 s | 17.39 s | 1.22× | 81.9% | 59.9 | 6.1% | 3,136 | 401k | 12.3% |
| 256 | 64 | 4,096 | 128 | 32 | 3829439 | 21.72 s | 17.39 s | 1.25× | 80.1% | 58.7 | 5.9% | 3,071 | 786k | 13.2% |
| 512 | 128 | 8,192 | 256 | 64 | 3829442 | pending | | | | | | | | |

The step goes from 17.39 s to 21.72 s at 256 GPUs (80% of the 16-GPU step). Most of that is the first doubling. The CV climbs from 6% to 13% because steps are bimodal: at 16 GPUs a fast step is about 16.1 s and 78 TFLOP/s, and a slow step is about 18.6 s and 67 TFLOP/s. At 256 GPUs the fast steps are about 17.8 s and 70 TFLOP/s, while slow steps reach about 26 s. The mean includes both. The 4-node HybridEP smoke at global batch 128 (job 3828590) was about 8.7 s and 72 TFLOP/s/GPU. This sweep uses twice the micro-batches per GPU, and the 4-node mean step is about twice that.

Job 3829442 (128 nodes, 512 GPUs, global batch 8192) has no iteration log. It is pending on the group node limit.

## Qwen3.5-35B-A3B, MoonEP and Megatron-FSDP

Config: `config/experiments/swagatam/alps/qwen3.5_moe_35B_A3B_q35c_weak_scaling.yaml` (model from `qwen3.5_moe_35B_A3B_q35c_smoke.yaml`, taken from Korbinian's `q35_conservative_mn5`). Launcher: `pretrain_hybrid.py`. Logs from 2 Nov 2026. Only weak scaling was run. A separate 8-node and 16-node pair with a narrower FSDP group is at the end of this section.

### Model

Same widths as the pipeline-2 model: hidden 2048, 16 heads, 2 query groups, head dimension 256, expert FFN 512, 256 experts, top-8, one shared expert of width 512. The stack is a `HybridModel`. The pattern `LELELE*E` repeated is the same 3 Gated DeltaNet plus 1 attention schedule, 40 layers, with attention on layers 4, 8, …, 40. Gated DeltaNet widths: key head 128, value head 128, 16 key heads, 32 value heads, short conv kernel 4. Sigmoid router, sequence aux loss `1e-4`, expert bias updated at `1e-3`. Interleaved partial RoPE, RMSNorm epsilon `1e-6`, zero-centered gamma, QK-norm, attention output gate, attention softmax in fp32. bf16. Full recompute of 2 layers. Untied embeddings. Sequence 4096, mock data. The production micro-batch of 4 at sequence 8192 ran out of memory in the Gated DeltaNet Triton autotune (job 3832303), so these runs stay at sequence 4096.

### Parallelism

Tensor and pipeline parallelism are 1, so data-parallel size is `4 × nodes` and expert data-parallel size is the node count. Expert parallelism is 4 and stays inside the node. The dispatcher is MoonEP (`moe_flex_dispatcher_backend: moonep`, 64 SMs). `MOONEP_MULTICAST=0` because a 4-GPU GH200 node has no NVLink multicast. MoonEP is built into `pylibs_moonep_pkg` and also replaces the routed-expert GEMMs.

Parameters, gradients, and optimizer state are Megatron-FSDP ZeRO-3 (`data_parallel_sharding_strategy: optim_grads_params`). `num_distributed_optimizer_instances = max(1, nodes // 8)`: one shard group while the job is under 16 nodes, then groups of 8 nodes, replicated. Below 16 nodes the whole job is one ZeRO-3 group. From 16 nodes up, each group is 32 GPUs and a gradient all-reduce runs across the copies. `CUDA_DEVICE_MAX_CONNECTIONS=8`. Overlap of parameter gather and gradient reduction is on, with `fsdp_double_buffer`. Gradient communication dtype is fp32. Micro-batch 4, 4 micro-batches per GPU, so each GPU sees 16 sequences per step. Memory at 4 nodes was about 0.92 of the GPU.

### Weak scaling

Global batch is `64 × nodes`. Mean of iterations 3–20 (18 steps). Iterations 1–2 are startup, about 158–164 s and 87–91 s. From iteration 3 the series is flat (step-time CV 0.1–0.5%). All four finished jobs ran 20/20. Ideal step is the 16-GPU time, 15.00 s. A prior 4-node MoonEP smoke at this batch (job 3834529) was about 14.7 s and 83 TFLOP/s/GPU. The steady mean here is 15.00 s and 81.3 TFLOP/s/GPU.

| GPUs | Nodes | GBS | DP | Expert DP | FSDP groups | Job | Mean step | Ideal | Step / ideal | Efficiency | TFLOP/s/GPU | MFU | Tok/s/GPU | Aggregate tok/s | Step CV |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 4 | 256 | 16 | 4 | 1 | 3837717 | 15.00 s | 15.00 s | 1.00× | 100% | 81.3 | 8.2% | 4,370 | 69.9k | 0.1% |
| 32 | 8 | 512 | 32 | 8 | 1 | 3837720 | 16.95 s | 15.00 s | 1.13× | 88.5% | 71.9 | 7.3% | 3,866 | 124k | 0.5% |
| 64 | 16 | 1,024 | 64 | 16 | 2 | 3837715 | 17.31 s | 15.00 s | 1.15× | 86.6% | 70.4 | 7.1% | 3,787 | 242k | 0.4% |
| 128 | 32 | 2,048 | 128 | 32 | 4 | 3837714 | 17.71 s | 15.00 s | 1.18× | 84.7% | 68.8 | 7.0% | 3,700 | 474k | 0.3% |
| 256 | 64 | 4,096 | 256 | 64 | 8 | 3837718 | pending | | | | | | | | |
| 512 | 128 | 8,192 | 512 | 128 | 16 | 3837716 | pending | | | | | | | | |

Almost all of the loss is the first doubling, 15.00 s to 16.95 s, which is also where the ZeRO-3 group widens from 16 GPUs to 32. After that the step grows to 17.71 s at 128 GPUs (85% of the 16-GPU step). Replicating the 8-node group adds a few tenths of a second per doubling. Aggregate tokens/s rises from 70k to 474k.

Jobs 3837718 (64 nodes, 256 GPUs, global batch 4096) and 3837716 (128 nodes, 512 GPUs, global batch 8192) have no iteration logs. Both are pending on the group node limit.

### Narrower FSDP groups

Same MoonEP recipe, with the shard group held at 4 nodes (`num_distributed_optimizer_instances = nodes / 4`) instead of 8. These two points are the pair already written up in `docs/qwen3.5_35B_A3B_clariden_throughput.md`. They are single measured steps from iteration 3 on, quoted here so this file has every scaling point.

| Nodes | GPUs | FSDP groups | Job | Step | TFLOP/s/GPU | Tok/s/GPU | Memory |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 8 | 32 | 2 × 4 nodes | 3837956 | 14.0 s | 87.2 | 4,690 | 0.95 |
| 16 | 64 | 4 × 4 nodes | 3838038 | 16.2 s | 75.1 | 4,050 | 0.95 |

A 4-node shard group is faster than an 8-node group at both sizes (14.0 s against 16.95 s at 8 nodes, 16.2 s against 17.31 s at 16 nodes). The per-layer all-gather stays inside 16 GPUs, and one gradient all-reduce per step crosses the copies. Memory stays near 0.95, because every GPU keeps a 16-GPU shard. MoonEP is unchanged: its four ranks are still the four GPUs on one node.

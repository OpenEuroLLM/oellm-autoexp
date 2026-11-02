# Qwen3.5-35B-A3B throughput on Clariden

Mock data, so the loss is not a training result. Throughput is Megatron's model-FLOP count (`TFLOP/s/GPU` in the log). Iterations 1–2 are the Gated DeltaNet Triton autotune and are left out.

Two layouts. The first is `pretrain_gpt.py`: a repeating 3 Gated DeltaNet + 1 attention schedule, pipeline 2, micro-batch 1, global batch 128. The second is `pretrain_hybrid.py`: the same layer schedule, pipeline 1, Megatron-FSDP, micro-batch 4. The dispatcher ranking flips between them. The hybrid config was later aligned to the GPT schedule; the micro-batch table below is from before that alignment, and the weak-scaling section is after it.

Container for both: `nemo-26.08.sqsh`. Megatron-Core `0.19.2` from `submodules/Megatron-LM` (`oellm/v0.19-hybrid`). Clariden, 4 nodes × 4 GH200. Expert parallelism 4 stays inside one node (NVLink). Cross-node HybridEP and DeepEP are not used. Both want GPU-initiated RDMA. Clariden's inter-node network is Slingshot, which NCCL already uses for the pipeline send/recv.

## pretrain_gpt

Config: `config/experiments/swagatam/alps/qwen3.5_moe_35B_A3B_gdn_smoke.yaml`. Launcher: `submodules/Megatron-LM/pretrain_gpt.py`. Logs and traces: `output/qwen3.5_35B_A3B_gdn_smoke_TP1_PP2_EP4_N4_clariden/`.

Quoted numbers are iterations 10–12 when the run got that far. Global batch 128 only.

### Model and layout

Qwen3.5-35B-A3B text model (40 layers, hidden 2048, 16 heads, 2 query groups, head dim 256). Each block of 4 layers is 3 Gated DeltaNet layers and 1 full attention layer (`experimental_attention_variant=gated_delta_net`, `linear_attention_freq=4`). Mixture of experts: 256 routed experts, top-8, expert FFN 512, one shared expert of the same width. Partial interleaved RoPE (64 of 256 channels, base 10⁷). bf16, Adam, full activation recompute. Sequence length 4096. Mock data, `NullTokenizer` vocabulary 262144.

The log reports 34.19B parameters total, about 5.28B on each pipeline stage.

Clariden, 4 nodes × 4 GH200. Tensor parallelism 1, pipeline parallelism 2, expert parallelism 4, micro-batch 1, global batch 128. Expert parallelism stays inside one node (4 GPUs, NVLink). The second pipeline stage is the cross-node traffic. Full recompute stays on: turning it off ran out of memory at this layout for both batch 128 and batch 1024.

Common flags on every row except the plain baseline: `overlap_grad_reduce`, `overlap_param_gather`, `moe_permute_fusion`, `NVTE_USE_CUTLASS_GROUPED_GEMM=1`. Expert all-to-all overlap is off. It requires virtual pipeline parallelism and forbids full recompute, which this layout needs in order to fit.

### Dispatchers

| Name | Megatron settings | What moves the tokens |
| --- | --- | --- |
| NCCL all-to-all | `moe_token_dispatcher_type=alltoall` | NCCL. In the profile this is `ncclDevKernel_SendRecv`. |
| HybridEP | `flex` + `moe_flex_dispatcher_backend=hybridep` | DeepEP's HybridEP buffer, NVLink inside the node. Already in the image. |
| DeepEP | `flex` + `moe_flex_dispatcher_backend=deepep` | DeepEP's intra-node buffer. Same package. |
| MoonEP | `flex` + `moe_flex_dispatcher_backend=moonep` | MoonshotAI MoonEP. Not in the image. Built into `pylibs_moonep_pkg` with `container/patches/moonep-no-multicast.patch`, because a 4-GPU GH200 node has no NVLink multicast. Also replaces the routed-expert GEMMs, so its checkpoints are a different layout. |

Cross-node HybridEP and DeepEP are not used. Both want GPU-initiated RDMA. Clariden's inter-node network is Slingshot.

### Results at global batch 128

Memory stays about 64 GB of 96 GB on every finished run.

| Dispatcher | Job | Step time | TFLOP/s per GPU | Tokens/s per GPU | Notes |
| --- | --- | --- | --- | --- | --- |
| NCCL, no overlap flags | 3730762 | 11–14 s | ~50 (last step 55) | ~2,600 | Iterations 3–10 only. No grouped-GEMM env, no permute fusion, no grad/param overlap. |
| NCCL all-to-all | 3827971 | 9.94 s | 63.0 | 3,300 | Iterations 10–12, flat. |
| HybridEP | 3828590 | 8.7–9.9 s | 63.5, 71.6, 72.0 | 3,300–3,770 | Iterations 10–12. The last two steps are the high ones. |
| DeepEP | 3829050 | 9.9–10.5 s | 59.6, 63.3, 63.1 | 3,120–3,310 | Iterations 10–12. Same band as NCCL, below HybridEP. |
| MoonEP | 3829179 | 9.7–10.7 s | 63.8, 64.4, 58.7 | 3,070–3,370 | Iterations 10–12. Same band as NCCL and DeepEP. `MOONEP_MULTICAST=0`, 64 SMs. 32 SMs (job 3829398) was 63.3, 63.2, 57.3. |

Profiled steps are not in this table. The profiler slowed steps 6–9 to about 14–16 s (about 40–43 TFLOP/s). The step after profiling stopped matched the unprofiled run.

### What the traces show

Rank 0, steps 6–8. Files: `torch_profile/rank-0-nccl.json.gz`, `torch_profile/rank-0-hybridep.json.gz`, `torch_profile/rank-0-deepep.json.gz`, `torch_profile/rank-0-moonep.json.gz` (profile job 3829219).

GPU kernel time over those three steps was 19.1 s (NCCL), 17.3 s (HybridEP), 20.1 s (DeepEP), and 14.8 s (MoonEP). GEMM time was 4.3 s on the first three. Attention was 0.7 s. MoonEP's GEMM bucket is 4.1 s and 55,680 launches, against 4.3 s and 153,600 launches on the DeepEP trace: the routed-expert GEMM runs inside the MoonEP kernels.

| | NCCL | HybridEP | DeepEP | MoonEP |
| --- | --- | --- | --- | --- |
| Collectives | 9.4 s (49%) | 8.5 s (49%) | 10.9 s (54%) | 5.9 s (40%) |
| `SendRecv` | 5.9 s, 8,691 launches | 2.6 s, 51 launches | 3.3 s, 51 launches | 1.6 s, 51 launches |
| Dispatcher kernels | the extra send/recv | `device_sync` 1.2 s | notify 1.8 s, dispatch 0.9 s, combine 0.7 s | dispatch 0.6 s, combine 0.4 s, prologue 0.2 s |

The 8,691 send/recv launches were the expert exchange. The 51 that remain are the pipeline messages between the two stages. MoonEP's own device kernels add up to 1.6 s (the three above, plus prefetch, planning, grad reduce, and inter-rank sync). That is less than DeepEP's notify, dispatch, and combine (3.4 s). It is also the shortest kernel total of the four.

The step did not get shorter. After the profiler stopped, iteration 10 was 10.0 s and 62.5 TFLOP/s, the same band as the unprofiled MoonEP run. On this `pretrain_gpt` layout HybridEP is the fastest unprofiled result, about 72 TFLOP/s. NCCL reduce-scatter (1.7 s, 165 launches) and all-gather (0.9 s, 165 launches) are still in the MoonEP trace.

32 SMs is MoonEP's own default and did not move the step. Job 3829398, iterations 10–12: 9.89 s, 9.90 s, 10.92 s (63.3, 63.2, 57.3 TFLOP/s).

## pretrain_hybrid

Config: `config/experiments/swagatam/alps/qwen3.5_moe_35B_A3B_q35c_smoke.yaml`, taken from `config/experiments/korbi/production/q35_conservative_mn5.yaml`. Launcher: `submodules/Megatron-LM/pretrain_hybrid.py`. Logs: `output/qwen3.5_35B_A3B_q35c_smoke_MBS*_EP4_N4_clariden/`.

Global batch 256, sequence 4096, 20 iterations. Quoted numbers are the flat steps from iteration 3 on. Liger's fused LM-head is off.

### What changes from pretrain_gpt

| | `pretrain_gpt` | `pretrain_hybrid` |
| --- | --- | --- |
| How the stack is chosen | `num_layers=40` and `linear_attention_freq=4` on the GPT model. Layer `i` is full attention when `(i+1) % 4 == 0`. | `is_hybrid_model` and `hybrid_layer_pattern`. `*` is attention, `L` is Gated DeltaNet, `E` marks that block as MoE. `num_layers` is taken from the pattern. |
| Layer mix | 30 Gated DeltaNet, 10 attention. Layer 1 is Gated DeltaNet. | Same schedule, written as `hybrid_layer_pattern` `LELELE*E` repeated. The earlier hybrid runs used Korbinian's pattern instead: 29 Gated DeltaNet, 11 attention, block 1 attention. |
| Parallelism | Pipeline 2, distributed optimizer, no FSDP. `CUDA_DEVICE_MAX_CONNECTIONS=1`. | Pipeline 1, Megatron-FSDP ZeRO-3 (`optim_grads_params`) over all 16 GPUs. `CUDA_DEVICE_MAX_CONNECTIONS=8`, which FSDP requires. |
| Recompute | Full, 1 layer. | Full, 2 layers. |
| Batch in the quoted runs | Micro-batch 1, global batch 128. 32,768 tokens per GPU per step. | Micro-batch 4, global batch 256. 65,536 tokens per GPU per step. |
| Router | Softmax, global aux loss `1e-3`. | Sigmoid, sequence aux loss `1e-4`, expert bias updated at `1e-3`. |
| RoPE and norm | Interleaved RoPE, LayerNorm epsilon `1e-6`. Max position 262144. | Same interleaved RoPE and epsilon `1e-6` in the current config. Max position 4096. The earlier hybrid runs left RoPE non-interleaved and epsilon at `1e-5`. |
| Extra fusions | Permute fusion and Cutlass grouped GEMM on the dispatcher runs. | Shared-expert overlap, fused residual RMSNorm, fused short conv, fused Gated DeltaNet output norm. |

Hidden size, heads, Gated DeltaNet widths, 256 experts, top-8, and the shared expert of width 512 are the same. Expert parallelism stays 4.

Micro-batch 4 at sequence 8192 (job 3832303) ran out of memory in the Gated DeltaNet Triton autotune on the first backward. Sequence 4096 fits.

### Results at global batch 256

MoonEP, 64 SMs, `MOONEP_MULTICAST=0`. Raising the micro-batch at a fixed global batch shortens the step almost in proportion: the GPU was underfed at micro-batch 1.

| Micro-batch | Dispatcher | Job | Step time | TFLOP/s per GPU | Tokens/s per GPU | Peak reserved |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | MoonEP | 3832449 | 49.0 s | 24.9 | 1,340 | 51 GB |
| 2 | MoonEP | 3833601 | 25.7 s | 47.4 | 2,550 | 59 GB |
| 4 | MoonEP | 3834529 | 14.7 s | 82.8 | 4,450 | 74 GB (usage 0.92) |
| 4 | HybridEP | 3834637 | 16.6 s | 73 | 3,950 | 75 GB (usage 0.93) |

HybridEP used the backend's own SM default (`moe_flex_dispatcher_num_sms` unset). Iterations 10–20 drift from 74.4 down to about 72. MoonEP at micro-batch 4 stays flat at 82.6–83.2.

On the `pretrain_gpt` layout HybridEP was ahead of MoonEP (about 72 against about 63). Here MoonEP is ahead. At micro-batch 1 the MoonEP step was launch-latency bound: the profile showed the GPU idle after each MoonEP kernel while Python built views. At micro-batch 4 those expert GEMMs are long enough to cover that, and MoonEP runs them inside dispatch and combine. HybridEP still does the exchange and the grouped GEMM as separate kernels. Memory is full enough that micro-batch 8 will not fit.

### Weak scaling, groups of 8

Same aligned hybrid config. `config/experiments/swagatam/alps/qwen3.5_moe_35B_A3B_q35c_weak_scaling.yaml`. Micro-batch 4, sequence 4096, 4 micro-batches per GPU, so the global batch is `64 × nodes`. MoonEP. `num_distributed_optimizer_instances = max(1, nodes // 8)`: one ZeRO-3 group through 8 nodes, then groups of 8 nodes replicated across the job. Quoted steps are iteration 3 onward.

| Nodes | Global batch | FSDP groups | Job | Step | TFLOP/s per GPU | Tokens/s per GPU | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 256 | 1 × 4 nodes | 3837717 | 15.0 s | 81.3 | 4,370 | 0.92 |
| 8 | 512 | 1 × 8 nodes | 3837720 | 16.9 s | 72.0 | 3,870 | 0.74 |
| 16 | 1024 | 2 × 8 nodes | 3837715 | 17.3 s | 70.5 | 3,790 | 0.76 |

Widening the shard group from 4 nodes to 8 costs about 1.9 s. Copying that group at 16 nodes adds about 0.4 s. Memory drops once the shard is 32 GPUs wide, and stays there when the group is replicated.

### Groups of 4 instead

Same batch and MoonEP, with the shard group held at 4 nodes (`num_distributed_optimizer_instances = nodes / 4`). Logs are under `output/qwen3.5_35B_A3B_q35c_smoke_MBS4_EP4_N*_clariden/`.

| Nodes | FSDP groups | Job | Step | TFLOP/s per GPU | Tokens/s per GPU | Memory |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 2 × 4 nodes | 3837956 | 14.0 s | 87.2 | 4,690 | 0.95 |
| 16 | 4 × 4 nodes | 3838038 | 16.2 s | 75.1 | 4,050 | 0.95 |

A group of 4 is faster than a group of 8 at both sizes. The per-layer all-gather stays inside 16 GPUs. The extra cost is one gradient all-reduce across the copies, once per step. Two copies were hidden under the backward (14.0 s, a bit under the single 4-node group). Four copies added about 2 s, to 16.2 s. That is still ahead of two groups of 8 (17.3 s, 70.5 TFLOP/s).

Memory does not improve as nodes are added. Every GPU keeps a 16-GPU shard, so usage stays about 0.95, the same ceiling as the 4-node run. The cross-group all-reduce is the part that grows with the number of copies. MoonEP is unchanged by this: its four ranks are still the four GPUs on one node. Expert matrices are sharded across the four nodes of the FSDP group, not across those four GPUs.

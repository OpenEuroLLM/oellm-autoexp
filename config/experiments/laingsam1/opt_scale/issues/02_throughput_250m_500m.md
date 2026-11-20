# Throughput of the 250M and 500M hybrid MoE on JUPITER: expert placement, batch size, cost per run

Status: 2026-11-20. Measured with short jobs (smoke test 18). A follow-up test of further speed
settings (smoke test 19) is queued.

## Summary

- Splitting the 96 experts over the 4 GPUs of a node, with the HybridEP dispatcher, is the fastest
  setting in every case tested. At 500M it is about twice as fast as holding all experts on every
  GPU, and it uses 20 GB less GPU memory.
- With that setting the models run at 105 to 150 TFLOP/s per GPU (Megatron's own figure), which
  is 11 to 15 % of the GPU's peak. The 100M model on one node runs at about 100 (10 %).
- Larger batches are faster per token: 10 to 35 % from batch 256 to 1024.
- AdamW and Muon cost the same within the measurement noise.
- One run with all three budgets costs about 150 to 200 GPU-h at 250M and 450 to 600 GPU-h at
  500M.

## Setup

| | |
|---|---|
| Models | same family as the 100M model (96 experts, top-4 plus shared, expert width = half the model width). 250M: 896 wide, 5 blocks, 246M active / 2.5B total. 500M: 1152 wide, 6 blocks, 486M active / 4.9B total |
| Machine | JUPITER, 4 nodes = 16 GH200 GPUs per job |
| Parallelism | data parallel across all GPUs; experts either on every GPU or split over the 4 GPUs of one node |
| Micro-batch per GPU | 8 sequences at 250M, 4 at 500M (sequence length 4096) |
| Other settings | bf16, flash attention, Liger fused output layer and loss, grouped GEMM |
| Measurement | 20 to 80 training steps, median over steps after the first 5 |

The model shapes are placeholders until the sizes for the scaling study are fixed.

## Results

### Tokens per second per GPU (thousands)

| Size | Optimizer | Experts | Batch 256 | Batch 512 | Batch 1024 |
|---|---|---|---|---|---|
| 250M | AdamW | on every GPU | 21.3 | 34.2 | 23.9 |
| 250M | AdamW | split, all-to-all | 32.6 | 35.9 | 45.2 |
| 250M | AdamW | split, DeepEP | 31.1 | 40.3 | 40.7 |
| 250M | AdamW | split, HybridEP | **34.6** | **45.3** | **46.9** |
| 250M | Muon | on every GPU | 24.8 | 29.9 | 28.8 |
| 250M | Muon | split, all-to-all | 38.7 | 42.6 | 48.3 |
| 250M | Muon | split, DeepEP | 35.7 | 30.0 | 43.6 |
| 250M | Muon | split, HybridEP | **39.4** | **43.9** | **49.0** |
| 500M | AdamW | on every GPU | 17.1 | 12.3 | 19.0 |
| 500M | AdamW | split, all-to-all | 23.3 | 17.6 | 25.6 |
| 500M | AdamW | split, DeepEP | 22.0 | 22.6 | 17.2 |
| 500M | AdamW | split, HybridEP | **25.2** | **26.6** | **27.9** |
| 500M | Muon | on every GPU | 15.4 | 17.4 | 14.8 |
| 500M | Muon | split, all-to-all | 20.6 | 15.8 | 18.5 |
| 500M | Muon | split, DeepEP | 14.0 | 15.2 | 18.0 |
| 500M | Muon | split, HybridEP | **22.8** | **24.6** | **28.6** |

Each number is one short job. Several rows other than HybridEP go down and up again with batch
size, which is noise between jobs (different nodes), so differences under about 20 % in those
rows should not be trusted. HybridEP is the only setting whose speed rises steadily with batch
size.

### HybridEP in detail

TFLOP/s is the value Megatron prints in its log (it includes the output layer and attention).
Peak is taken as 989 TFLOP/s.

| Size | Optimizer | Batch | Seconds per step | TFLOP/s per GPU | Share of peak | Peak GPU memory |
|---|---|---|---|---|---|---|
| 250M | AdamW | 256 | 1.90 | 107 | 10.8 % | 66 GB |
| 250M | AdamW | 512 | 2.89 | 140 | 14.1 % | 66 GB |
| 250M | AdamW | 1024 | 5.59 | 145 | 14.7 % | 66 GB |
| 250M | Muon | 256 | 1.66 | 122 | 12.3 % | 66 GB |
| 250M | Muon | 512 | 2.98 | 136 | 13.7 % | 66 GB |
| 250M | Muon | 1024 | 5.35 | 151 | 15.3 % | 66 GB |
| 500M | AdamW | 256 | 2.60 | 127 | 12.8 % | 61 GB |
| 500M | AdamW | 512 | 4.92 | 134 | 13.5 % | 61 GB |
| 500M | AdamW | 1024 | 9.41 | 140 | 14.1 % | 61 GB |
| 500M | Muon | 256 | 2.87 | 115 | 11.6 % | 59 GB |
| 500M | Muon | 512 | 5.32 | 124 | 12.5 % | 59 GB |
| 500M | Muon | 1024 | 9.17 | 143 | 14.5 % | 59 GB |

With all experts on every GPU, peak memory is 76 GB (250M) and 80 GB (500M) of 95 GB.

### Larger micro-batch does not fit

Doubling the micro-batch (16 at 250M, 8 at 500M) with HybridEP ran out of GPU memory in all 12
jobs. At 100M, raising the micro-batch from 4 to 16 was the largest single speed-up (2.3x), so
memory is what limits these sizes.

### Where the step time goes

Step time against the number of micro-batches per step is close to a straight line: about 0.6 s
per micro-batch, plus 0.3 to 0.8 s that is paid once per step. The fixed part is as large for
AdamW as for Muon, so it is not the optimizer step. It explains why larger batches are faster per
token.

## Cost of a training run

Tokens for one run with the 1x, 2x and 4x budgets (20, 40 and 80 tokens per active parameter,
branching from one constant-rate run): 92 x active parameters. Estimate = measured speed x 1.1.
The factor 1.1 comes from the finished 100M runs, whose billed GPU-h were 1.06 to 1.2 times the
pure training time (startup, evaluation, checkpoints).

| Size | Batch | Tokens | GPU-h | Hours on 4 nodes |
|---|---|---|---|---|
| 100M (measured, 1 node) | 256 | 9.3B | 45 to 52 | 7.5 to 8.5 per constant-rate run |
| 250M | 256 | 22.6B | 175 to 200 | 11 to 12.5 |
| 250M | 512 | 22.6B | about 155 | 10 |
| 250M | 1024 | 22.6B | about 145 | 9 |
| 500M | 256 | 44.7B | 540 to 600 | 34 to 37 |
| 500M | 512 | 44.7B | 515 to 555 | 32 to 35 |
| 500M | 1024 | 44.7B | 480 to 490 | about 30 |

Read the 250M and 500M rows as plus or minus 15 %. JUPITER jobs are limited to 12 hours, so a
500M run needs three chained jobs on 4 nodes.

## For comparison

Numbers recorded in other configs in this repository, same machine:

| Model | TFLOP/s per GPU | Source |
|---|---|---|
| Dense 1.8B, CUDA graphs | 412 (41 %) | `korbi/speed_tests/llama1_8b_jupiter_tp1_mbs6_nosp_cg.yaml` |
| MoE 30B (3B active), Megatron | 118 (12 %) | `korbi/qwen3_30B_A3B_speed_comp/` |
| This work, 250M and 500M, HybridEP | 107 to 151 (11 to 15 %) | this issue |

## Open points

- **Scaling beyond 4 nodes is not measured.** At a fixed batch, more GPUs means fewer
  micro-batches per GPU per step, so the fixed cost per step weighs more. Estimated loss on 8
  nodes at batch 256: about 15 to 20 % per GPU.
- **Further speed settings** are under test (smoke test 19): diagnostics off, fused norm kernels,
  CUDA graphs for the MoE router, recomputing MoE activations to fit a larger micro-batch, a
  timer breakdown and a profile.
- **MoonEP** is supported by the Megatron fork but the package is not in the container used here
  (`MegatronTraining-JUPITER-te218-fa3`). Is there a container that has it?
- **Larger sizes** (850M, 1.8B) have not been measured with HybridEP.
- **Megatron's FLOP count** for this architecture is taken as printed. It has not been checked
  independently here.

## Where things are

| | |
|---|---|
| Test configs | `config/experiments/laingsam1/opt_scale/smoke_test/18_tput_250m_500m.yaml`, `19_tput_tuning.yaml` |
| Read-out script | `config/experiments/laingsam1/opt_scale/smoke_test/collect_scale_throughput.py --logs <log dir>` |
| 100M throughput history | `config/experiments/laingsam1/opt_scale/hyb100m/README.md` |

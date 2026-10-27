# Block diffusion for Qwen3 (MegaDLMs)

This feature converts pretrained autoregressive (AR) Qwen3 models into block-wise discrete
diffusion language models (BD3/SDAR style) and trains them with MegaDLMs. It adds a launcher in
`oellm_autoexp/dllm/`, a Hugging Face (HF) <-> MegaDLMs converter, evaluation tools, and Hydra
configs for Qwen3-0.6B, 1.7B, 4B, 8B and 14B.

## What the model computes

For a training sequence of effective length `L` (Megatron `seq_length = L + 1`) and a block size
`B` (`L % B == 0`):

- Noise: every block `k` draws `t_k ~ U[t_min, t_max]` and masks each position with probability
  `t_k` (at least one mask per block). Masked positions get the mask token `M = len(tokenizer)`
  (151669 for Qwen3). Optional complementary masking adds a second copy that masks exactly the
  positions the first copy kept.
- Input: `[xt, x0]` of length `2L` (noisy copy, clean copy). RoPE uses position `i mod L` for both
  copies.
- Attention (FlexAttention BlockMask): a noisy block sees its own noisy block (bidirectional) and
  the clean blocks before it; a clean block sees clean blocks up to and including itself. With
  `--bd-doc-mask 1` attention stays inside a document (EOD = `<|endoftext|>`).
- Loss: only noisy positions predict `x0`. Token shift 0 predicts `x0[i]` from position `i`;
  token shift 1 predicts it from position `i - 1` (and from the last clean token of the previous
  block at a block start). The ELBO weighting divides each block's cross entropy by its exact
  masked fraction; `uniform` uses plain masked cross entropy.
- Validation: a deterministic estimator (`--bd-val-seed`) reports `val nelbo bits/token`, an
  upper bound of the NLL that is comparable across block sizes and with AR NLL.
- `--bd-objective ar` trains the same weights as a causal model (the AR control).
- `--bd-objective nld` (NLD-style hybrid) makes the clean copy token-causal, so the model keeps
  an exact AR mode, and trains `AR cross entropy + 0.3 x block-diffusion loss` over the AR token
  count (`--bd-nld-ar-weight`, `--bd-nld-diffusion-weight`), after an AR-only stage
  (`--bd-nld-ar-only-fraction 0.2`). It is evaluated in diffusion mode and in AR mode.

The normative description is the dllm-bd3-spec (spec sections are cited in the code).

## Code map

| Path | Content |
|---|---|
| `oellm_autoexp/dllm/pretrain_block_difflm.py` | launcher: model provider, forward step, loss, throughput log |
| `oellm_autoexp/dllm/model.py` | `BlockDiffusionGPTModel` (subclass of the MegaDLMs difflm GPTModel) |
| `oellm_autoexp/dllm/flex_core_attention.py`, `context.py` | flex core attention; BlockMask passed through a module-level context |
| `oellm_autoexp/dllm/masks.py`, `noise.py`, `targets.py`, `schedule.py` | mask_mods, noise and validation estimator, sources and loss terms, block-size schedule |
| `oellm_autoexp/dllm/args.py` | `--bd-*` flags chained with the MegaDLMs flags |
| `oellm_autoexp/dllm/convert/` | Qwen3 layout, HF -> MegaDLMs template fill, MegaDLMs -> HF export, `convert_chain.sh` |
| `oellm_autoexp/dllm/eval/` | parity check, sampler, offline NELBO, GSM8K, results collector, `eval_chain.sh` |
| `oellm_autoexp/dllm/data_prep.py`, `tokenize_fineweb_edu.sbatch` | FineWeb-Edu sharding and Qwen3 tokenization |
| `oellm_autoexp/dllm/reference.py` | tiny pure-PyTorch Qwen3-like model for the invariant tests |
| `oellm_autoexp/dllm/build_native_stack_snellius.sbatch`, `requirements-native-snellius.txt` | native GPU stack |
| `config/backend/megadlm_bd_torchrun.yaml` | backend: launcher, `aux.difflm_args`, `aux.bd_args` |
| `config/backend/megatron/dllm/bd/` | `bd_train_defaults`, `qwen3_<size>_base`, data config |
| `config/{container,slurm}/snellius_megadlm_native.yaml` | jobs on the native venv (no container) |
| `config/experiments/dllm/bd/` | conversion, smoke, probe, tier 0 and tier 1, method preset, and evaluation experiments |

## Software stack (Snellius, no container)

The jobs run on a pip venv linked as `oellm-autoexp/.venv-gpu`: torch 2.13.0+cu126, Transformer
Engine 2.16.0 compiled against that torch (sm_90), transformers 5.14.1, with the exact pins in
`oellm_autoexp/dllm/requirements-native-snellius.txt`. Build it once on a CPU node:

```bash
python3 -m venv <venv dir> && ln -s <venv dir> .venv-gpu
sbatch -o "$OUTPUT_DIR/dllm-ablations/logs/p4/native-stack-%j.log" \
    oellm_autoexp/dllm/build_native_stack_snellius.sbatch
```

Jobs load no modules: torch and Transformer Engine use the CUDA, cuDNN and NCCL libraries of the
pip wheels. The launcher makes MegaDLMs' legacy `fused_kernels.load` a no-op, because it only
runs `$CUDA_HOME/bin/nvcc -V`. MegaDLMs compiles its dataset helpers with the system g++ at the
first start (`submodules/MegaDLMs/megatron/core/datasets`, ignored `*.so`). W&B runs are written
offline to `<run dir>/wandb`; upload them with `wandb sync`.

## Launcher flags

`--bd-objective {bd3,ar,nld}`, `--bd-block-size N`, `--bd-block-schedule "0:4,0.15:8"`,
`--bd-token-shift {0,1}`, `--bd-complementary-mask {0,1}`, `--bd-loss-weighting {elbo,uniform}`,
`--bd-t-min F`, `--bd-t-max F`, `--bd-doc-mask {0,1}`, `--bd-val-seed N`, and for
`--bd-objective nld` also `--bd-nld-ar-weight F`, `--bd-nld-diffusion-weight F` and
`--bd-nld-ar-only-fraction F`. The MegaDLMs flags go in
`backend.aux.difflm_args` (at least `--mask-token 151669 --difflm-varilen-prob 0`), the `--bd-*`
flags in `backend.aux.bd_args`. Stock Megatron keys go under `backend.megatron`.

## Data

FineWeb-Edu `sample-10BT` (9,672,101 documents) is split into 16 shards by parquet row group and
tokenized with the Qwen3 tokenizer (EOD appended): 9.28B training tokens in shards 00-14, and
0.62B held-out tokens in shard 15 (validation and test).

```bash
python -m oellm_autoexp.dllm.data_prep write-jsonl --input-dir $DATA_DIR/raw/fineweb-edu/sample/10BT \
    --num-shards 16 --shard <i> --output $DATA_DIR/raw/fineweb-edu-jsonl/shard_<ii>.jsonl
sbatch -o "$OUTPUT_DIR/dllm-ablations/logs/p2/tok-%A_%a.log" oellm_autoexp/dllm/tokenize_fineweb_edu.sbatch
python -m oellm_autoexp.dllm.data_prep inspect --prefix <prefix> --tokenizer Qwen/Qwen3-0.6B-Base
```

The data config is `config/backend/megatron/dllm/bd/data_qwen3_fineweb_edu.yaml`.

## Converting a Qwen3 checkpoint

Conversion uses template fill: a 1-iteration MegaDLMs run with mock data and the target
architecture and tensor-parallel size saves a `torch` checkpoint whose key set, shapes, TP shards
and padded vocabulary the converter fills from the HF weights. The mask-token embedding row is the
mean of the real-token rows. One job per size runs the whole chain (`convert_chain.sh`: template,
convert on CPU, then the GPU parity check) on TP GPUs:

```bash
PYTHONPATH=. python scripts/run_autoexp.py --config-name experiments/dllm/bd/convert_qwen3_0.6B \
    --submit-and-exit [backend.megatron.tensor_model_parallel_size=2]
```

The result is `$OUTPUT_DIR/dllm-ablations/converted/qwen3_<size>_tp<TP>`; the converter refuses an
output directory that already holds a converted checkpoint. The parity check
(`eval/parity_check.py`) runs both sides in float32 (bf16 rounding alone moves top-1 below the
bar) and compares AR-mode logits with HF (top-1 >= 99 percent, mean KL <= 1e-3 nats), checks
invariant I3 (mean and maximum per-token cross-entropy difference <= 2e-3 and 5e-2 nats) and the
nld clean stream against HF. Qwen3 0.6B (TP 1 and 2), 1.7B, 4B and 8B pass with top-1 >= 0.9995
and KL <= 4e-7. The steps by hand:

```bash
python -m oellm_autoexp.dllm.convert.hf_qwen3_to_megadlm --hf Qwen/Qwen3-0.6B-Base \
    --template <template ckpt dir> --out $OUTPUT_DIR/dllm-ablations/converted/qwen3_0.6B_tp1
python -m oellm_autoexp.dllm.convert.megadlm_to_hf_qwen3 --megatron <ckpt dir> \
    --hf-base Qwen/Qwen3-0.6B-Base --out <hf dir> --block-size 8 --token-shift 0 --mask-token-id 151669
```

Training loads the converted checkpoint with `--pretrained-checkpoint` (finetune semantics, no
optimizer state) and resumes from its own `--save` directory afterwards. Jobs that load these
files set `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`. For a trained checkpoint (`iter_*`) the exporter
checks the block size, token shift and clean-stream attention against the checkpoint's arguments
and takes `dllm_clean_attention` from `bd_objective` by default.

## Training

```bash
PYTHONPATH=. python scripts/run_autoexp.py --config-name experiments/dllm/bd/smoke_qwen3_0.6b --dry-run
PYTHONPATH=. python scripts/run_autoexp.py --config-name experiments/dllm/bd/smoke_qwen3_0.6b --submit-and-exit
```

Size defaults (tensor parallel, micro batch, nodes): 0.6B (1, 4, 1), 1.7B (1, 2, 2), 4B (2, 2, 4),
8B (2, 1, 8), 14B (4, 1, 8). Global batch 256 sequences of 2048 tokens, WSD schedule (1 percent
warmup with at least 50 steps, 20 percent linear decay to 0.1 x LR), bf16, torch checkpoints.
A job resubmitted with the same config resumes from its last checkpoint. Jobs read the tokenized
data from GPFS with 8 loader workers per rank (tier 0: about 30k tokens/s/GPU at 0.6B);
`oellm_autoexp/dllm/stage_data.sh` can copy it to a node-local disk on systems that have one
(opt-in `DLLM_STAGE_DATA=1`; Snellius gpu_h100 nodes have none).

| Experiment | Purpose |
|---|---|
| `probe_qwen3` | 20-iteration throughput probe; size, TP and micro batch as overrides |
| `tier0_qwen3_0.6B` | learning rates 1e-5, 3e-5, 1e-4 and the edge step 3e-4 at 0.5B tokens (list sweep, `--array-subset i`) |
| `tier1_qwen3_0.6B` | R00-R11 at 2B tokens and lr 1e-4 (one factor at a time, AR control R10, nld hybrid R11) |

Run names follow `bd3-q3_<size>-B<B>-sh<shift>-cm<comp>-w<weight>-t<tmin>_<tmax>-dm<doc>-sc<schedule>-lr<lr>-T<tokens>`
(schedule `0:4,0.15:8` is written `0-4_0.15-8`), `ar-q3_<size>-lr<lr>-T<tokens>`, and
`nld-q3_<size>-B<B>-sh<shift>-a<diffusion weight>-ao<AR-only fraction>-lr<lr>-T<tokens>`; a test
checks that every tier job's flags match its name. With complementary masking the global and micro batch
are halved, so the FLOPs per step match the base run.

Method presets combine these knobs (`config/backend/megatron/dllm/bd/method/`), one experiment
per size and method (`qwen3_<size>_<method>.yaml`, 0.6B to 8B):

| Preset | Settings (rest: B = 8, shift 0, ELBO weights, t in [0, 1], doc mask on) |
|---|---|
| `ar` | causal AR control |
| `sdar` | B = 4 |
| `bd3lm` | B = 4, clipped noise t in [0.45, 0.95] |
| `fastdllm_v2` | B = 32, token shift 1, complementary masks |
| `llada2` | block-size schedule 2 -> 2048 (full sequence) -> 32, complementary masks |
| `mdlm` | B = L = 2048 (masked diffusion over the whole sequence; the clean copy is unused) |
| `nld` | hybrid AR + 0.3 x block diffusion, B = 32, 20 percent AR-only |

With complementary masks, the global and micro batch are halved (same FLOPs per step).

## Evaluation

| Metric | Tool |
|---|---|
| M1 NELBO bits/token (AR: exact NLL) on 1000 held-out sequences | `python -m oellm_autoexp.dllm.eval.eval_nelbo` |
| AR-mode M1 of nld runs (exact NLL of the clean stream) | `eval_nelbo --objective ar`, `<run>.m1_ar.json` |
| M2 GSM8K 4-shot accuracy, static and dynamic@0.9 decoding | `python -m oellm_autoexp.dllm.eval.eval_gsm8k` |
| M3 tokens per forward at dynamic@0.9 | `eval_gsm8k` output |
| M4 training tokens/s/GPU | `bd throughput` lines in the training log |

`eval_qwen3.yaml` runs export, M1 and GSM8K for one model on one GPU (`eval/eval_chain.sh`); the
`EVAL_*` keys under `backend.env` select the checkpoint and settings, and the results land in
`$OUTPUT_DIR/dllm-ablations/eval` as `<run>.m1.json` and `<run>.gsm8k_<mode>.json`. Tier jobs run
the same chain in their own allocation after the last iteration (`train_then_eval.sh`, one GPU
per task). `EVAL_DTYPE=float32` runs both evaluators in float32 (used for the GPU check of I8, which
bf16 kernel rounding breaks); `EVAL_CLEAN_ATTENTION` overrides the clean attention of the export:

```bash
PYTHONPATH=. python scripts/run_autoexp.py --config-name experiments/dllm/bd/eval_qwen3 --submit-and-exit \
    backend.env.EVAL_RUN=<run name> backend.env.EVAL_CKPT=<run dir>/checkpoints \
    backend.env.EVAL_OBJECTIVE=bd3 backend.env.EVAL_MODES=static,dynamic \
    backend.env.EVAL_BLOCK_SIZE=8 backend.env.EVAL_TOKEN_SHIFT=0
python -m oellm_autoexp.dllm.eval.collect_results --eval-dir $OUTPUT_DIR/dllm-ablations/eval \
    --logs-glob "$OUTPUT_DIR/dllm-ablations/runs/*/slurm-*.log" --results-md <file> --write
```

## Tests

```bash
pytest tests/unit/dllm tests/integration/test_dllm_config_feasibility.py
```

They cover the spec invariants on CPU: I1 masks against a loop reference, I2 equal N- and C-stream
outputs without masking, I3 bd3 with block size 1 and shift 1 equals AR, I4 noise, I5 sources,
I6 deterministic validation, I7 exact converter round trip at TP 1 and 2, I8 sampler with block
size 1 equals AR greedy decoding, I9 every flag of every dllm experiment is accepted by the
launcher's argparser, I10 the nld mask against a loop reference, I11 the nld clean stream equals
the AR model (and HF Qwen3 in the GPU parity check), and I12 the nld loss reduces to the bd3 loss
(AR weight 0) and to the AR loss (AR-only stage).

## Known limits (v1)

- Pipeline parallel 1, context parallel 1, no sequence parallelism (the source gather needs
  full-sequence hidden states; the BlockMask context relies on one micro-batch at a time).
- MegaDLMs vendors megatron-core 0.10; its checkpoints use the `torch` format.
- The sampler runs on HF transformers with dense 4D masks; it is not optimized for speed.

## Results

Qwen3-0.6B-Base, 2B tokens of FineWeb-Edu at L = 2048 and lr 1e-4 (tier 0 picked it from 1e-5 to
3e-4). M1 is the NELBO in bits/token on 1000 held-out windows (AR: exact NLL), M2 the GSM8K 4-shot
accuracy on 250 problems, M3 the tokens per forward pass at dynamic@0.9. The base model scores M1
3.809 and GSM8K 52.4 with AR greedy decoding.

| Arm | M1 | M2 static | M2 dyn@0.9 | M3 |
|---|---|---|---|---|
| R00 base: B 8, shift 0, ELBO, t in [0, 1], doc mask | 4.2275 | 10.4 | 10.0 | 1.51 |
| R01 / R02 / R03: B 4 / 16 / 32 | 4.0756 / 4.3313 / 4.4011 | 10.8 / 2.0 / 1.2 | 10.0 / 2.4 / 1.2 | 1.41 / 1.61 / 1.41 |
| R04 token shift 1 | 4.3997 | 8.8 | 10.4 | 1.48 |
| R05 complementary masks (FLOP-matched) | 4.2308 | 8.0 | 8.8 | 1.51 |
| R06 uniform weighting | 4.2035 | 8.0 | 7.6 | 1.55 |
| R07 t in [0.2, 0.8] | 4.2083 | 9.2 | 8.8 | 1.56 |
| R08 block-size schedule 0:4,0.15:8 | 4.2273 | 14.0 | 11.6 | 1.41 |
| R09 doc mask 0 | 4.2290 | 12.0 | 12.0 | 1.50 |
| R10 AR control | 3.7121 | 13.2 (AR greedy) | - | - |
| R11 nld hybrid, B 32 | 4.4906 (AR mode 3.7332) | 14.4 | 12.8 | 1.38 |

- M1 grows with the block size, and GSM8K collapses from B = 16 on; the selection rule keeps B = 8.
- No binary factor passes the rule (M1 at least 0.5 percent better, M2 at most 1.5 points worse).
  The block-size warmup (R08) and doc mask 0 (R09) tie on M1 but gain 3.6 and 1.6 points of M2.
- The AR control loses most of the base model's GSM8K too (52.4 to 13.2): the drop comes largely
  from continued pretraining on FineWeb-Edu at this LR, not only from the diffusion objective.
- The nld hybrid keeps an AR mode within 0.6 percent of the AR control and decodes GSM8K best
  (14.4 static, 12.8 at 1.38 tokens per forward), at +6.2 percent diffusion M1 against R00.

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
| `config/experiments/dllm/bd/` | conversion, smoke, probe, tier 0 and tier 1, and evaluation experiments |

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

`--bd-objective {bd3,ar}`, `--bd-block-size N`, `--bd-block-schedule "0:4,0.15:8"`,
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
(`eval/parity_check.py`) compares AR-mode logits with HF (top-1 >= 99 percent, mean KL <= 1e-3
nats) and checks invariant I3. The steps by hand:

```bash
python -m oellm_autoexp.dllm.convert.hf_qwen3_to_megadlm --hf Qwen/Qwen3-0.6B-Base \
    --template <template ckpt dir> --out $OUTPUT_DIR/dllm-ablations/converted/qwen3_0.6B_tp1
python -m oellm_autoexp.dllm.convert.megadlm_to_hf_qwen3 --megatron <ckpt dir> \
    --hf-base Qwen/Qwen3-0.6B-Base --out <hf dir> --block-size 8 --token-shift 0 --mask-token-id 151669
```

Training loads the converted checkpoint with `--pretrained-checkpoint` (finetune semantics, no
optimizer state) and resumes from its own `--save` directory afterwards. Jobs that load these
files set `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`.

## Training

```bash
PYTHONPATH=. python scripts/run_autoexp.py --config-name experiments/dllm/bd/smoke_qwen3_0.6b --dry-run
PYTHONPATH=. python scripts/run_autoexp.py --config-name experiments/dllm/bd/smoke_qwen3_0.6b --submit-and-exit
```

Size defaults (tensor parallel, micro batch, nodes): 0.6B (1, 4, 1), 1.7B (1, 2, 2), 4B (2, 2, 4),
8B (2, 1, 8), 14B (4, 1, 8). Global batch 256 sequences of 2048 tokens, WSD schedule (1 percent
warmup with at least 50 steps, 20 percent linear decay to 0.1 x LR), bf16, torch checkpoints.
A job resubmitted with the same config resumes from its last checkpoint.

| Experiment | Purpose |
|---|---|
| `probe_qwen3` | 20-iteration throughput probe; size, TP and micro batch as overrides |
| `tier0_qwen3_0.6B` | learning rates 1e-5, 3e-5, 1e-4 at 0.5B tokens (list sweep, `--array-subset i`) |
| `tier1_qwen3_0.6B` | R00-R11 at 2B tokens (one factor at a time, AR control R10, nld hybrid R11) |

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
`$OUTPUT_DIR/dllm-ablations/eval` as `<run>.m1.json` and `<run>.gsm8k_<mode>.json`:

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

See the ablation tables in the pull request description.

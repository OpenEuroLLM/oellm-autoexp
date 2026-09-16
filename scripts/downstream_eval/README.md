# Downstream evaluation of Megatron checkpoints

Convert Megatron `torch_dist` checkpoints to HF safetensors, then evaluate them with
[oellm-eval](https://github.com/OpenEuroLLM/oellm-eval), branch `feat/vllm-data-parallel`:
12 open-sci lm-eval tasks on the HF backend, and GSM8K / MATH500 / MBPP / HumanEval /
GPQA-Diamond through vLLM and Evalchemy. oellm-eval is a submodule of this repository, and the scripts refuse to
run unless it is clean and at the pinned revision.

Only code lives here. Exports, caches and results live under `WORK_ROOT`.

## Setup

oellm-eval is a submodule pinned on branch `feat/vllm-data-parallel`. Check it out and install it
(the scripts import it as a Python package); `EVAL_REPO=<path>` overrides it with your own checkout.

```bash
git submodule update --init submodules/oellm-eval
uv tool install -p 3.12 submodules/oellm-eval
```

You also need membership of the Slurm account you want to charge (`e-ext-2025e02-108` or
`e-sta-openeurollm`, chosen with `ACCOUNT=`).

## Usage

```bash
cd scripts/downstream_eval
MODEL=32b_dense ./eval_checkpoints.sh list               # runs, aliases, what is converted
MODEL=32b_dense ./eval_checkpoints.sh list v2           # iterations of one run
MODEL=32b_dense ./eval_checkpoints.sh prepare           # once: tokenizer + Megatron-Bridge copy
MODEL=32b_dense ./eval_checkpoints.sh convert v2:40000  # any iteration; also <run>:latest or an alias
MODEL=32b_dense ./eval_checkpoints.sh check v2_40k      # NLL ~1.3; ~12 means a broken conversion
MODEL=32b_dense ./eval_checkpoints.sh hf v2_40k         # renders; SUBMIT=1 to submit
MODEL=32b_dense ./eval_checkpoints.sh reasoning all     # renders only, never submits
MODEL=32b_dense ./eval_checkpoints.sh table             # both result tables
MODEL=32b_dense ./eval_checkpoints.sh plot              # both figures
```

Checkpoints are addressed as `<run>:<iteration>`; the export is named `<run>_<iteration/1000>k`
(`v2:40000` -> `v2_40k`). `models/<MODEL>.env` lists the runs, plus `ALIASES` for names that
predate this scheme (`control_64k` = `v1:64000`).

`TASKS="HumanEval GPQADiamond"` selects other reasoning tasks (default `gsm8k MATH500 mbpp`).
Run MBPP with `QUEUE_LIMIT=1`: its `code_eval` metric shares one temp file between concurrent tasks.

## Adding a model

Copy `models/9b_example.env`, set the architecture file, tokenizer, `RUNS` and
`EVAL_DP` (replicas per node: 4 for a 32B on GH200, 1 for a 9B). Nothing else changes —
prompt views, task lists, tables and plots are shared.

## Adding a site

Copy `sites/leonardo.env.example` to `sites/<site>.env` and fill in the container paths, account
and partition; `lib.sh` picks a profile from the hostname. Conversion needs containers built for
that architecture (the JUPITER images are aarch64). Everything after conversion goes through
oellm-eval, which already knows Leonardo, JURECA, JUWELS, LUMI, Snellius and UFAL
(`oellm/resources/clusters.yaml`), so only the image and the account differ.

## Files

| file | what |
|---|---|
| `eval_checkpoints.sh` | entry point, all commands |
| `lib.sh` | profile loading and shared paths |
| `sites/*.env`, `models/*.env` | site and model profiles |
| `convert.py`, `convert.sbatch`, `check.sbatch` | Megatron -> HF conversion and its sanity check |
| `hf_evals.sh` | 12 open-sci lm-eval tasks (HF backend) |
| `reasoning_evals.sh` | vLLM/Evalchemy renderer; validates the pinned revision, image and caches |
| `table_hf.py`, `plot_hf.py`, `table_reasoning.py`, `plot_reasoning.py` | tables and figures |
| `table_plot.py` | shared table drawing (ties within 1 se are marked as ties) |

## Pitfalls these scripts already handle

- **Chat template.** A Qwen3 `chat_template.jinja` in an export makes the evals apply a chat
  protocol the base model never saw; conversion renames it.
- **HF caches.** Someone else's cache is not writable for you (`.lock` files), so each `WORK_ROOT`
  has its own `hf_home` and `hf_home_reasoning`. The GPQA entry under `hub/` must be a real copy,
  not a symlink into another cache: check with `find $HF_HOME -type l`.
- **fp32 loading.** lm-eval 0.4.10 passes `dtype=`, which transformers 4.53 ignores, so the HF
  launcher is patched to `torch_dtype=bfloat16` (a 32B in fp32 does not fit).
- **MBPP races.** Concurrent tasks share one `code_eval` temp file; run MBPP with `QUEUE_LIMIT=1`
  unless the launcher gives each array task its own `HF_METRICS_CACHE`.
- **`<bos>`.** OpenEuroLLM training data has no `<bos>`, but the tokenizer adds one to every eval
  prompt. Measured on the 32B: few-shot and multiple-choice scores move by less than a point,
  while 0-shot MATH500 and HumanEval swing by up to 12, so compare like with like.

## Relation to the 32B results

`$WORK_ROOT/RESULTS.md` (for the 32B: `/e/scratch/e-sta-openeurollm/production_training/downstream_eval_32b`)
holds the findings and the setup notes, including the `<bos>` A/B. The numbered scripts there
(`0_prepare.sh` … `7_bos_ab.py`) are the originals these were generalized from; they stay as the
record of how those results were produced.

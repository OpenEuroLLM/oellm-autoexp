# Standalone B1 annealing on JUPITER

For MareNostrum 5, use the [BSC launcher, mixture recipes and runbook](bsc/README.md).

This package prepares separate B1 32B annealing experiments from a retained native tensorwise checkpoint, using either the Prelude `multilingual1+dominant1` or `multilingual1+dominant2` blend. TP4/PP4, sequence length 4,096, GBS 4,096, FP8 tensorwise with BF16 edge layers, Adam/RNG restoration and sample-based WSD `minus_sqrt` follow B1 production. The new data phase starts at the fork and consumes its mixture from offset zero. The executable preparer supports B1 32B. The original 9B linear-WSD examples need a separate JUPITER port and validation. [Original configs, mixture inputs and provenance](origins/README.md)

## Start from a fresh JUPITER login

You need GitHub read access to `OpenEuroLLM/oellm-autoexp` and its Megatron fork, JUPITER Container Runtime Engine access, a Slurm Booster allocation account and storage visible from login and compute nodes. Your account also needs read access to the B1 checkpoint, pinned training image, tokenizer and both frozen blends. The example below uses the B1 project `e-ext-2025e02-108`. Set `ACCOUNT` to an account you can charge and `RUN_PARENT` to your own writable scratch directory. Project storage and Slurm account names can differ. [JUPITER access](https://apps.fz-juelich.de/jsc/hps/jupiter/access.html) · [Container runtime access](https://apps.fz-juelich.de/jsc/hps/jupiter/container-runtime.html)

Run the walkthrough inside a named tmux session. Clone the branch into shared scratch and initialize the Megatron revision recorded by that branch.

```bash
tmux new -s b1-annealing
```

```bash
export ACCOUNT=e-ext-2025e02-108
export RUN_PARENT="/e/fscratch/e-ext-2025e02-108/$USER/annealing"
mkdir -p "$RUN_PARENT"
test -w "$RUN_PARENT"
df -h "$RUN_PARENT"
df -i "$RUN_PARENT"
sacctmgr -nP show assoc where user="$USER" account="$ACCOUNT" format=Account,Partition,QOS
command -v git apptainer sbatch squeue sacct flock

git clone --branch prod/oellm_32b_dense https://github.com/OpenEuroLLM/oellm-autoexp.git \
  "$RUN_PARENT/oellm-autoexp"
cd "$RUN_PARENT/oellm-autoexp"
git submodule update --init submodules/Megatron-LM
git rev-parse HEAD
git submodule status submodules/Megatron-LM
```

Create and activate your own Python 3.12 environment on JUPITER. If you are starting without Python or a package manager, use the repository's Micromamba bootstrap and the environment creation and activation commands. Set its example scratch paths to your own and use the install commands below in the B1 checkout. The linked walkthrough's Qwen-specific image and model setup apply to its own example. [Micromamba bootstrap](../../JUPITER_SETUP_GUIDE.md#1-micromamba) · [Environment creation and activation](../../JUPITER_SETUP_GUIDE.md#3-env)

```bash
cd "$RUN_PARENT/oellm-autoexp"
test -n "${CONDA_PREFIX:-${VIRTUAL_ENV:-}}"
test "$(command -v python)" = "${CONDA_PREFIX:-${VIRTUAL_ENV:-}}/bin/python"
python -m pip install -e .
python -m pip check
python -c 'import compoconf, oellm_autoexp, omegaconf'
```

The host environment runs the preparer and Slurm monitor. The pinned training image supplies the GPU runtime. These are the actual JUPITER inputs used for the validated checkpoint and both mixtures. Check readability as your own account before submitting anything. `prepare.py` checks the checkpoint's indexed shards and all 294 `.bin`/`.idx` pairs for the selected mixture.

```bash
export SEED=/e/scratch/e-ext-2025e02-108/oellm_32b_dense_b1/b1_conservative60k_100k_tensorwise/training_ckpts/checkpoints/iter_0416000
export TRAIN_IMAGE=/e/project1/e-sta-openeurollm/container/MegatronTraining-JUPITER-te218-fa3_aarch64_202608280932.sif
export TOKENIZER=/e/data1/datasets/playground/mmlaion/oellm/oellm_tokenizer_256k
test -r "$SEED/.metadata"
test -r "$TRAIN_IMAGE"
test -r "$TOKENIZER/tokenizer.model"
```

`TRAIN_IMAGE` and `TOKENIZER` above are access checks for the inputs already pinned by the resolved config. Check the generated batch scripts against those paths. The dataset paths live in the [dominant1](datamix/dominant1.txt) and [dominant2](datamix/dominant2.txt) files. Allow roughly 71 GiB per smoke cache based on the measured caches, plus space for logs, and budget separately for indices and checkpoints at your full run size. Check your project quota as well as filesystem free space and inodes. Keep the linked seed available until your arm has a validated replacement checkpoint.

## Validate both mixtures with short runs

The coordinator below reproduces the published smoke shape for iteration 416000. Each arm restores the optimizer and RNG, resets the validation cursor for the new blend, and takes three updates on 32 nodes in one ten-minute allocation, with checkpoint writes and retries disabled. Cache construction is a separate job and must pass before training. The two preparations reserve up to 32 GPU-hours total, and the two smokes up to 42.67 GPU-hours total.

```bash
CAMPAIGN=$(mktemp -d "$RUN_PARENT/b1-smoke.XXXXXX")
ARM1="$CAMPAIGN/smoke-dominant1"
ARM2="$CAMPAIGN/smoke-dominant2"
STATE="$CAMPAIGN/validation"

for mix in dominant1 dominant2; do
  arm="$CAMPAIGN/smoke-$mix"
  python annealing/b1_prelude/prepare.py --root "$arm" --seed "$SEED" \
    --mix "$mix" --tokens 20000000000 --nodes 32 --smoke --account "$ACCOUNT"
  python annealing/b1_prelude/cache.py prepare --root "$arm"
done
```

Review `recipe.json`, `config/annealing_arm.yaml`, `training_core/job_*.sbatch`, and the `settings.json`, `render.json` and batch script in each arm's `cache-preparation/`. Both `prepare` commands render files without submitting. After reviewing the resolved shape and allocation, submit the preparations and start the cache-gated coordinator in the same tmux session.

```bash
for arm in "$ARM1" "$ARM2"; do
  sbatch --test-only "$arm"/cache-preparation/job_*.sbatch &&
    python annealing/b1_prelude/cache.py submit --root "$arm" || break
done
python annealing/b1_prelude/validate.py init --root "$STATE" --arm "$ARM1" --arm "$ARM2" &&
  python annealing/b1_prelude/validate.py watch --root "$STATE" --execute &&
  python annealing/b1_prelude/validate.py verify --root "$STATE"
```

The coordinator waits for the two cache jobs and checks each arm's cache before submitting its smoke once. Inspect `status.json`, the per-arm step CSVs and `all_passed.json` under `$STATE`. After a disconnect, reconnect with `tmux attach -t b1-annealing`. If the watcher exited, use the same repository, environment and state path to restart `watch`. A `STOP` file in the validation state directory holds new admissions. A passing smoke requires the expected checkpoint restore and cursor reset, three consecutive finite loss and gradient updates with positive LR and throughput, zero skipped/NaN updates and a Slurm `COMPLETED` job with exit code zero. It records the consecutive loss changes. [November 20 validation result](validation_result.json)

## Prepare and launch a full arm

Use a new run root and size `--tokens`, `--nodes`, `--attempts` and `--walltime` to your allocation. This example renders a dominant1 run with a 20B target. At 512 nodes each allocation uses 2,048 GPUs, so four 12-hour allocations cap this example at 98,304 reserved GPU-hours. Check the live partition, QoS and project budget before choosing these parameters.

```bash
FULL_CAMPAIGN=$(mktemp -d "$RUN_PARENT/b1-full.XXXXXX")
RUN="$FULL_CAMPAIGN/dominant1-20b"
python annealing/b1_prelude/prepare.py --root "$RUN" --seed "$SEED" \
  --mix dominant1 --tokens 20000000000 --nodes 512 --attempts 4 \
  --walltime 12:00:00 --account "$ACCOUNT"
python annealing/b1_prelude/cache.py prepare --root "$RUN"
```

| Token budget | Updates | Actual tokens |
|---|---:|---:|
|20B|1,193|20,015,218,688|
|50B|2,981|50,012,880,896|
|100B|5,961|100,008,984,576|
|300B|17,882|300,010,176,512|
|2T|119,210|2,000,011,919,360|

The endpoint is seed iteration plus `ceil(tokens / 16,777,216)`. The reserved upper bound is GPUs × walltime × maximum allocations. The prior 2T plan used up to 32 allocations of 12 hours, with roughly 13 allocations forecast from historical speed. Each full arm retains its own checkpoints, logs and monitor state. The supplied smoke covers startup. Interrupted full-state resume and terminal rolling cleanup require their own validation before running at scale.

Review the rendered training and preparation files as for the smokes. Build the full arm's indices with its own resolved budget, then check the completed allocation and cache before using `launch.sh` in the named tmux session.

```bash
sbatch --test-only "$RUN"/cache-preparation/job_*.sbatch &&
  python annealing/b1_prelude/cache.py submit --root "$RUN"
CACHE_JOB=$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["job"])' \
  "$RUN/cache-preparation/submitted.json")
sacct -X -j "$CACHE_JOB" --format=JobID,State,ExitCode
```

Once the cache job reports `COMPLETED` with exit code `0:0`, check the cache and start training.

```bash
python annealing/b1_prelude/cache.py check --root "$RUN" &&
  "$RUN/launch.sh"
```

The initial launch checks the cache again before submitting. Full runs save rolling checkpoints every 125 updates and use the bounded autoexp continuation policy. Retain `$RUN/monitor` and restore its existing session if the monitor exits. Keep each arm's optimizer, RNG, data cursor and schedule state together on continuation. A `STOP` file in `$RUN` prevents new admissions. [Autoexp monitor restoration](../../README.md#monitoring-sessions)

The prebuild uses the production `data_cache_build_fscratch_trainshape.yaml` recipe with one rank, a two-layer model, the actual tokenizer and data builder, one update and no checkpoint writes. Each preparation can reserve up to 16 GPU-hours. The gate checks the exact data phase and validation sample counts, successful Slurm completion, the production header/array-size check, and ordered blend and child coverage. `--cache` can point to a known cache from an earlier arm. To reuse a completed preparation, render the new arm and preparation with the same cache path, then run `cache.py reuse --root "$RUN" --from-root "$PRIOR"` with `PRIOR` set to the original arm directory containing its successful preparation receipt. The gate verifies both sets of inputs. A smoke cache is only reusable for a matching target shape. The shared writer lease covers cache preparation and training for each cache. If a writer was interrupted, quarantine its partial groups after writers are quiescent and keep the valid groups.

## Set up downstream evaluation separately

Training and its in-run validation use the Megatron image. For benchmark scoring, follow the `oellm-eval` JUPITER guide in a separate environment. It starts with a fresh clone and covers the CLI, pinned ARM64 vLLM image, offline benchmark assets and a GPU acceptance run. [Install and run oellm-eval on JUPITER from scratch](https://github.com/OpenEuroLLM/oellm-eval/blob/feat/vllm-data-parallel/docs/machines/jupiter/from-scratch.md)

Export the native `torch_dist` checkpoint to Hugging Face format using the B1 model configuration and tokenizer. When following the downstream conversion guide, register your annealing arm's checkpoint directory, set `WORK_ROOT` to your own writable conversion workspace, point `MEGATRON_REPO` at the checkout used for training and use the arm's resolved training config as `ARCH_YAML`. Use the exported model with the recipe and evaluator revision pinned by the benchmark campaign you want to reproduce. [Checkpoint conversion and model registration](../../scripts/downstream_eval/README.md#adding-a-model) · [Evaluating a custom model](https://github.com/OpenEuroLLM/oellm-eval/blob/feat/vllm-data-parallel/docs/machines/jupiter/from-scratch.md#12-evaluate-other-models-or-benchmark-groups)

## Reproduce source mixtures and inspect provenance

```bash
REBUILT=$(mktemp -d "$RUN_PARENT/b1-mixtures.XXXXXX")
for mix in dominant1 dominant2; do
  python annealing/b1_prelude/datamix/rebuild.py --mix "$mix" --output "$REBUILT/$mix.txt"
  cmp "$REBUILT/$mix.txt" "annealing/b1_prelude/datamix/$mix.txt"
done
python -m unittest discover -s annealing/b1_prelude -p 'test_*.py' -v
```

Both blends retain the original weights and prefix order, including the original AgentTrove prefix. At the November 17 LUMI/JUPITER byte-comparison snapshot, 431 of 588 files matched, with zero mismatches and 157 files still pending (39.194 TB). [Verification continuation and receipts](origins/README.md)

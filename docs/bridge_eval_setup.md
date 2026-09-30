# Megatron-Bridge + oellm-evals chain setup

This guide covers the cluster-side prerequisites for running the
`MegatronBridgeBackend` (Megatron → HF checkpoint conversion) and
`OELLMEvalBackend` (lm-eval-harness evaluation by way of `oellm-eval schedule
--local`) as chained stages after a Megatron training run. The reference
chain experiment is
`config/experiments/korbi/chain_qwen3_bridge_train_eval_<cluster>.yaml`.

## Mental model

The chain has three SLURM jobs gated by `FileExistsCondition`:

1. **train** — standard `MegatronBackend`, writes `iter_NNNNNNN/`
   (torch_dist) + `latest_checkpointed_iteration.txt`.
2. **convert** — `MegatronBridgeBackend`, runs
   `python -m oellm_autoexp.backends.megatron_bridge.run_export …`
   inside the training container, writes `hf/iter_NNNNNNN/model.safetensors`.
3. **eval** — `OELLMEvalBackend` with `local: true`, runs
   `oellm-eval schedule --local true …` which calls `lm_eval` per task and
   drops `results/<hash>_<ts>.json` files.

The convert stage **does not** require a separate container — it reuses
the training container, with two cluster-agnostic shims applied:

- `container/megatron/patch_bridge_lazy_imports.py` is run once against
  `submodules/Megatron-Bridge/src/megatron/bridge/models` so missing model
  bridges (mamba, certain VL/Omni variants) fail soft at import time.
- `oellm_autoexp/backends/megatron_bridge/run_export.py` monkey-patches
  `nvidia_resiliency_ext.__version__` (≥ 0.6.0), shims
  `_clean_metadata_for_serialization`, and registers a placeholder
  `LayerWiseDistributedOptimizer` before `from megatron.bridge import …`.
  This lets Bridge import against the training container's vendored
  Megatron-LM even when the host's OpenEuroLLM fork is older.

## Common prerequisites (all clusters)

```bash
# Clone with submodules (and force HTTPS so private mirrors don't matter)
git -c url.https://github.com/.insteadOf=git@github.com: \
    clone https://github.com/OpenEuroLLM/oellm-autoexp.git \
    -b oellm_evals_integration --recurse-submodules

cd oellm-autoexp

# One command for the rest: builds the Python env, patches
# Megatron-Bridge's tolerant-imports, downloads the Qwen3 tokenizer
# heavy files, and (with --prefetch) populates the HF dataset cache.
bash scripts/install_eval_env.sh --prefetch
```

The installer auto-detects the cluster from hostname by way of
`scripts/detect_cluster.py`; pass `--cluster NAME` to override.
Dependencies for the eval env are declared upstream in
`submodules/oellm_evals/pyproject.toml` under the `[eval]` and
`[eval-base]` extras (single source of truth, see
[`submodules/oellm_evals/docs/VENV.md`](../submodules/oellm_evals/docs/VENV.md)):

- **`[eval]`** — full venv install: `lm_eval[hf,vllm,api,tasks]>=0.4.12`
  plus `datasets>=4.0`. Pulls torch + transformers + accelerate + peft
  (`[hf]`), PyPI vLLM (`[vllm]`, CUDA-only), HTTP API backends (`[api]`),
  and the full task aggregate (`[tasks]` → ifeval/math/multilingual/…).
- **`[eval-base]`** — container-friendly subset: `lm_eval[api,tasks]` +
  `datasets>=4.0`. Drops `[hf]` and `[vllm]` so the container's
  pre-built torch (and on Lumi the custom ROCm-vllm shipped in
  `laif-rocm-…sif`) aren't replaced by PyPI wheels.

The sections below document what `install_eval_env.sh` does per cluster
so you can audit the install or run the pieces manually if something
goes sideways.

## Eval environment

`oellm-eval schedule --local true` shells out to `python -m lm_eval`, so
the eval stage needs an environment with:

- `oellm` CLI on `$PATH`
- `lm_eval >= 0.4.12` + its deps (`datasets >= 4.0`, `transformers`,
  `accelerate`, `scipy`, `threadpoolctl`, `scikit-learn`, `chardet`,
  `pytz`, `tabulate`, `colorama`, …)
- a working `torch` build (CUDA on NV, ROCm on AMD)

Two ways to provide that:

- **Venv** (juwels, jupiter): `uv venv --python 3.12 <path>` + `pip install`,
  point `OELLM_EVAL_VENV` at it. The eval slurm launcher (`*_venv.yaml`)
  sources `<venv>/bin/activate` and runs `oellm` from there.
- **Container** (leonardo, lumi): bake or extend an eval container that
  already ships `lm_eval`. Install `oellm` by way of `pip install --user -e
  submodules/oellm_evals` from inside the container so its `entry_points`
  console script lands in `~/.local/bin`. Add `~/.local/bin` to PATH by way of
  `--env PATH=…` in the eval slurm launcher (`*_eval.yaml`). The user-site
  `.pth` files are editable installs pointing inside the container, so
  the eval launcher must bind the host repository at the same path used at
  install time (`/workspace/oellm-autoexp` is the convention).

### Pre-fetching datasets

Compute nodes on all three target clusters have no internet, so
`load_dataset` calls must hit a populated cache. The shipped helper does
this from a login node:

```bash
# Run inside whatever env will be used at eval time so the
# datasets-library version matches.
python scripts/prefetch_datasets.py open-sci-0.01 \
    submodules/oellm_evals/oellm/resources/task-groups.yaml
```

Two important details:

- `HF_HUB_OFFLINE` / `HF_DATASETS_OFFLINE` must be unset (or `=0`) during
  the prefetch; the script clears them, but the login shell's bashrc
  often sets them.
- oellm-evals's generated eval script unconditionally exports
  `HF_DATASETS_CACHE=$HF_HOME/datasets`. Pick `HF_HOME` so that
  `$HF_HOME/datasets/<owner>___<repository>/…` lines up with the populated
  legacy cache layout that `lm_eval` reads from. On Leonardo this means
  pointing `backend.env.HF_HOME` at the dotted `.cache/huggingface`
  directory; see `chain_qwen3_bridge_train_eval_leonardo.yaml`.

`scripts/prefetch_datasets.py` will fail on tasks whose `task-groups.yaml`
entry is missing a `subset` (notably `cais/mmlu`). The eval still runs
for everything else; mmlu just gets a `Couldn't reach …` error in its
per-task log.

## Per-cluster recipes

### Juwels Booster

```bash
# Eval venv (oellm + lm-eval + deps)
module load Stages/2025  # gives Python 3.12
uv venv --python 3.12 ~/work/eval_venv
PYTHONPATH= ~/work/eval_venv/bin/pip install \
    torch transformers "lm_eval>=0.4.12" \
    -e ~/work/Projects/oellm-autoexp/submodules/oellm_evals \
    -e ~/work/Projects/oellm-autoexp \
    "compoconf==0.1.14" \
    scipy threadpoolctl scikit-learn chardet pytz tabulate colorama \
    accelerate jinja2 more_itertools pandas rich pyyaml jsonargparse
```

The chain config uses `slurm: juwels_venv` + `container: juwels_venv`,
which `unset PYTHONPATH` (the Stages-2025 modules leak Python 3.13
site-packages into PATH otherwise) and sources the venv.

Required env at submit time: `SLURM_PARTITION_DEBUG=develbooster`,
`SLURM_ACCOUNT=cstdl`, `OUTPUT_DIR`, `HF_HOME`, `CONTAINER_CACHE_DIR`.

### Leonardo

```bash
# 1) Install oellm-evals + datasets/lm-eval into the eval container's user-site
singularity exec --bind /leonardo_scratch --bind /leonardo --bind /leonardo_work \
    --env HF_HUB_OFFLINE=0 \
    /leonardo_work/OELLM_prod2026/container_images/eval_env-leonardo.sif \
    bash -c "pip install --user --no-cache-dir \
        -e /leonardo/home/$USER/work/Projects/oellm-autoexp/submodules/oellm_evals \
        --upgrade 'datasets>=4.0' 'lm_eval>=0.4.12'"

# 2) Pre-fetch datasets inside the eval container (legacy cache layout)
singularity exec --bind /leonardo_scratch --bind /leonardo --bind /leonardo_work \
    --env HF_HUB_OFFLINE=0 --env HF_DATASETS_OFFLINE=0 \
    --env HF_HOME=/leonardo_scratch/fast/<EUHPC>/cache/huggingface \
    /leonardo_work/OELLM_prod2026/container_images/eval_env-leonardo.sif \
    python3 scripts/prefetch_datasets.py

# 3) Stub a venv directory so oellm-evals's `source $VENV/bin/activate` is a no-op
#    (container is already the activated env).
mkdir -p ~/eval_venv_stub/bin && touch ~/eval_venv_stub/bin/activate
```

The chain config uses `slurm: leonardo_eval` + `container: leonardo_eval`
which forces `PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin` (so the
ronlr venv on the host PATH doesn't leak in) and binds
`$HOME/work/Projects/oellm-autoexp:/workspace/oellm-autoexp` (so the
`pip install --user -e` `.pth` files resolve).

Convert stage uses `slurm: leonardo_bridge` and sets
`backend.env.PYTHONNOUSERSITE=1` to keep the host's `~/.local`
(broken `antlr4` from the lighteval install) out of the convert
container.

### Lumi

Lumi follows the same pattern as Leonardo but the container is
`laif-rocm-…` (ROCm pytorch) and the user-site install goes to
`$HOME/eval_local` by way of `pip install --target` (Lumi's container has its
own `/opt/venv` so `--user` is rejected; `--target` plus a `PYTHONPATH`
override in `slurm/lumi_eval.yaml` is the workaround).

```bash
mkdir -p ~/eval_local/lib/python3.12/site-packages
singularity exec --bind $HOME --bind /pfs \
    /scratch/project_462000963/containers/laif-rocm-….sif \
    bash -c "pip install --no-cache-dir --upgrade \
        --target \$HOME/eval_local/lib/python3.12/site-packages \
        -e /scratch/project_462000963/user/$USER/Projects/oellm-autoexp/submodules/oellm_evals \
        -e /scratch/project_462000963/user/$USER/Projects/oellm-autoexp \
        compoconf==0.1.14 'lm_eval>=0.4.12' 'datasets>=4.0' \
        jsonargparse rich pandas pyyaml"
```

Status (as of writing): Lumi `sbatch` returns `AssocMaxSubmitJobLimit`
for `project_462000963` because the project allocation has been
exhausted. The chain config submits cleanly through `--dry-run`; an
allocation refresh is required before the chain can run.

## Submitting

```bash
export OUTPUT_DIR=…
export HF_HOME=…
export OELLM_EVAL_VENV=…           # venv-mode clusters only
export CONTAINER_CACHE_DIR=…
export SLURM_PARTITION_DEBUG=…     # use the debug partition for fast turnaround
export SLURM_ACCOUNT=…

PYTHONPATH=. python scripts/run_autoexp.py \
    --config-name experiments/korbi/chain_qwen3_bridge_train_eval_<cluster>
```

Without `--no-monitor` the orchestrator submits the train stage, polls
for the checkpoint, then submits convert and eval as their start
conditions trigger. With `--submit-and-exit` only the train stage is
submitted — convert/eval sbatch scripts are still rendered, but you have
to `sbatch` them yourself once the train checkpoint lands.

## Where things land

```
$OUTPUT_DIR/chain_qwen3_bridge_train_eval_<cluster>_0/   # train
    iter_0000050/                                         # Megatron ckpt
    latest_checkpointed_iteration.txt                     # gate for convert
    hf/iter_0000050/                                      # HF safetensors (written by convert)

$OUTPUT_DIR/chain_qwen3_bridge_train_eval_<cluster>_1/   # convert (sbatch + log)
$OUTPUT_DIR/chain_qwen3_bridge_train_eval_<cluster>_2/   # eval
    eval/<timestamp>/results/<hash>_<ts>.json             # per-task lm-eval output
```

## Standalone batch conversion + HF Hub upload

The chain above assumes convert is one stage of an orchestrated
train→convert→eval pipeline. For publishing checkpoints from an
**already-running** training job (no orchestrator chain, no eval), use
`scripts/mass_convert_checkpoints.py` + `scripts/upload_to_hf.py` instead.
Built for the OpenEuroLLM Prelude 9B ("baby") release, but cluster/model
agnostic.

### What's new here versus the chain's `run_export`

- **`--max-shard-size`** on `run_export.py` / `create_dummy_model.py`
  (default `5GB`). The real export mirrors the dummy reference model's
  shard layout, so this must be set on the dummy model's `save_pretrained`
  call, not just the real one; that's why it threads through both files.
- **`validate_export.py`**: loads the converted HF checkpoint and runs a
  canonical-prompt generation battery (multilingual, EU languages) as a
  post-conversion sanity check, writing a `validation.json` next to the
  checkpoint.
- **`convert_and_validate_task.py`**: single-checkpoint Slurm task entry
  point. Reads one entry from a JSON manifest (by `--task-index` or
  `$SLURM_PROCID`), runs convert then validate in-process, skips convert if
  the output dir already exists.
- **`scripts/mass_convert_checkpoints.py`**: discovers `iter_*` dirs across
  one or more `--checkpoints-dir`, groups the ones not yet converted into
  Slurm jobs (`--group-size` checkpoints per job, one task per checkpoint),
  and submits them respecting a QOS's per-user job cap. Idempotent: re-run
  any time to pick up new checkpoints; already-converted ones are skipped.
  `--singularity-bind` (repeatable) is required and cluster-specific; there's
<!-- google-doc-style-ignore -->
  no default. Pass `--hf-repo-id` when multiple operators each convert into
  their own `--output-dir` against the same run (a scratch dir generally
  can't be shared across HPC accounts): a checkpoint already published as a
  complete branch on that repo is skipped too, even if this operator's own
  `--output-dir` has never seen it, so nobody redundantly reconverts what
  someone else already finished.
<!-- google-doc-style-resume -->
- **`scripts/upload_to_hf.py`**: pushes each converted `<output-dir>/<name>/`
  to its own branch (`iter_NNNNNNN`, or `<label>_iter_NNNNNNN` when the
  converter ran with `--run-label`) of a target HF Hub repository, by way of
  `upload_large_folder` (resumable). A branch only counts as complete if it
  actually has the weight files, `config.json`, and `validation.json`, not
  just whether the branch ref exists, so a crash or network blip mid-upload
  gets retried automatically rather than silently treated as done.

### Prerequisites beyond the common ones above

- **Submodule over HTTPS**: if the account cloning `oellm-autoexp` has no
  GitHub SSH key registered (common for a shared HPC project account), the
  `Megatron-Bridge` submodule clone fails with `Permission denied
  (publickey)`. Fix before `git submodule update`:

  ```bash
  git config submodule.submodules/Megatron-Bridge.url \
      https://github.com/OpenEuroLLM/Megatron-Bridge.git
  git submodule update --init submodules/Megatron-Bridge
  ```

- **Tokenizer**: `run_export.py` needs a working `tokenizer.json` for the
  target tokenizer. If `transformers` falls back to a TikToken parser on a
  raw SentencePiece `.model` file (missing `sentencepiece` in the
  container), the repository's own
  `oellm_autoexp/postprocess/resources/megatron_bridge/tokenizers/…/tokenizer.json`
  should already cover this for the standard OpenEuroLLM tokenizers; verify
  it's present and loads before assuming a manual copy is needed.
- **Login-node Python**: the discovery/submission steps in
  `mass_convert_checkpoints.py` need Python ≥ 3.7 (`from __future__ import
  annotations`, f-strings); Leonardo's login-node default `python3` is 3.6.
  Use `python3.11` (`module load python/3.11.7`) directly.
<!-- google-doc-style-ignore -->
  Without `--hf-repo-id`, no container is needed for this step, since it
  only touches the local file system and `subprocess`. `upload_to_hf.py`,
  and `mass_convert_checkpoints.py` whenever `--hf-repo-id` is passed, need
  `huggingface_hub` and must run inside the training container
  (`singularity exec … python3 scripts/mass_convert_checkpoints.py …`).
<!-- google-doc-style-resume -->
- **HF Hub token**: `upload_to_hf.py --token-file` defaults to
  `~/.cache/huggingface/token`; make sure it holds a token with write
<!-- google-doc-style-ignore -->
  access to the target `--repo-id`.
<!-- google-doc-style-resume -->

### Example: full backfill, then incremental catch-up

<!-- google-doc-style-ignore -->
```bash
# One-off backfill of everything already on disk, batched for throughput
# (2-job QOS cap x group-size 16 = 32 checkpoints converting in parallel).
# --singularity-bind has no default: list whatever this cluster's containers
# need bound; --hf-repo-id is optional, only needed when more than one
# operator converts into their own separate --output-dir for the same run.
python3.11 scripts/mass_convert_checkpoints.py \
    --checkpoints-dir /leonardo_scratch/fast/<project>/production_training/<run>/checkpoints \
    --training-config /leonardo_scratch/fast/<project>/production_training/<run>/logs/current.yaml \
    --output-dir /leonardo_scratch/large/userexternal/$USER/<run>-ckpts \
    --container-image /leonardo_work/<project>/container_images/<training>.sif \
    --singularity-bind /leonardo_scratch --singularity-bind /leonardo --singularity-bind /leonardo_work/ \
    --hf-repo-id <org>/<repo> \
    --account <SLURM_ACCOUNT> --qos boost_qos_dbg --max-concurrent-jobs 2 --group-size 16

# Push everything converted so far to the Hub, one branch per iteration:
singularity exec --bind /leonardo_scratch --bind /leonardo --bind /leonardo_work \
    /leonardo_work/<project>/container_images/<training>.sif \
    python3 scripts/upload_to_hf.py \
    --output-dir /leonardo_scratch/large/userexternal/$USER/<run>-ckpts \
    --repo-id <org>/<repo>

# Ongoing: re-invoke with --group-size 1 as new checkpoints appear, so each
# gets its own Slurm job. That lets a caller chain a single-checkpoint
# upload job onto each conversion job by way of `sbatch --dependency=afterok:<jobid>`
# instead of polling. See the `checkpoint_watcher` tool in oellm-monitoring
# for a periodic wrapper that does exactly this end to end.
```
<!-- google-doc-style-resume -->


### Several training runs in one repository (`--run-label`)

A branch name is the output directory name, and by default that is the
Megatron iteration directory (`iter_0086000`). That is unambiguous for a
single run, but not when several runs publish into one repository: two runs
that reach the same iteration produce the same name, and the second one is
silently dropped during discovery rather than reported as a conflict.

Pass `--run-label <label>` to `mass_convert_checkpoints.py` and the outputs
(and therefore branches) become `<label>_iter_NNNNNNN`. Invoke the tool once
per run, each with its own label and its own
`--checkpoints-dir`/`--training-config`. `upload_to_hf.py` discovers both the
bare and the labelled form; pass `--run-label` there to restrict a pass to a
single run.

The Prelude 9B repository already uses this shape by convention
(`iter_0953312` for the base run, `anneal300b_iter_0989075` for the anneal
variant); `--run-label` makes it a first-class option instead of something
assembled by hand.

### Example: JUPITER, the 32B dense runs

The 32B runs publish to one repository with every run labelled, so no run is
privileged and a branch always states where it came from:

| label | run | checkpoints |
|---|---|---|
| `v1` | `oellm_32b_dense_prod_dataopt5_gbs4096_lr3e-4` | 23 |
| `v2` | `oellm_32b_v2_prod_gbs4096_lr1.76e-4` | 26 |
| `v2zloss` | `oellm_32b_v2_zloss_unfused_i60k_gbs4096_lr1.76e-4` | 18 |
| `fork` | `oellm_32b_dense_prod_dataopt5_deferoff_3e-4_i64k_seed1234` | 21 |
| `b1` | `B1-reborn-production` (jitsev1's reanimation from 64k) | 89 |

What differs from the Leonardo example:

- **Checkpoints and training configs live on different file systems.**
  `save:` writes to `$SCRATCH`, while `base_output_dir` (and therefore
  `logs/current.yaml`) is on `$PROJECT`. The two paths do not share a prefix,
  so they have to be given separately.
- **`singularity` is Apptainer** under a compatibility name, so the submitted
  command works unchanged.
- **`--gpus-per-task` is rejected** by this Slurm (`Invalid GRES
  specification`), so pass `--gpu-binding gres` to request
  `--gres=gpu:<tasks per node>` instead.
- **One bind is enough**: `--singularity-bind /e` covers home, project,
  scratch and fscratch.
- **The QOS cap is much higher.** `normal` allows 512 jobs per user with a
  12 h wall limit, against Leonardo's 2-job debug QOS, so throughput is set by
  how much of the machine is reasonable to occupy rather than by the cap.
- **A 32B needs more than the 30 min default**; allow about 1 h per checkpoint,
  and prefer a tight limit: short jobs backfill far better on a machine that
  usually has several thousand nodes allocated.
- **Test rounds belong in the development reservation.** `--reservation
  develbooster` (8 nodes, `jpbo-101-[01-08]`) starts in seconds instead of
  queueing behind production work. Production passes leave it off.
- **The tokenizer needs a `tokenizer.json` before the first conversion.**
  `openeurollm/tokenizer-256k` publishes only a SentencePiece `tokenizer.model`,
  and without `sentencepiece` in the image `transformers` falls back to the
  TikToken parser and fails with `Error parsing line b'\x0e'`. Generate it once
  on a login node, then restore the tracked metadata, which
  `save_pretrained` rewrites in passing:

  ```bash
  uv run --with transformers --with sentencepiece --with protobuf python3 -c \
    "from transformers import AutoTokenizer as T; p='oellm_autoexp/postprocess/resources/megatron_bridge/tokenizers/openeurollm/tokenizer-256k'; T.from_pretrained(p).save_pretrained(p)"
  git checkout -- oellm_autoexp/postprocess/resources/megatron_bridge/tokenizers/openeurollm/tokenizer-256k
  ```

  The result is byte-identical to the `tokenizer.json` published with the 9B.
- **No outgoing SSH.** Clone over HTTPS with a personal access token, and
  rewrite any `ssh://` submodule URL (see the submodule note above) before
  `git submodule update`.

<!-- google-doc-style-ignore -->
```bash
# One run. Repeat per label; each writes into the same output tree.
python3 scripts/mass_convert_checkpoints.py \
    --run-label b1 \
    --checkpoints-dir /e/fscratch/e-sta-openeurollm/jj1/reanimation_64k_20260911/B1-reborn-production/training_ckpts/checkpoints \
    --training-config /e/fscratch/e-sta-openeurollm/jj1/reanimation_64k_20260911/B1-reborn-production/config/autoexp.yaml \
    --output-dir /e/project1/e-sta-openeurollm/$USER/oellm-32b-ckpts \
    --container-image /e/project1/e-sta-openeurollm/container/MegatronTraining-JUPITER-bridge-bridge_aarch64_202605191331.sif \
    --singularity-bind /e \
    --hf-repo-id openeurollm/oellm-32b \
    --hf-model Qwen/Qwen3-32B --tokenizer openeurollm/tokenizer-256k \
    --account e-sta-openeurollm --partition booster --qos normal \
    --gpu-binding gres --time-limit 03:00:00 --group-size 4 --max-concurrent-jobs 16

# Push what is converted so far, one branch per checkpoint.
singularity exec --bind /e \
    /e/project1/e-sta-openeurollm/container/MegatronTraining-JUPITER-bridge-bridge_aarch64_202605191331.sif \
    python3 scripts/upload_to_hf.py \
    --output-dir /e/project1/e-sta-openeurollm/$USER/oellm-32b-ckpts \
    --repo-id openeurollm/oellm-32b
```
<!-- google-doc-style-resume -->

A complete pass over all five runs is roughly 180 checkpoints at about 64 GB
each, which does not fit alongside anything else in a project quota. Convert,
upload and remove in batches rather than backfilling every export first.

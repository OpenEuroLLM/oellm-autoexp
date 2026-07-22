# Bootstrapping the Muon / Qwen3-multilingual work on a new cluster

This branch (`exp_sam_muon`) carries everything reproducible: the Megatron v0.17
submodule pin, the Qwen3 0.1B_ne + Muon configs, and the smoke scripts. It does
**not** carry environment-specific artifacts (container image, overlay, installed
packages, data). Those are cheap to recreate — steps below.

## What travels in git vs what you rebuild

| Travels in git | Rebuild on the new cluster |
|---|---|
| `submodules/Megatron-LM` @ v0.17 (`4a81356`) | base container (new cluster's Megatron-capable image) |
| `config/experiments/laingsam/…` (the run) | `emerging_optimizers` (one pip command, below) |
| `config/backend/.../data/oellm_256k_lumi_longctx.yaml` | data paths inside the configs (repoint to new FS) |
| `muon_smoke/…` (scripts + docs) | — |
| — | the old `muon-overlay.img` — **not needed anymore** (see step 2) |

## 1. Get the repo + submodule

If cloning from the bundle:

    git clone oellm-muon.bundle oellm-muon
    cd oellm-muon
    git checkout exp_sam_muon
    git submodule update --init --recursive   # needs network once, fetches Megatron v0.17

Confirm the pin:

    git -C submodules/Megatron-LM rev-parse --short HEAD   # expect 4a81356

## 2. emerging_optimizers — NO overlay needed

`emerging_optimizers` is pure Python, so skip the 6 GB apptainer overlay entirely.
Install it into a plain directory and put it on `PYTHONPATH` (run inside the new
cluster's container, in an allocation — not on a login node):

    pip install --no-deps --target=$PWD/muon-pylibs \
      "git+https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git@v0.2.0"

`--no-deps` matters — without it pip drags a whole torch into that folder.

In every run/launch script:

    export PYTHONPATH=$PYTHONPATH:$PWD/muon-pylibs:submodules/Megatron-LM

Verify:

    python -c "import emerging_optimizers as e; print('OK', e.__file__)"

Because there's nothing compiled, this works the same on NVIDIA and ROCm.

## 3. Repoint data paths

The configs and `muon_smoke/*.sbatch` point at LUMI paths under
`/scratch/project_465002530/...`. Update to the new cluster's filesystem:

- `config/backend/megatron/multilingual_scaling/data/oellm_256k_lumi_longctx.yaml`
  — `data_args_path`, tokenizer model/cache.
- the `SIF=`, `OVL=` (drop it), `DATAMIX=`, `DATACACHE=`, `HFCACHE=` lines at the
  top of `muon_smoke/06_real_data_muon.sbatch` and `04_autoexp_qwen3_muon.sbatch`.

## 4. Environment differences to watch (LUMI → new cluster)

The LUMI recipe has ROCm/Cray-specific bits that likely change on a new machine:

- `CC=gcc-12 CXX=g++-12` and the AITER JIT warm-up (`~/.aiter`) are **ROCm-only** —
  drop on NVIDIA.
- the `-B /boot/... -B /opt/cray -B /var/spool/slurmd` binds are LUMI-specific —
  use the new cluster's required binds.
- `--use-flash-attn` / fusion flags: keep the ones the new stack supports.

## 5. Smoke test (prove it before a real run)

Start small, same as the LUMI ladder:

1. `muon_smoke/muon_gpu_smoke.py` — Newton-Schulz on one GPU (optimizer core works).
2. `04_autoexp_qwen3_muon.sbatch` — Muon trains real Qwen3 0.1B_ne on mock data.
3. real data once you have a valid dataset on the new cluster.

Existing detail lives in `INSTRUCTIONS.md`, `RESULT.md`, and `REAL_DATA_RUNBOOK.md`.

## Note on the data blocker (carried over)

The reason for the move: on LUMI the real multilingual mix lives in
`project_462000963`, which is inaccessible (no group membership, root `2770`), and
the readable `long-ctx-sample` fallback has a systematic off-by-one in every `.idx`
(`document_indices[-1] == sequence_count - 1`), which fails Megatron's assert at
`indexed_dataset.py`. On the new cluster you need a dataset whose `.idx` satisfies
`sequence_count == document_indices[-1]` before real-data training will load.

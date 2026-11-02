# oellm-autoexp on CSCS Alps (Clariden)

Clariden is the CSCS Alps machine-learning cluster: GH200 nodes with 4 GPUs (about 96 GB
usable each), 288 Grace cores (aarch64), and the Slingshot network. This document covers
the cluster integration added to oellm-autoexp, how to set it up, and how it was validated.

The integration is configuration only; no Python code in `oellm_autoexp/` changed. It was
developed and tested on `feat/hybrid-v0.19` with `submodules/Megatron-LM` at
`oellm/v0.19-hybrid` (`a2f7be60`) and the NeMo 26.08 container.

## How it differs from the other clusters

On MareNostrum, JUPITER and similar clusters, `slurm.launcher_cmd` wraps the training
command in `singularity exec` or `apptainer exec` with `--bind`/`--env` flags. Alps has no
Apptainer. It uses the CSCS Container Engine, where `srun --environment=<edf.toml>` starts
every task inside a container described by an EDF (Environment Definition File: a TOML file
with the image, mounts, environment and hooks).

| | MareNostrum (Singularity) | Clariden (Container Engine) |
| --- | --- | --- |
| Container entry | `singularity exec ... image.sif` in `launcher_cmd` | `srun --environment=<edf>` in `slurm.srun_opts` |
| `container.image` | `.sif` path | EDF path (`container/alps/nemo-26.08.toml`) |
| Mounts | `container.bind` | `mounts` in the EDF |
| `launcher_cmd` | container exec, `--env` flags, `nslookup` | `cd $SLURM_SUBMIT_DIR` plus `export` of `backend.env` |
| Multi-node NCCL | InfiniBand variables (`NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME`, ...) | `aws_ofi_nccl` hook in the EDF, which sets `NCCL_NET` and `FI_CXI_*` |
| `MASTER_ADDR` | `nslookup` of the first node | `scontrol show hostnames ... \| head -n 1` |

Inside `srun ... bash -c '...'` the shell already runs in the container, so the launcher
only changes into the repository and exports the backend environment. The `cd` is needed
because the Container Engine does not start in the submit directory. Without it, relative
paths such as `./submodules/Megatron-LM/pretrain_gpt.py` and `PYTHONPATH=.:submodules/Megatron-LM`
would break.

## Files

| File | Purpose |
| --- | --- |
| `container/alps/nemo-26.08.toml` | EDF: NeMo 26.08 `.sqsh`, mounts `/capstor`, `/iopsstor`, `/users`, `aws_ofi_nccl` hook (`cuda-dl` variant), `PMIX_MCA_psec=^munge` |
| `config/container/clariden.yaml` | `image` points at the EDF (absolute, from `$PROJECT_DIR` or `$PWD`); `runtime: edf`; `MACHINE_NAME: CLARIDEN` |
| `config/slurm/clariden.yaml` | sbatch defaults, `srun_opts`, `launcher_cmd`, cluster environment |
| `templates/clariden.sbatch` | `base.sbatch` plus `ulimit -c 0` and a job and node header |
| `scripts/detect_cluster.py` | `clariden-ln*` maps to cluster `clariden` |
| `config/experiments/swagatam/alps/qwen3_sparse_0.9B_wider_20L_smoke.yaml` | Smoke test: 10B-A0.9B Qwen3 MoE, mock data |
| `config/experiments/swagatam/alps/qwen3_sparse_0.9B_wider_20L_strong_scaling.yaml` | Strong-scaling sweep over 1 to 32 nodes |

### Notes on `config/slurm/clariden.yaml`

- `srun_opts: "--environment=${container.image} --cpus-per-task=288 --kill-on-bad-exit=1 "`.
  This is a string rather than `srun_args` so the same file also loads on branches whose
  `SlurmConfig` has no `srun_args` (for example `main`). The trailing space is required on
  those branches, because the template has no space before `bash`.
- `sbatch.gres: null`. The base group sets `gres: gpu:1`; GPUs are requested with
  `gpus_per_node: 4` instead.
- One task per node (`ntasks_per_node: 1`, 288 CPUs); `torchrun` starts the 4 GPU processes
  with `--node-rank $SLURM_NODEID`, so expert-parallel groups of 4 stay on NVLink.
- `account` and `partition` default to `a0248` and `normal`; override them with
  `SLURM_ACCOUNT` and `SLURM_PARTITION`. Clariden partitions: `normal` (12 h), `debug`
  (at most 4 nodes and 1.5 node-hours), `preemptable` (24 h).
- `HF_HOME` defaults to `/iopsstor/scratch/cscs/$USER/hf_home`.
- `MACHINE_NAME` must match `container.env.MACHINE_NAME` (the root schema asserts this).
- Not carried over from MareNostrum: `module load singularity`, the InfiniBand `NCCL_*`
  variables, `CXX=g++-13`, the libcuda bind mounts and `TRITON_LIBCUDA_PATH`. The Container
  Engine injects the driver, and the hook configures NCCL.

### Notes on the EDF

- The EDF sets no `PYTHONPATH` and no `workdir`. The launcher exports
  `PYTHONPATH=.:submodules/Megatron-LM`, which shadows the Megatron-Core bundled in the image
  at `/opt/Megatron-Bridge/3rdparty/Megatron-LM`.
- The image's own environment puts `/opt/venv/bin` first on `PATH`. The experiments still set
  `PATH=/opt/venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin` in `backend.env`, so
  `torchrun` always comes from the NeMo venv and not from `/usr/local/bin`.
- The image path is project specific (`/capstor/store/cscs/swissai/a0248/images/`). Other
  projects should copy the EDF or point `container.image` at their own.

## Setup

1. **Import the image once**, on a compute node (enroot will not create missing directories):

   ```bash
   srun -A <account> -p debug -N1 -t 01:00:00 --pty bash
   mkdir -p /capstor/store/cscs/swissai/<project>/images
   enroot import -o /capstor/store/cscs/swissai/<project>/images/nemo-26.08.sqsh docker://nvcr.io#nvidia/nemo:26.08
   ```

2. **Python for autoexp on the login node.** The system Python is 3.6, and autoexp needs 3.10
   or newer. Install `uv`, which adds itself to `~/.bashrc`, and create the venv:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   cd oellm-autoexp && git submodule update --init submodules/Megatron-LM
   uv sync
   ```

3. **Environment** (in `~/.bashrc`):

   ```bash
   export OUTPUT_DIR=/capstor/scratch/cscs/$USER/autoexp   # job outputs; keep them on scratch, not the store
   export PROJECT_DIR=/path/to/oellm-autoexp              # optional; defaults to $PWD
   ```

   `OUTPUT_DIR` is required: the experiments derive `TORCHINDUCTOR_CACHE_DIR` from it.

## Running

From the repository root on a Clariden login node:

```bash
# Render only
uv run python scripts/run_autoexp.py --config-name experiments/swagatam/alps/qwen3_sparse_0.9B_wider_20L_smoke --dry-run

# 1-node smoke test (debug partition, 50 iterations, mock data)
uv run python scripts/run_autoexp.py --config-name experiments/swagatam/alps/qwen3_sparse_0.9B_wider_20L_smoke

# 2 nodes, with NCCL transport logging
uv run python scripts/run_autoexp.py --config-name experiments/swagatam/alps/qwen3_sparse_0.9B_wider_20L_smoke \
  slurm.sbatch.nodes=2 slurm.env.NCCL_DEBUG=INFO

# Strong-scaling sweep (1, 2, 4, 8, 16, 32 nodes; GBS 2048, MBS 1)
uv run python scripts/run_autoexp.py --config-name experiments/swagatam/alps/qwen3_sparse_0.9B_wider_20L_strong_scaling \
  backend.megatron.train_iters=100
```

In the sweep, `slurm.sbatch.time` and the GPU-hour estimate scale with `train_iters` and the
node count. The estimate uses `job.est_step_time_s`, based on a measured 152 s per step on
1 node at GBS 2048, so that config needs a branch with the `est_*` job fields.

## Validation

Both runs used the smoke config: the 10B-A0.9B Qwen3 MoE (20 layers, hidden size 2048, 64
experts, top-4, expert parallelism 4), Muon optimizer, sequence length 4096, GBS 128, MBS 1,
mock data with a 256k vocabulary, and 50 iterations.

| | 1 node (job 3720239) | 2 nodes (job 3720572) |
| --- | --- | --- |
| Result | COMPLETED, 10.6 min | COMPLETED, 7.2 min |
| Megatron-Core | `0.19.2+a2f7be60b` (from the submodule) | same |
| Step time (iterations 6–50) | 9.51 s | 5.47 s (1.74x, about 87% efficiency) |
| Throughput per GPU | ~133 TFLOP/s, 13.8k tokens/s | ~115 TFLOP/s, 12.1k tokens/s |
| Peak memory per GPU | 54 GB | 44 GB |
| lm loss at iteration 50 | 0.00698 | 0.00705 |
| NaN or skipped iterations | 0 | 0 |

- Multi-node NCCL uses Slingshot: `NET/OFI Selected provider is cxi ... (found 4 nics)`,
  `Using network AWS Libfabric`, and cross-node channels on `NET/AWS Libfabric/<n>/GDRDMA`.
  There are no `NET/Socket` lines.
- The Megatron dataset helper compiles inside the container, and `torchrun --numa-binding node`
  works on GH200.
- Mock data consists of counting sequences (`1, 2, ..., L-1, EOD`), so the loss collapsing
  towards 0 is expected. It checks plumbing and throughput, not training quality.

Messages that are safe to ignore:
- `Failed to import Triton kernels ... triton_kernels.matmul_ogs`, an optional import in the image.
- `NET/OFI GIN only supports RDMA transport protocol`.
- `Disabling sequence parallelism because tensor model parallelism is disabled`.
- The pynvml, TVM/TileLang field, `GPTModel IS DEPRECATED`, `one_logger` and
  `destroy_process_group()` warnings.

## Known limitations and follow-ups

- **Postprocess steps that run in the same job** build a `singularity exec` prefix
  (`_build_container_exec_prefix` in `oellm_autoexp/orchestrator.py`), so they do not work
  with an EDF yet. Training jobs are unaffected; conversion and eval chains need a
  `runtime == "edf"` branch there.
- **Real data:** the tokenizer and tokenized datasets still have to be copied to Alps. The
  experiments then switch from `mock_data`/`NullTokenizer` to `HuggingFaceTokenizer` with a `data_path`.
- **Project-specific defaults:** the `a0248` account and the image path in the EDF. Other
  projects override them with `SLURM_ACCOUNT` and their own EDF or `container.image`.
- **Bristen** (Alps A100 nodes, x86_64) is not covered. It needs an x86 image and its own EDF.
- **Untested:** `preemptable` with requeue, job arrays, and the monitor's restart path on Clariden.

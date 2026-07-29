# Tiny mLSTM/GDN Hybrid Smoke

## Objective
Test Megatron-LM experimental mLSTM and GDN hybrid attention variants locally under about 1 GB free VRAM.

## Hypothesis
A very small mock-data Megatron run can validate that the branch wiring and kernels are importable without requiring the full architecture-scaling model sizes or production data.

## Setup
- Configs: `config/experiments/architecture_scaling_variants/multilingual/{ml_base.yaml,gdn_base.yaml}`
- Script: `scripts/tiny_variant_smoke.sh`
- Environment: local RTX 3090 Ti; `nvidia-smi` reported 1075 MiB free before testing.

## Run Log
- 2026-07-29: Created labbook and copied the smoke script.
- 2026-07-29: `nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader` reported `NVIDIA GeForce RTX 3090 Ti, 24564 MiB, 23038 MiB, 1075 MiB`.
- 2026-07-29: `./scripts/tiny_variant_smoke.sh gated_delta_net 2>&1 | tee experiment/001_tiny_mlstm_gdn_hybrid_smoke/results/gdn.log` completed 10/10 iterations.
- 2026-07-29: `./scripts/tiny_variant_smoke.sh mlstm 2>&1 | tee experiment/001_tiny_mlstm_gdn_hybrid_smoke/results/mlstm.log` completed 10/10 iterations.
- 2026-07-29: `PYTHONPATH=. python scripts/run_autoexp.py --dry-run --array-subset 0 --config-name experiments/architecture_scaling_variants/multilingual/gdn5_0.1B_low` rendered one sbatch script.
- 2026-07-29: `PYTHONPATH=. python scripts/run_autoexp.py --dry-run --array-subset 0 --config-name experiments/architecture_scaling_variants/multilingual/mlstm5_0.1B_low` rendered one sbatch script.
- 2026-07-29: Synced the branch to JUPITER with `./sync_to_jupiter.sh`.
- 2026-07-29: Verified JUPITER container exists: `$CONTAINER_CACHE_DIR/MegatronTraining-JUPITER-linattn_aarch64_202607290923.sif`.
- 2026-07-29: Remote dry-runs for `gdn5_smoke_tiny_jupiter` and `mlstm5_smoke_tiny_jupiter` rendered one sbatch script each.
- 2026-07-29: Submitted first 2-node JUPITER smoke jobs: GDN `1090196`, mLSTM `1090197`. Both started and failed during Megatron import with `AssertionError: Minimum required nvidia-resiliency-ext package version is 0.6.0.`
- 2026-07-29: Patched `submodules/Megatron-LM/megatron/core/dist_checkpointing/strategies/nvrx.py` so too-old NVRx disables NVRx async support instead of aborting import.
- 2026-07-29: Submitted rerun 2-node JUPITER smoke jobs after the NVRx patch: GDN `1090267`, mLSTM `1090270`. Both started and passed the previous NVRx import failure, then failed in the first forward pass because Triton/FLA could not locate or use CUDA correctly inside the container.
- 2026-07-29: Added `TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib` to the tiny JUPITER smoke configs and submitted third-attempt jobs: GDN `1090348`, mLSTM `1090349`. As of the last poll, both are `PENDING` with Slurm reason `Priority`.
- 2026-07-29: The current JUPITER hybrid image reports `nvidia-resiliency-ext==0.5.0`, below the `0.6.0` requirement in the checked-out Megatron-LM. Added `container/megatron/MegatronTraining-JUPITER-linattn-nvrx.def.in`, a derived-image recipe that uses the existing hybrid SIF as `BASE_IMAGE` and replaces only NVRx with `0.6.0`.
- 2026-07-29: Built `/e/project1/e-sta-openeurollm/container/MegatronTraining-JUPITER-linattn-nvrx_aarch64_202607291158.sif` from a temporary rendered definition and the original `MegatronTraining-JUPITER-linattn_aarch64_202607290923.sif`. The recipe test and `apptainer exec --nv` both reported `nvidia-resiliency-ext 0.6.0`; Megatron reported `is_nvrx_min_version=True` and `has_nvrx_async_support=True`. Updated the JUPITER linear-attention container config and GDN/mLSTM overlays to select the new image.
- 2026-07-29: Re-rendered the tiny JUPITER GDN and mLSTM smoke jobs after syncing the corrected config paths. Both scripts reference `MegatronTraining-JUPITER-linattn-nvrx_aarch64_202607291158.sif` and retain their respective experimental-attention variant flags.
- 2026-07-29: Submitted fresh tiny 2-node runs against the NVRx image: GDN `1090774` and mLSTM `1090775`. Both are pending with Slurm reason `Priority`; the earlier queue entries were not changed.

## Results
- `results/gdn.log`: `experimental_attention_variant` resolved to `gated_delta_net`; model had 1,352,912 parameters on rank 0; iteration 10 completed with `number of skipped iterations: 0` and `number of nan iterations: 0`; peak reserved memory reported by Megatron was 110 MB.
- `results/mlstm.log`: `experimental_attention_variant` resolved to `mlstm`; model had 1,351,056 parameters on rank 0; iteration 10 completed with `number of skipped iterations: 0` and `number of nan iterations: 0`; peak reserved memory reported by Megatron was 110 MB.
- `results/oellm_gdn_dry_run.log`: oellm-autoexp composed `gdn5_0.1B_low` and rendered `/mnt/data/training_outputs/architecture_scaling_variants_multilingual/qwen3_gdn5_0.1B_ne_lr0.002_gbsz128_stable/job.sbatch`, which includes `--experimental-attention-variant gated_delta_net` and `--linear-attention-freq 6`.
- `results/oellm_mlstm_dry_run.log`: oellm-autoexp composed `mlstm5_0.1B_low` and rendered `/mnt/data/training_outputs/architecture_scaling_variants_multilingual/qwen3_mlstm5_0.1B_ne_lr0.002_gbsz128_stable/job.sbatch`, which includes `--experimental-attention-variant mlstm` and `--linear-attention-freq 6`.
- `results/jupiter/dry_gdn_smoke.log`, `results/jupiter/dry_mlstm_smoke.log`: remote oellm-autoexp dry-runs rendered tiny 2-node smoke sbatch scripts.
- `results/jupiter/slurm/slurm-1090196.log`, `results/jupiter/slurm/slurm-1090197.log`: first JUPITER runs failed before model construction because the container has an importable `nvidia-resiliency-ext` below Megatron's required `0.6.0`.
- `results/jupiter/submit_gdn_smoke_nvrxfix.log`, `results/jupiter/submit_mlstm_smoke_nvrxfix.log`: rerun jobs were submitted as Slurm jobs `1090267` and `1090270`.
- `results/jupiter/slurm/slurm-1090267.log`: GDN rerun reached first forward pass, then failed in `fla` CPU fallback with `AttributeError: module 'torch.cpu' has no attribute 'device'`.
- `results/jupiter/slurm/slurm-1090270.log`: mLSTM rerun reached first forward pass, then failed in Triton with `AssertionError: libcuda.so cannot found`, while reporting `/usr/local/cuda/compat/lib/libcuda.so.1` as the available location.
- `results/jupiter/submit_gdn_smoke_tritonpath.log`, `results/jupiter/submit_mlstm_smoke_tritonpath.log`: third-attempt jobs were submitted as Slurm jobs `1090348` and `1090349`.
- `results/jupiter/MegatronTraining-JUPITER-linattn-nvrx_aarch64_202607291158.build.log`: build, embedded test, and runtime package-version verification for the derived NVRx image.

## Interpretation
Both tiny hybrid smoke tests passed locally under the available-VRAM constraint. The oellm-autoexp production-style GDN and mLSTM 0.1B configs also compose and render the expected Megatron flags. On JUPITER, sync, container discovery, remote config rendering, Slurm submission, and multi-node allocation were validated. The first multi-node runs exposed a container/Megatron compatibility issue in the NVRx async-checkpoint probe; the branch was patched to handle that as a feature absence. The post-NVRx reruns reached the first forward pass, so distributed startup and model construction are working, but the container still has CUDA/Triton discovery problems for the linear-attention kernels. A third pair with `TRITON_LIBCUDA_PATH` set is queued and still needs to complete.

## Related
- Architecture configs: `config/experiments/architecture_scaling_variants/`

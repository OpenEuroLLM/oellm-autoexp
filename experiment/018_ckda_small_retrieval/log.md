# Native Megatron CKDA/KDA retrieval interventions

## Objective
Establish the focused retrieval baseline for the native 983M / 80B-token
Megatron CKDA and KDA hybrid pair, then diagnose and test repair mechanisms.

## Setup
- Jupiter source: `/e/project1/e-sta-openeurollm/poeppel1/signed_kda`.
- Native checkpoints: `/e/scratch/e-sta-openeurollm/poeppel1/signed_kda_runs/megatron_ladder_finegrained/ckpt_decay/{skda-shipped-hybrid-lowrank,kda-sig-hybrid-lowrank}_983M_gbs256/decay80B/iter_0076294/mp_rank_00/model_optim_rng.pt`.
- Existing evaluation exports: `/e/scratch/e-sta-openeurollm/poeppel1/eval_finegrained/export/{skda-shipped-hybrid-lowrank,kda-sig-hybrid-lowrank}_983M_gbs256_decay80B`.
- MN5 workspace: `/gpfs/projects/ehpc533/users/korbip/ckda_retrieval_018`.
- Preserve the original FLA implementation and explicit model-class loader.
  Adaptation uses native checkpoints with fresh optimizers for both arms.
- New jobs go to p264, not o260c. Existing Slurm jobs are never modified.

## Run log
- 2026-11-10: User authorized preparations, initial focused hybrid comparison,
  and implementation of the analysis/intervention workflow.
- 2026-11-10: Started copying the original code, native model checkpoints and
  existing evaluation exports through workstation staging `/tmp/ckda018`.
- 2026-11-10: User requested moving the not-yet-submitted bgqar 23499 evals
  to host004. Replacement watcher `scripts/wait_final_eval.sh` targets p264.
  Stop only the local previous evaluation launcher and deferred 248-node
  health-check launcher; preserve submitted jobs, including 23450 evals.

## Results
Pending. Commands, manifests, baseline metrics and per-example outputs belong
under `results/`, with remote artifacts referenced here.

## Related
- [017: first fully switched large-model evaluation](../017_ckda_branch_early_eval/log.md).
- `config/experiments/korbi/complex_kda/README.md` describes native Megatron sweeps.

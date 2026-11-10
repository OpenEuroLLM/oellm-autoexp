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
Baseline metrics pending. Commands, manifests and per-example outputs belong
under `results/`, with remote artifacts referenced here.

### 2026-11-11 preparation and validation
- Native model files downloaded, checked against Jupiter SHA256 hashes and
  copied to MN5. `finish_stage.py` independently checks hashes again on MN5;
  the readiness gate opened at approximately 00:08 CET.
- Source hashes and native metadata: [source manifest](results/source_manifest.json),
  [CKDA checkpoint](results/native_ckda.json), [KDA checkpoint](results/native_kda.json).
  Both are step 76294, checkpoint format 3, with 490 model keys and the original
  spec `lm_scaling.megatron_ext.specs.arm_spec`. Original micro-batches differed
  (2 vs 4); fresh continuation controls must use an identical batch layout.
- MN5 training image: torch 2.14.0a0+b2c75dd062.nv26.09, transformers 5.20.1,
  lm-eval 0.4.13, H100. First runtime probe exposed the removed transformers
  CoQA scoring module. `metric_compat.py` supplies its standard normalized
  exact/F1 functions; normalization, repeated tokens and empty answers checked.
- Probe `mn5-pool-p264.0665` passed finite signed-model GPU output and offline
  loading of all six datasets. Further probe `ckda018_probe3_620eed` additionally
  checks chunk/cache agreement and exact KDA no-op behavior for signed controls.
- Baseline pool job `ckda018_baseline_67df60` was enqueued on p264 at 23:58 CET,
  gated on verified assets and the successful runtime probe. It uses one node,
  four workers: CKDA/KDA focused and CKDA/KDA recall, and has a four-hour limit.
- Source checkpoint files contain native weights; the evaluation exports are
  regenerated from those native checkpoints. No HF-to-Megatron conversion.
- The original tokenizer vocabulary and BPE merges match MN5's NeoX tokenizer.
  Existing 318BT training and separate validation Nemotron prefixes are recorded
  in [the plan](notes/plan.md); no additional training data transfer is needed.
- Scoped inference diagnostics, counterfactual analysis commands and a native
  two-step resume smoke script are implemented/prepared. Broad screens and
  adaptation training remain unsubmitted pending baseline review and native
  load/backward compatibility. See [plan](notes/plan.md).
- bgqar 23499: replacement launcher successfully enqueued nine export/eval jobs
  on p264 at 23:49:55 CET; the export and workers placed on host004. Original
  23450 jobs on o260c remain untouched. No existing Slurm allocation was resized,
  cancelled, requeued or updated.

## Related
- [017: first fully switched large-model evaluation](../017_ckda_branch_early_eval/log.md).
- `config/experiments/korbi/complex_kda/README.md` describes native Megatron sweeps.

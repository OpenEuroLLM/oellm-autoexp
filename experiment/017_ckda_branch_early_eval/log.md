# CKDA branch early downstream evaluation

## Objective
Measure Winogrande accuracy, SQuAD v2 F1 (10-shot), and CoQA F1 (0-shot)
for the randomly initialized attention arm bgqar at its first fully switched checkpoint.

## Setup and decision
- User accepted the early 23450 evaluation on 2026-11-10; retain the existing 23499 evaluation.
- Training run: mn5-pool-o260c.0041, allocation 48535429, 256 nodes.
- Surgery at 23300, crossfade over 150 steps: 23400 still mixes one-third CKDA
  and two-thirds attention and cannot use the current HF/vLLM exporter.
- 23450 is fully attention at layers 9/25/41, after 10.0663296B training tokens.
- Existing config: `config/evals/ckdab_fast_mn5.yaml`, snapshot
  `/gpfs/projects/ehpc533/users/korbip/hybrid_v019/autoexp_ckdab27`.
- Export: `ckdab_bgqar_fp8_23450`; pool: mn5_pool_o260c.
- Existing Slurm jobs and the existing 23499 watcher must remain untouched.

## Run log
- 2026-11-10: Inspected training through 23361. Same-batch mean loss difference
  bgqar minus drop: +0.000105 nats; bgqar minus bgqa: -0.000120 nats (61 batches).
  No convincing training-loss improvement at this stage.
- Launcher waits for checkpoint `.metadata` plus the committed latest-checkpoint
  pointer >= 23450, then invokes the existing evaluation launcher once. The
  evaluation launcher schedules the export and workers gated on export readiness.
- Command: `nohup bash experiment/017_ckda_branch_early_eval/scripts/wait_and_launch.sh > experiment/017_ckda_branch_early_eval/results/launcher.log 2>&1 < /dev/null &`.
- Logs: [results/launcher.log](results/launcher.log). Successful submission creates
  `results/submitted`; a failed submission requires inspection before a manual retry.

## Results
Remote results will be under
`/gpfs/scratch/ehpc533/hybrid_nucleus/flag_evals_oellm/ckdab_bgqar_fp8_23450`.

## Interpretation
Pending. Compare against main 23500, drop 23499, original bgqa 23499, and bgqar 23499.
Differences across checkpoint steps are not controlled estimates of the branch effect.

## Related
- Earlier downstream analysis: `docs/ckda_branch_focus.md`.
- Final evaluation: existing watcher for bgqar 23499, retained.

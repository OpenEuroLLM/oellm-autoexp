# Ambitious plain K: fresh 35900 rebuild and five-step MN5 training

## Objective
Rebuild the most promising shared-KV intervention from the original ambitious
step-35900 checkpoint, avoiding the existing DK1/DK1T surgery checkpoints, and
train steps 35901–35905 with a final synchronous save on 128 healthy MN5 nodes.

## Hypothesis
Plain K is the best supported intervention in the small31 Jupiter experiments.
Dropout 0.3 (DK1) had some extraction gains but worse validation loss and CoQA;
head temperature T was neutral. Check the latest paired results before launch.

## Setup
- Source: `/gpfs/scratch/ehpc533/hybrid_nucleus/production/main_mhc4/ambitious_e360_eng80_v4_2ph_hd256/ckpt_rolling/iter_0035900`.
- Fresh root: `/gpfs/scratch/ehpc533/hybrid_nucleus/production/kvshare_k_fresh35900_20261116`.
- Snapshot: `/gpfs/projects/ehpc533/users/korbip/hybrid_v019/autoexp_k_fresh35900_20261116`.
- Configs: `kvshare_k_fresh_surgery_mn5`, `kvshare_k_fresh5_mn5`.
- Six additive branches: 9/25/41/58/74/90 read K/V from 0/17/33/50/66/82.
- Copied Q/gate/output/Q-norm, folded input norm, gamma zero, CKDA full weight.
- Rest dropout 0; no head-temperature addition. GBS 1024, sequence 65536,
  microbatch 2, reference DP 512 / microbatch 1; five steps, sync save 35905.
- Existing pool only; no Slurm submissions, cancellations, requeues or updates.

## Run Log
- 2026-11-16: Inspected docs §14.8/§20 and fetched live Jupiter readouts.
  Existing MN5 DK1T save test is running; leave it untouched. New work waits for
  available nodes. Rebuild into a fresh root and verify generated tensors.
- 18:59 CET: Surgery enqueued as `ambitious_K_fresh35900_surgery_8368e0`,
  pool task `mn5-pool-p264.1207`, host006 / allocation 48653415, node as01r2b17.
- 18:59 CET: Training enqueued as `ambitious_K_fresh35900_5step_6ee369` at
  pool priority 100 (only this new task), gated on the verified fresh tracker.
- Existing DK1T attempt `.1206` failed loading the missing key
  `model.module.decoder.layers.9.self_attention.branch.head_temperature`.
  No numerical corruption was established; plain K has no head-temperature keys.

Exact launch commands (on MN5, from the isolated snapshot):

```bash
scripts/korbi/mn5_autoexp.sh scripts/run_autoexp.py --submit-and-exit \
  --pool /gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_p264 \
  --config-name experiments/korbi/recall_surgery/kvshare_k_fresh_surgery_mn5 --yes
scripts/korbi/mn5_autoexp.sh scripts/run_autoexp.py --submit-and-exit \
  --pool /gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_p264 \
  --config-name experiments/korbi/recall_surgery/kvshare_k_fresh5_mn5 --yes
```

The snapshot is a physical copy of `autoexp_kvshare`, with the two new configs,
this experiment directory, and the local surgery helpers overlaid. The copy's
`monitor_state` destination already existed after rendering; that source monitor
state was not copied. The active pool monitor remains the existing one.

## Results
- 19:07 CET: The snapshot supports rank-local DCP planning and the existing
  DK1T save test already uses it. Enable `FSDP_DCP_LOCAL_LOAD_PLAN=1` plus
  load/save phase timing for K. Replace only our not-yet-started pool entry
  (submitted=false, attempts=0, runtime_job_id=null); no Slurm step is cancelled.

Live Jupiter paired readouts through 48638 are in `results/jupiter_paired.txt`.
At 48638, K vs C: F1|answering +1.87, HasAns +2.82, CoQA +1.76,
validation -0.0278. K improves validation at all seven paired readouts.
DK2 has the strongest final recall (+3.67/+3.78/+5.52), but requires first
training a dropout donor and stitching its branches into the control; its final
training uses no dropout. DK1 has only one readout, CoQA -1.23 and validation
+0.0462. T is neutral. Decision: plain K, no dropout or head temperature, is
the supported direct insertion from original 35900 for this five-step test.

## Related
- `docs/recall_surgery_ambitious.md`, §14.8 and §20.

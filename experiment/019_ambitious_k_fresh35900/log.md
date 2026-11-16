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
- 2026-11-16 19:22 CET, user explicitly authorized stopping pool workloads,
  shrinking the current allocation to 136 nodes, then a 128-node K continuation
  with a checkpoint at 35999 and critical recall evaluation of its checkpoints.
- Froze host006 and cancelled eight remaining internal pool records (including
  old pending evaluations, control, DK1T retries and the K smoke). No queued
  Slurm host was cancelled. A diagnostic step launched from the login node hit
  Slurm I/O launch errors; discarded that attempt and reran from the batch node.
- Fresh thorough check: 240 candidates, median 600.85 TFLOP/s; 232 compute
  passes. Network group 1 measured 19.48 GB/s; the two slowest pairs identify
  as01r5b16. Removed it from consideration as well as all compute failures.
- Resized 48653415 from 264 to 136, with no active workload/test steps or inuse
  reservations. Kept the batch node and the best 135 remaining nodes. Worst
  retained GPU 591.63 TFLOP/s. See `results/resize136_selection.json` and
  `results/keep136.txt`. Queued job IDs and submit timestamps diff unchanged.
- Prepared the documented 1000-step continuation to 36900; saves at 35999,
  36400 and 36900 are all retained. In-training validation stays disabled
  because of its known 64k-mode failure; checkpoint recall readouts are separate.
- Readouts: baseline 35900 plus all three saves, six spare-node workers on
  the 30-row critical recall suite (CoQA, Winogrande, SQuAD v2, SWDE, FDA,
  squad_completion and 24 Belebele languages). Each export is parallel and
  gated on its completed checkpoint; evaluations wait for structural export
  checks and actual-export HF/vLLM parity, including chunked prefill.

- 19:19:22 CET: Fresh K `.1210` started on 128 nodes in host006, after
  DK1T `.1209` failed on the same missing branch-temperature key. No existing
  Slurm step/allocation was stopped. Both configs have sequence 65536, GBS
  1024, microbatch 2, final iteration 35905, synchronous save. DK1T uses rest
  dropout 0.3 and head temperature; fresh K uses 0.0 and false.

- 19:11:08 CET: Surgery `.1207` completed with rc=0. First pass 407 s,
  including 361 s reading 96 source tensors. Wrote 108 new keys into eight
  delta files; hard-linked 1024 original shards, zero full-shard copies.
  Independent verification: 108 tensors, zero hash differences. Source
  `.metadata` hash checked unchanged; fresh tracker published at 35900.
  See `results/surgery.log` and `results/surgery_manifest.json`.
- Instrumented training entry: `ambitious_K_fresh35900_5step_af465d`.

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

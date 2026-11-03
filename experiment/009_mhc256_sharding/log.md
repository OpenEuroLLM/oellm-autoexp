# mHC256 microbatch and hybrid sharding

## Objective
Reduce GPU-hours/token without changing production's global batch or reference sample order; quantify startup cost and exposed communication.

## Hypothesis
MBS4 restores two accumulation microbatches at DP1024. HSDP2 reduces inner parameter-gather group sizes but adds replica gradient synchronization. Neither improvement is assumed before measurement.

## Setup
- Production: checkpoint 4807 migrated from 128 nodes; 256 nodes, EP4, global batch8192 x8192 tokens, MBS8, reference DP512/MBS8. Stable ~20.2s versus old ~27.2s.
- Successful startup: 19:47:29 library setup, 19:53:12 distributed initialized, 20:07:54 model+optimizer+checkpoint ready, 20:08:54 dataloaders ready, 20:17:45 first step. Rank-local DCP planning fixed the pathological gather but did NOT eliminate the 21min setup +9min first-step cost.
- Reuse shared data and compiler caches; keep rank-local DCP planning and batched metadata fixes. No claim of guaranteed short restart. Same-DP fresh checkpoint avoids old-layout resharding; still needs measured timing.
- Small test config: `config/experiments/korbi/production/hybrid_mhc256_hsdp_check_mn5.yaml`. Four spare nodes, production architecture/optimizer paths with 8 experts, Engram65539 uneven rows, seq1024, GBS32. Ref saves rolling20 and ends30. Control and HSDP2 load20 and train21–30. Separate outputs, no singleton preemption. HSDP optimizer row exchange explicitly supports Replicate/Shard placements, but end-to-end correctness unproven.
- Full config: `config/experiments/korbi/production/hybrid_mhc256_mbs4_profile_mn5.yaml`. MBS4 only, GBS8192, original reference512/8 retained, Engram lookup grouping1 unchanged, original train_iters/cache keys retained. Profile resumed steps13–15 on ranks0/512 (schedule is relative to process start despite schema comment). Continue training afterward to avoid another restart if stable/faster.
- Profiles: small ranks0/8; production ranks0/512, CPU+CUDA Chrome traces. No shapes/stacks to limit overhead. Exclude capture and trace-export steps from throughput statistics; compare >=20 clean steps after warmup. Measure startup separately from steady state.

## Run Log
- 2026-11-03: user authorized MBS4 first plus parallel small-scale HSDP soundness/profiling; requested avoiding another 20–30min startup. Existing phase transition code exits/reloads rather than a safe in-process microbatch switch. No live-process mutation attempted.

## Results
Pending.

## Interpretation
Full-scale restart is gated on small-scale resume/sampler checks and a concrete startup assessment. Keep CNS allocations held. Do not release or close hosts. Small HSDP timing is diagnostic, not a 256-node throughput prediction.

## Related
- `experiment/005_fsdp_init_metadata/log.md`
- `docs/pp_vs_fsdp.md`
- 20:36 CET: enqueued three 4-node jobs on existing held cns264 allocation48203953. Ref started as mn5-pool-p264.0014; control/HSDP wait for rolling20. Production .0013 continues independently. Exact CPU HSDP load passed both default and rank-local planning. Production checkpoint4807 metadata is463MiB, so startup I/O/deserialization remains material; no quick-restart guarantee.
- Full production comparison should use original `autoexp_v4move_20261103` code paths with only the new config added, to retain path-sensitive compiler caches; do not deploy a changed Megatron tree over the running job. Fresh same-layout checkpoint is required before any production handover.
- Later 2026-11-03: user reports another agent released allocation48203953. Scaling/profiling work superseded by restoring normal128-node mHC training on held CNS132 allocation48094115, p129/host004. Cancelled queued256-node production record .0018 to prevent competing restart. Small ref/control/HSDP2 jobs had already completed successfully (rc0); interpretation/profile ingestion deferred to production recovery.
- Copy job p129.0351 was NOT a continuation: observed iteration58/consumed475136, loss4.981. User requested it be preempted; pool cancel requests graceful USR2 stop without requeue, host held. Original production persistent4938 and rolling4900 remain separate. Asked whether latest4938 or explicit rewind4800 is intended, while preparing normal128-node config with fatal-on-missing checkpoint guard.
- 22:39 CET: normal continuation submitted to p129 as `mn5-pool-p129.0352`, job `ambitious_v4_2ph_128n_resume_p1_13aae8`. Production load/save root, latest persistent4938 (verified successful save21:09:01, consumed40452096), rolling4900 older; 128nodes, MBS8, GBS8192, reference512/8, normal ZeRO3, full optimizer/RNG, strict load, fatal if checkpoint missing. Config synced alone into original production snapshot to preserve its compiler paths. Copy0351 cancellation in progress; host004 allocation48094115 held, no allocation release requested. Only phase1 enqueued.
- Root cause of copy's fresh start: `--load` pointed to `ckpt_rolling/iter_0004800` instead of the checkpoint root. Megatron looked for `iter_0004800/latest_checkpointed_iteration.txt` and explicitly logged random start. Original production root was not overwritten by the copy.

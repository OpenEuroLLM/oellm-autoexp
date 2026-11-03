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

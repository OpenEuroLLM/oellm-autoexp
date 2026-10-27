# Handover: LDM speed / profiling, GQA rematch, production state (2026-10-27, ~10:45)

From session oellm-autoexp-hybrid-89 to the main agent.

## 1. Open jobs (MareNostrum 5, ehpc533)

| what | where | state |
|---|---|---|
| 32-node profiles: `mhc4`, `mhc4_m2`, `ldm_m2`, `ldm_m4`, `mhc4_pp4`, `ldm_pp4` | jobs 48126146-51, acc_ehpc, direct sbatch (not a pool); snapshot `$R/autoexp_ldmprof3`; monitor `monitor_logs/ldm_prof32c` | pending |
| 128-node rerun `ldm_speed_ldm_m4_128n` (log_interval 1) | pool mn5_pool_p129, run 0010, host 48117532 | running / next |
| conservative `q35c_main_gqa` | p33 (32n, running, 6,515 tok/s/GPU); p129 (128n, pending behind 0010) | — |
| LDM long-context evals `ds5lcresgqa(logn)_s2_ldm*` | results written (downstream.json, 13 keys each); jobs finishing | done |

- Config: `config/experiments/korbi/production/hybrid_ldm_prof_mn5.yaml`. It runs 10 steps, profiles
  step 8 on ranks 0 and 127, and logs every step.
- Traces land in `/gpfs/scratch/ehpc533/hybrid_nucleus/production/ldm_prof/<arm>/torch_profile/rank-{0,127}.json.gz`.
- Analysis tools:
  - `scripts/korbi/ep_speed/trace_breakdown.py`: exposed communication.
  - `scripts/korbi/ep_speed/trace_launch_bound.py`.
  - The LDM authors' `tools/trace_summary.py`.
- PP arms:
  - PP4 without interleaving, 13/12/12/11 blocks, distributed optimizer, mbs 2 (64 micro-batches per
    pipeline).
  - LDM under PP uses `depth_memory_segments` with 4 layers per chunk: the unwind path refuses PP.
  - PP + muown_classical + Engram + MoonEP is untested on the full model. The job cancels itself on
    FAILED.

## 2. Findings

**Speed tests were averaging the warm-up.** The 12-step speed tests logged at interval 10, so their one
line averaged steps 2-10.
- Old 32-node numbers: mHC-4 5,243 / LDM mbs 2 3,680 / LDM mbs 4 2,558 tok/s/GPU. They are not clean.
- `hybrid_ldm_speed_mn5` now logs every step.

**128 nodes in pool p129 are 2x slow.** Clean rerun of mHC-4 at 128 nodes (p129 run 0008), from step
3 on: 50.5 s/step = 2,595 tok/s/GPU.
- The non-pool production job 48029517 had the same model / mbs 4 / gbs 4096 and ran 26.2 s/step
  (~5,000 tok/s/GPU).
- The configs differ only in GQA, eval/log and train_iters.
- p33 is fine at 32 nodes.
- Suspect host 48117532 (a node?) or the overlap-step environment at 128 nodes. Check it before the
  conservative run relies on it. The campaign agent (Current edits review) is informed.
- The 128-node LDM mbs 4 run fits (mem 0.85). Its warm-up-averaged step time equals mHC-4's
  (59.6 vs 59.5 s).

**GQA rematch s2, final (step 4000).**
- LDM ×1 vs mHC-4:
  - Loss: −0.014 now, slope +0.025 per ln t, extrapolated +0.027 at 10B.
  - Best long-document NLL: 2.228 vs 2.260 at 32-64k.
  - Weaker retrieval at 64k: 16-fact retrieval with log-n 0.50 vs 0.76; mHC-4 ×0.125 0.80.
- s1: LDM and mHC-4 tie.
- Tables: `scripts/korbi/campaign/rematch_report.py` and `rematch_longctx.py` (copies in /tmp/korbip_rep
  on MN5).

**CUDA graphs:** launch-bound share is 0.3-0.4 % of the step. Not worth pursuing.

## 3. Production state

- The 128-node MHA production job 48029517 ended CANCELLED at 2026-10-26 23:48 (step ~2630, not by
  this session).
- The GQA fresh start (`main_gqa`, `scripts/korbi/production/flip_mha_to_gqa.sh`, not started) is on
  hold for the user's go.
- cns144 job 47998967: still pending, adopted by p129. `monitor_logs/watch_cns144_record.sh` records its
  start.
- Save fixes (async + batched uneven-DTensor metadata + nvrx plan cache, Megatron ca6431476 / bdbda626a)
  are in the GQA configs. At 32 nodes the blocking time per save is 7 s from the 3rd save on.

## 4. Next steps

1. When the 32-node profiles land:
   - Per arm: step time, kernel families, exposed FSDP communication, LDM kernels, allocator stalls.
   - Key question: is `mhc4_m2` as slow as `ldm_m2`? Then the gap is ZeRO-3 traffic at mbs 2, and the
     fix is fitting LDM at mbs 4 (~9.4 GB/GPU extra: M + dM workspace), not the LDM kernels.
2. PP vs FSDP on the full model from `mhc4_pp4` / `ldm_pp4`.
3. The cause of the p129 128-node slowdown.
4. The LDM decision for the user. Retrieval (the stated criterion) favours mHC-4.

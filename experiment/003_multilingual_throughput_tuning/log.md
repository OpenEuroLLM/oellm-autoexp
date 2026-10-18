# Multilingual 0.1B Throughput Tuning

## Objective

Raise JUPITER GH200 training throughput for the 0.1B SWA multilingual model
before releasing the 6B-token architecture comparison.

## Hypothesis

The completed smoke used microbatch 4 with two accumulation steps and reached
about 152 TFLOP/s/GPU. Increasing the per-GPU microbatch improves kernel
occupancy and reduces per-step fixed overhead; microbatch 8 preserves global
batch 32 while reducing accumulation to one.

## Setup

- Config: `config/experiments/architecture_scaling_variants/multilingual/throughput_0.1B.yaml`
- Cluster: JUPITER Booster, one node / four GH200 GPUs per point.
- Points: `(mbs, gbs) = (4,32), (8,32), (16,64), (32,128)`.
- Measurement: steady-state `throughput per GPU (TFLOP/s/GPU)` after startup.

## Run Log

- 2026-07-30: The repaired multilingual source pairs passed complete
  Leonardo-to-JUPITER SHA-256 verification.
- 2026-07-30: SWA smoke `1104313` completed 500/500 iterations at roughly
  152 TFLOP/s/GPU with `mbs=4`, `gbs=32`, and two accumulation steps.
- 2026-07-30: Prepared the four-point microbatch throughput sweep. It is not
  committed, per user instruction.
- 2026-07-30: Re-enabled TorchDynamo and tested MCore local CUDA graphs with
  job `1105656` at the viable `mbs=8`, `gbs=32` point.
- 2026-07-30: Moved the throughput run to the e-sta-openeurollm JUPITER
  project/scratch paths. The first Dynamo retry exposed missing linkable CUDA
  driver libraries in Apptainer; adding the explicit `/opt/nvidia/lib` binds
  and `TRITON_LIBCUDA_PATH` resolved it. Gloo then required an explicit `ib0`
  interface on some allocated nodes.
- 2026-07-31: Enabled native vocabulary-parallel CE, gradient-accumulation,
  and masked-softmax fusions in the shared multilingual base, then tested the
  full-attention control at mbs 4 and 8.
- 2026-07-31: Profiled the steady-state TP=1 Liger MBS=32 path with rank-0
  PyTorch Profiler job `1135455`. The first profiler attempt `1135171` stopped
  before training because its 16 iterations were shorter than the inherited
  20-step LR warm-up; the profiling-only warm-up was changed to eight steps.
- 2026-07-31: Prepared an uncommitted optimization sweep: MBS=48, TE operation
  fuser, DDP communication overlap, explicit 8192-token Liger chunks, and
  MBS=64 with full recomputation. Also prepared a TP=4 sequence-parallel,
  native-vocabulary-CE MBS=32/GBS=32 control. Submission is waiting for the
  JUPITER login project mount to recover.

## Results

- `mbs=4`, `gbs=32`: job `1105114` completed at 150.6 TFLOP/s/GPU.
- `mbs=8`, `gbs=32`: job `1105113` completed at 160.9 TFLOP/s/GPU.
- `mbs=16`, `gbs=64` and `mbs=32`, `gbs=128`: jobs `1105116` and `1105115`
  OOM'd in vocabulary-parallel cross entropy, which materializes the padded
  262,272-token logits in fp32.
- Dynamo + `cuda_graph_impl: local`, `mbs=8`, `gbs=32`: job `1105656`
  completed 100/100 iterations with no skipped or NaN iterations, 69.55%
  memory use, and 160.2 TFLOP/s/GPU at the final iteration (roughly 161
  TFLOP/s/GPU steady state). The log contained no Dynamo or graph-capture
  failure; it did emit a non-fatal AccumulateGrad stream-mismatch warning.
- Four-model comparison, `mbs=4`, `gbs=32`, seq=4096, DP=4, Dynamo + local
  CUDA graphs (all 100/100): full attention `1118733` = 160.8 TFLOP/s/GPU,
  98.3k tokens/s/GPU; SWA 7:1 `1118781` = 170.5, 104.2k; mLSTM 7:1
  `1118783` = 133.7, 98.2k; GDN 7:1 `1118729` = 126.0, 91.7k. Memory use
  ranged from 71.4% (full/SWA) to 76.1% (GDN). The initial `mbs=8` attempt
  OOM'd all four variants because local CUDA-graph pools left insufficient
  room for the 32 GiB fp32 vocabulary-logits temporary.
- Fused full-attention, `mbs=4`, `gbs=32`: job `1127752` completed 100/100
  at 186.3 TFLOP/s/GPU and 113.9k tokens/s/GPU (88.5% memory). This is a
  15.9% token-rate gain over the unfused full-attention control (`1118733`,
  98.3k tokens/s/GPU). The mbs=8 fused point `1127753` still OOM'd: native CE
  compiles an fp32 `(4096, 8, 262272)` temporary, requiring 32.02 GiB while
  only about 31 GiB is free.
- Liger fused LM-head/CE direct CUDA parity check: loss and gradients match
  the FP32 reference for the binary Megatron loss mask. The first benchmark
  attempt ran native CE because this sweep constructs `GPTModel`, not
  `HybridModel`; the adapter was then added to both classes. The initial MBS=8
  retry `1132060` consequently still OOM'd in the native path. The actual
  Liger jobs completed: MBS=4 `1132418` (100/100, validation loss 7.1180),
  peak allocated 8.96 GB / reserved 10.22 GB, 37.5 TFLOP/s/GPU and 22.9k
  tokens/s/GPU; MBS=8 `1132840` (100/100, validation loss 6.7633), peak
  allocated 15.45 GB / reserved 16.92 GB, 68.4 TFLOP/s/GPU and 41.8k
  tokens/s/GPU. Both are well below native's 49.17 GB peak allocation and the
  native MBS=8 OOM threshold, but substantially slower.
- Liger large-microbatch follow-up: MBS=16, GBS=64 job `1133872` completed
  100/100 with 28.43 GB peak allocated / 30.41 GB reserved, 117.9
  TFLOP/s/GPU and 72.0k tokens/s/GPU at the final iteration (validation loss
  6.7080). MBS=32, GBS=128 job `1134437` completed 100/100 with 54.39 GB peak
  allocated / 57.37 GB reserved, 174.9 TFLOP/s/GPU and 106.9k tokens/s/GPU
  (validation loss 6.8214). Both started within three seconds of submission.
- Liger MBS=32 rank-0 profile: job `1135455` completed and emitted
  `torch_profile/rank-0.json.gz` in its JUPITER output directory. At the
  180 TFLOP/s/GPU steady state, each step has 131,072 tokens. Liger 0.8.0
  therefore chooses a 256-token chunk (vocab/hidden factor 513) and loops 512
  times. The four captured steps contain 2,048 Liger CE kernels, 6,144
  output-head GEMMs, and thousands of fp32 add/copy kernels from chunked
  LM-head weight-gradient accumulation. Summed GPU-kernel durations (which
  may overlap across streams) are 75 ms/step for CE, 209 ms/step for the
  three head-GEMM families, and 496 ms/step for the leading fp32
  accumulation/copy families.
- Custom Liger CE chunk-size retry: JUPITER parity-validated an opt-in
  8,192-token fixed chunk implementation against FP32 loss and gradients.
  Job `1137454` (MBS=32, GBS=128) completed 100/100 at 317.8
  TFLOP/s/GPU and 194.2k tokens/s/GPU, with 61.69 GB peak allocated
  (64.94 GB reserved), versus 174.9 TFLOP/s/GPU for Liger's automatic
  256-token chunks. Its final validation loss was 6.9637. The direct MBS=48
  capacity follow-up `1137473` also completed 100/100, but at only 228.0
  TFLOP/s/GPU and 139.4k tokens/s/GPU; it reached 80.42 GB allocated and
  84.02 GB reserved. Thus the larger microbatch fits but is already
  memory-bandwidth/capacity limited rather than increasing utilization.
- Follow-up optimization sweep: the isolated Transformer Engine op-fuser
  point `1137571` completed at 177.4 TFLOP/s/GPU and the gradient
  reduce/parameter-gather overlap point `1137588` at 176.9 TFLOP/s/GPU.
  Neither improves on the automatic-chunk 174.9-TF baseline meaningfully, and
  neither is competitive with explicit 8,192-token Liger chunks. Native
  vocabulary-parallel CE was exercised with TP=4: MBS=32 `1137536` correctly
  launched after adding `CUDA_DEVICE_MAX_CONNECTIONS=1`, but its 32.06-GB
  per-rank fused CE temporary OOM'd; MBS=16/GBS=32 `1137551` completed but
  reached only 70.4 TFLOP/s/GPU and 43.0k tokens/s/GPU because TP
  communication and two accumulation passes dominate at this scale.
- MBS=64 full-recompute follow-up: this Megatron branch requires explicit
  full/uniform recompute (not legacy `--recompute-activations`) and
  `cuda_graph_impl=full_iteration`. Liger 0.8 contained CUDA-graph-hostile
  scalar checks in both its forward and backward paths; the adapter replaces
  them with shape-derived/no-op-safe behavior, preserving the numerical
  parity test. Final run `1137738` reached 100/100 and validation at 281.9
  TFLOP/s/GPU and 172.3k tokens/s/GPU, with 15.21 GB peak allocated and
  19.79 GB reserved. Its outer Slurm launcher then hung after all ranks had
  torn down and was manually cancelled to release the GPUs; model training
  and validation had already completed. MBS=64 is therefore feasible but
  slower than the 317.8-TF MBS=32 explicit-chunk recipe due to recomputation.
- Post-change revalidation: after making the explicit Liger adapter safe for
  full-iteration CUDA graph capture, MBS=32/GBS=128 job `1137772` completed
  at 317.2 TFLOP/s/GPU and 193.9k tokens/s/GPU (validation loss 6.7635),
  confirming that the graph-safe adapter preserves the winning throughput.
  `jpbo-001-48` is excluded from this throughput sweep after repeatedly
  exposing zero CUDA devices inside the working container.

## Interpretation

- 2026-07-31: Added a four-architecture Liger throughput gate before the
  requested 6B-token comparison. `throughput_four_models_liger_0.1B.yaml`
  reuses the existing full-attention, 7:1 mLSTM, 7:1 GDN, and 7:1 SWA points,
  but uses the Liger-capable linear-attention container, TP=1, MBS=32,
  GBS=128, and an explicit 8,192-token LM-head chunk. Each point is a
  100-iteration (about 52M-token) benchmark with an isolated blend-index
  cache. Gate: all four must complete and reach at least 300 TFLOP/s/GPU
  before submitting any 6B-token training run.

- 2026-07-31: Began an opt-in TP=1 integration of Liger-Kernel's chunked
  fused LM-head plus cross-entropy path. Unlike Megatron's native/TE CE
  fusions, this avoids materializing the full vocabulary logits. The adapter
  keeps Megatron's per-token loss contract so the existing loss mask remains
  authoritative. It is intentionally rejected for TP>1 until a
  vocabulary-parallel implementation is added. Correctness gate:
  `scripts/test_liger_fused_linear_ce.py`; JUPITER benchmark config:
  `config/experiments/architecture_scaling_variants/multilingual/throughput_liger_fused_lmhead_0.1B.yaml`.

- Liger successfully removes the output-loss memory bottleneck and enables
  MBS=8 at TP=1. Its small-chunk path is slow (MBS=8 reaches only 41.8k
  token/s/GPU versus 113.9k for native fused CE at MBS=4), but this amortizes
  strongly: MBS=32 reaches 106.9k token/s/GPU without an OOM. It should remain
  opt-in for the four-model speed comparison until a direct full-attention
  MBS=32 control is available; the required GBS=128 also changes its optimizer
  semantics from the GBS=32 loss-comparison runs.

- The profile identifies the generic Liger chunk loop—not RMSNorm—as the
  remaining dominant limitation. The active model is already using
  Transformer Engine, FlashAttention, fused RoPE, fused RMSNorm+QKV/FC1,
  fused SwiGLU, native softmax and gradient-accumulation fusions, plus local
  CUDA graphs. `use_transformer_engine_op_fuser` remains disabled and is worth
  measuring, but it cannot eliminate Liger's 512 LM-head chunks. MBS=64 is
  not viable without activation recomputation: observed peak allocation rises
  from 28.43 GB at MBS=16 to 54.39 GB at MBS=32, projecting to about 106 GB at
  MBS=64 on an 87.5-GiB GPU. MBS=48 is the largest plausible no-recompute
  point (about 80 GB projected) and would reduce the chunk-loop count to 384.

Local CUDA graphs and Dynamo are compatible with this workload, but did not
produce a material throughput gain over the eager `mbs=8` baseline. The next
meaningful batch-size test is TP=2 with sequence parallelism, which halves the
per-rank vocabulary-logits activation that causes the `mbs=16` OOM.

The four-model result shows that the shared 262,272-token vocabulary/loss path
dominates a substantial part of end-to-end time: architecture-only token-rate
differences are modest despite their different TFLOP accounting. The fused
execution path is now the default and is a material, semantics-preserving win.
Larger TP=1 microbatches remain constrained by the fp32 CE workspace; profile
the LM head/loss before attempting TP, which is expected to add communication
overhead at this small model scale.

## Related

- Previous: `experiment/002_7to1_fullattn_mlstm_gdn/`
- Next: update the 6B-token comparison only if the chosen microbatch preserves
  its intended global batch and passes the full distributed smoke.
# 2026-07-31 — Four-model Liger gate baseline and recurrent-hybrid retry

- The non-6B, 100-iteration TP=1 Liger gate completed cleanly: FullAttn 324,
  SWA 355, mLSTM 260, and GDN 232 TFLOP/s/GPU at MBS=32 / GBS=128. The latter
  two do not satisfy the >300 TFLOP/s/GPU launch condition, so the 6B-token
  runs remain unsubmitted.
- Added `throughput_rnn_hybrids_liger_recompute_0.1B.yaml`: targeted mLSTM and
  GDN retries at MBS=64 / GBS=256, one-layer full recomputation, and
  full-iteration CUDA graph capture. This holds architecture, sequence length,
  precision, Liger head, and dataset fixed while testing whether freed
  activation memory can increase occupancy enough to clear the gate.
- The retry also completed numerically cleanly but regressed to mLSTM 235.5 and
  GDN 195.7 TFLOP/s/GPU. Recompute therefore is not a viable way to obtain the
  required hybrid throughput. After all ranks and validation had completed,
  Slurm wrappers 1138570/1138571 were cancelled only to release their hung
  outer allocations; their benchmark artifacts are complete.

# 2026-07-31 — Production 6B-token four-model launch

- User authorized four 6B-token training stages and their four dependent
  `datamix4-val` evaluation stages. The final comparison preserves the selected
  optimization recipe exactly: GBS=128 and LR=5e-4.
- Updated the production config to the validated one-node DP=4/MBS=32 geometry,
  local CUDA graphs, explicit 8192-token Liger chunks, and the current
  Liger+FLA/mLSTM container. This changes neither the global batch nor token
  order; it replaces the old four-node/MBS=4 execution geometry with the
  measured high-throughput per-GPU path.
- Added completed throughput results as inline comments to the four-model gate,
  the rejected recurrent recompute retry, and the prior full-attention
  optimization sweep configs. No commit was made, per user instruction.
- The first non-interactive DAG submit was blocked before job creation because
  the old output group contained `current.log` symlinks into the retired
  projectnucleus scratch mount. The production config now uses a fresh
  `architecture_scaling_variants_multilingual_7to1_liger_mbs32` group so the
  eval-stage DAG conditions can be evaluated without touching stale artifacts.
- Submitted the fresh eight-point DAG successfully: training jobs are SWA
  `1140566`, FullAttn `1140567`, GDN `1140568`, and mLSTM `1140569`; the four
  `eval6BT` jobs remain checkpoint-gated until their corresponding
  `iter_0011445` directory is present.

# 2026-07-31 — First 6B batch crashed at eval; fixed and resubmitted

- The first four training jobs (SWA 1140566, FullAttn 1140567, GDN 1140568,
  mLSTM 1140569) all FAILED at iteration 1600 — the first periodic-eval step —
  after ~20 min each. Every rank died with
  `TypeError: 'NoneType' object is not an iterator` in `get_batch`. Training
  was healthy up to that point (FullAttn 323 TFLOP/s/GPU, loss ~3.9).
- Root cause: the `final6BT` training stage uses `split: 100,0,0` (no
  validation data), but `eval_interval` (204800//128 = 1600) and `eval_iters`
  (100) were inherited from `ml_base`. Megatron sets
  `do_valid = valid_dataloaders is not None and (full_validation or eval_iters
  > 0)` (`training.py:4283`), so `eval_iters=100` forced `do_valid=True`
  despite the empty valid dataloader; the eval at iter 1600 dereferenced a
  `None` iterator.
- Fix: set `backend.megatron.eval_iters: 0` on the `final6BT` stage in
  `fullattn_mlstm7_gdn7_0.1B_50BT.yaml`. This makes `do_valid` False and
  disables in-loop eval. The `eval6BT` stage keeps its own non-zero `eval_iters`
  (204800//gbs = 1600) and runs the `datamix4-val` blend after annealing, so
  validation coverage is unchanged. Verified in the rendered sbatch:
  final6BT `--eval-iters 0`, eval6BT `--eval-iters 1600`.
- Resubmitted the eight-point DAG: training jobs FullAttn `1143970`, GDN
  `1143971`, mLSTM `1143972`, SWA `1143973`; eval stages remain checkpoint-gated
  on `iter_0011445`. No commit was made, per user instruction.
- Confirmed healthy ~26 min in: all four passed iteration 1600 with zero
  NoneType/Traceback/OOM/FATAL errors and zero nan/skipped iters. Steady-state:
  FullAttn iter 2130 loss 3.66 @ 306; mLSTM iter 2170 loss 3.67 @ 260; GDN iter
  1840 loss 3.76 @ 228; SWA iter 2440 loss 3.59 @ 353 TFLOP/s/GPU.

# 2026-07-31 — Second crash (wandb artifact cache) + mLSTM igate-init sweep

- The post-eval-fix resubmission (1143970-73) trained cleanly past iter 1600 but
  all four FAILED at iter 4000 — the first checkpoint save (save_interval =
  512000/128 = 4000). The checkpoint itself persisted (iter_0004000 on disk);
  the crash was in the post-save hook `wandb_utils.on_save_checkpoint_success`,
  which logs each checkpoint as a W&B model artifact and initialises W&B's
  artifact file cache at $HOME/.cache/wandb — the same read-only home .cache
  that broke the uv cache/logger dir. Error:
  "PermissionError: Unable to write to .../.cache/wandb/artifacts".
- Resolution: user repointed the ~/.cache symlink to the writable project dir
  (/e/project1/e-sta-openeurollm/poeppel1/.cache). No config change (a
  WANDB_CACHE_DIR redirect was drafted then reverted at user request).
- Resubmitted the four (1150175 fullattn, 1150176 gdn, 1150177 mlstm, 1150178
  swa); they auto-resumed from iter_0004000 (--load == --save) and ran past the
  save with the wandb hook now succeeding (fullattn iter 4720, loss 3.25, 321
  TFLOP/s/GPU).
- Preliminary science (user obs): training-loss curves of the four architectures
  are ~indistinguishable at later steps; mLSTM lags slightly with anomalous
  behaviour near ~1k steps, suspected to be the very low default input-gate bias
  init (-10.0) keeping the input gate shut early.
- Implemented `mlstm_igate_bias_init` as a first-class knob in the Megatron
  branch (v0.18.2-openeurollm-mlstm): added the field to TransformerConfig,
  made MLSTMLayer resolve `igate_bias_init` from config when not passed
  (mirroring mlstm_gate_soft_cap), so Megatron auto-derives --mlstm-igate-bias-init.
  Regenerated oellm_autoexp/backends/megatron/{config_schema,cli_metadata}.py.
- New sweep config `multilingual/mlstm7_igate_sweep_0.1B.yaml`: 7:1 mLSTM+NoPE,
  identical 0.1B/6B recipe (GBS=128, LR=5e-4), sweeping igate init over
  {-12,-10,-8,-3} (group architecture_scaling_variants_multilingual_mlstm7_igate_sweep,
  wandb dense_multilingual_scaling_mlstm7_igate_sweep). Submitted: igm12 1150294,
  igm10 1150292, igm8 1150293, igm3 1150295 (eval stages checkpoint-gated).
  Verified each run's args dump carries the intended value.

# 2026-08-01 — Config cleanup + speed-optimized main scaling sweep + geo smokes

- Removed 19 outdated pre-Liger configs (swa5_*, swa7_*, gdn5_*, mlstm5_*,
  gdn_base, mlstm_base, tiny smokes) — superseded by the new liger main_* set.
  Throughput probe configs kept as the optimization record.
- Built the speed-optimized 4-architecture main sweep on the validated liger
  setup (jupiter_liger, 1 node/MBS32, Liger fused CE): main_7to1_base.yaml +
  main_{fullattn,swa7,gdn7,mlstm7}_0.1B_{low,high}.yaml. WSD ladders (stable +
  3 cooldowns); dry-run validated (4 nodes/3 edges; correct arch flags, GA1/GA2
  geometry, branch iters).
- mLSTM uses mlstm_igate_bias_init = -8.0 (best of the igate sweep) everywhere.
- GQA note: at 0.4B/0.9B, mLSTM pins linear_num_key_heads to aux.heads (k==v);
  GDN keeps GQA (k=groups, v=heads). Verified against schema default (key=16).
- Geometry smokes submitted (100-iter, 4 arch each) to lock MBS/nodes/TP for
  the larger ladders: geo_smoke_0.2B (mbs16, 1158013-16), geo_smoke_0.4B (mbs8,
  1158017-20), geo_smoke_0.9B (mbs8, 1158021-24). All 12 RUNNING. GDN/mLSTM
  memory is the binding constraint (GDN was 99.7% mem at 0.1B/MBS32).

## Geometry smoke results (2026-08-01) — TFLOP/s/GPU at 100 iters, 1 node/TP1

| size | MBS | fullattn | swa7 | mlstm7 | gdn7 |
|------|----:|---------:|-----:|-------:|-----:|
| 0.2B | 16  | 334.1 | 377.0 | 296.9 | 263.7 |
| 0.4B |  8  | 352.4 | 395.2 | 300.1 | 270.2 |
| 0.9B |  8  | 389.5 | 421.6 | 357.3 | OOM   |

- fullattn / swa7 / mlstm7 fit and run well at these geometries across all
  sizes -> lock 0.2B@MBS16, 0.4B@MBS8, 0.9B@MBS8 (1 node, TP1) for them.
- GDN is the memory-binding architecture: fine at 0.2B/0.4B (MBS8/16), but OOM
  at 0.9B/MBS8/TP1 (Triton CUDA OOM). Retry submitted at 1 node gbs16:
  A) TP2/MBS8 (job 1158618), B) TP1/MBS4 (job 1158617) — pick the fitting one
  with higher Tok/s/GPU for the 0.9B GDN ladder.
- GA is throughput-neutral, so production gbs (128/256/512/1024) just scales GA
  on the locked MBS.

### GDN 0.9B geometry resolved (2026-08-01)

- TP1/MBS4 (job 1158617): FITS, 312.9 TFLOP/s/GPU, mem 0.668 (67%). -> use this
  for the GDN 0.9B ladder (full/swa/mlstm stay at MBS8).
- TP2/MBS8 (job 1158618): FAILED, but only because CUDA_DEVICE_MAX_CONNECTIONS
  was not set to 1 (Megatron requires it for TP/CP) — not a GDN/kernel limit.
  Unnecessary to pursue since MBS4/TP1 works.

## LOCKED geometry for the main scaling ladders (1 node, TP=1, per GPU)

| size | fullattn/swa7/mlstm7 MBS | gdn7 MBS | notes |
|------|-------------------------:|---------:|-------|
| 0.1B | 32 (validated 6BT)       | 32       | GA scales gbs 128/256 |
| 0.2B | 16                       | 16       | all four fit |
| 0.4B |  8                       |  8       | all four fit |
| 0.9B |  8                       |  4       | GDN halves MBS (OOM at 8) |

Production gbs (128/256/512/1024) just scales GA on the locked MBS; more nodes
optional to cut GA/wall-time (throughput-neutral for the science).

# 2026-08-01 — Single-file nested scaling sweep (main_scaling_all.yaml)

- Consolidated the whole 4-architecture scaling grid into ONE nested-sweep file
  `main_scaling_all.yaml`: sweep = architecture (4) x (size,tier) ladder (7);
  each ladder = 1 stable + one WSD cooldown per budget. 128 points, 100 edges,
  dry-run validated. Guarantees every architecture runs identical stages/cells.
- Removed the interim per-arch files (main_7to1_base + main_{arch}_0.1B_{low,high})
  as superseded; kept geo_smoke_* and main_scaling_all.
- Framework fix (oellm_autoexp/hydra_staged_sweep/dag_resolver.py): a single file
  with multiple stable+cooldown ladders collapsed siblings (the stage-mask hides
  the whole ladder dimension), so a cooldown could branch from the WRONG ladder's
  stable (proved: 0.1B_high gbs256 cooldown loaded the gbs128 stable). Fix: when
  >1 sibling matches, prefer the one sharing the most parameters (same
  size/gbs/lr/variant => same ladder). Backward-compatible (single-ladder files
  have exactly one match). Updated test_multiple_siblings_warning_log; all 162
  hydra_staged_sweep tests pass. (Pre-existing unrelated failure:
  test_config_feasibility[throughput_sweep_qwen3_1.7b] — legacy_tokenizer absent
  from schema at HEAD too.)
- Config gotcha: unescaped ${aux.heads} in a sweep-group override resolves early
  against the base placeholder; the arch group's linear head counts are escaped
  (\${aux.heads}) so they bind the ladder's size. Verified: mLSTM/GDN heads are
  16 at 0.4B/0.9B, 8 at 0.1B/0.2B; GDN keeps k=8/v=16 GQA.
- Locked geometry baked into the file: MBS uniform per size (0.1B 32, 0.2B 16,
  0.4B 8, 0.9B 4), nodes 1/1/1/2/2/4/4 per tier (throughput-only; GA scales gbs).

# 2026-08-01 — Mamba2 hybrid added (5th architecture)

- Mamba2 is NOT an experimental_attention_variant option upstream (only
  gated_delta_net/mlstm/dsa) and its MambaMixer needs mamba-ssm + causal-conv1d,
  absent from jupiter_liger. Two-part integration:
  1. Megatron adapter: megatron/core/ssm/mamba2_attention.py — Mamba2Attention
     wraps the native MambaMixer to fit the TransformerLayer self_attention slot
     (fused input-LN via column_parallel_layer_norm_linear in_proj, returns
     (output,bias), ignores mask/RoPE). Registered get_mamba_module_spec +
     dispatch 'mamba' + is_linear_attention_variant + config Literal 'mamba'.
     Schema regenerated (mamba_state_dim 128, mamba_head_dim 64, mamba_num_groups
     8 already present). Import-validated.
  2. Container: built MegatronTraining-JUPITER-linattn-mamba_aarch64_202608010917.sif
     (def: container/megatron/MegatronTraining-JUPITER-linattn-mamba.def.in) =
     jupiter_liger + mamba-ssm 2.3.2.post1 + causal-conv1d. All five archs now
     share ONE container. Config: config/container/jupiter_liger_mamba.yaml.
     (Build-time OSError writing ~/.tilelang was harmless — /e/home is writable
     at runtime; pre-flight import of mamba2 ssd_combined kernel + causal_conv1d
     succeeded in-container.)
- 7:1 Mamba2 + NoPE (experimental_attention_variant=mamba, linear_attention_freq
  8; expand 2 -> 16 heads at 0.1B). Added to the 6B comparison
  (fullattn_mlstm7_gdn7_0.1B_50BT.yaml) and the single-file main_scaling_all.yaml
  (now 5 archs), both switched to jupiter_liger_mamba.
- Submitted the Mamba2 6B run via mamba2_0.1B_50BT.yaml (mamba2-only, same group/
  recipe so it joins the completed four): train 1170998 (+ checkpoint-gated eval).
  Validating the adapter end-to-end.

# 2026-08-01 — Native Mamba2 hybrid speed reference

- Built config/experiments/.../throughput_native_mamba2_0.1B.yaml: native
  Megatron HybridModel via pretrain_hybrid.py (backend.launcher_script override),
  Nemotron-H pattern "M-M-M-M-*-M-M-M-M-*-M-M-M-M-" (28 layers: 12 Mamba, 14 MLP,
  2 attention), 0.1B dims, same MBS32/GBS128/Liger/mamba-container geometry.
- Two required args discovered by running: launcher_script=pretrain_hybrid.py,
  and --spec megatron.core.models.hybrid.hybrid_layer_specs hybrid_stack_spec
  (HybridModel needs an explicit stack spec). Job 1171547.
- Result (steady state, 1 node MBS32): 203M params, ~246 TFLOP/s/GPU,
  ~189k Tok/s/GPU, mem 71%.
- vs the GPT-block Mamba2 adapter (job 1171140, 7:1, ~206M params): ~234 TFLOP/s,
  ~172k Tok/s, mem 76%. Native is ~10% faster in Tok/s and lighter on memory.
- CAVEAT (not apples-to-apples): the native config has 12 Mamba mixers vs the
  adapter's 14, so the delta mixes (a) fewer expensive SSD mixers and (b) leaner
  native wiring (no self_attn_bda / fused-LN-in-attn-slot / per-layer MLP
  coupling). A pure wiring comparison would need a mamba-mixer-count-matched
  native config (user chose the canonical Nemotron-H recipe instead).

# 2026-08-01 — dag_resolver bug fixed at the root (expander stage-detection)

- Replaced the earlier parameter-matching band-aid in dag_resolver with the
  correct root fix in the sweep EXPANDER.
- Cause: the nested-group branch of expander._expand_group used
  _nested_has_stage_key, which flags a group-path position as a "stage axis" if
  a `stage` key exists ANYWHERE below it (look-down). An outer per-(size,gbs)
  ladder selector merely CONTAINS stable+cooldown sub-sweeps, so it got
  mis-flagged and masked out of the sibling match_key — collapsing distinct
  ladders. A gbs256 cooldown then matched (and loaded) the gbs128 stable.
- Fix: a position is a stage axis iff `stage` is introduced AT that level (its
  own defaults / a direct config), OR inherited from an ancestor stage context.
  Threaded an `in_stage_context` flag downward through _expand_group; nested
  configs now use the same _has_stage_key(merged-defaults) test as leaves.
  Removed the now-unused _nested_has_stage_key.
- dag_resolver: removed the param-overlap tie-break; a >1 sibling match now
  RAISES (fail-loud) — with the expander correct this never fires for valid
  configs, but it can never again silently branch from the wrong stable.
- Validation: 163 hydra_staged_sweep tests pass (+1 new regression test
  test_multiladder_cooldown_branches_from_own_stable asserting each cooldown
  edges from its own-gbs stable; updated test_multiple_siblings_raises for the
  fail-loud). main_scaling_all re-dry-run WITHOUT the band-aid must still give
  160 pts / 125 edges with correct cross-size (gbs512 0.2B vs 0.9B) links.

# 2026-08-01 — mLSTM causal conv integrated (GDN-style, CP-ready) + 6B runs

- Added a short depthwise causal conv over q,k,v to the Megatron mLSTM
  (megatron/core/ssm/mlstm.py), gated by new config `mlstm_conv1d`; width =
  linear_conv_kernel_dim. Mirrors GatedDeltaNet exactly: nn.Conv1d
  (depthwise, TP partition_dim=0), applied in forward AFTER the CP all-to-all
  (full-sequence head-sharded regime) so it is CP-safe — weight CP-sliced at
  runtime via get_parameter_local_cp(split_sections=[qk,qk,v]); FLA causal_conv1d
  (silu) fast path + torch F.conv1d deterministic path. Added the conv term to
  the mLSTM FLOP estimator; regenerated schema (--mlstm-conv1d).
- 6B reference runs (mlstm7_conv_0.1B_50BT.yaml, igate -8, same group/recipe as
  the other 6B points): conv kernel 4 (job 1174837) and 8 (job 1174838).
- Runtime-validated (steady state, 1 node/MBS32):
  * k4: 234.8 TFLOP/s, 172.3k Tok/s/GPU, mem 0.94, loss falling, 0 nan.
  * k8: 224.2 TFLOP/s, 164.4k Tok/s/GPU, mem 0.94, loss falling, 0 nan.
- Confirms width 8 WORKS via FLA's Triton causal_conv1d (Tri Dao's CUDA kernel,
  used by Mamba, is capped at 4). k8 is ~5% slower in Tok/s than k4.
- Note: the conv raises memory notably (0.94 vs ~0.76 for no-conv mLSTM at the
  same geometry) — watch it at larger sizes / MBS (GDN-like OOM risk).
- Both train to iter_0011445 (~2 h); datamix4-val evals are checkpoint-gated and
  will join the 6B comparison alongside fullattn/gdn7/swa7/mamba2 + no-conv mlstm7.

# hyb100m: 100M-active hybrid MoE, WSD ladder for 4 optimizers

How to run `11_wsd_ladder.yaml`, why it is set up the way it is, and what went wrong before.
First-time setup (accounts, container, Megatron, venv) is in `../README.md`; do that first.

## TL;DR

```bash
# once: Liger kernel into your pyext (see "Setup" below), then on a login node:
tmux new -s wsd                      # note the login node name (hostname): tmux lives there
cd $OPT_SCALE_BASE/oellm-autoexp
source $OPT_SCALE_BASE/venvs/autoexp/bin/activate
python scripts/run_autoexp.py --config-name experiments/laingsam1/opt_scale/hyb100m/11_wsd_ladder --dry-run
python scripts/run_autoexp.py --config-name experiments/laingsam1/opt_scale/hyb100m/11_wsd_ladder
# detach: Ctrl-b d      re-attach: tmux attach -t wsd  (on the SAME login node)
```

No `--submit-and-exit` here: the monitor must keep running (it releases the cooldowns and
restarts crashed jobs). Expected cost ~55 GPU-h per optimizer, ~220 GPU-h for all 4.

## What the config runs

Model: Swagatam's hybrid block (3 Gated DeltaNet + 1 gated attention, each followed by an MoE),
640 wide x 4 blocks, 96 experts top-4 -> 101M active / 1.0B total parameters. Data: v8ml mix.
Batch: **256 sequences x 4096 tokens = 1.05M tokens per step** (global batch size 256).

Per optimizer, 4 jobs (16 in total, indices `0-3 adam, 4-7 muon, 8-11 angular_muown,
12-15 muown_classical`):

| Job | Steps | Tokens | What |
|---|---|---|---|
| stable | 0 -> 6168 | 6.5B | 40-step warmup, then constant lr; saves the branch points 1542 / 3084 / 4626 / 6168 |
| decay1x | 1542 -> 1928 | 2.0B (= 20N, 1x Chinchilla) | loads stable @1542, linear lr cooldown to 0 over the last 386 steps |
| decay2x | 3084 -> 3855 | 4.0B (2x) | cooldown over 771 steps |
| decay4x | 6168 -> 7710 | 8.1B (4x) | cooldown over 1542 steps |

The decays wait for their checkpoint (`job.start_condition`) and are cancelled if the stable
job dies with a fatal error (`job.cancel_condition`). Both are checked by the monitor.

Optimizer settings (agreed 2026-11-13): AdamW lr 1e-3, beta1 = beta2 = 0.95, wd 0.1;
Muon / AngularMuown lr 5e-3 (LUMI settings); muown_classical lr 1e-3, wd 0.

## History: what went wrong in the first run (2026-11-14), and what was changed

The first stable AdamW run (old settings: micro-batch 4, eval every 500, checkpoint every 2000)
**crashed after 5.5 h (~22 GPU-h) without a single checkpoint**. Investigation:

**1. It crashed because the node ran out of CPU RAM, not GPU memory.** On GH200 the GPU
memory and CPU memory are one address space; when host RAM runs out the Linux OOM killer
SIGKILLs a process, and Megatron's "mem usages" climbs beforehand. Two causes:

- *Building the data index inside the job* (first run with a new batch/steps combination,
  ~20 min on rank 0) left rank 0 at 95-135 GB of RAM. **Fixed:** the index is now cached in
  `$OPT_SCALE_BASE/cache/megatron_index` and reused. With the cache, rank 0 starts at ~5 GB.
- *PyTorch data-loader worker processes grow* by ~1-1.5 GB per step summed over the node
  (test 12: node RAM in use 19 GB -> ~300 GB over 200 steps, each worker ~21 GB; growth was
  slowing down by the end). **Not fixed, contained** (see "Safety" below). This is the main
  open issue; every run also gets 10-30 % slower as RAM fills, probably the same cause.

**2. Throughput was poor** (11.5 s/step, ~40 TFLOP/s per GPU, 4 % MFU). Smoke test 12
(`../smoke_test/12_throughput_memory.yaml`, 8 variants x 200 steps, 1 node, AdamW) measured:

| Variant | s/step | TFLOP/s/GPU | MFU | GPU mem | loss @200 |
|---|---|---|---|---|---|
| control (old settings, micro-batch 4) | 11.5 | 39.6 | 4.0 % | 70 % | 5.22 |
| 2 data workers instead of 4 | 11.5 | 39.6 | 4.0 % | 70 % | 5.25 |
| fused RMSNorm / GDN out-norm kernels | 11.4 | 40.0 | 4.0 % | 69 % | 5.17 |
| Liger, micro-batch 4 | 10.7 | 42.5 | 4.3 % | 45 % | 5.14 |
| DeepEP dispatcher, experts split over 4 GPUs | 9.2 | 49.3 | 5.0 % | 64 % | 5.11 |
| HybridEP dispatcher, experts split over 4 GPUs | 7.9 | 57.5 | 5.8 % | 64 % | 5.12 |
| Liger, micro-batch 8 | 7.4 | 61.9 | 6.3 % | 61 % | 5.14 |
| **Liger, micro-batch 16 (used now)** | **4.7** | **97.5** | **9.9 %** | 91 % | 5.18 |

(median over steps 30-200; MFU = TFLOP/s / 989; loss differences are run-to-run noise.)

**Why Liger + micro-batch 16 is 2.5x faster, and why it does not change the training:**
- The *micro-batch* is how many sequences one GPU processes per forward/backward pass. The
  *global* batch (what the optimizer sees, 256) is unchanged: before 4 GPUs x 16 passes x 4
  sequences, now 4 GPUs x 4 passes x 16 sequences. Same gradient, same update.
- A 100M model's matrices are tiny, so at micro-batch 4 the GPU mostly waits (kernel launches,
  expert routing, syncs). Bigger micro-batches keep it busy.
- Micro-batch 16 did not fit before: the 262k-vocabulary output layer would materialise
  16 x 4096 x 262144 logits (~34 GB). Liger's fused linear + cross-entropy computes the loss in
  chunks, so that tensor never exists. Liger is the enabler; the micro-batch is the speed-up.
- Liger alone at micro-batch 4 is only ~7 % faster.

## What `11_wsd_ladder.yaml` contains now (vs the first run)

Same model, data, batch size, learning rates, schedule, step counts -> same training. Changed:

| Setting | Before | Now | Why |
|---|---|---|---|
| micro-batch | 4 | 16 | 2.5x speed (needs Liger) |
| Liger fused LM-head CE | off | `--liger-fused-linear-cross-entropy --liger-fused-linear-cross-entropy-chunk-size 8192` (in `backend.aux.arch_args`) | makes micro-batch 16 fit |
| data workers | 4 | 2 | same speed, fewer growing worker processes |
| checkpoints | every 2000 + 1542/3084/6168 | every 514 steps (~45 min), `save_retain_interval 1542` keeps only 1542/3084/4626/6168 + the latest | a crash loses <= 45 min |
| clean exit | none | `exit_duration_in_mins: 690` (saves and exits at 11.5 h) | never killed by the 12 h limit mid-save |
| validation | every 500 steps, 20 batches | every 240 steps, 10 batches (10.5M tokens, ~15 s) | 26 points on the stable curve; same total validation samples -> cached index reused |
| monitoring | - | node RAM sampler -> `<output dir>/node_mem_<jobid>.log`; `log_memory_to_tensorboard`, `log_params_norm` | see memory problems coming |
| auto-restart | time limit, crash, NaN, ... | + `host_oom` (data worker killed / SIGKILL) and `clean_exit` events | all capped at 3 submissions per job |
| time limits | stable 12 h, decays 2.5 / 4.5 / 8 h | stable 12 h, decays 1.5 / 2.5 / 4 h | jobs are 2.5x shorter |

Megatron also runs one final validation at the end of every job (the numbers for the scaling
fit). The monitor's "training finished" trigger does not cancel the job, so that final
validation and save always run.

## Setup (in addition to `../README.md`)

Liger kernel 0.8.0 goes into **your own** `$OPT_SCALE_BASE/pyext` (already first on
`PYTHONPATH` in the configs). The container is read-only and is not touched; nobody else is
affected. `setup_jupiter_hybrid.sh` does this too; manually:

```bash
SIF=/e/project1/e-sta-openeurollm/container/MegatronTraining-JUPITER-te218-fa3_aarch64_202608280932.sif
apptainer exec $SIF /opt/venv/bin/python -m pip install --no-deps --target $OPT_SCALE_BASE/pyext liger-kernel==0.8.0
# check: must print "HAVE_LIGER True" and a path under your pyext
apptainer exec --nv --env PYTHONPATH=$OPT_SCALE_BASE/pyext:$OPT_SCALE_BASE/NVIDIA-Megatron-LM-hybrid $SIF \
  /opt/venv/bin/python -c "import liger_kernel; from megatron.core.fusions.liger_fused_linear_cross_entropy import HAVE_LIGER; print('HAVE_LIGER', HAVE_LIGER, liger_kernel.__file__)"
```

Without Liger, Megatron silently falls back (`HAVE_LIGER = False`) and micro-batch 16 runs out
of GPU memory.

The data-index cache: the first job with a new (global batch x steps, eval) combination builds
it (~20-30 min on rank 0, all other ranks wait; RAM-hungry). If yours is empty, run
`--array-subset 0` (stable AdamW) alone first and start the rest once it is training.

## Running

1. `--dry-run`, check one `job.sbatch` (under `$OPT_SCALE_BASE/experiments/opt_scale/hyb100m/`):
   `--micro-batch-size 16`, `--liger-fused-linear-cross-entropy`, `--save-interval 514`,
   `--save-retain-interval 1542` (stable only), `--eval-interval 240`, your account and paths.
2. Start the monitor in tmux (TL;DR above). All 16 jobs, or one optimizer first with
   `--array-subset 0-3` (adam), `4-7` (muon), `8-11` (angular_muown), `12-15` (muown_classical).
   **Right after starting, also pre-build the cooldown data indexes** (one-off per `$OPT_SCALE_BASE/cache`,
   ~7 GPU-h; not needed again later with the same data / batch / eval settings):
   `python scripts/run_autoexp.py --config-name experiments/laingsam1/opt_scale/hyb100m/11b_prebuild_decay_index --submit-and-exit`.
   Otherwise the decay1x jobs of all optimizers start at about the same time (~2 h in) and
   each builds the same index in the same folder (race, and ~100 GB extra RAM on rank 0).
3. **To stop: Ctrl-C the monitor FIRST, then `scancel`.** Otherwise the monitor treats the
   cancel as a crash and resubmits.
4. Re-attaching: `tmux attach -t wsd` works only on the login node where tmux was started
   (`ssh <that node>` from another login node). Before concluding "no monitor is running",
   check that node, not just the one you are on.

## Checking a running job

```bash
squeue --me
L=$OPT_SCALE_BASE/logs; ls -t $L | head          # <job name>-<jobid>.out / .err
grep "iteration " $L/<job>-<jobid>.out | tail -3 # step time, mem usages, loss
grep "validation loss" $L/<job>-<jobid>.out | tail -3
# node RAM (the open issue): MemAvailable should not keep falling towards 0
grep MemAvailable $OPT_SCALE_BASE/experiments/opt_scale/hyb100m/<job name>/node_mem_<jobid>.log | tail -5
```

Expected: ~4.7-5 s/step (Muon-type a few % slower), "mem usages" ~0.9, a checkpoint every
~45 min. Total node RAM is ~480 GB; worry if MemAvailable drops below ~50 GB.

wandb runs offline (no internet on compute nodes). Upload later from a login node:
`wandb sync $OPT_SCALE_BASE/experiments/opt_scale/hyb100m/<job name>/wandb/wandb/offline-run-*`.
Tensorboard files are in `<job name>/tensorboard`.

Logged every step: train loss, lr, grad norm, params norm, MoE load-balancing loss / router
logit spread, TFLOP/s and tokens/s per GPU, GPU memory, skipped/NaN iterations; timers and
allocator memory in tensorboard; validation loss + perplexity at every evaluation.
Per MoE layer (as on LUMI, `moe_per_layer_logging` + `moe_expert_viability_metrics`): aux
loss, router scores, expert utilisation, expert viability. Not enabled:
`moe_masked_layer_validation` (draws extra validation batches the validation set is not sized
for -> runs out mid-run; and ~17x the cost of a normal evaluation).
wandb entity `openeurollm-project`, project `optimizer_scaling_laws` (applies on `wandb sync`).

## Cost

| | Steps | GPU-h (1 node = 4 GPUs) |
|---|---|---|
| stable | 6168 | ~33 (~8.5 h) |
| decay1x / 2x / 4x | 386 / 771 / 1542 | ~3 / ~5 / ~9 (incl. index build) |
| **per optimizer** | 8867 | **~55** (old settings: ~110) |
| all 4 optimizers | | ~220 |

For planning sweeps at 100M with these settings: a 1x-Chinchilla run (1928 steps) is ~10 GPU-h,
a full ladder ~55 GPU-h. Bigger models cost more per token; `../smoke_test/13_throughput_ladder.yaml`
measures that (250M / 500M / 850M active, 1-4 nodes).

Check what you used:
```bash
sacct -u $USER -S 2026-11-01 -X -n -o ElapsedRaw,AllocTRES%60 \
  | awk '{match($2,/gres\/gpu=([0-9]+)/,m); s+=$1*m[1]} END{printf "%.1f GPU-h\n", s/3600}'
```

## Open issues / next steps

- **Data-loader worker RAM growth** (above): contained by checkpoints + auto-restart, not
  fixed. Watch `node_mem_*.log` in the first long run; if it does not level off, test
  `num_workers: 1` and look at the dataset objects the workers copy.
- HybridEP (experts split over the 4 GPUs) was the second-biggest win and has not been
  combined with Liger + micro-batch 16 yet; together maybe ~12-15 % MFU.
- 10 % MFU is low in absolute terms because the model is tiny and the vocabulary is 262k
  (the output layer dominates the FLOPs); larger models in test 13 should do better.
- If you write new configs: keep global batch 256 x 6168 steps and validation
  `(train_iters // eval_interval + 1) x eval_iters = 260` to reuse the cached data index;
  anything else rebuilds it (submit one job alone first).

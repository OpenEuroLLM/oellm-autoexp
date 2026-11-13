# opt_scale smoke tests (JUPITER, GH200)

First runs of the optimizer scaling-law study on JUPITER: a Qwen3.5-style hybrid
attention MoE (3 Gated DeltaNet : 1 gated full-attention layer) on the 0.1B_ne sparse
rung, with AdamW, Muon and AngularMuown.

They check that the environment works, the hybrid model trains, all three optimizers
run, and multi-node communication works, and they give first throughput numbers for
budgeting the full ladder.

## ▶ Status / resume here (2026-11-11 evening)

**Done and working**
- 01 (Nemotron, 1 node, 6 jobs): all pass. 02 (Nemotron, 2 nodes, 4 jobs) and
  03 (Nemotron, 1-node diagnostics, 2 jobs): all pass. Results below.
- Multi-node scaling is flat (same tokens/s/GPU on 1 and 2 nodes).

**04 (full v8ml mix): job 0 DONE, jobs 1-5 submitted on the evening of 2026-11-11**
- Job 0 (gdn + adam) first attempt 2840915 FAILED after 13 min: rank 0 was still building
  the dataset indices when the other ranks hit the 0.1B rung's 10-minute
  `distributed_timeout_minutes`. Fixed (`distributed_timeout_minutes: 120` in
  `04_v8ml_single_node.yaml`).
- Job 0 rerun 2841590 COMPLETED (17 min 46 s, ~1.2 GPU-h): index build for the remaining
  datasets ~12 min, all 522 datasets (1044 train/valid index sets) now cached; then 100 steps,
  loss 12.40 -> 6.13, val 6.81 (step 50) -> 6.29 (step 100). Results in the 04 table below.
- Jobs 1-5 were submitted with (cap 5 x 2 = 10 GPU-h, no time override):
  ```bash
  cd /e/scratch/e-aif-2026sc04-211/users/slaing/oellm-autoexp
  source /e/scratch/e-aif-2026sc04-211/users/slaing/venvs/autoexp/bin/activate
  python scripts/run_autoexp.py --config-name experiments/laingsam1/opt_scale/smoke_test/04_v8ml_single_node \
    --array-subset 1-5 --submit-and-exit
  ```
- **Next step (2026-11-12):** check jobs 1-5 and fill the 04 table:
  ```bash
  sacct -u laing1 -S 2026-11-11T16:45 -X -o JobID%10,JobName%40,State,ExitCode,Elapsed
  ls -t /e/scratch/e-aif-2026sc04-211/users/slaing/logs/smoke_v8ml_*      # training output is in .out
  ```
  Look for: State COMPLETED, `iteration 100/100` in the .out log, finite falling loss.
  If a job failed, the reason is in the matching `.err` log.

**Open issues**
1. Intermittent Muon slowdown (see "Muon slowdown" below): ~half of Muon/AngularMuown runs
   lose ~30 % throughput in the optimizer step. Leading hypothesis: CPU/NUMA placement
   (no per-GPU CPU binding, `OMP_NUM_THREADS=288` per process). Next: a 1-node test with
   per-rank CPU binding.
2. 0.2B rung has 22 layers (not divisible by 4) -> needs 20 or 24 for the GDN 3:1 pattern.
3. v8ml data here is NOT decontaminated (see "Data"); ask where flag-decon-v2-agenttok lives
   (possibly `/e/data1/datasets/playground/mmlaion/oellm/`, ACL from jitsev1 still pending).
4. Index cache size: ~47 GB for the Nemotron runs; the full v8ml mix will add ~150-200 GB
   per (batch, seq, split, samples) configuration in
   `/e/scratch/e-aif-2026sc04-211/users/slaing/cache/megatron_index`. Clean up old sets.
5. v8ml data loading: in 04 job 0, ~10 % of steps were slow (3.8-5.0 s vs median 2.40 s),
   likely data-loader stalls reading from 522 files (cold cache). Watch in jobs 1-5; if it
   persists, try more `num_workers` (currently 4).

**Uncommitted:** everything under `config/experiments/laingsam1/opt_scale/` and
`config/backend/megatron/opt_scale/` on branch `optimizer_scaling_laws` (not pushed).
Megatron fork fix is committed locally on fork branch `jupiter/te214-dev-compat` (not pushed).

**GPU-hours used so far:** ~10.4 before jobs 1-5 (01 ~3, 02+03 ~5.3, 04 job 0 failed ~0.9 +
rerun ~1.2); jobs 1-5 add at most 10 (expected ~3).
Budget: 90K node-hours = 360K GPU-hours on `e-aif-2026sc04-211`.

## Setup (done once; `setup_jupiter.sh` is safe to re-run but see note)

| What | Where |
|---|---|
| Account / partition | `e-aif-2026sc04-211` / `booster` |
| Container | `/e/project1/e-ext-2025e02-108/container/nemo_26.04.sif` (torch 2.11, TE 2.14 dev build, fla 0.4.2) |
| Megatron fork | `/e/scratch/e-aif-2026sc04-211/users/slaing/NVIDIA-Megatron-LM-angular`, branch `jupiter/te214-dev-compat` = `oellm/v0.19-moe-lumi-angular-muown` @ `1f88af5d` + TE fix `2165e5755` |
| emerging_optimizers 0.2.0 | `/e/scratch/e-aif-2026sc04-211/users/slaing/pyext` (the container's 0.1.0 is too old; the fork then silently disables Muon) |
| autoexp env | `/e/scratch/e-aif-2026sc04-211/users/slaing/venvs/autoexp` (uv, Python 3.12) |
| Logs | `/e/scratch/e-aif-2026sc04-211/users/slaing/logs/<job>-<jobid>.{out,err}` (training output is in `.out`) |
| Outputs (tensorboard, offline wandb) | `/e/scratch/e-aif-2026sc04-211/users/slaing/experiments/opt_scale/smoke_test/<job>/` |
| Data index cache | `/e/scratch/e-aif-2026sc04-211/users/slaing/cache/megatron_index` |

Note: `setup_jupiter.sh` still pins `1f88af5d`; re-running it would check out the commit
WITHOUT the TE fix. Don't re-run it as is (or switch back with
`git -C <fork> switch jupiter/te214-dev-compat`).

Data groups needed: `e-sta-openeurollm` (full flag collection; for 04) and container access.
A shell opened before joining a group doesn't have it (`id -Gn` vs `id -Gn laing1`); log in
fresh, or use `sg <group> -c '...'`. Slurm jobs pick up the account's groups.

## Run order

From the repo root on a login node, with the autoexp env active:

```bash
D=config/experiments/laingsam1/opt_scale/smoke_test
E=experiments/laingsam1/opt_scale/smoke_test

bash $D/00_env_check.sh                                                     # env check on login-node GPU (free)

# Nemotron-only (done)
python scripts/run_autoexp.py --config-name $E/01_single_node --dry-run
python scripts/run_autoexp.py --config-name $E/01_single_node --array-subset 0 --submit-and-exit
python scripts/run_autoexp.py --config-name $E/01_single_node --array-subset 1-5 --submit-and-exit
python scripts/run_autoexp.py --config-name $E/02_multi_node --submit-and-exit
python scripts/run_autoexp.py --config-name $E/03_muon_diag_1node --submit-and-exit

# Full v8ml mix (in progress)
python scripts/run_autoexp.py --config-name $E/04_v8ml_single_node --dry-run
python scripts/run_autoexp.py --config-name $E/04_v8ml_single_node --array-subset 0 --submit-and-exit slurm.sbatch.time=02:00:00
python scripts/run_autoexp.py --config-name $E/04_v8ml_single_node --array-subset 1-5 --submit-and-exit
```

Always run job 0 of a sweep alone first: it builds the shared data-index cache once (parallel
jobs would each build it) and limits the cost if something is wrong.

## Jobs

| File | Data | Jobs | Nodes | What it checks |
|---|---|---|---|---|
| `01_single_node.yaml` | Nemotron | {gdn, full_attn} x {adam, muon, angular_muown} = 6 | 1 | env, hybrid model, all optimizers, baseline |
| `02_multi_node.yaml` | Nemotron | gdn+adam, gdn+muon, gdn+angular_muown, full_attn+muon = 4 | 2 | multi-node scaling + Muon diagnostics |
| `03_muon_diag_1node.yaml` | Nemotron | gdn+muon, full_attn+muon = 2 | 1 | Muon diagnostics at 4 GPUs |
| `04_v8ml_single_node.yaml` | full v8ml | same 6 as 01 | 1 | everything on the real mix |

All jobs: 100 iterations, global batch 64 sequences per node x 4096 tokens, eval at 50/100,
no checkpoints, 30-minute limit (04 job 0: 2 h via command-line override).
02 and 03 run with `timing_log_level: 2` and `TORCH_LOGS=recompiles` (diagnostics).

Hard GPU-hour caps (Slurm time limit, no auto-restart with `--submit-and-exit`):
1 node x 30 min = 2 GPU-h per job; 2 nodes x 30 min = 4 GPU-h per job.

## Data

**01-03: OpenEuroLLM Nemotron-CC, Flagship One copy**
`/e/fscratch/e-ext-2025e02-108/training/collection/flag/nemotron-cc-1.0/megatron-lm/`.
Only 8 shards are used, the ones whose `.bin` size matches their `info.json`:
`high/actual/shard_0000, 0003, 0005` and `medium-high/actual/shard_0000`-`0004`
(~770B tokens). Do NOT add the others: `high` 0001 is truncated, `high` 0002/0004 and
all `medium` shards are 0-byte files.

**04: full v8ml mix** via `../data/datamix_v8ml_flagfull_jsc.txt`
(Megatron `--data-args-path`), generated by `../data/make_datamix_v8ml_flagfull.py`:
- official v8ml weights (`oellm_testing/datamix/datamix_v8ml_jsc.txt`: 80 % v8 English +
  20 % flag-ml, tokenizer-256k-agent);
- flag sources pointed at the complete flag collection
  `/e/fscratch/e-sta-openeurollm/training_v2/training/collection/flag/` (needs group
  `e-sta-openeurollm`; 488/488 shards verified intact, ~16.8T tokens);
- v8-only sources (hftok, vtok, mv1sub) from `oellm_testing` (verified intact);
- 522/531 entries; 9 missing mv1sub instruct sets (1.95 % of the mix) dropped, weights rescaled.

Caveats: the flag part is the ORIGINAL collection, not the decontaminated
`flag-decon-v2-agenttok` copy the official mix points to. Validation uses the per-dataset
`split`; do NOT use `oellm_testing/v8ml-valid` with this mix (its held-out documents are
still in the original flag collection).

`data_integrity_check` is off in every config on purpose: it would hash all data files on
the login node and try to write a checksum file into the (read-only) data directories.

**Index build:** the first run of a new (data, seq, split, num_samples) configuration builds
per-dataset indices on rank 0 while the other ranks wait. Nemotron (8 datasets): minutes.
v8ml (522 datasets): ~25-30 min estimated (201 datasets in ~10 min). Needs a large
`distributed_timeout_minutes` (04 uses 120) and is cached afterwards.

## Results

### 01_single_node (Nemotron, 1 node / 4 GPUs, last-50-step means)

| Job | Slurm ID | loss@100 | val@50 | val@100 | s/step | TFLOP/s/GPU* | tokens/s/GPU | params (rank 0) |
|---|---|---|---|---|---|---|---|---|
| gdn / adam | 2835440 | 6.07 | 6.58 | 6.05 | 2.28 | 40 | 28.8k | 207,965,376 |
| gdn / muon | 2837088 | 5.19 | 5.79 | 5.14 | 3.20 ⚠ | 39 | 20.5k | 207,965,376 |
| gdn / angular_muown | 2837089 | 5.00 | 5.67 | 4.93 | 3.14 ⚠ | 40 | 20.8k | 207,965,376 |
| full_attn / adam | 2837090 | 6.66 | 7.17 | 6.65 | 1.99 | 54 | 32.9k | 205,672,960 |
| full_attn / muon | 2837087 | 5.50 | 6.30 | 5.46 | 1.99 | 54 | 33.0k | 205,672,960 |
| full_attn / angular_muown | 2837091 | 5.67 | 6.51 | 5.63 | 1.95 | 55 | 33.7k | 205,672,960 |

### 02_multi_node / 03_muon_diag_1node (Nemotron; level-2 timing on)

| Job | Slurm ID | Nodes | s/step | tokens/s/GPU | samples/s/GPU | forward-backward | rest of step (≈ optimizer) | grad sync |
|---|---|---|---|---|---|---|---|---|
| gdn / adam | 2838781 | 2 | 2.31 | 28.3k | 6.92 | | | |
| gdn / muon | 2838782 | 2 | 2.31 | 28.4k | 6.93 | 2.03 s | ~0.3 s | 0.016 s |
| gdn / angular_muown | 2838779 | 2 | 2.25 | 29.1k | 7.11 | | | |
| full_attn / muon | 2838778 | 2 | 3.11 ⚠ | 21.0k | 5.14 | 1.69 s | ~1.4 s | 0.015 s |
| gdn / muon (diag) | 2838784 | 1 | 2.27 | 28.8k | 7.04 | 1.98 s | ~0.3 s | 0.006 s |
| full_attn / muon (diag) | 2838783 | 1 | 3.08 ⚠ | 21.3k | 5.20 | 1.66 s | ~1.4 s | 0.006 s |

⚠ = hit by the intermittent Muon slowdown.
*Megatron's TFLOP/s formula does not model Gated DeltaNet and evidently excludes the
optimizer step; use tokens/s (or samples/s = tokens/s / 4096) for comparisons.

### Findings so far
- All 12 Nemotron runs pass; all three optimizers work on hybrid and full attention.
  Parameter counts (207.97M vs 205.67M on rank 0) confirm the GDN layers are built.
- Typical throughput (0.1B MoE, seq 4096): hybrid ~28.5-29k tok/s/GPU (~7.0 samples/s/GPU),
  full attention ~33k tok/s/GPU (~8.1 samples/s/GPU). Hybrid is ~13 % slower at seq 4096
  (expected at short context). 1 -> 2 nodes: no per-GPU loss.
- MFU is low (~40-55 TFLOP/s/GPU vs ~989 peak) as expected for a 0.1B model; the repo's
  JUPITER guide reports 255-285 TFLOP/s/GPU for a dense 1.7B.
- Losses are NOT an optimizer comparison (100 steps, 10-step warmup, AdamW at lr 2e-3 vs 5e-3).

### Muon slowdown
- 5 of 10 Muon/AngularMuown runs were ~0.9 s/step slower; AdamW never. It is NOT tied to
  the hybrid architecture (slow runs switched arch between rounds).
- The extra time is outside forward-backward and gradient sync, i.e. in the optimizer step
  (~1.4 s vs ~0.3 s). Forward-backward times are identical in slow and fast runs, so the
  GPUs themselves are not slower.
- Not torch.compile recompilation: `TORCH_LOGS=recompiles` shows no recompiles of Muon
  functions and no recompile-limit hits.
- Ruled out earlier: which weights Muon handles (2D non-embedding only), matrix sizes,
  batched vs per-matrix Newton-Schulz (AngularMuown batched, Muon not, both affected).
- Leading hypothesis (untested): CPU-side kernel-launch cost of Muon's many small
  matmuls, sensitive to CPU/NUMA placement: no per-GPU CPU binding and
  `OMP_NUM_THREADS=288` for each of 4 processes per node (oversubscription).
  Test: per-rank CPU binding to the GPU's socket, ~72 threads per rank.

### Failed attempts
- 2834927 (01 job 0): the fork passed `single_grouped_weight` to TE's `GroupedLinear`, which
  the container's TE 2.14.0 dev build lacks. Fixed in the fork (`jupiter/te214-dev-compat`).
- 2840915 (04 job 0): collective timeout while rank 0 built v8ml indices (see Status).

### 04_v8ml_single_node (full v8ml mix)

s/step = median of the last 50 steps (some steps are slow, see open issue 5).

| Job | Slurm ID | loss@100 | val@50 | val@100 | s/step | tokens/s/GPU | Notes |
|---|---|---|---|---|---|---|---|
| gdn / adam | 2841590 | 6.13 | 6.81 | 6.29 | 2.40 | ~27.3k | mean 2.85 s incl. slow steps (p90 3.8 s, max 5.0 s); includes ~12 min index build |
| gdn / muon | | | | | | |
| gdn / angular_muown | | | | | | |
| full_attn / adam | | | | | | |
| full_attn / muon | | | | | | |
| full_attn / angular_muown | | | | | | |

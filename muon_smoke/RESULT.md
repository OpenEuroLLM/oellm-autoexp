# Muon 50M smoke test — Capella (NVIDIA H100) progress

See `RESULT_lumi.md` for the original LUMI/ROCm result (PASS, 2026-07-15) — kept
as-is, not touched. This file tracks the Capella port under `NEW_CLUSTER_SETUP.md`.

## Verdict: RUNG 1 + RUNG C (mock) + RUNG C (real Nemotron data) all PASS (2026-07-28). Muon trains end-to-end on Capella on real data.

**Real-data confirmation (job 3806836, `muon_smoke/03_real_data_test_capella.sbatch`)**:
`mock_data: false`, `data_path` pointing at the converted `train.bin`/`.idx`
(26.4M sequences, see `DATA_CONVERSION.md`). Real training iterations, loss
decreasing cleanly: iter 1 lm loss 10.863 → iter 30 lm loss 10.480, grad norm
9.99 → 4.60, zero NaN/skipped iterations. Confirms the full pipeline —
real Nemotron data → real GPT2 tokenizer → Muon+Adam optimizer → real
training steps — end to end on Capella.

| Step | Result |
|------|--------|
| Submodule pin (Megatron-LM v0.17, `4a81356`) | OK — verified matches risto's `v0.17-mcore` |
| Container/sandbox build | OK — 33G sandbox at `/data/horse/ws/sala597i-slaing/containers/.tmp_sandbox/oellm_sandboxv514iw4i/sandbox`, verified runs `torch` + imports `oellm_autoexp`. Packed `.sif` built too but can't be mounted on this node (`squashfuse_ll`/FUSE blocked) — using the sandbox via `singularity exec --underlay` for everything |
| `emerging_optimizers` install (`NEW_CLUSTER_SETUP.md` step 2) | OK — installed to `muon-pylibs/`, `import emerging_optimizers` succeeds |
| RUNG A — schema regen (`optimizer: muon` valid key) | OK — ran `scripts/generate_megatron_config.py` + `generate_megatron_dataclass.py` locally in this worktree (these are gitignored/local-only, not committed — see below) |
| RUNG B — dry-run render (`config experiments/laingsam/muon_50M_50BT`) | OK, exit 0 — sweep expands to 2 points (stable + decay50B), DAG resolves cleanly |
| RUNG 1 — `muon_gpu_smoke.py` GPU smoke test | **PASS** (job 3797690, `sbatch muon_smoke/01_smoke_capella.sbatch`, `capella-interactive`) — `NVIDIA H100 MIG 1g.12gb` visible, Newton-Schulz orthogonalized (singular values 0.977–1.032), `HAVE_EMERGING_OPTIMIZERS: True`. First real test of `--nv` GPU passthrough into the sandbox — worked first try. Log: `muon_smoke/logs/01_optimizer_construct_3797690.log` |
| RUNG C — mock-data training steps (real `pretrain_gpt.py` run via `muon_50M_50BT.yaml`) | **PASS** (job 3800698) — iteration 1/10/20 all logged, `lm loss` decreasing (10.900 → 10.884 → 10.685), grad norm finite/reasonable, zero NaN/skipped iterations. Muon+Adam optimizer builds and trains for real. |
| Real Nemotron data | Blocked on format conversion (see below) |

### RUNG C debugging log (all fixed, all still relevant to real-data training too)
Six real, independent bugs found and fixed getting here — every one of them would
have blocked real Nemotron training identically, not just the mock-data smoke:
1. Schema regen needed (RUNG A) — `optimizer: muon` / `exit_signal` / `layernorm_epsilon` (see above).
2. Tokenizer: `AutoTokenizer.from_pretrained` needs a local `config.json` *and* the
   directory needs to actually be bind-mounted into the container (`/data/horse/ws/...`
   isn't auto-bound like `$HOME` is) — see tokenizer section below.
3. `torch.compile`/Triton JIT-fusion warmup (`bias_swiglu`) can't find `libcuda.so`
   in this container; `disable_jit_fuser` config flag is a no-op (decorator binds
   at import time, before args are parsed) — fixed via `TORCHDYNAMO_DISABLE=1` env var.
4. `run_autoexp.py`'s orchestrator injects its own `PYTHONPATH: .:submodules/Megatron-LM`
   for the training subprocess, silently overriding whatever `--env PYTHONPATH` we
   pass to `singularity exec` — fixed via `backend.env.PYTHONPATH` override in the
   experiment yaml (needs `muon-pylibs` appended, matching `NEW_CLUSTER_SETUP.md`).
5. `LayerWiseDistributedOptimizer.set_bucket_layerwise_params_list()` crashes
   (`TypeError: 'NoneType' object is not iterable`) whenever `overlap_param_gather=True`
   on a single GPU (dp_cp group size 1) — real gap in Megatron's own code, not
   ours; fixed by setting `overlap_param_gather: False` (multi-GPU-only optimization anyway).
6. OOM on the 1g.12gb MIG slice: `num_workers: 7` (LUMI's 28-cpu/4-gpu ratio,
   doesn't match Capella's 2-cpu allocation) and `micro_batch_size: 16` (output-layer
   vocab-projection logits too large for 12GB) — fixed via `num_workers: 2` and a
   CLI override `micro_batch_size=4` (base config's 16 untouched, for later real GPUs).

Known cosmetic issue, not yet root-caused: `--array-subset 0` + CLI overrides for
`train_iters`/`eval_iters`/`save_interval` don't actually take effect — the job
runs against the sweep's own formula-derived `train_iters` (95368, from the 50B-token
stable-stage formula) instead of the intended short smoke value. Harmless for now
(just cancel the job once pass/fail is clear from the log), worth investigating
before any longer real-data run so it doesn't silently run the full token budget.

### New: `muon_smoke/01_smoke_capella.sbatch`
Runs `muon_gpu_smoke.py` as a proper `sbatch` job instead of an interactive
`salloc` — logs to `muon_smoke/logs/`, no need to hold a terminal open.
`--cpus-per-task=2 --mem=4G --gres=gpu:h100_1g.12gb:1` — matches the values
already proven to work on `capella-interactive`; larger requests (tried 4
CPU / 16G once) tripped `QOSMaxCpuPerUserLimit`.

Note: Capella's `/etc/slurm/cli_filter.lua` has a real, intermittent bug
(`attempt to div a 'number' with a 'string'` at `limits.lua:120`) that
sometimes falsely rejects `sbatch`/`srun` submissions with a QOS violation
even when nothing else is queued. Retrying the identical command usually
succeeds — seen repeatedly, not something to chase in our own config.

## What changed vs LUMI (`muon_50M_50BT_lumi.yaml` kept as-is for reference)

New `config/experiments/laingsam/muon_50M_50BT.yaml` (Capella):
- `/container: none`, `/slurm: local` instead of `/container: lumi`, `/slurm: lumi`
  — no `capella` cluster target exists in this repo yet, so we're driving
  `singularity exec` by hand (same as the LUMI ladder's RUNG C2 pattern) rather
  than building a new autoexp cluster target right now.
- `mock_data: true`, `data_path: []` — real data path below is still TODO.
- `seq_length`/`max_position_embeddings`: 4096 → 2048, matching the Nemotron
  sample's actual `ctx_2048` packing.
- `tokenizer_model` points at the *local* tokenizer dir under
  `/data/horse/ws/jasi149i-fastlm/data/nemotron-cc-sample-mtsynth/tokenized_gpt2/tokenizer`
  (read-only access) instead of `EleutherAI/gpt-neox-20b` — the sample data was
  actually tokenized with a `GPT2Tokenizer` (vocab_size 50257 + 47 pad tokens),
  confirmed by reading `tokenizer_config.json` and one `.arrow` shard's schema.
- Dropped the LUMI/ROCm-only fusion-flag block (`bias_swiglu_fusion: False`, etc.)
  per `NEW_CLUSTER_SETUP.md` step 4.
- Dropped the LUMI `slurm.sbatch` block (partition `standard-g`, account
  `project_465002530`) — not used with `/slurm: local`.
- Two schema/field fixes needed that are unrelated to Muon or Capella, just
  gaps in this checkout's generated schema vs the actual v0.17 pin:
  - `exit_signal` (Megatron v0.17's `signal.Signals` enum field) — generator
    wrote default `'15'` but the type wants a name (`'SIGTERM'`); overrode
    explicitly in the yaml.
  - `norm_epsilon` → renamed to `layernorm_epsilon` (Megatron v0.17 field
    rename that the LUMI-era config predates).

## Data: real Nemotron path, and why it's not wired in yet

`/data/horse/ws/jasi149i-fastlm/data/nemotron-cc-sample-mtsynth/` (coworker's
grant, read-only for us — confirmed via `getfacl`, never written to):
- `raw_dataset/`: 972GB raw text, HF `datasets` arrow format.
- `tokenized_gpt2/ctx_2048/train/`: 203GB, **already tokenized** (`input_ids`
  int32 arrays, 2049 long = 2048 ctx + 1, plus `docs_lengths`) with
  `GPT2Tokenizer`, still arrow format.

Decision (2026-07-27): use the already-tokenized GPT2 data rather than
re-tokenizing the raw text with `gpt-neox-20b` — re-tokenizing is a genuine
CPU-bound pass over 972GB (no GPU needed either way; real throughput not yet
benchmarked). Re-tokenizing to match the rest of the repo's convention is
deferred ("worry about it later").

Tokenizer files (`tokenizer.json` + `tokenizer_config.json`, ~3.5MB) were
copied — read-only from `jasi149i-fastlm`, written only to our own workspace —
into `/data/horse/ws/sala597i-slaing/data/nemotron_gpt2_tokenizer/`, plus a
minimal `config.json` added there so `AutoConfig` resolves locally instead of
mistaking the path for a Hub repo id. `tokenizer_model` in the yaml points at
this copy, not at `jasi149i-fastlm` directly.

Still needed before flipping `mock_data: false`:
- A conversion script: read `input_ids` out of the arrow shards, write them
  into Megatron `.bin`/`.idx` via `IndexedDatasetBuilder`. Mechanical (data is
  already tokenized), but not yet written.
- Output must land only under our own workspace
  (`/data/horse/ws/sala597i-slaing/data/...`), never into `jasi149i-fastlm`.
- `data_path: []` in the config needs the resulting `.bin`/`.idx` prefixes.

## Next steps
1. ~~Run `muon_gpu_smoke.py` in a GPU allocation~~ — DONE, PASS (job 3797690).
2. ~~Run RUNG C (mock-data training steps)~~ — DONE, PASS (job 3800698).
3. Write the arrow → `.bin`/`.idx` conversion script, run it (writes only to
   our own workspace), benchmark real throughput first.
4. Flip `mock_data: false`, fill in `data_path`, rerun the smoke ladder on
   real data.
5. (Minor, non-blocking) figure out why `--array-subset`/CLI overrides for
   `train_iters` etc. don't take effect against the composable sweep — matters
   for not silently running the full 50B-token budget on a real-data run.

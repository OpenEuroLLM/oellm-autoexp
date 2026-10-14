# Downstream evaluation of the multilingual architecture-scaling variants

## Objective

Score the five architecture variants of the multilingual scaling comparison
(`config/experiments/architecture_scaling_variants/multilingual/main_scaling_all.yaml`)
on downstream benchmarks with `oellm-evals`, rather than only on validation
loss / PPL (which `eval_all_scaling_all.yaml` covers). Suites:

- **`dclm-core-22`** — 21 lm-eval-harness tasks (English, mostly log-likelihood
  plus 6 `generate_until` tasks).
- **`oellm-multilingual`** — super group: belebele-eu (24 langs), global-mmlu-eu
  (18), include (25), mgsm-eu (4), generic-multilingual (xwinograd / xcopa /
  xstorycloze), and flores-200 in both directions (35 EU langs) through
  lighteval.

Scope: all **124 cooldown-final checkpoints** (5 variants × 25 budget points,
minus one missing `gdn7_nope` point), so downstream metrics can be plotted
against token budget the same way the val-loss ladder is.

## The blocker, and how it was resolved

`oellm-eval` runs `lm_eval --model hf --model_args pretrained=<dir>`, so every
checkpoint has to be loadable by `transformers`. Of the five variants only
`fullattn` maps onto a stock HF architecture (Qwen3, reachable by way of
Megatron-Bridge). `swa7_rope100k`, `gdn7_nope`, `mlstm7_nope` and `mamba2_nope`
have no HF counterpart.

**Resolution: one custom HF architecture for all five variants**, in
`oellm_autoexp/hf_export/oellm_hybrid/`:

- `configuration_oellm_hybrid.py` — `OellmHybridConfig`, whose `layer_types`
  list assigns a mixer per layer out of `full_attention`, `sliding_attention`,
  `gdn`, `mlstm`, `mamba2`.
- `modeling_oellm_hybrid.py` — `OellmHybridForCausalLM`. A transcription of the
  Megatron modules, *calling the same kernels the runs trained with*:
  `fla.ops.gated_delta_rule.chunk_gated_delta_rule`,
  `mlstm_kernels.torch.get_mlstm_kernel`,
  `mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined`. All three
  packages are already installed in the JUPITER eval venv. The fused input
  projections stay fused, exactly as stored, so no tensor is re-laid-out.

Using one architecture for all five (rather than Bridge/Qwen3 for two and
custom code for three) means every variant takes an identical eval code path —
which is what makes the numbers comparable across variants.

Conversion: `oellm_autoexp/hf_export/convert_megatron_to_hf.py` reads the
`torch_dist` checkpoint through PyTorch DCP (no Megatron runtime, no CUDA,
single process — the global shapes in the metadata resolve TP/PP/DP for us),
remaps to HF names, and writes a self-contained dir (`model.safetensors`,
`config.json` with `auto_map`, the two modeling modules, the 256k tokenizer).

### Checkpoint layout facts the converter had to absorb

- **Two different layer layouts.** The hybrid runs store parameters per layer
  (`decoder.layers.0.…`) because their layer specs are heterogeneous. The
  homogeneous runs (`fullattn`, `swa7_rope100k`) stack all layers into a single
  tensor with a leading layer axis (`decoder.layers.mlp.linear_fc1.weight`,
  shape `[16, 3072, 512]`). `unstack_layers()` normalises the latter.
- **Fused projections are stored split into named sections** by the sharded
  state-dict factories, and the section order is the mixer's concat order:
  GDN `[query, key, value, z, beta, alpha]`, mLSTM
  `[query, key, value, ogate, igate, fgate]`, Mamba2 `[z, x, B, C, dt]`.
- **`linear_qkv` is grouped-interleaved**, not three contiguous q/k/v blocks:
  per query group `[q_1 … q_{h/g}, k, v]`. Matters at 0.4B/0.9B (GQA 16/8).
- **The input layernorms are fused into the projections**
  (`in_proj.layer_norm_weight` / `linear_qkv.layer_norm_weight` = pre-mixer
  norm; `mlp.linear_fc1.layer_norm_weight` = pre-MLP norm).
- **SwiGLU**: Megatron's `swiglu()` is `silu(chunk[0]) * chunk[1]`, so the first
  half of `linear_fc1` is `gate_proj` and the second is `up_proj`.
- **Layer assignment**: linear-attention variants put *softmax* attention where
  `(i + 1) % linear_attention_freq == 0`; SWA uses the same rule by way of
  `window_attn_skip_freq`. Both ⇒ every 8th layer is full attention.
- **Sliding window**: Megatron/TE `window_size=(1024, 0)` lets query `i` attend
  to `j ∈ [i - 1024, i]` — 1025 keys, *inclusive* bound.

### Verification: 1:1 parity against the Megatron path

`scripts/korbi/check_hf_megatron_parity.sh <train_run_dir> [hf_dir]` runs the
**same token ids** (taken from the pretokenized training mix) through both
paths in one allocation and diffs them:

1. `scripts/korbi/dump_megatron_logits.py` builds the model inside the training
   container from the run's *own* `job.sbatch` flags — so the architecture is
   not re-specified anywhere — loads the `torch_dist` checkpoint and dumps
   logits. Only data-pipeline / logging / schedule flags are dropped.
2. `validate_hf_export.py --reference <dump>` runs the converted HF dir on the
   identical ids and reports max/mean |Δlogit|, mean+max KL, top-1 agreement and
   the NLL delta.

Measured on 0.1B/20BT `mlstm7_nope` (2 × 512 real tokens):

| metric | value |
|---|---|
| max \|Δlogit\| | 0.375 |
| mean \|Δlogit\| | 0.0224 |
| mean KL(mega‖hf) | 5.1e-4 (max 5.7e-3) |
| top-1 agreement | 98.73 % |
| **NLL megatron / hf** | **2.76168 / 2.76187 (Δ 2e-4)** |

That is bf16-forward noise. The NLL delta — the quantity evals actually consume
— is ~200× smaller than the 0.045-nat spread between the architectures being
compared, so it cannot reorder the comparison. Top-1 flips come from near-ties
and are expected between two different kernel/reduction orders.

### gdn3_qwen35 (gated attention) — 2026-08-14, job 1365131

The Qwen3.5 arm exercises three export paths the arms above never touched:
`attention_output_gate`, `layernorm_zero_centered_gamma` and partial RoPE
(`rotary_percent 0.25`). Re-run on the completed 0.1B/gbs128 stable
(`iter_0076294`, 182 tensors, strict load clean):

| metric | gdn3_qwen35 | reference band (other arms) |
|---|---|---|
| max \|Δlogit\| | 0.297 | 0.375 |
| mean \|Δlogit\| | 0.0178 | 0.0224 |
| mean KL(mega‖hf) | 2.6e-4 (max 1.8e-3) | 4.9–5.2e-4 |
| top-1 agreement | 99.61 % | 98.4–99.5 % |
| **NLL megatron / hf** | **2.86297 / 2.86279 (Δ 1.8e-4)** | \|ΔNLL\| 0.0–0.0027 |

Better than the established band on every metric, which is the real result here:
the risky part was `_split_qkv`. With the gate the fused projection is laid out
`[q, gate, k, v]` per KV group, so the old 3-way split would have handed back
the gate as k and k as v — and the SHAPES STILL LINE UP, so nothing would raise.
That failure returns NLL near the 12.48 random baseline, not a subtle drift, so
this table is what rules it out. `zero_centered_gamma` is the opposite hazard: it
is a pure off-by-one on every gamma, would load silently, and shows up only as a
mildly wrong loss — hence the unit check that `zcg(gamma=0) == plain(gamma=1)`.

### The bug this caught (and how it hid)

The first version of the port scored `fullattn` and `swa7_rope100k` at ~5.8 nats
on real data while the three linear variants sat at ~3.19 — but *every* variant
passed the strict-load check. Cause: `OellmRotaryEmbedding` registered `inv_freq`
as a **non-persistent buffer**. transformers 5.x builds models on the meta device
and only materialises tensors present in the checkpoint, so `inv_freq` came back
as **zeros** — turning RoPE into the identity (`cos≡1, sin≡0`) silently, with no
missing-key error. It only hit the two RoPE variants; the NoPE hybrids were
unaffected, which is exactly why a cross-variant comparison exposed it.
`inv_freq` is now recomputed from two scalars at first use rather than stored.

Lesson encoded in the checks: a strict-load pass proves nothing about tensors
that are *supposed* to be absent from the checkpoint. Only the reference
comparison catches that class of bug.

### Known limitation

The HF port has **no KV / recurrent-state cache**: `use_cache=False`, so
generation recomputes the prefix at every step. Log-likelihood tasks (the bulk
of both suites) do a single forward pass and are unaffected; the generative
tasks — 6 in dclm-core-22, and all of flores-200 — pay for it. This is the main
thing the pilot has to measure. Also, with left-padded generation batches the
recurrent mixers see the pad prefix; padded positions are zeroed at the model
input, which keeps GDN and Mamba2 states at exactly zero, but the mLSTM
max-state can still drift, so mLSTM generative numbers should be taken at
batch size 1 until a cache is implemented.

Separately, the `chunkwise--triton_xl_chunk` mLSTM kernel asserts a
sequence length divisible by 16 (training always fed 4096). lm-eval feeds
arbitrary lengths, so the mixer right-pads to a chunk boundary and truncates —
safe because the recurrence is causal.

## Cluster setup (JUPITER)

`scripts/setup_eval_env_jupiter.sh` (login node — compute nodes are offline):

1. Installs `oellm-evals` into `$OELLM_EVAL_VENV` with `--no-deps`. The venv's
   `bin/pip` and the previous `bin/oellm` carried a **stale shebang** pointing at
   `/e/project1/projectnucleus/poeppel1/venv` (the venv was relocated), so the
   old install was not importable; pip is driven through the interpreter
   instead. `--no-deps` protects the venv's `datasets` and the
   fla/mlstm_kernels/mamba_ssm builds the converted models need.
2. Builds `$OELLM_EVAL_HF_HOME` (`…/cache_eval`): a real `datasets/` dir of
   **symlinks** into the shared cache (`scripts/link_eval_dataset_cache.py`),
   plus a symlink to the account `token`. oellm-evals hardcodes
   `HF_DATASETS_CACHE=$HF_HOME/datasets`, while JUPITER's populated cache keeps
   the `owner___name` dirs at the cache ROOT; the symlink dir reconciles the two
   without copying, and keeps the eval cache isolated so nothing in the user's
   shared cache is modified.
3. `--prefetch` downloads everything still missing, **tolerating failures** and
   printing a coverage report (oellm-eval's own `--download_only` aborts the
   whole run on the first unavailable dataset).

### The `datasets` version question — resolved by measurement

oellm-eval's pyproject pins `datasets<4.0.0` because some suite datasets used to
be loading-script based. Following that pin turned out to be **worse**, not
better: the Hub now serves dataset-card metadata in the 4.x format, which 3.x
cannot parse (`Feature type 'List' not found`), so under 3.x even plain
`allenai/ai2_arc` fails to resolve. A 24-probe sweep over both suites under each
version settled it:

| version | outcome |
|---|---|
| 3.6.0 (overlay) | many datasets unresolvable — Hub metadata is 4.x-format |
| 4.8.5 (venv)    | 23/24 fine; only `winograd_wsc` (script-based) is lost |

So the eval runs on the venv's `datasets` 4.x and the 3.x overlay idea was
dropped. Related trap: cache entries *built* by 4.x are unreadable by 3.x, so
mixing the two versions against one cache corrupts it for the older one.

### Known gaps (both external, neither is a code problem)

1. **`facebook/flores` is gated.** The token at `$HF_HOME/token` is present but
   the account has not accepted the dataset terms, so the two flores-200 groups
   (the bulk of `oellm-multilingual`, and its most expensive part) cannot be
   downloaded. Fix: accept the terms at
   <https://huggingface.co/datasets/facebook/flores> with that account, then
   re-run `setup_eval_env_jupiter.sh --prefetch`.
2. **`wsc273`** loads through the removed `winograd_wsc` loading script and is
   dead under `datasets>=4`. 1 of dclm-core-22's 21 tasks; the other 20 run.

## Run Log

- 2026-08-11: Surveyed the training tree — all 124 cooldown finals
  (`*_firstcd_decay*BT`) plus 35 stable runs have checkpoints. `gdn7_nope` has
  24 of 25 budget points.
- 2026-08-11: Updated the `oellm_evals` submodule from `5e8225c` to `c6c1e05`
  (20 commits): adds `jeopardy` to dclm-core-22 (20 → 21 tasks), the
  `--max_array_len` fix, per-group language filtering, and new EU groups
  (xnli, pawsx, xcsqa, mmlu-prox, global-mgsm, opensubtitles, polymath,
  global-piqa). The `oellm-multilingual` super group itself is unchanged.
- 2026-08-11: Wrote the `oellm_hybrid` HF architecture + DCP converter. All
  five 0.1B/20BT variants convert cleanly (164–206 tensors each).
- 2026-08-11: Repaired the eval venv, built the isolated `cache_eval` HF_HOME,
  prefetched 91/201 dataset configs (the rest were already cached or are the
  two known gaps).
- 2026-08-11: Fixed three port bugs found on GPU — `layer_types` renamed to
  `mixer_types` (transformers 5.x strictly validates that field name),
  `_tied_weights_keys` must be a mapping, and the mLSTM kernel's
  divisible-by-16 sequence-length assert.
- 2026-08-11: **Found and fixed the RoPE-as-identity bug** by way of NLL on real
  pretokenized data. After the fix, HF NLL on a dclm shard reproduces the
  training ranking exactly:

  | variant | training val loss | HF NLL (dclm shard) |
  |---|---:|---:|
  | mamba2_nope    | 2.9867 | 3.1898 |
  | gdn7_nope      | 2.9882 | 3.1864 |
  | mlstm7_nope    | 3.0126 | 3.2002 |
  | swa7_rope100k  | 3.0314 | 3.2178 |
  | fullattn       | 3.0330 | 3.2233 |

  (Different data slice, hence the constant offset; the *ordering* and the
  ~0.04-nat spread both reproduce.)
- 2026-08-11: Built `scripts/korbi/check_hf_megatron_parity.sh` and ran it on
  all five variants. Every one sits at the same bf16 noise floor — mean KL
  4.9–5.2e-4, top-1 agreement 98.4–99.5%, |ΔNLL| 0.0–0.0027 — regardless of
  whether the layer is attention, GDN, mLSTM or Mamba2. That uniformity is the
  evidence the port is faithful.
- 2026-08-11: Dropped `convert` from the sweep — it is now a one-shot
  `scripts/convert_arch_scaling_to_hf.py` pass, since conversion is a 2-second
  CPU job and JUPITER has no CPU partition (124 whole-node allocations for it
  would be pure waste).
- 2026-08-11: The DAG resolver's **parallel** path cannot unpickle its own
  results for this config — `TypeError: RootConfig.__init__() missing 3 required
  keyword-only arguments: 'slurm', 'job', 'backend'` is raised inside the pool's
  result thread and the run then hangs. I first blamed the mixed `BashBackend`
  sweep; that was wrong — it reproduces with a single-backend sweep too, so it
  is a pre-existing resolver/pickling issue rather than anything specific to the
  convert stage. Workaround: `HYDRA_STAGED_SWEEP_WORKERS=1`, which renders all
  250 points cleanly in a couple of minutes. Worth fixing upstream in
  `hydra_staged_sweep/parallel.py`.
- 2026-08-11: Full dry-run green: 250 points, 250 sbatch scripts rendered.
  The GPU-h estimate (2250) is the sum of the configured wall-time caps
  (4 h dclm + 12 h multilingual), not a prediction — the pilot replaces it.

## 2026-08-12 follow-ups

- **flores un-gated** → prefetch now covers **200/200** dataset configs. Also
  taught the downloader to enumerate configs for task entries with no `subset`
  (xwinograd / xcopa / xstory_cloze), which otherwise would have been missing on
  the offline compute nodes.
- **wsc273 removed** from `dclm-core-22` (now 20 tasks) in the `oellm_evals`
  submodule — its `winograd_wsc` loading script cannot run under `datasets>=4`.
  Needs upstreaming to OpenEuroLLM/oellm-evals.
- **The pickling bug is a compoconf bug, already fixed upstream.**
  `compoconf.ConfigInterface.__reduce__` returned `(cls, (), asdict(self))`,
  which (a) calls `cls()` with no args — impossible for `RootConfig`, whose
  `slurm`/`job`/`backend` are `field(default_factory=dataclasses.MISSING)`, that is
  required — and (b) flattens nested configs to plain dicts by way of `asdict`.
  compoconf **0.2.2 removed that `__reduce__`** (its docstring now names both
  failure modes); both environments were pinned at 0.1.16 while this repository's
  `pyproject.toml` already required `>=0.2.2`. Upgrading the JUPITER venv to
  0.2.2 fixes it: the sweep now resolves 250 points across 72 workers and
  renders all 250 scripts. `HYDRA_STAGED_SWEEP_WORKERS=1` is no longer needed.
  No patch is carried in this repository.
- **Caching implemented** (`OellmHybridCache`): KV for attention layers, conv +
  delta state for GDN, `(C, n, m)` for mLSTM, conv + SSM state for Mamba2.
  Prefill uses the chunkwise kernels with `output_final_state`; each decode step
  uses the matching single-step kernel (`fused_recurrent_gated_delta_rule`,
  `get_mlstm_step_kernel`, `selective_state_update`). Two subtleties:
  attention masks are built in absolute positions so prefill and decode share
  one path, and the mLSTM chunk padding now forces `f=1, i=0` on pad steps so
  they leave the returned state exactly unchanged (previously they corrupted
  it, which only matters once the state is reused).
  `scripts/korbi/check_hf_cache_parity.py` greedy-decodes with and without the
  cache and requires identical tokens. **All five variants PASS (12/12 tokens
  each.)** One fix was needed: the mLSTM step kernel wants its gates as
  `[b, nh, 1]`, not `[b, nh]`.
- **No regression from the cache work**: re-diffing all five against the stored
  Megatron logit dumps reproduces the earlier numbers *exactly* (mean KL
  4.9-5.2e-4, |dNLL| 0.0-0.0027), so the log-likelihood path is untouched.
  Both gates are green:

  | variant | Megatron parity | cache parity |
  |---|---|---|
  | fullattn | PASS (dNLL 0.0024) | PASS 12/12 |
  | swa7_rope100k | PASS (dNLL 0.0002) | PASS 12/12 |
  | gdn7_nope | PASS (dNLL 0.0027) | PASS 12/12 |
  | mlstm7_nope | PASS (dNLL 0.0002) | PASS 12/12 |
  | mamba2_nope | PASS (dNLL 0.0000) | PASS 12/12 |

  Note the JUPITER login node has a GH200, which is where both checks ran while
  the compute nodes were down — useful for quick model-level verification.

## Next

1. Run the parity checker over all five variants (and one 0.9B GQA point — the
   grouped-interleaved QKV split is only exercised at 0.4B/0.9B).
2. Pilot subset of `eval_downstream_all.yaml`; measure per-suite wall time, then
   decide whether flores-200 needs `backend.limit`, a KV cache, or splitting the
   super group.
3. Get `facebook/flores` un-gated (account terms) — otherwise the two flores
   groups will fail in every `evalml` job.

## Follow-up: port to `oellm-eval` (branch `feat/hybrid-v0.19`)

- 2026-10-14: `prod/oellm_32b_dense` replaced the `submodules/oellm_evals` submodule (OpenEuroLLM/oellm-evals) with `submodules/oellm-eval`, which has the same layout. On the merged branch, `eval_downstream_all.yaml`, the eval container definition and `setup_eval_env_jupiter.sh` point at the new path.
- The two local `oellm_evals` edits were not carried over. The `agieval_lsat_ar` `{{gold[0]}}` fix ships with `oellm-eval` (`custom_lm_eval_tasks/agieval_lsat_ar/`). `wsc273` is back in `dclm-core-22` (21 tasks again), now read from the parquet `lighteval/winograd_wsc`, not from the `winograd_wsc` loading script that datasets 4.x refused to run. Results produced before this point have 20 dclm tasks; compare on the common task set (the aggregator's `n_common` column).
- Before the next launch: prefetch `lighteval/winograd_wsc` into the eval HF cache (`setup_eval_env_jupiter.sh --prefetch`) and rebuild the eval image, whose `%files` now stage `submodules/oellm-eval`.

# Multilingual SWA Scaling Rerun (JUPITER, <10k GPU-hrs)

Tests the hypothesis: **a 1024-token sliding-window attention at 5:1 (and 7:1)
SWA:full ratio is not worse than full attention and scales the same** — on the
current multilingual baseline, to unblock inference-optimized training
(KV cache ~2.7× smaller at 4k context, ~6×+ at long context).

## Baseline

W&B `openeurollm-project/dense_multilingual_scaling` — Qwen3-style dense
models (`qwen3_dense_{0.1B,0.2B,0.4B,0.9B}_ne`) on the 1TT-option-4
multilingual mix, oellm 256k tokenizer (padded vocab 262272), WSD to min_lr 0
(20% linear decay), warmup 2000, β=(0.9, 0.95), seed 1234, cyclic dataloader,
split 99,1,0. Authoritative configs: `origin/exp_diana` branch
(`config/backend/megatron/multilingual_scaling/`, `config/sweep/multilingual_scaling/`).

Exported baseline + analysis (this repo): `dump/scaling_experiments_multilingual/`
— `runs_summary.csv` (1166 runs), `configs.json`, `curves/`,
`post_decay_losses_multilingual.csv`, `best_bs_lr_multilingual.csv`,
`all_losses_multilingual.csv` (near-tie context).

**Comparison metric**: end-of-decay `lm loss validation` on the deterministic
1% split of the train mix (the metric present on 139 completed baseline decay
cells). The baseline's separate validation-stage campaign (datamix4-val)
crashed on W&B, so per-cell comparison uses the 1%-split metric; identical
blend + split + seed ⇒ same slice.

| size | hidden | layers | heads/groups | ffn  | N_ne | N_total | SWA 5:1 | SWA 7:1 |
|------|-------:|-------:|--------------|-----:|-----:|--------:|---------|---------|
| 0.1B |  512   | 16     | 8/8 (MHA)    | 1536 |  71M |   206M  | 14/16   | 14/16   |
| 0.2B |  768   | 22     | 8/8 (MHA)    | 2304 | 186M |   387M  | 19/22   | 20/22   |
| 0.4B | 1024   | 28     | 16/8 (GQA)   | 3072 | 440M |   709M  | 24/28   | 25/28   |
| 0.9B | 1536   | 28     | 16/8 (GQA)   | 4608 | 859M |  1262M  | 24/28   | 25/28   |

kv_channels 128 everywhere; no biases; qk-layernorm; tied embeddings.

## Run plan — speed-optimized 4-architecture comparison (rev. 2026-08-01)

The study is a **five-architecture** comparison (full attention, 7:1 mLSTM,
7:1 GDN, 7:1 SWA, 7:1 Mamba2) on the **validated high-throughput Liger setup**,
in the `jupiter_liger_mamba` container (jupiter_liger + mamba-ssm/causal-conv1d,
so all five share one identical image). The **entire grid is one nested-sweep
file**, `main_scaling_all.yaml` — architecture (5) × (size,tier) ladder (7), each
ladder = 1 stable + one WSD cooldown per budget. One file guarantees every
architecture runs the identical stages per cell. **160 points, 125 dependency
edges**. The old pre-Liger ladders and the interim per-arch `main_*` files were
removed.

Mamba2 slots into the same linear-attention dispatch as mLSTM/GDN
(`experimental_attention_variant=mamba`, `linear_attention_freq=8`) via the
`Mamba2Attention` adapter added to the Megatron branch; it uses its own
`mamba_state_dim/head_dim/num_groups` dims (not the `linear_*` head knobs).

Every ladder uses a single (gbs, lr) per tier — the fully-covered baseline
"near-winner" combos (grid analysis in `dump/scaling_experiments_multilingual/`;
matches `best_bs_lr_multilingual.csv` on 0.1B/0.4B and most 0.2B/0.9B cells).
GBS/LR are identical across all four architectures per cell (same-combo
comparison); geometry (nodes/MBS/GA) is throughput-only.

| tier | gbs / lr | budgets (BT) | MBS / nodes (locked by geo smoke) |
|---|---|---|---|
| 0.1B_low  | 128 / 2e-3  | 20/30/50            | 32 / 1 |
| 0.1B_high | 256 / 2e-3  | 80/200/300          | 32 / 1 |
| 0.2B_low  | 128 / 1e-3  | 6/12/20/30          | 16 / 1 |
| 0.2B_high | 512 / 2e-3  | 50/80/120/200/300   | 16 / 2 |
| 0.4B      | 256 / 2e-3  | 12/20/30/50         | 8 / 2 |
| 0.9B_mid  | 512 / 5e-4  | 30/50/80/120        | 4 / 4 |
| 0.9B_high | 1024 / 1e-3 | 200/300             | 4 / 4 |

- **MBS is uniform per size** (min that fits all four archs): 0.1B 32, 0.2B 16,
  0.4B 8, 0.9B 4 — GDN OOM'd at 0.9B/MBS8, so 0.9B uses MBS4 for everyone.
- **mLSTM input-gate bias init = -8.0** everywhere (best of the
  `mlstm7_igate_sweep_0.1B` {-12,-10,-8,-3} sweep).
- **GQA sizes (0.4B/0.9B)**: GDN keeps the GQA head layout (k=groups, v=heads);
  mLSTM has no GQA mode (`k==v`), so its head count pins to `aux.heads`.
- The 6BT anchor cell already ran as `fullattn_mlstm7_gdn7_0.1B_50BT.yaml`.

Submit the whole grid (or a subset with `--array-subset`):
`scripts/run_autoexp.py --submit-and-exit --config-name
experiments/architecture_scaling_variants/multilingual/main_scaling_all`.

**Geometry smokes** (`geo_smoke_{0.2B,0.4B,0.9B}.yaml`, `geo_smoke_0.9B_gdn_retry.yaml`):
short 100-iter throughput/OOM probes that locked the MBS/nodes above. Results in
`experiment/003_multilingual_throughput_tuning/log.md`. Note: a single-file
multi-ladder sweep required a `dag_resolver.py` fix so a cooldown branches from
its own ladder's stable (disambiguate by shared size/gbs/lr/variant).

**Submission order**: geometry smokes → 0.1B low/high (all 4 arch) → 0.2B/0.4B
ladders → 0.9B ladders last (biggest ticket). Submit each ladder with
`scripts/run_autoexp.py --submit-and-exit --config-name
experiments/architecture_scaling_variants/multilingual/main_<arch>_<size>_<tier>`.

## Prerequisites / staging status (2026-07-28)

1. **Pretokenized data** — STAGING IN PROGRESS from Leonardo
   (`/leonardo_work/OELLM_prod2026/preprocessed/openeurollm-tokenized-256k{,-val}`)
   to `/e/data1/datasets/products/openeurollm/pretokenized/…` (7.3 TB train +
   0.69 TB val; layout mirrors Leonardo verbatim; prefix knob
   `backend.megatron.aux.datamix_prefix`). Blend weights verified IDENTICAL
   between the Leonardo `1TT-option-4.sh` (605 baseline runs) and the LUMI
   variant recovered from `origin/exp_diana`. Transfer runs on leonardo
   login02 under `~/staging_jupiter/` (10 rsync chunk streams + val stream +
   12 byte-range dd streams for the two >0.8 TB .bin files); check/verify with
   `~/staging_jupiter/verify_transfer.sh` (needs the user's agent socket in
   `transfer_env.sh` to still be valid — re-source a fresh SSH_AUTH_SOCK if
   the session was restarted). **Current blocker (2026-07-30): do not submit
   training jobs until this verification succeeds.** The staged
   `opus-mt-10p-sample/kat_Geor.bin` and
   `finepdfs-10p-sample/eng_Latn.bin` are inconsistent with their `.idx`
   files: they are respectively 62,876,751,216 and 178,784,100,704 bytes
   shorter than the final indexed sequence requires. Resume or replace those
   source transfers, then verify again. File
   count 312, sizes vs `train_files_sizes.txt` (the two range-copied files are
   pre-truncated to full size — size alone doesn't prove completion for them;
   check the `END range … rc=0` lines in `range_*.log`).
2. **Tokenizer**: DONE — `/e/data1/datasets/products/openeurollm/tokenizers/tokenizer-256k`
   (43 MB, from Leonardo `openeurollm-tokenizer-256k`; knob `aux.tokenizer_path`).
3. **Validation mix**: also staged (`openeurollm-tokenized-256k-val`, blend in
   `config/backend/megatron/data_multilingual_1tt_val_jupiter.yaml`) for
   optional eval-only runs on final checkpoints. Declared as `valid_data_path`
   (Megatron per-split blend, evals the full val tree) — NOT the baseline's
   `data_path + split 1,99,0` hack. Mutually exclusive with `data_path+split`,
   so it lives in separate eval-only jobs (`skip_train: true`, ~204800
   samples), not in the training runs: training keeps `split: 99,1,0` because
   both the comparison metric (1%-slice val loss of the 139 baseline cells)
   AND the training token order depend on it. The val-mix metric is the
   forward-looking one for SWA-variant-vs-variant comparisons.
4. First job per gbs builds the blend index cache
   (`$OELLM_CACHE_DIR/multiling_1tt_262k`) — took ~30 min on the baseline;
   `distributed_timeout_minutes: 60` accounts for it.
5. Smoke test must confirm TE/flash **sliding-window support on GH200** in the
   JUPITER container before anything else.

## Mechanics

Same validated WSD ladder machinery as `../swa_base.yaml` (see `../README.md`):
one stable run per ladder to 0.8×D_max with checkpoints exactly at every
branch point, cooldown jobs branch via FileExistsCondition + `ckpt_step` +
`override_opt_param_scheduler` and decay linearly over the final 20% to 0.
All 14 configs validated through `build_execution_plan` (stage counts, branch
iters, 20% decay ratio, gbs divisibility).

## Analysis

Per (size, budget) cell: our end-of-decay val loss vs the same-combo baseline
value in `all_losses_multilingual.csv` (near-ties listed there; treat <~0.005
nats as noise — baseline has no seed repeats). Then fit loss(N, D) per variant
and compare exponents: "not worse" = per-cell deltas ~0; "scales the same" =
matching fit exponents. FLOPs/token caveat: SWA reduces attention FLOPs
slightly — recompute FLOPs/token for compute-optimal fits; loss-vs-tokens
comparisons are unaffected. For KV-cache claims quote window 1024 vs 4096
context on 5/6 (or 7/8) of layers.

## Downstream evaluation (`eval_downstream_all.yaml`)

`eval_all_scaling_all.yaml` scores val loss / PPL. `eval_downstream_all.yaml`
scores the same 124 cooldown finals on **`dclm-core-22`** and the
**`oellm-multilingual`** super group with oellm-evals — two stages per point
(`evaldclm`, `evalml`), each gated on that point's converted
`model.safetensors`. Full write-up:
`experiment/004_downstream_eval_arch_scaling/log.md`.

The enabling piece is a **custom HF architecture** covering all five variants,
`oellm_autoexp/hf_export/oellm_hybrid/` — attention / sliding-window attention /
GatedDeltaNet / mLSTM / Mamba2 selected per layer via `mixer_types`, calling the
same `fla` / `mlstm_kernels` / `mamba_ssm` kernels the runs trained with.
`convert_megatron_to_hf.py` reads the `torch_dist` checkpoint through PyTorch DCP
(no Megatron runtime needed) and writes a `trust_remote_code` model dir.

One-time cluster prep (login node, needs internet):

```bash
bash scripts/setup_eval_env_jupiter.sh --prefetch     # venv + HF cache + datasets
python scripts/convert_arch_scaling_to_hf.py          # 124 Megatron ckpts -> HF dirs
```

Verify a conversion against the Megatron path before trusting any numbers:

```bash
bash scripts/korbi/check_hf_megatron_parity.sh <train_run_dir>
```

Launch (pilot first, then the rest):

```bash
HYDRA_STAGED_SWEEP_WORKERS=1 PYTHONPATH=. \
  python scripts/run_autoexp.py --submit-and-exit --config-name \
  experiments/architecture_scaling_variants/multilingual/eval_downstream_all \
  --array-subset 0,1         # one point, both suites; widen once timings are known
```

Known gaps, both external:
- `facebook/flores` is **gated**; the account behind `$HF_HOME/token` has to
  accept the terms or the two flores-200 groups cannot download.
- `wsc273` loads through the removed `winograd_wsc` loading script, so it fails
  under `datasets>=4` (1 of dclm-core-22's 21 tasks).

# Recall surgery on the ambitious v4 run (2026-11-11)

Goal: lift **CoQA / SQuAD v2 / Winogrande** on the `ambitious_v4_2ph_hd256` run by an architecture
intervention applied to the existing checkpoint (brain surgery, not a restart), without giving up the
complex eigenvalues the CKDA layers exist for.

Everything below is measured, not projected. Numbers are % unless stated.

---

## 1. Where the model is

`ambitious_v4_2ph_hd256`, 23-24k steps ≈ 1.56T tokens of a 12T budget, 92 layers, **6 attention layers**
(0, 17, 33, 50, 66, 82) and 86 `complex_kda` layers (`hybrid_layer_pattern` in
`config/experiments/korbi/production/hybrid_main_mn5.yaml`), head dim 256, 16 query heads / **2 KV groups**,
**NoPE**, 360 experts top-k 5, seq 8192, gbs 8192.

FLAG evals (`flag_evals_oellm`, `config/eval_tasks/ckdab_fast.yaml`) — the three rows with the largest gap
to the 32B reference:

| model | CoQA EM | CoQA F1 | Winogrande | SQuAD v2 F1 |
|---|---:|---:|---:|---:|
| v4 hd256, 23000 | 49.53 | 66.24 | 64.01 | — |
| v4 hd256, 23500 | 50.42 | 66.13 | 62.51 | 26.89 |
| v4 hd256, 24000 | 49.82 | 65.90 | 65.82 | — |
| Qwen3-14B-Base (reference) | 56.25 | 73.43 | 73.80 | — |
| Qwen3-30B-A3B (reference) | 56.93 | 73.02 | 72.45 | — |
| Qwen2.5-32B (reference) | 62.25 | 76.57 | 75.85 | 42.91 |

Gap to the 32B: **≈ +8 pp Winogrande, +15 F1 CoQA, +16 F1 SQuAD v2**. That is the bar.

The same suite on the other rows is *at or above* the 32B (HellaSwag 79.84 vs 79.35, PIQA 83.79 vs 83.90,
BoolQ 87.06 vs 87.03). So the deficit is not "the model is worse at commonsense"; it is concentrated in
**span copy (CoQA, SQuAD v2)** and **pronoun/coreference (Winogrande, WSC273 75.46 vs 86.81)**.

## 2. What was already tried, and why it could not have worked

`production/ckda_branch/*` (2026-11-09/10), read out at 23450 / 23499 against `drop`:

| arm | what | CoQA F1 | Winogrande | SQuAD v2 F1 |
|---|---|---:|---:|---:|
| main 23500 | — | 66.13 | 62.51 | 26.89 |
| `drop` | complex_kda faded out at 9/25/41, nothing in its place | 66.26 | 63.06 | 28.85 |
| `bgqa` | attention **instead of** complex_kda at 9/25/41 | 65.61 | 62.59 | 27.44 |
| `cgqa` | the same at all six 3:1 positions | 64.48 | 63.54 | 27.84 |
| `d` | KronLSTM instead of complex_kda at 9/25/41 | 63.31 | 64.17 | 26.00 |
| `e` | kvshare attention instead of complex_kda | 65.07 | 63.38 | — |

Every delta is inside ±1.5 pp of the `drop` control, i.e. noise at n = 500 (CoQA) / 1267 (Winogrande).

Two structural reasons, both visible in the configs:

1. **All of them replaced.** `ckda_branch_alpha_end: 0.0` with `crossfade` drives
   `out = alpha·ckda + (1-alpha)·branch` to `branch` alone: the measurement is "attention *instead of* the
   linear mixer", which also removes the layer's memory. "Attention *in addition to* it" was never run.
2. **They were 99-199 steps.** 99 steps at gbs 8192 × 8192 tokens is 6.6B tokens of adaptation for a layer
   whose function has to be re-learned; the `drop` control shows the network barely notices 3 of 86 layers
   (deep redundancy), so a 3-6 layer change *cannot* move a benchmark in 200 steps.

## 3. What the 1.7B ladder says about attention for these tasks

The jupiter campaign (`eval_megatron_ladder/results`, same data/recipe/tokenizer across arms; Winogrande at
the stable phase end, 50BT):

| arm (attention fraction) | 588M | 983M | 1.7B |
|---|---:|---:|---:|
| `skda-shipped-lowrank` (0%) | 57.38 | 59.91 | **63.14** |
| `kda-sig-lowrank` (0%) | 58.56 | 60.77 | 62.98 |
| `skda-shipped-hybrid` (25%) | 56.67 | 60.69 | 60.85 |
| `kda-sig-hybrid` (25%) | 59.19 | 60.54 | 60.06 |
| `attn` (100%) | 55.09 | 57.54 | 58.80 |
| `attn-qknorm` (100%) | 55.41 | 56.35 | 59.35 |

Winogrande **falls monotonically with attention fraction**, ~4 pp from 0% to 100% at 1.7B. Attention is not a
free good for coreference; a hybrid has to *add* retrieval capacity without diluting the linear path. (CoQA and
SQuAD v2 are not in that campaign — they are being run on the same 1.7B exports, see §6.)

## 4. The intervention: additive parallel attention (+ optional RoPE)

`config/experiments/korbi/production/hybrid_v4_addattn_mn5.yaml`, checkpoint surgery from iter 23300.

```
out = complex_kda(x) + m(t) · (gamma_scale · gamma · calib) · attention(x)
```

* `ckda_branch_crossfade: false`, `alpha_end: 1.0`, `alpha_ramp: 0` → **complex_kda stays at full weight**;
  every complex eigenvalue of every layer survives, including the early layers the architecture depends on.
  This is the constraint the campaign is built around: nothing here trades the signed/rotating spectrum for
  attention.
* `calib` is measured on the first forward (RMS(ckda)/RMS(branch)); `gamma` starts at 0.5 (surgery
  `--gamma-init 0.5`), so the layer's output starts at ~1.5× its RMS instead of jumping to 2×.
* The branch's weights are **not random**: `add_ckda_branch.py` derives q / k / v / o / gate / qk-norm gains
  from the layer's own complex_kda projections, so at step 0 it is already a functioning attention layer over
  the same features, and 300 steps of training are spent adapting a working path rather than bootstrapping one.

Arms (each 18 nodes of the 72-node `mn5_pool_o134c` host, gbs 8192, 23300 → 23900, 300 steps):

| arm | what it tests |
|---|---|
| `ctrl` | paired control: same resume, same batches, nothing changed |
| `add6` | attention added at all six 3:1 positions (9, 25, 41, 58, 74, 90) |
| `add6_rope` | `add6` + `position_embedding_type: rope` on the stack's attention layers |
| `rope` | RoPE only (no branch): the attention path is NoPE today, and position is what span copy and pronoun order need. Zero new tensors |
| `add3` / `add6b` | dose: top three positions only / full mixer weight (queued behind the first four) |

Provenance: surgery `ckda_branch_surgery_mn5 aux.mode=prod aux.arms="add6 add3 add6b"
aux.root=.../production/ckda_addattn` (1 node, `scripts/korbi/production/ckda_branch_surgery.sh`), each output
carrying `ckda_branch_surgery.json` (source `.metadata` SHA-256, seed 1234, per-tensor SHA-256; `verify` re-derives
it). Launch:

```
cd $R/autoexp_add1 && scripts/korbi/mn5_autoexp.sh scripts/run_autoexp.py --no-monitor \
    --monitor-state-dir $R/pool_state --pool /gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_o134c \
    --config-name experiments/korbi/production/hybrid_v4_addattn_mn5 --yes "aux.only=[ctrl,add6,rope,add6_rope]"
```

No Megatron argument was added, so `config/backend/megatron/*` and `oellm_autoexp/backends/megatron/
config_schema.py` (the generated artifacts) are unchanged and do not need regenerating.

## 5. Operational notes

* The 72-node host `mn5_pool_o134c/host002` is **held** (`pool.json hold_hosts: true`, `control/hold`,
  `POOL_HOLD_HOSTS=1`) so it is never released while idle between arms.
* Runs in that pool that were not part of this campaign were cancelled (pool runs only — no SLURM job was
  touched, no job was requeued or cancelled), and the finished ones archived to `runs_done/`.
* Failures seen: `rc 86` = the preflight node check excluded bad nodes and the pool restarted the run on
  others (the host has lost ~14 of 72 nodes to it); `rc 1` = a node watchdog SIGTERM during startup. Both are
  handled by the pool's retry; they are not arm errors.
* Monitoring: `scripts/korbi/production/monitor_addattn.sh` appends a status block every 2 minutes to
  `$R/monitor_addattn.log` (pool state, per-arm ckpts, `iteration N/M`, the `[ckda_branch]` lines with
  `alpha / gamma / rms_ckda / rms_branch`).
* A node watchdog kill leaves the arm to be retried; a run that ends `rc 0` at 23900 is the readout.

## 6. Evaluation plan (same harness as the baselines)

Per arm: HF export of the 23900 checkpoint → FLAG `ckdab_fast` (`coqa`, `winogrande`, `squadv2` 10-shot) →
compare against `ctrl` and against `ambitious_v4_2ph_hd256_fp8_23500/24000`. Readouts at 23500 and 23900 are
saved (`save_interval: 200`), so a winner can be shown at the same step as the existing 23500 row.

Smaller scale: the 1.7B ladder's stable-phase-end exports (`wsd50B`) are being copied to MN5
(`hf_models/ladder17b/`) with the gpt-neox-20b tokenizer (`scripts/korbi/production/copy_ladder17b_to_mn5.sh`)
to put CoQA and SQuAD v2 on the same attention-fraction axis as Winogrande in §3 — the cheaper setting that
says whether "more attention" is worth pushing further before it is paid for at 3B-active.

## 7. Status (2026-11-11 01:45)

* Surgery: `add6`, `add3`, `add6b` **done** (iter 23300, all three in `production/ckda_addattn/`).
* Running: `add6` at 32 nodes (the flagship) and `add6_rope` at 18 nodes, both loading the 23300 checkpoint;
  `ctrl` / `rope` / `add3` / `add6b` queued behind them.
* The host is the limiting factor, not the config: 15-23 of its 72 nodes fail the thorough node check or die
  during torchrun rendezvous, so the pool spends its retries on bad nodes. Sizing was reduced to one or two
  arms per round (18-32 nodes) after 18-node arms died 4-6 times in a row at startup.
* **1.7B three-task eval — blocked by facilities, not by code.** The exports already exist on jupiter
  (`eval_megatron_ladder/export/<arm>`, 3.2 GB each, `SignedKDAForCausalLM`, `attn.layers [3,7,11,15,19,23]`)
  and the gpt-neox-20b tokenizer is at `/e/scratch/jureap59/cache/tokenizer/gpt-neox-20b`, but:
  1. every jupiter job of this account is `PENDING` with `StartTime=N/A` (including 1-node eval jobs
     submitted hours ago and two 516-node 32B jobs), so its GPUs cannot run the eval;
  2. the copy cannot be moved instead: jupiter reads at 31 MB/s but this host's uplink to MN5 runs at
     **~0.2 MB/s** (305 MB in 28 min), i.e. ~4.5 h per 3.2 GB export.
  What is left of that requirement is the axis that already exists on jupiter: Winogrande for six arms, six
  budgets and six rungs (§3). CoQA / SQuAD v2 at 1.7B need either a jupiter allocation or a faster route.

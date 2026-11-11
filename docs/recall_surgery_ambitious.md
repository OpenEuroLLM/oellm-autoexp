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

## 1.1 Where inside SQuAD v2 the loss sits (measured, 2026-11-11)

The FLAG SQuAD v2 row is 10-shot extractive QA over 11,873 documents, **half of which are unanswerable**. The
per-subset numbers in the result JSONs (flagship `results/squadv2_10/*/results_*.json`) separate the two halves:

| checkpoint | SQuAD v2 F1 | HasAns F1 | **NoAns F1** | best_f1 (threshold-calibrated) |
|---|---:|---:|---:|---:|
| v4_2ph 6000 | 29.32 | 58.31 | **0.40** | 50.50 |
| v4 hd256 10000 | 30.03 | 59.57 | **0.57** | 50.43 |
| v4 hd256 11000 | 30.83 | 61.24 | **0.51** | 52.51 |
| v4 hd256 23500 | 26.89 | — | — | — |

Span extraction is not the problem: **HasAns F1 is 58-61 %**, close to the reference model's. What fails is the
label-following half: the model answers a passage-only question even when the prompt's ten demonstrations show
the "unanswerable" reply, and detects unanswerability **0.4-0.6 % of the time** (it is right about 17 % of the
time on that subset only because some of those answers miss). Two consequences:

* the SQuAD v2 gap to the 32B (F1 42.90) is almost entirely a **10-shot in-context label-following** gap, the
  capability an induction/copy path supplies — which is what the additive attention branch is aimed at, and why
  `add6` is the first arm rather than a capacity-only change;
* `best_f1` (the harness's threshold-calibrated variant) is 50-52, i.e. the model's own confidence *does*
  separate the halves. It is a readout/decision failure, not missing knowledge.

The readout for every arm therefore tracks `HasAns_f1` **and** `NoAns_f1` separately, not just the aggregate —
an arm that lifts `NoAns_f1` from 0.5 to 20 moves the aggregate by ~10 pp with no change in extraction.


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

### 3.1 CoQA at 1.7B: attention *presence* is worth +12 F1 (measured 2026-11-11)

Same rows and harness (HF backend, `recall_surgery/ladder17b_evals`), the pi agent's 1.7B exports:

| 1.7B arm | attention | Winogrande | CoQA EM | CoQA F1 |
|---|---:|---:|---:|---:|
| `skda-shipped-lowrank` (pure CKDA) | 0 % | **62.83** | 43.33 | **55.95** |
| `skda-shipped-hybrid-lowrank` | 25 % | 61.01 | 55.72 | 68.40 |
| `kda-sig-hybrid-lowrank` | 25 % | 59.91 | 56.82 | 70.70 |
| `attn` | 100 % | 58.33 | 52.97 | 65.04 |

* CoQA needs *some* attention: pure CKDA -> 25 % hybrid is **+12.4 F1**; more than 25 % does not help.
* Winogrande goes the other way (-2 at 25 %, -4.5 at 100 %).
* A 1.7B / 50 BT hybrid (CoQA F1 68.4) **matches the 84B-A2.9B ambitious run at 1.6T (66.1)**. Ambitious has
  48 mixer layers, 6 attention + 42 CKDA (**12.5 %**, 7:1; §1's "86 complex_kda" counts the MoE positions):
  halfway between the pure stack and the 25 % hybrid. `add6` (branches at 9/25/41/58/74/90) brings it to
  exactly 25 %; `cgqa` reached 25 % by *replacing* the same six CKDA layers and gained nothing in 200 steps
  (CoQA F1 64.48). So the open question is less "which dose" than "how many tokens does surgically added
  attention need": the decisive small-scale test adds attention branches to the **pure-CKDA** 1.7B model
  (every 4th layer, the hybrid's placement) and tracks how much of the +12 F1 comes back per BT (§8).

### 3.2 q35c on all three rows, and where the closer-mention bias comes from (2026-11-11)

q35c (25 % RoPE-gated attention, same data) at 59.6k / 1.0T: **CoQA 45.70 / 63.13, Winogrande 67.32,
SQuAD v2 23.41 / 30.05 (HasAns 59.64)** -- vs ambitious at 1.0T (15k): 50.43 / 65.99, 62.59, 23.67 / 29.80
(58.95). So 25 % attention buys *nothing* on CoQA / SQuAD at this scale (q35c is worse on CoQA); the 1.7B
+12 is the step from *no* attention to some, which ambitious' 12.5 % already has. Only Winogrande separates
(+4.7), and that is the closer-mention bias (exp 016: ambitious picks the closer entity ~68 %, q35c / 32B ~50 %).

Closer-mention pick rate of the **1.7B** models (lm-eval samples + winogrande validation, `closer_bias` rule):

| 1.7B | acc | picks closer | acc gold=closer | acc gold=farther |
|---|---:|---:|---:|---:|
| pure CKDA | 0.628 | 0.509 | 0.639 | 0.618 |
| CKDA hybrid (NoPE attention) | 0.610 | 0.503 | 0.615 | 0.605 |
| KDA hybrid | 0.600 | 0.498 | 0.600 | 0.601 |
| attention (RoPE) | 0.583 | 0.528 | 0.613 | 0.555 |

**No bias at 1.7B** -- neither NoPE attention nor CKDA produces it. What ambitious has that the 1.7B models,
q35c and the 32B do not: **Engram** (hashed 2/3/4-gram tables, 80B rows, feeding the residual at layers 2 and
49), present since the earliest checkpoints that already show the bias; also mHC-4 / LDM streams, XSA, MuOwn.
A local n-gram memory that boosts continuations of the most recent tokens is the natural mechanism for a
"pick the closer mention" preference. **Test (inference only, zero cost):** the 23500 export evaluated with
the Engram layers' output scaled by 0 and 0.5 (`OELLM_ENGRAM_OUT_SCALE`, vLLM c518d32e9d; result dirs
`flag_evals_oellm/ambitious_v4_2ph_hd256_fp8_23500_eng{0,05}`, `ckdab_fast` rows, o134c).
If the bias and the rows move, the intervention is an Engram *damping / gating* surgery (zero new
parameters, CKDA untouched), trained over 23300-23500.

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

## 4.1 Corrections and the in-house comparison (2026-11-11 morning, session -98)

**SQuAD v2 extraction is NOT at the reference level.** §1.1 compared against an older number; on the same
harness the answerable half is far behind every reference, and unanswerable detection is broken for all of
them (so it cannot be the gap):

| model | CoQA EM / F1 | Winogrande | SQuAD v2 EM / F1 | HasAns F1 |
|---|---:|---:|---:|---:|
| v4 hd256 23500 (main) | 50.42 / 66.13 | 62.51 | 21.48 / 26.89 | 52.94 |
| v4 hd256 24000 (main) | 49.82 / 65.90 | 65.82 | 20.66 / 26.14 | 51.52 |
| Qwen3-30B-A3B-Base (MoE, ~3B active) | 56.93 / 73.02 | 72.45 | 31.46 / 38.39 | 76.32 |
| Qwen3-14B-Base | 56.25 / 73.43 | 73.80 | 31.64 / 38.69 | 77.43 |
| Qwen2.5-32B | 62.25 / 76.57 | 75.85 | 30.23 / 38.22 | 76.49 |

The closest reference, Qwen3-30B-A3B (same active size, but ~36T tokens vs our 1.6T), leads by +7 CoQA F1,
+7-10 Winogrande, +11.5 SQuAD F1 (+23 on the answerable half): a span-extraction / coreference gap.

**All replace-arms, including attention initialized at random, are noise** (23499 readouts):
bgqa-random (`production/ckda_branch/bgqar`, 256 nodes, own random q/k/v/o) CoQA 50.77 / 65.99, Winogrande
63.54, SQuAD 21.54 / 27.14 -- the same as `bgqa` (copied init) and `drop`.

**In-house evidence that the layout matters:** the q35c conservative run (Qwen3.5-like: gated DeltaNet 3:1,
**25 % gated attention**, 16q / 2kv x 256 with output gate -- the *same* attention shape as ours -- but
**partial RoPE**, no XSA), trained on the same FLAG data, reaches **Winogrande 67.96 at 59.6k steps
(~1.0T tokens)**, against 62.5-65.8 for ambitious at 1.58T, and shows no closer-mention bias (exp 016).
Its CoQA / SQuAD v2 rows were never run; the `ckdab_fast` round on `q35c_gqa_59600` is queued (p264).
The differences to ours are exactly what the additive / RoPE arms of §4 test: attention fraction
(25 % vs 6.5 %) and position (RoPE vs NoPE); plus XSA and GDN vs CKDA.

**The main run has stopped improving on these rows since ~13k steps** (FLAG evals, same harness):

| main step (tokens) | CoQA F1 | Winogrande | SQuAD v2 F1 | HasAns F1 |
|---|---:|---:|---:|---:|
| 13000 (0.87T) | 68.21 | 64.56 | 30.59 | 59.78 |
| 15000 (1.0T) | 65.99 | 62.59 | 29.80 | 58.95 |
| 18000 (1.2T) | 65.10 | 62.12 | 31.42 | 61.26 |
| 21000 (1.4T) | 62.90 | 63.46 | 29.32 | 57.57 |
| 23500 (1.58T) | 66.13 | 62.51 | 26.89 | 52.94 |

At equal tokens (1.0T) q35c's Winogrande is 67.96 vs our 62.59 (+5.4). More tokens alone will not close
the gap; SQuAD's answerable half even fell 61 -> 53 since 18k.

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

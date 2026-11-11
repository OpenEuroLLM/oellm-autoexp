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

### 3.1b Same-size public reference on the 1.7B harness (2026-11-11 evening)

`ladder17b_eval_mn5` entries `ref_*` (stock HF through `--hf-auto`, same lm-eval wrapper, tasks, shots and SQuAD
v2 first-4000 subset as §3.1):

| model | Winogrande | CoQA EM / F1 | SQuAD v2 EM / F1 | HasAns F1 | belebele (5) |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B (~18T tokens) | 63.14 | **61.92 / 75.33** | **36.05 / 40.34** | **77.47** | 75.20 |
| Qwen3-1.7B-Base (~36T tokens) | 63.77 | **64.62 / 79.10** | 34.30 / 38.07 | 73.11 | 79.89 |
| 1.7B CKDA hybrid, 50 BT | 61.01 | 55.72 / 68.40 | 17.50 / 21.96 | 42.13 | ~27 |

A 1.5B Qwen is at its own 32B sibling's level on CoQA / SQuAD v2 (FLAG harness: 76.57 / 38.22) and well above the
84B-A2.9B ambitious run (66.13 / 26.89; harnesses differ, SQuAD subset here), while its Winogrande (63.1) is at
ours and far below the 32B (75.9). **CoQA and SQuAD v2 behave like data / recipe properties (reading-comprehension
content + annealing), not capacity or architecture**; Winogrande is the row that scales -- and the one our levers
(Engram fix, deep Engram off) move.

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

### 3.3 Engram scaled at inference (23500 export, no training) and the CKDA diagnostics

| 23500 export | CoQA EM / F1 | Winogrande | picks closer | acc gold=farther |
|---|---:|---:|---:|---:|
| as trained | 50.42 / 66.13 | 62.51 | 0.668 | 0.458 |
| Engram output x0.5 | **53.32 / 69.09** | 65.11 | 0.662 | 0.491 |
| Engram output x0 | 52.55 / 67.99 | **66.61** | **0.613** | **0.553** |

Engram carries part of the closer-mention bias and costs CoQA / Winogrande points; the rest of the bias sits
in weights that trained around Engram. User (2026-11-11): Engram helps most other tasks -- removal is not the
goal. Consequences: the full-removal training arm `eng0` was stopped before it trained (its step had to be
`scancel`ed: o134c host002's supervisor ignores the pool's `stop` marker -- cancels there need the run's step
ended); **`eng05`** (Engram output 1 -> 0.5 over 50 steps from 23300, `hybrid_v4_engram_damp_mn5`, Megatron
ea309b639) trains on the 32-node slot, readouts every 50 steps; its cost on the other tasks has to be
measured before it is a candidate.

**Is CKDA itself the problem?** The 1.7B evidence says not much (CKDA vs bounded KDA: Winogrande level,
CoQA -2.3 within noise; no closer-mention bias at 1.7B; GDN q35c worse on CoQA), but its distinctive pieces
were never ablated *inside the big run*. Inference diagnostics on the 23500 export (vLLM 802137570c, default
off): `OELLM_CKDA_BETA_CAP1` (beta <= 1 -- logits clamped at 0 under allow_neg_eigval) and
`OELLM_CKDA_NO_SIGN` (no signed-decay gauge: no negative / rotating eigenvalues), alone and together,
optionally from `OELLM_CKDA_DIAG_FROM_LAYER` on; result dirs `..._23500_ckda_{betacap1,nosign,both}`.

**Result of the CKDA inference diagnostics (23500 export, no training):**

| 23500 export | CoQA EM / F1 | Winogrande | SQuAD v2 F1 (HasAns) |
|---|---:|---:|---:|
| as trained | 50.42 / 66.13 | 62.51 | 26.89 (52.94) |
| beta capped at 1 | 44.48 / 59.89 | 58.41 | 23.20 (43.76) |
| no signed-decay gauge | 51.52 / 66.84 | 61.25 | 27.17 (53.38) |
| both | 41.75 / 57.00 | 58.56 | (running) |
| Engram x0 (for reference) | 52.55 / 67.99 | 66.61 | 27.36 (53.05) |

* The **negative / rotating eigenvalues are not the problem**: removing the sign gauge is neutral (+0.7 CoQA,
  -1.3 Winogrande, inside noise).
* **beta > 1 is load-bearing** for exactly these rows (capping it costs 4-6 points everywhere), consistent with
  exp 015 (beta>1 ablation +0.031 loss). CKDA's delta-rule strength is *helping* retrieval.
* So CKDA is not what holds these rows back; Engram is the one component whose damping improves them (CoQA /
  Winogrande; SQuAD unchanged, +0.5).

**Belebele (24 languages, 5-shot) of the same inference variants** vs the main run on the same languages:

| 23500 export | belebele | diff | languages down |
|---|---:|---:|---:|
| Engram x0.5 | 76.84 | **-5.95** | 22 / 23 |
| Engram x0 | 53.14 | **-29.6** | 21 / 22 |
| CKDA beta cap 1 | 72.99 | -9.8 | 23 / 23 |
| CKDA no sign gauge | 82.27 | -0.6 | 18 / 24 |
| CKDA both | 72.37 | -10.9 | 21 / 21 |

**Engram carries multilingual reading** (user's caution confirmed): at inference, x0.5 trades +3 CoQA / +2.6
Winogrande for -6 belebele -- a bad trade. The remaining question is whether *training* at x0.5 (`eng05`)
recovers belebele while keeping the CoQA / Winogrande gain; read out at 23350 / 23400 (belebele first).
The CKDA sign gauge is neutral everywhere (belebele -0.6); beta > 1 is load-bearing everywhere.

**Which Engram layer** (`OELLM_ENGRAM_SCALE_LAYERS`, vLLM 63b5fcc8fe; the other layer keeps scale 1):

| 23500 export | CoQA EM / F1 | Winogrande | SQuAD v2 F1 (HasAns) | belebele diff |
|---|---:|---:|---:|---:|
| Engram x0, both layers | 52.55 / 67.99 | 66.61 | 27.36 (53.05) | -29.6 |
| Engram x0, layer 2 only | 50.40 / 65.96 | 64.33 | 26.87 (52.19) | **-30.3** |
| Engram x0, layer 49 only | 50.88 / 66.25 | **66.06** | 26.66 (52.44) | **-0.05** |

The early Engram (layer 2) carries **all** of the multilingual-reading benefit (belebele -30 alone) and gives back
only +1.8 Winogrande. The **deep Engram (layer 49) is free to remove**: Winogrande +3.6, belebele -0.05 (8 up / 10
down), CoQA / SQuAD unchanged. The CoQA gain of removing both (+1.9) needs both layers off.

### 3.4 The constant learning rate: an untested confound (2026-11-11 afternoon)

The main run has never decayed its lr (`hybrid_main_mn5`: lr = min_lr = 8.5e-4, `lr_decay_style: constant`);
every reference it is held against (Qwen3, Qwen2.5) is annealed. On the 1.7B ladder the 10 BT WSD decay alone
moves exactly these rows more than any intervention (§8): stable end 40 BT -> decayed 50 BT export, CoQA F1
60.0 -> 68.4, SQuAD v2 F1 18.0 -> 22.0, Winogrande 59.5 -> 61.0; the surgical attention arms moved CoQA by
<= 1 F1 in 1000 steps (§8 readout). `hybrid_v4_lr_decay_mn5` (30530b38) measures it on the big run: the main
run's own batches (train_iters / sampler / gbs unchanged), lr annealed to 0 with WSD's 1-sqrt tail over the last
100 steps (`dec100`, from 23400) or 200 steps (`dec200`, from 23300) to 23500, `override_opt_param_scheduler`
(the checkpoint's scheduler is constant-style). `dec100` is paired with the main run's own 23500 (same batches).
This is not an architecture change, but every surgery arm's readout is confounded by it: an intervention that
helps a decayed model is what matters for the final model, and if the decay closes much of the gap, the bar for
the intervention is different.

**First `dec100` launch did not decay** (o134c 1806, snapshot ckdab35): `learning rate: 8.500000E-04` at 23401-23404
with `lr_decay_style WSD`, `lr_wsd_decay_iters 100`, `override_opt_param_scheduler True` all in the args. The
scheduler is right in isolation (same arguments, a dummy optimizer: 7.65e-4, 7.30e-4, 7.03e-4 at 23401-23403);
the optimizer load restores the checkpoint's *param groups*, including their per-group `max_lr` / `min_lr`, and the
main run's groups carry `min_lr = lr = 8.5e-4` -- every group interpolates between its own bounds, so the decay
was a no-op. `override_opt_param_scheduler` only overrode the scheduler's own fields. Fixed in Megatron c9c04235f
(`load_checkpoint` resets the per-group bounds to the arguments' under the override flag; unit test
`tests/unit_tests/test_override_lr_bounds.py`); `dec100` relaunched from ckdab37 (the 5 constant-lr steps
discarded, no save had been written). Any earlier schedule change on a resumed run under this override was
affected the same way.

**c9c04235f was itself a no-op; real fix Megatron 2c8a1a0f4.** The relaunch (o134c 1855) printed "per-group lr
bounds reset to the arguments (0 of 8 groups differed)" and still logged 8.5e-4, with `min_lr=0.00085` on groups
0-6 after the load. The reset kept references to the pre-load group dicts and wrote into them after the load, but
`torch.optim.Optimizer.load_state_dict` replaces every group dict. Every Megatron optimizer's load ends there: the
Megatron-FSDP path in `DistributedOptimizer.load_state_dict` and the plain distributed optimizer alike. So the
optimizer's live groups kept the checkpoint's `min_lr = lr`. The unit test missed this because its fake load
mutated the dict in place. 2c8a1a0f4 (`load_checkpoint` in `megatron/training/checkpointing.py`) captures the
bounds by position and writes them into `optimizer.param_groups` as the optimizer holds them after the load. The
unit test now loads through torch's `load_state_dict` and fails on c9c04235f.

The fix was verified locally on a tiny GPT through the dec100 code path: Megatron-FSDP, `fsdp_dtensor`,
`muown_classical`, `--phase-transition-variable-batch` and `--data-sampler-reference-dp-size`. It trained at
constant lr 1e-3 = min_lr for 6 steps, then resumed with WSD minus_sqrt over 4 steps, `--min-lr 0` and the
override:

| Megatron commit | groups reported as differing | logged lr at 7-10 |
|---|---|---|
| c9c04235f | 0 of 3 | 1e-3 at every step |
| 2c8a1a0f4 | 3 of 3 | 5.0e-4, 2.93e-4, 1.34e-4, 0 (the exact 1-sqrt values) |

`dec100` should now log "7 of 8 groups differed" and lr 7.65e-4 at 23401. Group 7 is the Engram table group: it
keeps its current-argument bounds on load, which is why the 25000 Engram x5 -> x1 restart did take effect.

The same restore affects more than the lr bounds. Under Megatron-FSDP the whole group metadata is restored from
the checkpoint, so a resume that changes a per-group lr multiplier (`residual_mixer_lr_mult`, embedding / output
lr mults) is silently ignored without the override flag. A changed `wd_mult` (`embedding_wd_mult` /
`moe_router_wd_mult`) is ignored with or without it. Adam betas / eps are re-applied only for
`optimizer == adam`, not for `muown_classical`'s Adam groups.

### 3.5 Attention temperature (inference, 23500 export; vLLM `OELLM_ATTN_LOGIT_SCALE`)

| attention softmax scale | CoQA EM / F1 | Winogrande | SQuAD v2 F1 (HasAns) |
|---|---:|---:|---:|
| x1 (as trained) | 50.42 / 66.13 | 62.51 | 26.89 (52.94) |
| x1.1 | 50.57 / 66.75 | 62.43 | (running) |
| x1.2 | 53.38 / **69.36** | 62.83 | (running) |
| x1.25 | 53.85 / **69.25** | 62.19 | 25.98 (50.92) |
| x1.5 | 51.85 / 66.89 | 62.35 | 25.38 (50.03) |
| XSA off | 29.85 / 43.95 | 65.04 | -- |

A sharper softmax in the six NoPE attention layers reads conversational spans better: CoQA F1 +3.2 at x1.2 and
x1.25 (two neighbouring points agree), overshooting at x1.5; Winogrande flat, SQuAD v2 -0.9, belebele -2.1 at x1.25
(22 / 23 languages down). Training arm `attn_temp/ls120` (hybrid_v4_attn_temp_mn5: x1.2 from 23400 to 23500, the
main run's batches; insertion cost +0.007 / +0.005 loss at 23401 / 23402) measures whether the network keeps the
CoQA gain and recovers belebele; combined inference variant `combo_L49off_ls120` (deep Engram off + x1.2) running.
Megatron 35f74ae9c adds `attention_logit_scale` (factor on the softmax scale) for a training arm. XSA is
load-bearing for CoQA (the model was trained with it); removing it is not an option.

**The main run after the Engram fix (25000 restart: Engram Adam lr x1 instead of x5, own clip, row-norm cap 13 -> 7):**
26000 CoQA 50.30 / 66.61, Winogrande 65.82, SQuAD v2 28.07 (HasAns 55.74); **30000 CoQA 50.75 / 67.05, Winogrande
67.32** (24000: 49.82 / 65.90, 65.82, 26.14). Winogrande at 30000 is q35c's level (67.32) -- the run improves on its
own after the Engram fix, which is the baseline every 23.4k-23.5k intervention has to be read against.

**1.7B decay probe** (`dec_*` arms, ladder17b_continue_mn5: each arm's step-2000 save resumed with optimizer and
data position, lr 5e-4 -> 1e-5 linearly over 1000 steps = 0.5 BT):

| arm | CoQA F1 s2000 -> s3000 | SQuAD v2 F1 | Winogrande |
|---|---:|---:|---:|
| `hybrid_ctrl_fresh` | 59.64 -> **64.67** | 15.56 -> 18.79 | 58.56 -> 59.75 |
| `hybrid_add_lin_late` | 59.68 -> 64.59 | 13.65 -> 18.98 | 58.17 -> 59.67 |
| `pure_ctrl_fresh` | 45.46 -> **51.14** | 12.79 -> 11.53 | 60.62 -> 61.88 |
| `pure_add_every4` | 42.92 -> 49.95 | 13.17 -> 11.41 | 60.77 -> 61.80 |

A 0.5 BT anneal is worth +5 to +6 CoQA F1 at 1.7B in both stacks (hybrid SQuAD v2 +3.2) -- more than any surgery
arm -- and the added attention branches are worth nothing under the decay either (pure: -1.2 vs its control,
hybrid: -0.1). **1.7B verdict: attention added by surgery does not become a copy path in 1.5 BT, with or without
the anneal; the anneal itself is the large lever.**

### 3.6 Why CKDA hybrids trail KDA hybrids on copy tasks: not attention routing (2026-11-11 evening)

Hypothesis: in the KDA hybrid the six NoPE attention layers carry the copy job (bounded decay cannot hold content),
while the CKDA hybrid (signed / complex eigenvalues, beta > 1) keeps more of it in the recurrent state and its
attention is less sharp. Predictions: (1) zeroing attention hurts the KDA hybrid's CoQA / SQuAD much more;
(2) sharpening attention helps the CKDA hybrid more. Test: the 1.7B 50 BT hf exports, same harness as §3.1
(`ladder17b_eval_mn5` entries `diag_{kda,ckda}_hybrid_{noattn,ls110,ls120,ls130}`; `--ablate-attn` = the
original codebase's `ablate_attention`, every attention sublayer's output zeroed, residual + MLP kept;
`--attn-logit-scale X` = flash-attn `softmax_scale` x X in all six layers, hit-checked per layer).

| 1.7B hybrid | variant | Winogrande | CoQA EM / F1 | SQuAD v2 F1 | HasAns F1 |
|---|---|---:|---:|---:|---:|
| KDA (`kda-sig-hybrid-lowrank`) | as trained | 59.91 | 56.82 / 70.70 | 21.57 | 41.42 |
| | attention zeroed | 57.93 | 19.05 / 28.22 | 9.76 | 18.54 |
| | scale x1.1 | 60.30 | 57.20 / 70.34 | 22.35 | 42.91 |
| | scale x1.2 | 59.83 | 57.95 / **71.18** | 22.58 | 43.35 |
| | scale x1.3 | 59.75 | 57.63 / 70.88 | **22.77** | **43.69** |
| CKDA (`skda-shipped-hybrid-lowrank`) | as trained | 61.01 | 55.72 / 68.40 | 21.96 | 42.13 |
| | attention zeroed | 59.75 | 19.65 / 28.76 | 10.05 | 18.86 |
| | scale x1.1 | 61.17 | 56.63 / 68.90 | 22.40 | 42.97 |
| | scale x1.2 | 61.17 | 56.87 / **69.07** | 22.77 | 43.73 |
| | scale x1.3 | 61.09 | 56.70 / 68.73 | **22.89** | **43.96** |

(SQuAD v2: first 4000 docs, 10-shot. Belebele 5-shot is at chance at 1.7B, 25.5-28.6 in every variant, not read.)

* **Prediction (1) fails.** Zeroing attention drops CoQA F1 by 42.5 (KDA) vs 39.6 (CKDA) to the *same floor*
  (28.2 vs 28.8), and SQuAD v2 F1 by 11.8 vs 11.9 (9.8 vs 10.1). Both hybrids route the copy job through attention
  to the same degree; the CKDA state does not carry measurably more of it. The floor is far below the pure CKDA
  model (CoQA F1 55.95, §3.1): a hybrid trained with attention never learns the recurrent copy path the pure stack
  has. Winogrande drops only 2.0 / 1.3, so the ablated networks still work (the collapse is specific to copying).
* **Prediction (2) fails.** Sharpening helps both about equally and only a little: CoQA F1 +0.5 (KDA) vs +0.7
  (CKDA) at x1.2; SQuAD v2 F1 +1.2 vs +0.9 and HasAns F1 +2.3 vs +1.8 at x1.3, monotone up to x1.3 in both;
  Winogrande flat. The CoQA gap stays (71.18 vs 69.07 at x1.2), so the CKDA hybrid's attention is not
  under-sharp relative to KDA's. The 1.7B temperature gain is much smaller than the ambitious one (+3.2 CoQA F1 at
  x1.2, §3.5).
* The premise holds only for CoQA: at 1.7B the CKDA hybrid is *not* behind on SQuAD v2 (21.96 vs 21.57, HasAns
  42.13 vs 41.42); the gap is CoQA F1 -2.3 (EM -1.1), with Winogrande +1.1 the other way. Whatever causes it,
  it is not a difference in how much the model leans on attention or how sharp that attention is; the remaining
  candidates are the recurrent part's contribution *with* attention present (conversation-history tracking
  in CoQA's multi-turn questions) or single-seed noise.

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

## 4.2 Additive-branch export (ckda_attn blocks, 2026-11-11)

The additive arms (`crossfade false`, `alpha_end 1`: complex_kda at full weight plus an attention branch)
were not exportable (`branch_fold` raised). Now:

* HF: a block kind `ckda_attn` (pattern symbol `A`) = the CKDA mixer (`fla_layer`) and an attention of the
  stack's kind (16q / 2kv x 256, gate, qk norm, XSA, NoPE) on the same normalised input, `ckda(x) + attn(x)`;
  decoding state `(ckda state, (k, v))`. The exporter (`branch_add`) folds alpha into the CKDA `o_proj` and
  `m * gamma_scale * gamma * calib` into the branch `o_proj` (autoexp ddb0e425).
* vLLM: `OellmCkdaAttn` (CKDA state slot + the branch's KV cache), off the decode-graph paths
  (vllm 2fe52d3c76).

Validation (small model `add6_t`: the add6 layout on the hd256 test model, 1100 -> 1300,
`hybrid_v4_ckda_branch_test_mn5`):

| check | result |
|---|---|
| HF full forward vs cached decoding (local, tiny, fp32) | max |dlogit| 6.5e-5 (scale 32) |
| HF `ckda_attn` == ckda + attn (separately) | 0.0 |
| sub64 loss HF vs Megatron | 3.322261 vs 3.321716, **Δ 5.4e-4** (d 5.2e-4, bgqa 4.3e-4) |
| vLLM vs HF on the add6_t export, tp1 | prefill max 0.133, chunked <= 0.115, 0 positions > 0.5, greedy identical, NLL Δ <= 0.011 |
| tiny components `ckda_attn` / `ckdab_add_mix`, tp1 and tp4 | greedy identical; one 300-token prompt with a 2.1 / 2.7 spike at 3 positions (210-214), NLL Δ 0.006: the tiny MoE routing flip seen for kvshare too |

**Production export:** `hf_exports/ckdaadd_add6_fp8_23406` (+ `views/identity/ckdaadd_add6_fp8_23406`),
`ckda_attn` at layers 9 / 25 / 41 / 58 / 74 / 90, check_export clean (0 missing / unexpected / shape),
`.downstream_ready`. The checkpoint is **23406**, not 23400: the add6 run (o134c.1569) was cancelled at 10:54 and
its graceful-stop save at 23406 is the only one in `ckda_addattn/add6/ckpt` (no 23400 directory exists).
Fast round:

```
cd $R/autoexp_ckaddexp1 && $R/autoexp_env/bin/python scripts/evals/launch.py config/evals/ckdab_fast_mn5.yaml \
    --export ckdaadd_add6_fp8_23406 --set paths.pool=/gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_o134c
```

## 4.3 KV-group duplication (2 -> 8 / 16 KV groups, function-preserving; 2026-11-11 evening)

**Why.** The copy path lives in the six NoPE attention layers, which run 16 query heads x 256 on only **2 KV
groups** (8 query heads share one key / value head). A sharper softmax at inference buys +3.2 CoQA F1 here (§3.5)
but only +0.5 at 1.7B, where attention is MHA (§3.6): the big model's attention looks diffuse / KV-starved. The
surgery gives every pair (kv8) or every single query head (kv16 = MHA) its own K/V head, starting as an exact copy
of the shared one, so the model is unchanged at insertion and the copies can specialise in a continuation. No
new mechanism: the same attention, more KV heads.

**Layout** (Megatron `attention.py`): `SelfAttention.get_query_key_value_tensors` (l. 2119-2259) views the fused
`linear_qkv` output as `[sq, b, ng, (2 np/ng + 2) hn]` and splits each group into `[q heads | gate heads | k | v]`
(gate = `attention_output_gate`; `linear_qkv_out_dim` l. 1948-1950, the fused qk-norm path `_mixed_qkv_for_fusion`
l. 885-896 uses the same layout). Query head h reads KV group h // (np/ng) everywhere: flash / TE GQA, Megatron's
fused XSA (`fused_xsa.py` l. 37-38 `repeat_interleave`, l. 60-62), the HF export's `split_gated_qkv`
(`export_q35.py` l. 106-117) / HF modeling (`modeling_oellm_ambitious.py` `repeat_interleave`) and vLLM
(`OellmXSAAttention`, `num_kv_heads=config.num_key_value_heads`). Nothing else depends on `num_query_groups`:
q/k_layernorm are `[256]` (shared over heads), `linear_proj`'s input is 16 x 256 in query-head order, XSA projects
each query head on its own group's value, no value embeddings / Canon / qk-clip / sinks in the run.
`muon_qkv_headwise` (`optimizer/__init__.py` l. 938-950, `_get_qkv_split_shapes` in `emerging_optimizers.py`
l. 161-177) orthogonalizes every 256-row head block alone, so block shapes and the Newton-Schulz scale do not
change.

**Transform** (`scripts/korbi/production/duplicate_kv_groups.py`, per attention layer 0 / 17 / 33 / 50 / 66 / 82):
`linear_qkv.weight` [9216, 2048] -> [12288, 2048] (kv8) / [16384, 2048] (kv16): new group j holds q / gate heads
j·h' .. (j+1)·h'-1 (h' = 16 / G', re-ordered, values unchanged) and k / v of old group j // r (copied r = G'/2
times). Optimizer state (`muown_classical`, checkpoint keys `...linear_qkv.weight.{g, v_norm, m_u, m_g, v_g}`,
`[rows, 1]` except `m_u` `[rows, 2048]`): q / gate rows move with their weights; on the k / v copies `g` and
`v_norm` are copied and the moments **split** over the copies, `m_u`, `m_g`, `v_g` / r: dL/dk_old = the sum of
its copies' gradients (selftest, exact), the copies are exchangeable at insertion (first moments / r exact in
expectation) and each copy's gradient is nearly uncorrelated with the others' (|grad copy| ≈ |grad| / sqrt(r),
measured 0.48-0.56 at r = 4), so Adam's second moment / r too. `--moments copy` keeps everything verbatim
(momentum r x too strong for ~1/(1-beta) steps). `args.num_query_groups` = G'. Every other key (317,973 in total)
is untouched: the new tree hard-links the source's 1024 `.distcp` shards (`dcp_surgery.write_surgery`, source-
sized row chunks) and adds files with the 36 rewritten tensors; the main run's directory is only read.
Deterministic (no randomness): `verify` re-derives every tensor and compares the manifest's SHA-256.

**Exactness evidence.**

| check | result |
|---|---|
| `selftest` (CPU, fp64 hand-written gated GQA + qk RMSNorm + XSA + NoPE reference), 2 -> 4 / 8 / 16 groups | max \|out - ref\| / max \|ref\| <= 3.7e-16; every q / gate / k / v block where Megatron reads it; states as specified; copies' gradients sum to the original's (4e-15) |
| `megatron-test` (local GPU: Megatron `HybridModel` '*' layer, TE linear_qkv with fused input norm, unfused core attention, fused XSA, output gate; random layer 16 x 128) | 2 -> 8 and 2 -> 16, fp32 and bf16, also `fused_qk_norm_rope`: attention output and logits **bit-identical (0.0)**; negative control (two q heads swapped) 0.40 rel L2 |
| `export-test` (local GPU) | exporter split of the duplicated layout: q / gate identical, k / v = the 2 heads repeated; tiny HF OellmAmbitious 2 vs 4 KV heads: max \|dlogit\| 4e-5 (scale 38), cached decoding 1.7e-5 |
| fake fsdp_dtensor checkpoint (local): apply / verify / gpucheck | read-back exact, verify 0 differ, Megatron output identical |
| production `gpucheck` (the real 23400 layers through Megatron's attention, TE flash, fp32 + bf16, seq 4096) | in the surgery job, pool run o134c.1954 (queued 20:10) |
| end-to-end, eval-only, token level (`fixedvalid_eval_kvdup_v4_mn5`, 8 nodes, EP 8, sub64 v8ml + Winogrande/WSC273 requests, `MEGATRON_EVAL_TOKEN_DUMP`; `duplicate_kv_groups.py compare-dumps`) | queued: s23400 = o134c.1953, s23400kv8 enqueued (starts when kv8_src exists) |
| end-to-end, training: kv8's first step (23401) vs `lr_decay/dec100`'s 23401 (same 32-node layout, same batch, unchanged model) | reference **1.123511** (dec100, identical in two attempts o134c.1806 / .1855; ls120 at x1.2: 1.130263); kv8 must land within bf16 noise of it when it runs |

The HF export (`export_ambitious.py` takes `num_key_value_heads` from the run's `num_query_groups`) and vLLM
(`oellm_ambitious.py` reads `config.num_key_value_heads`; 8 KV heads at TP 4 = 2 per rank) need no change.

**Paths.** Source `production/main_mhc4/ambitious_e360_eng80_v4_2ph_hd256/ckpt_eval/iter_0023400` (read only);
outputs `/gpfs/scratch/ehpc533/hybrid_nucleus/production/attn_kv/kv{8,16}_src/ckpt/iter_0023400` (+
`latest_checkpointed_iteration.txt`, `kv_dup_surgery.json`: source path + `.metadata` SHA-256, arguments, code
stamp, old / new shapes, SHA-256 of every rewritten tensor); surgery logs `attn_kv/surgery_logs`.

**Launch.** Surgery (1 node, priority 40) and the eval-only check (8 nodes, priority 45, kv arms gated on the
surgery's `latest_checkpointed_iteration.txt`) from snapshot `autoexp_kvdup1`:

```
cd $R/autoexp_kvdup1 && scripts/korbi/mn5_autoexp.sh scripts/run_autoexp.py --no-monitor \
    --monitor-state-dir $R/pool_state --pool /gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_o134c \
    --config-name experiments/korbi/production/kv_dup_surgery_mn5 --yes
cd $R/autoexp_kvdup1 && scripts/korbi/mn5_autoexp.sh scripts/run_autoexp.py --no-monitor \
    --monitor-state-dir $R/pool_state --pool /gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_o134c \
    --config-name experiments/korbi/production/fixedvalid_eval_kvdup_v4_mn5 --yes "aux.only=[s23400,s23400kv8]"
```

Training arms (`config/experiments/korbi/production/hybrid_v4_kvdup_mn5.yaml`, 32 nodes, mbs 2, gbs 8192, 23400
-> 23500 on the main run's batches, saves every 50; **not enqueued** -- the 32-node slot is busy with
attn_temp/ls120 and lr_decay/dec100):

```
cd $R/autoexp_kvdup1 && scripts/korbi/mn5_autoexp.sh scripts/run_autoexp.py --no-monitor \
    --monitor-state-dir $R/pool_state --pool /gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_o134c \
    --config-name experiments/korbi/production/hybrid_v4_kvdup_mn5 --yes "aux.only=[kv8]"
```

`kv8` = 8 groups, `kv16` = MHA. No `kv8_lr` arm: Megatron has no attention-only lr multiplier (only Engram /
embedding / output layer / residual mixer), and none was invented for this.

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

## 8. 1.7B continuation testbed (2026-11-11, session -98 fork)

The small-scale testbed for the surgery arms: the native 1.7B ladder checkpoints at the **stable-phase end**
(signed_kda `megatron_ladder_medium/ckpt/<model>_1.7B_gbs128/iter_0076295`, ~40 BT, WSD lr 5e-4 constant,
gbs 128 x 4096, 24 layers, hidden 2048, NeoX tokenizer), continued in the **ladder's own code** on MN5.

**Assets on MN5** (`/gpfs/scratch/ehpc533/hybrid_nucleus/recall_surgery/`):
* `ladder17b_native/{skda-shipped-hybrid-lowrank,skda-shipped-lowrank,attn}/iter_0076295` -- native torch-format
  checkpoints with optimizer (23 GB each), copied with hpctransfer (`--split-size 1G`: 7.5 min per checkpoint).
* `native_megatron_ladder/` -- the ladder's exact Megatron tree (jupiter `signed_kda/external/oellm-autoexp/
  submodules/Megatron-LM`, 514b31b, the path in the 1.7B logs). 018's `native_megatron` is only its `megatron/` +
  `pretrain_gpt.py` (no `gpt_builders` / `model_provider`); `native_megatron_bb5febb82` (the other oellm-autoexp
  checkout on jupiter) is the WRONG revision: no `_import_mcore_async`, and the checkpoints' pickled args need
  `InferenceCudaGraphScope` which it lacks.
* `ladder17b_hf/<arm>_1.7B_wsd50B` -- clean copies of the jupiter HF exports (config.json from jupiter, weights
  linked from `hf_models/ladder17b`, whose config.json had been overwritten by a gpt-neox config link).
* lm_scaling / fla: experiment 018's copy (`/gpfs/projects/ehpc533/users/korbip/ckda_retrieval_018/source`);
  intervention arms use an immutable patched code dir (`code/v1`, `native_patch/make_code_dir.sh`).
* Data: MN5's NeoX-tokenized Nemotron 318BT sample (train) and its valid prefix -- a different sample than
  jupiter's; tokenizer identical (018).

**Configs** (`config/experiments/korbi/recall_surgery/`):

| config | what |
|---|---|
| `ladder17b_continue_mn5` | continuation (`scripts/korbi/recall_surgery/ladder17b_train.sh`): `aux.arm` sweep; `aux.load_mode` resume (optimizer + scheduler + iteration) or model_only (`--finetune --no-load-optim --no-load-rng`, fresh Adam, linear warmup `aux.warmup`=100 to 5e-4, then constant); `aux.steps` relative to the start; `aux.gbs` / `aux.nodes` / `aux.save_interval`; `aux.code_dir` + `aux.intervention` (-> SKDA_INTERVENTION) + `aux.src` (start checkpoint) per arm. A pool restart resumes from the run's own saves. Loud: exits 1 unless the last saved iteration equals the target. |
| `ladder17b_surgery_mn5` | 1-node job, all surgery arms in sequence: builds `code/v1` once, then `native_patch/surgery.py` -> `ladder17b_surgery/<arm>` (model-only copy with the new tensors, o_proj zero, `recall_surgery.json`) |
| `ladder17b_eval_mn5` | readout: export (`export_megatron.py`, skipped for `aux.ckpt: none`) -> Winogrande (0-shot, 1267) / CoQA (0-shot, 500) / SQuAD v2 (10-shot, first 4000 docs, 2 shards, HasAns/NoAns F1) on one node, one task per GPU, through the original `eval_downstream.build_lm` + `lm_scaling/eval_tasks`. Loud: missing result file -> exit 1. `ladder17b_table.py <dirs>` prints the table. |

Arms wired in `ladder17b_continue_mn5`: `ctrl_resume`, `ctrl_fresh` (hybrid, from the native checkpoint);
`pure_ctrl_fresh`, `pure_add_every4`, `pure_add_every4_rope` (pure CKDA, ffn 8064); `hybrid_ctrl_fresh`,
`hybrid_add_lin_late`, `hybrid_attn_rope` -- the surgery arms start from `ladder17b_surgery/<arm>`.

**Validation (1 node, gbs 128):**
* `ctrl_resume` from 76295: loss 2.226 / 2.240 / 2.230 / 2.251 / 2.233 (steps 76296-76300), grad norm 0.10-0.15,
  lr 5e-4 -- continuous with the original run's end (jupiter log: mean of the last 30 logged losses 2.245, grad norm
  0.11-0.16). The data sample differs; the loss level does not.
* `ctrl_fresh` (model only): warmup 5e-6/step as configured, loss 2.21-2.28 over the first steps.
* Speed: **5.7 s/step on 4 GPUs** (524k tokens/step, ~23k tok/s/GPU, = jupiter's 22k); the first two steps take
  ~10 min (fla Triton autotuning). 1 BT = 1907 steps at gbs 128 ~ 3.0 h per node.

**Gotchas fixed on the way (all would have failed silently or late):**
* torch >= 2.6 `weights_only` default: the native loader prints "could not load the checkpoint" and exits **0**;
  `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` (train / surgery / eval).
* `scontrol` is not in the image (exit 127 with `set -e`); the pool step exports `MASTER_ADDR`.
* `lr-decay-iters` past the end (warmup must be < decay steps; WSD with no decay phase = constant lr).
* lm_scaling.titan_ext registers torchtitan train specs on import (titan_oellm, toml, tomli_w ...) when the image's
  torchtitan imports: shadowed by `scripts/korbi/recall_surgery/notitan/` (the exporter only needs
  `titan_ext.hybrid_layers`).
* lm-eval 0.4.13 calls `doc_to_text(doc, cfg)`; the SQuAD2 class takes `(doc)`: patched on the loaded task class and
  the patched task object passed to `simple_evaluate`.
* 018's `scripts/` dir holds an `evaluate.py` that shadows the `evaluate` library (SQuAD metric):
  `metric_compat.py` vendored to `scripts/korbi/recall_surgery/compat/`.
* The pi agent's `ladder17b_evals` (o134c 1571-1574) ended rc 0 with no results (CoQA scoring / SQuAD2 API).

**Load strictness:** the native checkpoints are **torch format** (`mp_rank_00/model_optim_rng.pt`, distributed
optimizer in `distrib_optim.pt`), not torch_dist. Its loader (`checkpointing.load_model_state_dict`) retries with
`strict=False` when a strict load raises (and prints `load_return`), so a checkpoint lacking new keys would load with
the new parameters at their init -- but the surgery arms do not rely on it: `surgery.py` writes the new tensors into
the copy, so the load is strict and complete. (`--dist-ckpt-strictness` applies only to torch_dist.)

**Baseline readout** (`ladder17b_evals/`, SQuAD v2 = first 4000 docs):

| model (attention share) | Winogrande | CoQA EM | CoQA F1 | SQuAD v2 EM | SQuAD v2 F1 | HasAns F1 | NoAns F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `hf_attn` 50 BT (100 %) | 58.33 | 52.97 | 65.04 | 11.20 | 14.04 | 26.92 | 0.05 |
| `hf_kda_hybrid` 50 BT (25 %) | 59.91 | 56.82 | 70.70 | 17.05 | 21.57 | 41.42 | 0.00 |
| `hf_ckda_hybrid` 50 BT (25 %) | 61.01 | 55.72 | 68.40 | 17.50 | 21.96 | 42.13 | 0.05 |
| `hf_ckda_pure` 50 BT (0 %) | 62.83 | 43.33 | 55.95 | (running) | | | |
| **`native_ckda_hybrid` 40 BT stable end** (the testbed's start) | 59.51 | 47.37 | 60.03 | 13.50 | 17.98 | 34.44 | 0.10 |

Winogrande agrees with the jupiter ladder within 1 pt (58.80 / 60.06 / 60.85 / 63.14). **The stable-phase end is well
below the 50 BT (decayed) export on the copy tasks**: CoQA F1 60.0 vs 68.4, SQuAD v2 F1 18.0 vs 22.0 (Winogrande 59.5
vs 61.0) -- the 10 BT lr decay alone is worth +8 CoQA F1. Arms continued at constant lr from the stable end must be
compared with `ctrl_fresh` / `pure_ctrl_fresh` at the same step, never with the decayed exports.

**Surgery + patched-code smokes** (3 steps, model-only load, identical batches; `ladder17b_smoke/`):

| arm | new tensors | step 1 loss | step 2 | step 3 | note |
|---|---:|---:|---:|---:|---|
| `hybrid_ctrl_fresh` | 0 | 2.275560 | 2.270438 | 2.236034 | = native model-only load (2.275551) |
| `hybrid_add_lin_late` | 45 (9 branches) | 2.275560 | 2.270438 | 2.236037 | exact start, moves after the first update |
| `pure_ctrl_fresh` | 0 | 2.284275 | 2.277529 | 2.242382 | |
| `pure_add_every4` (mbs 1) | 30 (6 branches) | 2.284283 | 2.277525 | 2.242393 | exact start; mbs 2 OOMs at the loss |
| `pure_add_every4_rope` (mbs 1) | 30 | 2.284283 | 2.277525 | 2.242368 | exact start |
| `hybrid_attn_rope` (mbs 1) | 0 | 2.404005 | 2.396817 | 2.357385 | +0.13 at insertion: RoPE on NoPE-trained attention |

Step time at mbs 1 (1 node, gbs 128): branch arms 6.9 s, hybrid 6.05 s (5.7 s at mbs 2). 1 BT (1907 steps) ~ 3.7 h on
1 node; on 2 nodes ~ 1.9 h (data-parallel, same global batch).

**Launch** (from a current snapshot; e.g. 2 nodes per arm, ~1 BT):

```
scripts/korbi/mn5_autoexp.sh scripts/run_autoexp.py --no-monitor --monitor-state-dir $R/pool_state \
    --pool /gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_o134c --yes \
    --config-name experiments/korbi/recall_surgery/ladder17b_continue_mn5 \
    "aux.only=[pure_ctrl_fresh,pure_add_every4,pure_add_every4_rope]" aux.steps=2000 aux.nodes=2 aux.save_interval=500
# readout of a run's checkpoint (exports it first):
... --config-name experiments/korbi/recall_surgery/ladder17b_eval_mn5 "aux.only=[native_ckda_pure]" \
    aux.name=pure_add_every4_s2000 aux.model=skda-shipped-lowrank \
    aux.ckpt=/gpfs/scratch/ehpc533/hybrid_nucleus/recall_surgery/ladder17b_runs/skda-shipped-lowrank/pure_add_every4/ckpt \
    aux.code_dir=/gpfs/scratch/ehpc533/hybrid_nucleus/recall_surgery/code/v1 aux.intervention=add_attn@every4
```
**Intervention-arm export** (`scripts/korbi/recall_surgery/export_native.py` + `hf_branch.py`, 2026-11-11):
* `export_native.py` wraps `export_megatron.py` (loader, config, renames, cross-checks) and adds the branches
  (`decoder.layers.N.self_attention.branch.X` -> `model.layers.N.attn.branch.X`, their layers read off the checkpoint
  and checked against SKDA_INTERVENTION's tokens), `config.recall_branches` (the hybrid sibling's fla attention spec +
  `use_rope` / `window_size`) and `attn_rope` (`use_rope` on the existing attention spec). **Strict key accounting**:
  every checkpoint tensor must get an HF name (TE extra state / a tied output layer aside) -- `export_megatron.rename`
  returns `[]` for any unknown key, which would have dropped the branches and evaluated the control under the arm's
  name. Then a strict load through the eval's own loader.
* `hf_branch.py`: the stock fla `SignedKDAForCausalLM` (upstream fla untouched) with an fla `Attention` attached as
  `attn.branch` to the named blocks' mixers, `out = mixer(x) + branch(x)` on the same normalized input (as the native
  `AddBranchMixer`); the branch keeps its own KV cache for generation. Installed into `export_hf.load_exported`, so
  `eval_downstream.build_lm` (and thus every readout) picks it up.
* **Paired loss check** (`ladder17b_paired_mn5`, `ladder17b_paired_check.sh`; native = `native_loss.py` through the
  training launcher, i.e. the arm's own spec, code dir and SKDA_INTERVENTION): 8 x 4096 validation tokens, step 500:

| arm | branches | native mean | HF mean | delta | max per-sequence |delta| |
|---|---|---:|---:|---:|---:|
| `pure_add_every4` | 6 (3, 7, ..., 23) | 2.610099 | 2.610101 | +2.4e-6 | 3.7e-4 |
| `pure_ctrl_fresh` | - | 2.610351 | 2.610111 | -2.4e-4 | 6.2e-4 |
| `hybrid_add_lin_late` | 9 (12-14, 16-18, 20-22) | 2.596326 | 2.596388 | +6.2e-5 | 6.7e-4 |
| `hybrid_attn_rope` | - (RoPE on 6 attention layers) | 2.612688 | 2.612500 | -1.9e-4 | 6.3e-4 |

  All PASS (tolerance 0.01 mean / 0.05 per sequence; bf16 kernel differences only).

**Arm readouts at step 1000** (0.52 BT after surgery, 2 nodes, gbs 128, mbs 1, fresh Adam + 100-step warmup to 5e-4,
constant; `ladder17b_evals/<arm>_s1000`; SQuAD v2 = first 4000 docs; Belebele = mean of eng / deu / fra / spa / ita,
5-shot -- **at chance (25 %) for every 1.7B model**, so it carries no signal at this scale). Delta = arm minus its
paired fresh control at the same step:

| arm | Winogrande | CoQA EM | CoQA F1 | SQuAD v2 EM | SQuAD v2 F1 | HasAns F1 | NoAns F1 | Belebele |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `pure_ctrl_fresh` | 62.35 | 37.23 | 49.00 | 6.65 | 9.91 | 18.93 | 0.10 | 24.58 |
| `pure_add_every4` | 61.48 (-0.87) | 37.87 (+0.64) | 49.92 (+0.92) | 6.70 (+0.05) | 10.03 (+0.12) | 19.21 | 0.05 | 24.60 |
| `pure_add_every4_rope` | 61.88 (-0.47) | 37.83 (+0.60) | 49.86 (+0.86) | 6.42 (-0.23) | 9.68 (-0.23) | 18.49 | 0.10 | 24.80 |
| `hybrid_ctrl_fresh` | 60.93 | 49.73 | 62.07 | 9.60 | 13.14 | 25.23 | 0.00 | 26.89 |
| `hybrid_add_lin_late` | 60.14 (-0.79) | 49.65 (-0.08) | 62.53 (+0.46) | 10.32 (+0.72) | 13.86 (+0.72) | 26.61 | 0.00 | 27.09 |
| `hybrid_attn_rope` | 60.14 (-0.79) | 42.63 (-7.10) | 55.43 (-6.64) | 8.12 (-1.48) | 11.67 (-1.47) | 22.36 | 0.05 | 26.98 |

Noise: Winogrande +-1.4 (n 1267), CoQA +-~1.5 F1 (500 dialogues), SQuAD v2 subset +-~0.6 F1.

* **Adding attention branches to the PURE CKDA stack does not recover the hybrid's copy ability in 0.5 BT.** `pure_add_every4`
  is +0.9 CoQA F1 / +0.1 SQuAD F1 over its control (within noise); the hybrid trained with attention from the start is at
  62.1 / 13.1 -- a 12-point CoQA gap that the branch arms do not move. RoPE on the branches changes nothing.
* **More attention on the hybrid (`add_lin_late`, 9 branches) is within noise** (+0.5 CoQA F1, +0.7 SQuAD F1, -0.8 Winogrande).
* **RoPE on the hybrid's existing NoPE-trained attention hurts** (-6.6 CoQA F1, -1.5 SQuAD F1) and has not recovered after
  1000 steps (insertion cost +0.13 loss).
* **The fresh-optimizer continuation itself moves the copy rows**: `hybrid_ctrl_fresh` at 1000 steps vs the native stable end
  (`native_ckda_hybrid`, 40 BT): CoQA F1 62.1 vs 60.0, SQuAD v2 F1 13.1 vs 18.0 (MN5's Nemotron sample differs from
  jupiter's, and the optimizer is reset). Arms are therefore only comparable to their paired control, as above.
* The step-2000 readouts (`<arm>_s2000`) are queued in o134c (priority 50) and print with
  `scripts/korbi/recall_surgery/ladder17b_table.py ladder17b_evals/*_s2000`.

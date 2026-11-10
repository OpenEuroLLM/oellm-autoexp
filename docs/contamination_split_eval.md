# Contamination split of the standard evals: ambitious v4 vs the flag 32B

Date: 2026-11-10. Question: the ambitious v4 (mHC-4, hd256) and the JUPITER `oellm_32b_dense`
flag run differ a lot on a few evals (SQuAD v2, CoQA, Winogrande, WSC273, IFEval). The ambitious
run uses the v8 mix plus the flag-decon-v2 filter; the 32B used the vanilla option5 flag mix. How
much of the difference can the extra decontamination explain?

## Data

- **Ambitious**: `ambitious_v4_2ph_hd256_fp8_24000` = 24k steps, 1.61T tokens (67.1M tokens/step).
  The main run is at 26k (1.75T), but no 25k/26k flag-evals CSV exists yet; 24k is the latest
  evaluated checkpoint.
- **Flag 32B**: `b1cons_90k` = 90k steps, 1.51T tokens, and `b1consTW_300k` = 300k steps, 5.03T
  tokens (GBS 4096 x seq 4096 = 16.78M tokens/step). `oldprod_60k` (60k, ~1.0T) is carried as an
  extra point. No `b1consTW_100k` result was found; 90k is the closest available to the 1.7T point.
- The MN5 `flag` copy is an exact copy of the JUPITER option5 flag data; only the
  `/gpfs/.../users/korbip/...` datasets are filtered. The contamination labels therefore apply to
  the 32B's data.
- Labels come from the v6 scan over the 452 vanilla flag shards (`seg_stats.npz`, rule
  70 % verbatim run / 200 abs / 20 min). `seen` = at least one reference segment of the eval item
  passes the drop rule in the vanilla mix (so it was in the 32B data and dropped from ours);
  `clean` = a matchable segment exists but none passes; `unknown` = only short fields, the matcher
  cannot tell.

The comparison tables use the same metric and n-shot for both models, and exact English task
labels (not a mean over the multilingual `hellaswag_*` / `global_mmlu_full_*` variants).

## Most different evals

Positive Δ = 32B - ambitious, in percentage points.

| task | metric | n-shot | ambitious 24k | 32B 90k (1.5T) | 32B 300k (5T) | Δ vs 90k | Δ vs 300k | direction |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| CoQA | EM | 0 | 49.82 | 66.67 | 69.12 | +16.85 | +19.30 | 32B better |
| SQuAD v2 | EM | 10 | 20.66 | 35.69 | 39.74 | +15.03 | +19.08 | 32B better |
| IFEval | prompt strict | 0 | 39.56 | 21.44 | 26.43 | -18.11 | -13.12 | ambitious better |
| SQuAD v2 | F1 | 10 | 26.14 | 39.22 | 42.91 | +13.08 | +16.77 | 32B better |
| CoQA | F1 | 0 | 65.90 | 79.72 | 81.72 | +13.81 | +15.82 | 32B better |
| WSC273 | acc | 0 | 78.02 | 84.62 | 86.81 | +6.59 | +8.79 | 32B better |
| Winogrande | acc | 0 | 65.82 | 70.32 | 70.56 | +4.50 | +4.74 | 32B better |
| GSM8K | exact (flex) | 4 | 77.26 | 72.78 | 78.70 | -4.47 | +1.44 | crossover |
| MMLU | acc | 5 | 70.28 | 66.00 | 72.89 | -4.28 | +2.61 | crossover |
| LAMBADA | acc | 0 | 72.42 | 70.15 | 71.45 | -2.27 | -0.97 | ambitious better |
| ARC-challenge | acc_norm | 25 | 63.14 | 61.43 | 64.08 | -1.71 | +0.94 | crossover |
| BoolQ | acc | 10 | 86.61 | 85.17 | 87.03 | -1.44 | +0.43 | crossover |
| HellaSwag | acc_norm | 0 | 79.40 | 78.10 | 79.35 | -1.29 | -0.05 | crossover |
| OpenBookQA | acc_norm | 0 | 45.80 | 44.80 | 45.00 | -1.00 | -0.80 | ambitious better |

The rest of the English standard crowd is on par: PIQA (-0.4 / -0.4), SocialIQA (+0.1 / +0.1),
ARC-easy (+1.1 / +2.2), CommonsenseQA (+0.1 / +1.6), Copa (+1.0 / -3.0, n=200).

Two metric caveats:
- SQuAD v2 `NoAns_exact` is 0.0 for both 32B points and 0.82 % for ambitious: the 32B never
  abstains. The overall SQuAD gap is a HasAns gap (40.6 -> 71.5 / 79.6 `HasAns_exact`), not a
  no-answer gap.
- The translation metrics (`flores200`, `opensubtitles_multi40`) in the FLAG CSV are not on a
  consistent scale in these exports (values ~100x) and are excluded from the ranking.

### Other large differences (not covered by the contamination split)

These are the other big gaps in the same CSVs. They are not reading / commonsense items, so the
v6 split does not label them; they are listed to separate the contamination question from the
broader model/data-mix differences.

| task | metric | ambitious 24k | 32B 90k | 32B 300k | Δ vs 90k | Δ vs 300k | likely |
|---|---:|---:|---:|---:|---:|---:|---|
| GPQADiamond | accuracy_avg | 54.04 | 27.10 | 27.10 | -26.94 | -26.94 | Closed-book generative QA, not reading comprehension. The 32B eval ran with `max_length=4096`; the ambitious eval with `max_model_len=40960`. The 32B outputs are truncated mid-reasoning / cut `\boxed{}` (59/594 empty at 300k), so the comparison is confounded. Re-run the 32B at the same context budget before reading this as capability. |
| MATH500 | accuracy | 22.80 | 50.60 | 43.40 | +27.80 | +20.60 | 32B much better; real math capability / data-mix difference, not flag contamination. |
| BigBench repeat-copy-logic | exact_match | 50.00 | 53.12 | 71.88 | +3.12 | +21.88 | 32B better; small n, likely capability/data mix. |
| BigBench operators | exact_match | 67.62 | 77.62 | 86.67 | +10.00 | +19.05 | 32B better; small n. |
| BigBench language-id | acc | 21.77 | 19.60 | 38.66 | -2.17 | +16.89 | 32B better at 300k; small n. |
| BigBench dyck | exact_match | 34.40 | 50.30 | 50.30 | +15.90 | +15.90 | 32B better; small n. |
| BigBench cs-algorithms | exact_match | 64.17 | 68.41 | 76.97 | +4.24 | +12.80 | 32B better; small n. |
| BigBench qa-wikidata | exact_match | 59.51 | 69.62 | 71.48 | +10.11 | +11.97 | 32B better; small n. |

The multilingual families show large mean gaps too (`global_piqa_prompted` ambitious 49.25 vs
31.28 / 19.56; `sib200` 36.18 vs 32.65 / 45.48), but they are means over 30+ languages and are not
part of the English contamination split. They should be checked per language before being read as
capability differences.

## Per-item contamination split (300k point)

`stream_split.py` streams `refs_v6.jsonl.gz`, matches the eval samples by normalized passage /
question text, and assigns each eval item its v6 label. Paired gap = 32B - ambitious, pp. The
split uses the per-item sample means; they can differ from the CSV aggregate by a few hundredths
of a point (e.g. WSC273 +8.86 here vs +8.79 in the CSV table).

| task | metric | n | seen % | gap all | gap seen | gap clean | gap unknown | contamination-explained* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SQuAD v2 | EM | 11873 | 69.5 | +19.08 | +18.23 | +21.00 | — | **0 % (negative)** |
| SQuAD v2 | F1 | 11873 | 69.5 | +16.77 | +16.08 | +18.36 | — | **0 % (negative)** |
| CoQA | EM | 500 | 67.4 | +19.30 | +20.28 | +17.28 | — | **10 %** |
| CoQA | F1 | 500 | 67.4 | +15.82 | +17.21 | +12.93 | — | **18 %** |
| Winogrande | acc | 1267 | 1.0 | +4.74 | -7.69 | +4.91 | +4.84 | **0 %** |
| WSC273 | acc | 271 | 1.1 | +8.86 | 0.00 | +15.07 | +6.67 | **0 %** |
| BoolQ | acc | 3270 | 86.5 | +0.43 | +0.14 | +1.88 | +13.33 | gap too small |
| Belebele | acc_norm | 900 | 36.0 | +1.89 | +1.54 | +2.08 | — | ~0 % |
| HellaSwag | acc_norm | 10042 | 18.5 | -0.05 | +0.32 | -0.14 | -0.09 | gap ~0 |
| PIQA | acc_norm | 1838 | 0.0 | -0.44 | — | +0.00 | -0.44 | no seen items |
| SocialIQA | acc | 1930 | 11.9 | +0.10 | +1.74 | +0.00 | -0.12 | gap ~0 |
| OpenBookQA | acc_norm | 500 | 1.4 | -0.80 | +14.29 | +2.50 | -1.32 | 7 seen items only |

\* Counterfactual: give the seen items the clean (and unknown) gap and recompute the overall gap;
the reported percentage is how much of the observed overall gap disappears. Negative means the
clean-group gap is larger than the overall gap, i.e. contamination explains none of it.

At the 90k point the picture is the same: SQuAD EM +15.03 all / +14.32 seen / +16.64 clean;
CoQA EM +16.85 / +18.64 / +13.14 (contamination ~22 %); Winogrande +4.50 / 0.00 / +3.57;
WSC273 +6.27 / -33.33 / +10.96. The extra 60k point (`oldprod_60k`) shows the gap growing with
the 32B token budget (SQuAD EM +11.4 -> +15.0 -> +19.1; CoQA F1 +13.5 -> +13.8 -> +15.8;
Winogrande +2.6 -> +4.5 -> +4.7; WSC273 +1.5 -> +6.6 -> +8.8), which is the signature of a real
capability gap rather than a fixed data leak.

## Reading

- **SQuAD v2: contamination explains none of the gap.** 69.5 % of the eval items have a
  reference segment that was in the 32B's vanilla mix, but the 32B is *more* better on the clean
  items (+21.0 EM) than on the seen items (+18.2 EM). The same holds for F1. The 32B's advantage
  is therefore a reading-comprehension capability gap, not a decontamination artifact.
- **CoQA: contamination explains only a small part.** The extra gap on seen items is +3.0 EM /
  +4.3 F1, i.e. ~10 % / ~18 % of the overall gap. The clean gap is still +17.3 EM / +12.9 F1.
- **Winogrande: contamination explains nothing.** Only 1 % of items are seen; the gap is in the
  clean (+4.9) and unknown (+4.8) groups. This agrees with experiment 016: the ambitious model has
  a closer-mention bias (picks the option mentioned last), the 32B does not. It is a model
  behavior, not a data-leak effect.
- **WSC273: contamination explains nothing.** 1.1 % seen; the gap is on clean items (+15.1) and
  unknown items (+6.7). Also consistent with the pronoun-resolution / closer-mention behavior.
- **The other standard tasks are on par**, so there is no hidden contamination signal to explain.
- **IFEval goes the other way** (ambitious +13 to +18 pp): that is the v8 instruction / reasoning
  mix, not contamination.
- **The other large gaps are outside this split.** GPQA Diamond (32B -27 pp) is closed-book
  generative QA and the 32B run was capped at `max_length=4096` vs the ambitious
  `max_model_len=40960`; its outputs are visibly truncated, so this is an eval-setting confound,
  not a contamination effect. MATH500 (32B +21 pp) and the BigBench tasks are capability /
  data-mix differences, not flag contamination.

Caveat: the v6 rule catches verbatim runs of >=70 % (or >=200 tokens). The `clean` group can still
contain short fragments or reordered copies that the rule does not flag. But since the gap is not
larger on the `seen` group, fragments are unlikely to be the main driver. A 26k ambitious eval
(1.75T, matching the 32B 100k / 1.68T point) and a `b1consTW_100k` eval would tighten the
token-matched comparison.

## Reproduction

`scripts/korbi/contamination_split/stream_split.py` is a stdlib-only streaming replacement for
`items.jsonl.gz` + `analyze.py` (the full `items.jsonl.gz` build was killed after `seg_stats.npz`;
the streaming version never materializes the 18.6M reference texts):

```bash
python3 scripts/korbi/contamination_split/stream_split.py \
  --work /gpfs/scratch/ehpc533/users/korbip/flag_decon_v6 \
  --stats /gpfs/scratch/ehpc533/users/korbip/contamination_split/v6/seg_stats.npz \
  --samples /gpfs/scratch/ehpc533/users/korbip/contamination_split/samples \
  --ref b1consTW_300k --ours main_24000 \
  --extra b1cons_90k oldprod_60k main_23500 \
  --out /gpfs/scratch/ehpc533/users/korbip/contamination_split/v6/split_full.json
```

Samples: `samples/main_{23000,23500,24000}` on MN5; `b1consTW_300k`, `b1cons_90k`, `oldprod_60k`
copied from JUPITER `contamination_split/samples/`. Output: `split_full.json` and
`stream_split.log` in the same directory. Local mirrors: the three Jupiter CSVs are in
`dump/flag_summaries/`, and the split outputs / paired table are in
`dump/flag_summaries/contamination_split/`.

# Decontamination comparison: flag-decon-v1 vs. the OpenEuroLLM "-pure" sets

Date: 2026-10-28. Data: `agenttrove-0.0` and `openthoughts-3` of the flag mix on MareNostrum 5, as
decontaminated by

- **ours**: `flag-decon-v1` (`scripts/flag_decontam/`, method and history in
  `docs/flag_post_decontamination.md`, sections 2 and 4), and
- **theirs**: the OpenEuroLLM pretraining-decontamination pipeline (NeMo Curator
  `TaskDecontamination`, repository commit 413335e), released as
  `flag/agenttrove-0.0-pure` and `flag/openthoughts-3-pure`.

## 1. Summary

- They remove **15x (agenttrove) and 7x (openthoughts) more documents** than we do. The overlap is
  small: they also remove 49 % / 18 % of our removals; we remove 3.1 % / 2.6 % of theirs.
- Both use **exact matching of normalised word sequences**. Neither uses edit distance, MinHash or
  embeddings. The difference is the **trigger** and the **reference list**:
  - theirs: **one** benchmark n-gram (8-13 words) that is rare in the dataset (<= 10 occurrences)
    removes the whole document;
  - ours: **one benchmark item mostly copied in a single verbatim run** (>= 70 % of the item, or
    >= 200 tokens, and >= 20 tokens) removes the document.
- **agenttrove: the gap is mostly the trigger.** 75 % of their extra removals contain a phrase from
  our references (control: 7.5 %), but only 67 of 4,062 contain >= 30 % of any one item. Samples
  are generic code and contest boilerplate.
- **openthoughts-3: the gap is mostly the reference list.** 60 % of their extra removals share no
  n-gram with any of our 5.9M items; the trigger is a benchmark we do not index (most likely full
  Hendrycks MATH, DeepMind Math, GSM-Symbolic) or a match shorter than our 20-token minimum.
- **GSM8K and MATH500 are clean after either filter** (no full copies remain). Our rule still flags
  546 documents in their sets for MATH500, but these are a false positive of ours (shared
  Asymptote plotting code, 544 docs) and one modified problem (2 docs).
- Each method has its own blind spot: theirs over-removes on generic phrases; ours misses copies
  whose sections are reordered and over-removes on boilerplate shared across benchmark items.

## 2. The two methods side by side

| | ours (flag-decon-v1) | theirs (-pure) |
|---|---|---|
| Unit matched | token ids mapped to a word-level canonical stream (lower-cased, non-alphanumerics removed) | words of the normalised text (lower-cased, punctuation removed) |
| N-gram size | 13 / 24 / 40 tokens, the smallest spanning >= 30 characters | 13 words; a whole short text of 8-12 words as one n-gram |
| Reference items | 5.89M segments, every text field of every row (question, context, options, answer, tests), deduplicated, from 165 sources | n-grams of 98 English task entries (`benchmarks/english_benchmarks.yaml`) |
| Splits | test / validation / dev; `train` only when it is the only split | per task, mostly test / validation; some full datasets (Jeopardy train) |
| Common-phrase filter | n-grams > 50 times in a 105B-token background sample, or in > 10 benchmark segments, are dropped | n-grams matched > 10 times in the dataset being cleaned are ignored |
| Trigger | the longest verbatim run of one item: >= 70 % of the item, or >= 200 tokens; >= 20 tokens | any one remaining n-gram |
| Action | drop the whole document | drop the whole document (pure sets are exact subsets: 0 modified documents) |
| Generic hold-back | items that would drop > 20,000 documents are not used (one AlpacaEval prompt) | none |
| Shortest item matched | 20 canonical tokens | 8 words |
| Output | rewritten Megatron `.idx`, original `.bin` symlinked | re-tokenised Megatron shards (`256k-without-bos`, `256k-reason-without-bos`) |

Consequences of the trigger:

- **Theirs** counts any one shared phrase. A 13-word phrase like "You can output the answer in any
  case (for example, the strings ...)" or a common Python idiom is enough. The frequency cut-off
  (<= 10, `--match-threshold`) is relative to the cleaned dataset: in a 1.2-1.6M-document set, a phrase repeated 11
  times survives, so a benchmark item copied 11+ times is **not** removed by its common n-grams
  (only by its rare ones).
- **Ours** needs one item mostly reproduced in order. A changed word in the middle splits the run,
  so modified items (cos -> sin, other numbers) stay by design. Copies with sections moved around
  also stay (each piece < 70 %). Long items whose boilerplate (plotting helpers, test harnesses,
  prompt templates) is shared with other texts trigger the >= 200-token branch.

## 3. Benchmark coverage

Their English list has, among others, these benchmarks that **we do not index** (checked against
all 165 source attributions of our references):

- math: **Hendrycks MATH test (5,000; MATH500 is a subset)**, DeepMind Math (test), GSM-Symbolic,
  GSM8K-Platinum (a re-labelled GSM8K test), AIME 2026, BasicSkills;
- code: DS-1000, LBPP, MT-MBPP;
- QA / reasoning: BBH (we index six BIG-bench tasks of the FLAG suite via `hails/bigbench`, not
  the 23 BBH tasks), LogiQA, DROP, SimpleQA, SuperGPQA, Humanity's Last Exam, LongBench-v2,
  MedQA, LAB-Bench, CulturalBench;
- chat / safety: MT-Bench, HarmBench, DoNotAnswer, ToxiGen, BBQ, Winogender.

Commented out in their list: LMArena (55k / 100k / 140k), OpenSubtitles, Tatoeba, HELMET,
MMLU-CF, SQuAD, BigBench.

We index many benchmarks **they do not**: the 438 FLAG evals (incl. the multilingual ones:
Global-MMLU, MGSM, PolyMath, Belebele, FLORES, ...), the oellm-eval groups outside FLAG, MixEval,
full SciRIFF, WikiAnn, MultiLoKo in all languages, Tatoeba, NorIdiom. (Their multilingual task files
exist, 38 languages; which were applied to these two English datasets is not recorded in the
release.)

## 4. Results

The pure sets are complete (`.bin` size equals what the `.idx` implies; leftover `*.partial`
second copies stopped growing at 15:24 on 2026-10-28) and exact subsets of the originals: every
kept document matches an original document token for token.

| | agenttrove-0.0 (1.57M docs, 20.2B tok) | openthoughts-3 (1.20M docs, 20.2B tok) |
|---|---|---|
| removed by them | 4,192 docs / 52.4M tok (0.26 %) | 5,013 / 84.2M (0.42 %) |
| removed by us | 265 / 3.0M (0.015 %) | 746 / 12.3M (0.061 %) |
| both | 130 / 0.87M | 132 / 1.86M |
| theirs only | 4,062 / 51.5M | 4,881 / 82.3M |
| ours only | 135 / 2.2M | 614 / 10.5M |

"Removed by us" here applies the finalize rule without the generic hold-back, so agenttrove
shows 265 instead of the 254 dropped in flag-decon-v1 (one AlpacaEval prompt).

**Both:** the clear copies: GSM8K / MGSM / PolyMath (51 agenttrove, 80 openthoughts) and
MMLU / MMLU-Pro / Global-MMLU items.

### 4.1 Why they remove more (their rule applied to our references)

`explain_pure.py`: for each group, the distinct n-grams of our table each document contains (no run
requirement, i.e. their trigger with our references), and the best-matching item.

| | agenttrove theirs-only | openthoughts theirs-only | control kept by both (5,000) |
|---|---|---|---|
| contains any table n-gram | 75 % | 40 % | 7.5 % / 7.6 % |
| >= 10 n-grams of one item | 37 % | 6.5 % | 1.2 % / 0.1 % |
| >= 30 % of one item's n-grams | 67 docs | 49 docs | - |
| >= 50 % of one item's n-grams | 11 docs | 0 | - |

Best-matching benchmarks of their extra removals: agenttrove LiveCodeBench 2,117, BigCodeBench
454, HumanEval(+) 230, AlpacaEval 125; openthoughts PolyMath 626, MATH500 384, LiveCodeBench 214.

Typical theirs-only documents (samples in `compare_pure/<dataset>/explain_samples.txt`):

- generic code: `def is_prime(n)` loops, sklearn import blocks, number-to-word arrays, a
  days-per-month list that matches a LiveCodeBench stdin test case;
- contest boilerplate: "You can output the answer in any case ...", "where m and n are positive
  integers and n is not divisible by the square of any prime. Find m+n";
- tables of powers of two;
- a minority of real partial copies, e.g. a LiveCodeBench/Codeforces problem copied with `$$$`
  math and its "Note" section moved into the middle. We keep it because no single run reaches 70 %.

### 4.2 Why we remove documents they keep

Best-matching benchmarks of our extra removals: agenttrove AlpacaEval 39, BigCodeBench 31, SciRIFF
24, MBPP(+) 28; openthoughts MixEval 480, MATH500 64.

- **Not in their list:** MixEval; full SciRIFF (they index the yes/no subset only).
- **In their list but kept by them** (AlpacaEval, BigCodeBench, MBPP): likely their frequency
  cut-off. An item copied in more than 10 documents has n-grams above the threshold, so the copies
  count against themselves. Other fields than theirs (we index outputs, answers and tests too) may
  add to it.
- **Our false positives through the >= 200-token branch:** in openthoughts, one MixEval item (496
  documents) and one MATH500 solution (544 documents; mostly the same documents, as 496 + 544 >
  614) match only through the `[asy]` plotting helper both contain (`rr_cartesian_axes`, runs of
  301-412 tokens); the problems differ.

Across the whole flag mix the >= 200-token branch alone drops 183,888 documents / 3.49B of our
18.0B tokens (`abs_rule_report.py`). Its top items are boilerplate shared across items:
HumanEval+/MBPP+ test harness (`def is_floats(x)`), SciRIFF instruction templates, the MixEval
`[asy]` item, an AlpacaEval loop snippet. MultiLoKo's Marathi Wikipedia passages are real copies of
the source pages (kept by design).

## 5. GSM8K and MATH500

| | agenttrove-0.0 | openthoughts-3 |
|---|---|---|
| GSM8K docs we drop | 50, all also removed by them | 0 |
| GSM8K in their pure set (our rule) | 0 | 0 |
| MATH500 docs we drop | 8 | 544 |
| MATH500 in their pure set (our rule) | 2 (one item: modified problem, "sin" for "cos", diagram code verbatim = 75 % of the item) | 544 (one item: `[asy]` boilerplate, false positive) |
| MATH500 partial matches we keep | 29 pairs at 50-70 % coverage: modified problems (other vector / equation after the same instruction) | 0 at >= 50 % |

Neither set contains a full verbatim GSM8K or MATH500 item.

## 6. Assessment

| | theirs | ours |
|---|---|---|
| recall on verbatim copies of indexed items | high (any phrase) | high (>= 70 % run) |
| recall on reordered / partly rewritten copies | high | low (single longest run) |
| benchmarks of the FLAG suite and multilingual evals | partial | complete |
| some popular benchmarks (full MATH, DeepMind Math, GSM-Symbolic, DS-1000, BBH, ...) | yes | no |
| false positives on generic phrases / code | many | few |
| false positives on shared boilerplate inside long items | few | yes (>= 200-token branch) |
| modified items (cos -> sin) | removed if a rare phrase is shared | kept (by design) |
| cost in tokens (these two sets) | 0.26 % / 0.42 % | 0.015 % / 0.061 % |

## 7. Recommendations for a v6 (not applied; flag-decon-v1 is unchanged)

1. **Add their missing benchmarks** to our references, test splits first (Hendrycks MATH test,
   DeepMind Math, GSM-Symbolic, GSM8K-Platinum, DS-1000, LBPP, MT-MBPP, BBH, LogiQA, DROP, SimpleQA,
   SuperGPQA, HLE, MedQA, MT-Bench, ...). This closes most of the openthoughts gap without taking on
   their single-phrase rule.
2. **Union coverage:** also drop a document when the runs of one item together cover >= 70 %
   (catches reordered copies such as the Codeforces example).
3. **Boilerplate in long items:** exclude template code shared by many items (`[asy]` helpers,
   test harnesses, prompt templates) from the >= 200-token branch, or apply that branch only to the
   item's non-template part.
4. For agenttrove and openthoughts only, a document could be kept only when both filters keep it:
   0.27 % / 0.47 % of their tokens.

## 8. Reproduction

Snapshots `autoexp_decon13`-`decon16`, one `acc_debug` node each (all via `run_autoexp`,
`config/experiments/korbi/decontam/flag_decon_mn5.yaml`):

```bash
# per dataset: O = flag/<ds>/megatron-lm/data/shard_0000_text_document,
#              P = flag/<ds>-pure/megatron-lm/openeurollm-256k-without-bos/data/shard_0000
aux.step=compare aux.args="--orig $O --pure $P --out $W/compare_pure/<ds>"
aux.step=explain aux.args="--orig $O --cmp $W/compare_pure/<ds> --procs 100 --samples 10"
aux.step=absrule                                   # whole mix
# login node: samples of matches
python scripts/flag_decontam/inspect_hits.py --shard <ds> --bench MATH500 --min-cov 0.5
```

`W = /gpfs/scratch/ehpc533/users/korbip/flag_decon_v5`. Outputs: `compare_pure/<ds>/{summary.json,
docs.npz, samples.txt, explain.json, explain_docs.npz, explain_samples.txt}`,
`_report_abs_rule.json`. Their method: `README.md`, `remove_matching_ngrams_parallel.py`
(`--match-threshold 10`), `benchmarks/english_benchmarks.yaml` and `benchmarks/code/*.py`
(`min_ngram_size=8, max_ngram_size=13`) of the pretraining-decontamination repository.

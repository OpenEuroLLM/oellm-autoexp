# Extractive-QA SFT sources for the copy-behaviour experiment

Goal: install **verbatim-span answering** and **calibrated abstention** in the free-form
reading-comprehension format, then measure it in a paired SFT experiment.

## 1. Why this data, and why not the current mix

`datamix_v8` contains **no extractive-QA source** (no SQuAD, CoQA, NQ, DROP, SWDE or FDA).
The nearest entries are `P3` (0.55 %), `misc_instruct` (0.20 %) and the Nemotron
specialized/instruct blocks (~7 %), all instruction- or reasoning-shaped.

Observed consequence on the recall suite (v4b@36900, same harness as the Qwen references):

| task | prompt shape | shots | ours | Qwen3-14B |
|---|---|---|---|---|
| SWDE | schema slots `... \| category: Feature \| year:` | 0 | 83.71 | 85.69 |
| FDA | schema slots `... \| Purpose for submission:` | 0 | 30.40 | 30.76 |
| CoQA | bare `Q: ... A: ...` | 0 | 67.69 | 73.43 |
| SQuAD v2 | bare `Title/Background/Question/Answer` | 10 | 30.85 | 38.69 |

Copy behaviour is **intact** when a structural slot pins the answer; it degrades when the
answer format is free. On SQuAD v2 answerable questions our model scores F1 = 0 on 30.1 %
of samples (Qwen 11.1 %) and refuses 14.3 % (Qwen 5.8 %). The deficit is free-form span
copying plus abstention calibration, not extraction in general.

## 2. Contamination rules

Reference evals that must not leak: **CoQA, SQuAD v2, Winogrande, Belebele, SWDE, FDA,
squad_completion, RULER**.

`docs/decontamination_eval_set.csv` already lists `coqa`, `squad_v2`, `belebele` and
`winogrande` in `in_refs_v5`. It does **not** list:

- **SQuAD v1.1** (`rajpurkar/squad`) — same paragraphs and questions as SQuAD v2, so it is
  contaminated by proximity even though it is absent from the reference list;
- **SWDE**, **FDA**, **squad_completion** — our own extraction evals, currently unguarded.

So the reference list must be extended before any of this data is trained on. Procedure:

1. Add SQuAD v1.1, SWDE, FDA and squad_completion to the decon reference list.
2. Run `scripts/flag_decontam/` on every candidate file (exact normalised-word matching, one
   benchmark item mostly copied in one verbatim run removes the document).
3. Second pass at sample level: dedupe on normalised-question hash **and** context hash
   against the eval samples themselves. The eval result JSONL files carry `doc_hash`,
   `doc.question` and `doc.context`, so this check is direct and catches paraphrased
   near-duplicates that n-gram matching misses.

## 3. Candidate datasets (verified via the HF API, 2026-11-19)

### Tier A — non-Wikipedia, clean against SQuAD v2 / CoQA / Belebele

| dataset | HF id | domain | verbatim spans | size | license |
|---|---|---|---|---|---|
| NewsQA | `Maluuba/newsqa` | CNN news | yes, plus genuine no-answer items | ~120 k QA | MIT |
| TweetQA | `ucsbai/tweet_qa` | tweets | yes, short spans | small | cc-by-sa-4.0 |
| BioASQ | `kroshan/BioASQ` | biomedical | yes, factoid | small | check card (non-commercial?) |
| CUAD | `theatticusproject/cuad-qa` | legal contracts | yes, clause spans | ~13 k | cc-by-4.0 |
| QASPER | `allenai/qasper` | NLP papers | yes, answer + evidence spans | 1 585 papers | cc-by-4.0 |
| FinQA | `mihirinamdar/finqa` | financial tables | yes, cell spans | small | check card |
| TAT-QA | `FangyuLei/tatqa` | financial tables | yes, cell spans | small | check card |
| TyDiQA | `google-research-datasets/tydiqa` | multilingual Wikipedia | yes (primary task) | 2.94 GB, 240 k | apache-2.0 |

NewsQA is the single most useful entry: non-Wikipedia text, real no-answer questions (so
abstention can be tuned without SQuAD v2), and permissive licence.

### Tier B — Wikipedia-derived, usable only after question/context dedupe

| dataset | HF id | note | size | license |
|---|---|---|---|---|
| HotpotQA | `hotpotqa/hotpot_qa` | multi-hop spans + yes/no | 0.75 GB, 203 k | cc-by-sa-4.0 |
| DROP | `ucinlp/drop` | numeric/date exact spans, excellent copy signal | 87 k | cc-by-sa-4.0 |
| Quoref | `allenai/quoref` | coreference spans | ~24 k | cc-by-4.0 |
| ROPES | `allenai/ropes` | background + span answers | 14 k | cc-by-4.0 |
| MultiSpanQA | `nc33/MultiSpanQA` | multi-span answers, discourages paraphrase | small | check card |
| TriviaQA (rc) | `mandarjoshi/trivia_qa` | entities, not always spans; 34.8 GB | 914 k | licence unknown — verify |

### Excluded

| dataset | reason |
|---|---|
| SQuAD v2 | is the eval |
| SQuAD v1.1 | same paragraphs/questions as SQuAD v2 |
| CoQA | is the eval |
| Belebele, Winogrande | are evals (and not span tasks) |
| SWDE, FDA | are evals; style-adjacent document extraction would leak |
| squad_completion | SQuAD variant, is the eval |
| MS MARCO | answers are generated, not verbatim → teaches the wrong behaviour |
| NarrativeQA, ELI5 | generative answers |

### Scale warning

After exclusions the public pool is roughly **0.2–0.5 B tokens**. That is enough for a 1 B-token
SFT experiment, not for large-scale continued pretraining. Scaling path: generate synthetic
span-QA from our own pretraining corpus (disjoint from eval contexts by construction), with an
explicit abstention rate — this is also the only route that gives full control of the
span-vs-paraphrase target format at scale.

## 4. Paired SFT experiment

Same base checkpoint (e.g. the v4b@40000 HF export), same hyperparameters, same token budget,
two arms differing only in SFT data source:

| arm | SFT data |
|---|---|
| **ambitious** | Tier A + Tier B + synthetic span-QA generated from our pretraining corpus, decontaminated |
| **flag** | same task selection drawn from the FLAG collection (`flag-decon-v2` trees) |

Suggested slice: 1 B tokens, ≥60 % of it verbatim-span extractive QA, abstention share
20–25 % (deliberately below SQuAD v2's 50 % so the false-abstention prior is pulled down, not
amplified).

Readouts, all already in the recall suite:

- **false-abstention rate on SQuAD v2 answerable questions** (baseline 14.3 %) — primary;
- **SQuAD v2 HasAns F1** (baseline 60.97);
- **CoQA F1/EM** (baseline 67.69 / 52.70);
- SWDE and FDA as no-regression checks (83.71 / 30.40);
- a copy-fidelity metric: exact-span match rate on held-out extractive QA.

Prediction: HasAns F1 +5–10 pp, false abstention 14.3 % → 6–8 %, CoQA +3–5 pp, SWDE/FDA flat.
If SWDE/FDA also move, the effect is general copy reinforcement; if not, it is specific to
free-form QA. Either outcome identifies the mechanism.

## 5. Fetching

Run on the cluster, not locally. Suggested root: `$R/sft_data/` on MN5.

```bash
export HF_HOME=$R/sft_data/hf
huggingface-cli download Maluuba/newsqa --repo-type dataset --local-dir $R/sft_data/raw/newsqa
huggingface-cli download hotpotqa/hotpot_qa --repo-type dataset --local-dir $R/sft_data/raw/hotpotqa
huggingface-cli download ucinlp/drop --repo-type dataset --local-dir $R/sft_data/raw/drop
huggingface-cli download allenai/quoref --repo-type dataset --local-dir $R/sft_data/raw/quoref
huggingface-cli download allenai/ropes --repo-type dataset --local-dir $R/sft_data/raw/ropes
huggingface-cli download allenai/qasper --repo-type dataset --local-dir $R/sft_data/raw/qasper
huggingface-cli download theatticusproject/cuad-qa --repo-type dataset --local-dir $R/sft_data/raw/cuad
huggingface-cli download google-research-datasets/tydiqa --repo-type dataset --local-dir $R/sft_data/raw/tydiqa
```

None of the Tier A entries is gated. `mandarjoshi/trivia_qa` is large (34.8 GB) and has an
unclear licence, so fetch it only if the licence is resolved.

## 6. Open question

"ambitious vs flag" is read here as *two SFT data sources, one base checkpoint*. If it instead
means *our model versus the FLAG-trained model*, the pairing changes (two checkpoints, one data
set) and section 4 must be rewritten.

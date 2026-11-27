# Where the English-math deficit comes from (decon overlap analysis, 2026-11-27)

Question: the English MGSM row scores far below its translations (49.2 % vs 74.8 / 68.0 / 65.4 %
at 56 k), and MATH500 is 22.0 % plain / 62.4 % CoT against 50.6 % for the 32 B dense reference.
Earlier work falsified template mimicry and prompt artefacts (`docs/mgsm_en_probe.md`). This note
tests the remaining contamination-shaped explanation: do the decontamination overlaps point at
*questions without answers*?

## Method

The v6 reference set (`flag_decon_v6`, 18.58 M segments) already holds every benchmark in the FLAG
suite, so the recorded hits answer the question without a rescan. Three reference sources carry
math text, and they differ in what a segment contains:

| source | segments | segment content |
|---|---|---|
| `openai/gsm8k` | 2385 | question, or question + worked solution ending `#### <n>` |
| `evalchemy/MATH500` | 980 | question segments, and separate `\boxed{}` solution segments |
| `juletxara/mgsm` | 72 (en 12, es 21, fr 22, de 17) | **question only** — no solution, no answer |

Coverage = longest verbatim run / segment tokens, taken from `hits/*.npz` (max over documents per
segment). A document that carries question + solution lands at ≈ 1.0; one that stops where the
question stops lands below 0.4, because the solution is several times the question length.

Two traps were hit and fixed, both worth remembering:

- `grep -n` is 1-based while segment ids are 0-based. Using the raw line number shifts every
  language label by one segment and scrambles the small MGSM block (72 segments).
- `np.savez` members are plain `.npy`; they can be parsed without numpy (`zipfile` + `struct`),
  which matters because the login-node and `autoexp_env` pythons have no numpy, and `acc_debug`
  is capped at one job per user.

## Result 1 — the residue is question-only

Per segment is the cleaner instrument (each reference segment takes its best document); per pair is the
same data weighted by how many documents echo a segment.

| source | kind | segments | any hit | median coverage | cov ≥ 0.95 |
|---|---|---|---|---|---|
| gsm8k | **question only** | 1069 | **1065 (99.6 %)** | **1.000** | 1038 (**97.1 %**) |
| gsm8k | question + solution | 1316 | 166 (12.6 %) | 0.000 | 95 (7.2 %) |
| MATH500 | question | 484 | 287 (59.3 %) | 0.764 | 227 (46.9 %) |
| MATH500 | solution (`\boxed{}`) | 496 | 120 (24.2 %) | 0.000 | 11 (2.2 %) |
| mgsm | question, **en** | 12 | **11 (92 %)** | 1.000 | 11 |
| mgsm | question, es/fr/de | 60 | **0 (0 %)** | 0.000 | 0 |

Pair-level view of the same hits (documents per segment):

| source | kind | pairs | cov ≥ 0.95 | cov < 0.4 |
|---|---|---|---|---|
| gsm8k | question only | 4004 | **3117 (77.8 %)** | 160 (4.0 %) |
| gsm8k | question + solution | 2336 | 126 (5.4 %) | 2063 (88.3 %) |
| MATH500 | question | 51513 | 2884 (5.6 %) | 12470 (24.2 %) |
| MATH500 | solution | 34307 | **34 (0.1 %)** | 25926 (75.6 %) |
| mgsm | question, **en** | 33 | 30 (90.9 %) | 0 |
| mgsm | question, es/fr/de | **0** | — | — |

English math **problems** sit in the training tree verbatim; worked **solutions** do not: 99.6 % of
GSM8K question-only segments are matched at median coverage 1.000, against 12.6 % of the
solution-bearing segments at median coverage 0.000 (7.2 % full). MATH500 repeats the pattern
(59.3 % vs 24.2 % any-hit, median 0.764 vs 0.000). The hypothesis "overlaps are questions without
answers" describes the residue correctly, and at dataset scale rather than anecdotally.

The asymmetry also runs opposite to the contamination story: English MGSM questions are present
(11 of 12 found verbatim), the Spanish/French/German translations have **no hits at any coverage**,
yet English is the low-scoring side. Translations score higher *without* contamination, so their
advantage cannot come from leakage.

## Result 2 — question-only overlap is inert

If seeing the question helped, contaminated items would score higher. They do not:

| eval row | question in tree (cov ≥ 0.95) | not in tree |
|---|---|---|
| `gsm8k_4` | **80.7 %** (n = 2076) | **82.2 %** (n = 562) |
| `mgsm_native_cot_en` | **54.5 %** (n = 22) | 49.0 % (n = 478) |

−1.5 pp on 2076 gsm8k items and +5.5 pp on 22 MGSM items: no benefit, no measurable harm. Copies
that carry no answer teach nothing.

## Result 3 — the deficit is capability-shaped, not format-shaped

From the 56 k samples (same 250 problems per language):

| row | acc | median out tokens | answer line present | wrong answers containing the right number |
|---|---|---|---|---|
| `mgsm_native_cot_en` | 49.2 % | 54 | 98.8 % | 26 % |
| `mgsm_native_cot_es` | 74.8 % | 59 | — | 27 % |
| `mgsm_native_cot_de` | 68.0 % | 53 | — | 34 % |
| `mgsm_native_cot_fr` | 65.4 % | 58 | — | 34 % |
| `global_mgsm_en` (direct) | 28.0 % | 2 | — | 6/180 |
| `global_mgsm_es` (direct) | 38.0 % | 2 | — | 7/155 |

The gap reproduces in a second, independent format (direct answering, −10 pp), outputs are the same
length, and the answer line is emitted 98.8 % of the time. Near-miss rates are equal
(15.7 % of English errors vs 16.8–20.0 %), so it is neither extraction, nor truncation, nor
arithmetic slip: English derivations are wrong more often, and the measured failure mode is
"long derivation, no final answer" (44.8 % of MATH500 outputs carry no `\boxed`, while CoT prefill
lifts MATH500 from 22.0 % to 62.4 %).

## Verdict

The English-math deficit is **not** a decontamination artefact. The overlap is question-only and
behaviourally inert, the English side is the *more* contaminated one yet scores lower, and the data
mix cannot explain an English-specific gap either (math-reasoning sources are 15.6 % of the mix and
overwhelmingly English). What the tree holds is English problems without English worked solutions,
and the model's English behaviour has exactly that shape.

Lever: English worked solutions, not more decontamination. This is the paired-SFT direction scoped in
`docs/extractive_qa_sft_sources.md`.

## Limits

- Coverage is the longest *verbatim* run, so a solution present in paraphrased wording is invisible.
  "Solutions absent" means "no verbatim solution copies".
- The English MGSM reference set is 12 segments against ~20 per translation; the direction is
  unambiguous, the magnitude is not.
- Everything here is correlational. Causality needs a training-side intervention: drop question-only
  math documents, or add worked solutions and measure the same rows.

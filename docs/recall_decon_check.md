# Recall-suite contamination check (2026-11-19)

Question: do the recall suite's extraction evals — **SWDE**, **FDA**, **squad_completion** — leak
into the data the ambitious run trains on?

## Method

1. `scripts/flag_decontam/refs_from_eval.py` extracts reference segments from our own eval runs
   (these three benchmarks have no public HF dataset): SWDE 1 119, FDA 1 506, squad_completion
   7 185 → **9 810 segments** in `flag_decon_v6/refs/refs_extra_recall.jsonl.gz`.
2. `build_table.py` over those refs → 6 247 segments with 13-grams, 13 with 24, 6 with 40,
   1 184 536 unique n-grams, 50 989 template n-grams dropped, 3 544 segments too short to match
   (mostly squad_completion).
3. `scan.py` over a **27-of-531 shard sample (≈5 %)** of the real training mix
   (`data/v8ml-stages/s3/{flag-decon-v2-agenttok,v8-decon-agenttok}`), 168.8 B tokens per worker,
   986 s.
4. Decision rule copied from `finalize.py`: a (document, segment) pair is contamination when its
   longest verbatim run is ≥ `max(20, min(ceil(0.7 · n_tokens), 200))` canonical tokens.
   Segments flagging more documents than the generic threshold are boilerplate and are left out;
   the 20 000-document threshold was scaled to ~1 016 for the 5 % sample.

`finalize --dry-run` cannot be used on a partial scan ("scan incomplete for 429 shards"), so the
numbers below come from the hit pairs plus the same rule finalize applies.

## Result — training tree

| source | flagged docs | after boilerplate exclusion | extrapolated (×19.7) |
|---|---|---|---|
| FDA | 8 361 | **235** | ~4 600 |
| squad_completion | 298 | **298** | ~5 900 |
| SWDE | 124 | **124** | ~2 400 |
| total | 8 783 | **657** | ~13 000 |

**FDA: 97 % of the signal is one template.** Ten generic segments, all the same 510(k) header
(`SUBSTANTIAL EQUIVALENCE DETERMINATION DECISION SUMMARY / A. 510(k) Number: … B. Purpose for
Submission: …`), six near-identical 1 644–1 645-token segments each flagging 8 187 documents.
That is form boilerplate, not item leakage, and it is the case `--max-docs` exists for. It does
mean the FDA eval shares its *document corpus* with web crawls, which matters for reading the FDA
metric independently of leakage.

**Residual contamination is real but tiny**: ~657 documents in a 5 % sample, ~13 000 over the mix,
≈0.001 % of tokens. It does not inflate the recall numbers.

## Why it survived decontamination

The training tree is `flag-decon-v2`, built against refs v6, which contains `squad_v2`, `coqa`,
`belebele`, `winogrande` — but **not** `squad_completion`, `SWDE`, `FDA`. The residual matches sit
exactly in the benchmarks the reference list was missing, which is direct evidence that extending
the list was necessary rather than precautionary. `squad_completion` is the clearest case: its
passage extraction differs from `squad_v2`'s, so v6's reference segments did not cover it.

## Caveats

- 5 % sample; the generic-threshold rescaling is an approximation and extrapolation assumes
  uniform shard composition.
- A raw-collection scan (same refs, 23 shards of `training-data-collection/flag`) gave the same
  shape: SWDE max coverage 0.55 (below threshold), FDA dominated by the same 510(k) template,
  squad_completion matching full passages — but that tree is *not* what the run trains on.

## Next steps

1. Rebuild refs as **v7** = `refs_v6` + the 9 810 appended segments (`--append-to` keeps segment
   ids valid).
2. A full decon pass then removes the ~13 k residual documents at negligible cost to the mix.
3. Do not expect any of this to move CoQA or SQuAD v2: those remain a free-form span-copy and
   abstention-calibration issue (`docs/extractive_qa_sft_sources.md`).

## Reproduce

```bash
python scripts/flag_decontam/refs_from_eval.py --eval-root $E/ambitious_v4b_learned_fp8_36900 \
    --task swde --task fda --task squad_completion --out refs_extra_recall.jsonl.gz
# build table, then scan the shard sample with CUDA_VISIBLE_DEVICES="" or a free-GPU node:
#   scan.py --mode scan --work $WORK --root <train tree> --shards <sample list>
```

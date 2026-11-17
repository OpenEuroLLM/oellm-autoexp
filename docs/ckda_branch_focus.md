# ckda_branch post-fit: downstream reading / commonsense signal

Date: 2026-11-10. The `ckda_branch` arms replace CKDA layers at the 23400 surgery checkpoint and
train the branch for ~99 steps (23499). The post-fit downstream round (`config/evals/ckdab_focus_mn5.yaml`
+ `ckdab_obqa_mn5.yaml`) covers the FLAG reading / commonsense / math rows and the recall suite.
The English lm-eval rows are below. All values are %; deltas are arm - `drop` (the branch-weight-0
control), in pp.

## Status

- `main` = `ambitious_v4_2ph_hd256_fp8_23500`; arms = `*_fp8_23499`.
- `bgqa`, `cgqa` = standard attention branches; `d` = KronLSTM blocks (the CKLSTM-style block);
  `e` = kvshare; `drop` = branch weight 0 control.
- `bgqa`, `cgqa`, `d`, `drop`: all 258 done markers / 185 result dirs. `e`: 256 done markers,
  missing `squadv2_10.ok` and one part marker, so its SQuAD v2 row is absent.
- The arms are short post-fit branches (99 steps on top of the 23400 checkpoint), not full runs.

## English reading / commonsense

| task | metric | main | drop | bgqa | cgqa | d (KronLSTM) | e (kvshare) | 32B 90k (1.5T) | 32B 300k (5T) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HellaSwag 0-shot | acc_norm | 79.84 | 79.41 | 79.35 | 79.27 | 79.43 | 79.22 | 78.10 | 79.35 |
| PIQA 10-shot | acc_norm | 83.79 | 83.79 | 83.19 | 83.35 | 83.24 | 83.13 | 83.90 | 83.13 |
| BoolQ | acc | 87.06 | 86.64 | 86.67 | 86.57 | 86.97 | 87.00 | 85.17 | 87.03 |
| SocialIQA | acc | 47.80 | 47.54 | 47.39 | 47.44 | 47.90 | 46.98 | 48.57 | 48.77 |
| COPA | acc | 89.00 | 89.00 | 88.00 | 88.00 | 90.00 | 89.00 | 89.00 | 85.00 |
| CommonsenseQA | acc | 76.09 | 75.43 | 73.22 | 73.38 | 74.94 | 75.35 | 77.81 | 79.28 |
| Winogrande | acc | 62.51 | 63.06 | 62.59 | 63.54 | 64.17 | 63.38 | 70.32 | 70.56 |
| OpenBookQA | acc_norm | 46.80 | 45.40 | 45.60 | 46.00 | 46.20 | 45.20 | 44.80 | 45.00 |
| WSC273 | acc | 75.46 | 75.46 | 75.82 | 75.09 | 73.99 | 76.92 | 84.62 | 86.81 |
| CoQA | F1 | 66.13 | 66.26 | 65.61 | 64.48 | 63.31 | 65.07 | 79.72 | 81.72 |
| SQuAD v2 | F1 | 26.89 | 28.85 | 27.44 | 27.84 | 26.00 | — | 39.22 | 42.91 |
| GSM8K | exact (flex) | 73.62 | 71.57 | 71.72 | 72.10 | 73.92 | 71.72 | 72.78 | 78.70 |

Deltas vs `drop` (pp):

| task | bgqa Δ | cgqa Δ | d (KronLSTM) Δ | e (kvshare) Δ |
|---|---:|---:|---:|---:|
| HellaSwag 0-shot | -0.06 | -0.14 | +0.02 | -0.19 |
| PIQA 10-shot | -0.60 | -0.44 | -0.55 | -0.66 |
| BoolQ | +0.03 | -0.07 | +0.33 | +0.36 |
| SocialIQA | -0.15 | -0.10 | +0.36 | -0.56 |
| COPA | -1.00 | -1.00 | +1.00 | 0.00 |
| CommonsenseQA | -2.21 | -2.05 | -0.49 | -0.08 |
| Winogrande | -0.47 | +0.47 | +1.10 | +0.32 |
| OpenBookQA | +0.20 | +0.60 | +0.80 | -0.20 |
| WSC273 | +0.36 | -0.37 | -1.47 | +1.46 |
| CoQA | -0.65 | -1.78 | -2.95 | -1.19 |
| SQuAD v2 | -1.41 | -1.01 | -2.86 | — |
| GSM8K | +0.15 | +0.53 | +2.35 | +0.15 |

Means over the 10 common tasks (9 commonsense + CoQA F1; SQuAD excluded because `e` is missing):

| | main | drop | bgqa | cgqa | d (KronLSTM) | e (kvshare) | 32B 90k | 32B 300k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| common-10 mean | 71.65 | 71.23 | 70.83 | 70.84 | 71.28 | 71.18 | 74.07 | 75.03 |
| commonsense-9 mean | 72.20 | 71.73 | 71.35 | 71.47 | 72.08 | 71.79 | 73.51 | 74.36 |

## Reading

- **No arm shows a clear improvement.** The main run is at the top of the means; the `drop` control
  is next, and all branch arms are at or below it. The attention arms (`bgqa`, `cgqa`) are slightly
  worse on CommonsenseQA (-2.2 / -2.1 vs drop) and do not recover HellaSwag / PIQA / CoQA.
- **KronLSTM (`d`) has the only weak positive signal:** Winogrande +1.10 pp vs drop and GSM8K
  +2.35 pp vs drop, but it loses CoQA -2.95 pp and SQuAD -2.86 pp, and its common-10 mean is only
  +0.05 pp over drop. With ~1.9 / 1.7 pp standard errors these are ~1-1.4 sigma.
- **kvshare (`e`)** is close to drop on the mean, improves WSC273 +1.46 pp, but is missing SQuAD
  and is worse on CoQA (-1.19 pp).
- **The `drop` control is often the best non-main row** (SQuAD, CoQA, HellaSwag, PIQA,
  CommonsenseQA). The short post-fit itself shifts the model (main vs drop: SQuAD +1.96 pp,
  GSM8K +2.05 pp), so the branch effect is not cleanly isolated at this budget.
- **The 32B reference is still ahead where it matters.** Token-matched `b1cons_90k` (1.5T) is at
  74.07 common-10 / 73.51 commonsense-9, and `b1consTW_300k` (5T) at 75.03 / 74.36. The best branch
  (`d`) is 71.28 / 72.08, so it closes essentially none of the ~2.4-2.8 pp common-10 gap. The big
  unresolved rows are SQuAD v2 F1 (26.9 vs 39.2 / 42.9), CoQA F1 (66.1 vs 79.7 / 81.7), Winogrande
  (62.5 vs 70.3 / 70.6) and WSC273 (75.5 vs 84.6 / 86.8); the branch arms move these by at most
  ~1-2 pp. The 32B is not uniformly better (HellaSwag, BoolQ, OpenBookQA, and 90k GSM8K favour the
  ambitious model), so the reference gap is specifically the reading / pronoun-resolution rows.
- **Conclusion:** the post-fit evals are essentially finished, but they do **not** show a sign that
  replacing CKDA layers with KronLSTM or attention fixes the commonsense / reading deficit. The
  signal is small, mostly within noise, and the control is as good or better. A longer post-fit or
  a full run would be needed before ruling the branches in or out.

Artifacts: the `.ok` markers were copied to `dump/flag_summaries/ckdab_focus/<run>/done/`; the
summary shards are in `dump/flag_summaries/ckdab_focus/<run>/summary_shard*.tsv`.

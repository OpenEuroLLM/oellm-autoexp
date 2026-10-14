# Competition math in the 32B anneals: mode, not capability

Measured 2026-10-06/07 on JUPITER. Question: are the v1-recipe anneals (FP8; v1, v1C, v1T/v1TT) pathological,
given that they trail v2 (bf16) on AIME / AMC in the release evals and loop more under greedy decoding?

**Answer: no. The gap is the models' tendency to start a worked derivation, not their ability to solve the
problems.** Forced to reason (prefill ` <think>\n`), all seven checkpoints solve competition math equally well.

## Checkpoints

| name | line | step |
|---|---|---|
| v1anneal_108k / v1anneal_120k | v1 = B1-production anneal (FP8 delayed) | 108k / 120k |
| v1annealC_108k | v1C = B1-conservative anneal (FP8 delayed) | 108k |
| v1annealC_120k_l0fix | v1C 120k with the layer-0 down_proj least-squares repair | 120k |
| v1annealTT_120k | v1T fork at 112k, FP8 tensorwise + bf16 first/last layers | 120k |
| v2anneal_108k / v2anneal_120k | v2 anneal (bf16, z-loss) | 108k / 120k |

All anneals saw identical batches (same mix, cache, seed, consumed samples).

## 1. The release math evals are greedy, and noisy

* Evalchemy sends `do_sample: False` with `temperature: 0.7`; lm-eval's vLLM adapter turns `do_sample=False`
  into temperature 0. The "10 repeats" of AIME/AMC therefore differ only through batch-dependent bf16
  arithmetic (near-tie argmax flips), not through sampling.
* Same checkpoint, same weights/config/container, two runs (v2 120k, 2026-09-30 vs 2026-10-04):
  AIME24 0.220 vs 0.130 (four borderline problems flipped from solved to truncated); every task with
  >= 500 items moved <= 0.007. `scripts/noise_diff.py`, `data/*.rerun.csv`.
* Rule of thumb: large tasks +-0.01 is noise; AIME/AMC differences up to ~0.1 are plausibly noise.
  The reported AIME stderr (~0.008) measures repeat-to-repeat batch noise and is ~10x too small.

## 2. Context limit

The models have a trained context of 4096 tokens (`max_position_embeddings: 4096`, no RoPE scaling).
Prompt + generation must fit in 4096 tokens, so "give it more tokens" is not an option for these
checkpoints.

## 3. Sampled evaluation (temperature 0.6, top_p 0.95)

16 samples per problem (MATH500: 4), Evalchemy's own prompts and tokenization (loaded through lm-eval's
VLLM class; greedy re-runs reproduce the release outputs, `scripts/check.py`). Lenient accuracy
(boxed -> answer phrase -> last number). `data/sampled/summary.csv`.

| | v1 108k | v1C 108k | v2 108k | v1 120k | v1C fix 120k | v1TT 120k | v2 120k |
|---|---|---|---|---|---|---|---|
| AIME24 | 0.12 | 0.08 | 0.13 | 0.05 | 0.08 | 0.07 | 0.20 |
| AIME25 | 0.08 | 0.06 | 0.15 | 0.05 | 0.05 | 0.03 | 0.18 |
| AMC23 | 0.28 | 0.25 | 0.42 | 0.24 | 0.21 | 0.19 | 0.41 |
| MATH500 | 0.54 | 0.43 | 0.55 | 0.44 | 0.43 | 0.28 | 0.52 |
| AMC23 pass@16 | 0.83 | 0.88 | 0.85 | 0.78 | 0.78 | 0.80 | 0.93 |

Under sampling the loop rate falls to 0.07-0.24 for every model (greedy: 0.3-0.9), and v2 hits the 4096
cap as often as the v1 lines. Loops and truncation no longer separate the models; v2 still leads.

## 4. Forced reasoning: prefill ` <think>\n`

The eval prompt is plain text ending in `Answer:` (identity template); the models open their own worked
solutions with ` <think>`. Same sampling, generation continues after the prefill. `data/forced_think/`.

| | v1 108k | v1C 108k | v2 108k | v1 120k | v1C fix 120k | v1TT 120k | v2 120k |
|---|---|---|---|---|---|---|---|
| AIME24 | 0.17 | 0.17 | 0.18 | 0.20 | 0.20 | 0.19 | 0.21 |
| AIME25 | 0.18 | 0.20 | 0.18 | 0.20 | 0.23 | 0.21 | 0.20 |
| AMC23 | 0.51 | 0.53 | 0.51 | 0.54 | 0.55 | 0.53 | 0.56 |
| MATH500 | 0.73 | 0.74 | 0.73 | 0.75 | 0.74 | 0.755 | 0.75 |

95% bootstrap CIs (over problems): AIME +-0.11, AMC23 +-0.12, MATH500 +-0.03. All seven are within noise
of each other on every task. Forced solutions that finish inside 4096 tokens are correct ~95% of the time
for all models; the remaining failures hit the context limit (82-93% of forced AIME samples).

**Why the unforced results differ:** share of unforced samples that open with `<think>`
(`scripts/think_rate.py`):

| | v1 108k | v1C 108k | v2 108k | v1 120k | v1C fix 120k | v1TT 120k | v2 120k |
|---|---|---|---|---|---|---|---|
| AIME24 | 0.44 | 0.15 | 0.74 | 0.12 | 0.08 | 0.26 | 0.79 |
| AMC23 | 0.34 | 0.14 | 0.51 | 0.12 | 0.09 | 0.08 | 0.53 |
| MATH500 | 0.51 | 0.16 | 0.47 | 0.19 | 0.14 | 0.05 | 0.38 |

v2's anneal increased the habit of reasoning first; the v1-recipe anneals reduced it. v1TT under greedy
decoding often answers AIME with a bare guess (` 100`) instead of a derivation.

## 5. Likelihood

Reference-solution NLL (per token, solution after the eval prompt; MATH500 500, AIME24 30; AIME25 and
AMC23 have no solutions in their data files). `data/solution_nll/`.

| | MATH500 | diff vs v2 120k [95% CI] | AIME24 |
|---|---|---|---|
| v1 108k | 0.539 | +0.028 [+0.025, +0.031] | 0.777 |
| v1C 108k | 0.526 | +0.011 [+0.008, +0.014] | 0.766 |
| v2 108k | 0.525 | +0.009 [+0.008, +0.011] | 0.771 |
| v1 120k | 0.538 | +0.029 [+0.026, +0.033] | 0.773 |
| v1C fix 120k | 0.513 | -0.002 [-0.005, +0.001] | 0.752 |
| v1TT 120k | 0.521 | +0.015 [+0.012, +0.018] | 0.749 |
| v2 120k | 0.514 | - | 0.759 |

Anneal-mix NLL (40 datasets, 1920 docs sampled from the training shards, so in-distribution and possibly
seen; identical exposure for all runs). `data/heldout_nll/`, `scripts/mix_nll.py`.

| | overall | math | code | English web | multilingual |
|---|---|---|---|---|---|
| v1 120k | 0.952 | 0.717 | 0.498 | 1.845 | 1.857 |
| v1C fix 120k | 0.907 | 0.677 | 0.476 | 1.785 | 1.801 |
| v1TT 120k | 0.903 | 0.673 | 0.474 | 1.782 | 1.799 |
| v2 120k | 0.911 | 0.682 | 0.480 | 1.787 | 1.809 |

v1 production is the only line that is weaker as a model (~0.03-0.05 NLL behind, English-web NLL rising
108k -> 120k). v1TT and the repaired v1C are at parity with or better than v2.

## Conclusions

* Numerics: clean for v1, v1T/v1TT (FP8 degradation tests); v1C's layer-0 FP8-underflow dependence was
  found, explained and repaired (see memory `v1annealC_late_degeneration`).
* Likelihood: v1TT / repaired v1C at parity with v2 (better on the anneal mix and AIME24 solutions).
* Capability under forced reasoning: identical to v2 on AIME24/25, AMC23, MATH500.
* The only difference: the prior over response style (opening a derivation), which post-training sets.
* Compare base models on competition math with forced reasoning or sampling, never with greedy alone.

## Reproduce

Scripts run inside `/e/project1/e-sta-openeurollm/container/oellm-eval-vllm.sif` (vLLM 0.28):

    sbatch --job-name=ms_<ck>  scripts/math_sample.sbatch <ck>   # greedy prompt check + sampled run
    sbatch --job-name=cot_<ck> scripts/math_cot.sbatch <ck>      # solution NLL + forced-<think> run
    python3 scripts/check.py out/          # prompt identity vs the release outputs
    python3 scripts/grade.py <dir>         # (in the container) summary.csv, per_problem.csv
    python3 scripts/think_rate.py          # share of samples opening with <think>

Raw generations (all samples, all checkpoints):
`/e/project1/e-sta-openeurollm/poeppel1/analysis/anneal-math-20261007/` on JUPITER.

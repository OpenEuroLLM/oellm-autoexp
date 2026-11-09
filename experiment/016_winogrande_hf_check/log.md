# 016: is ambitious' Winogrande deficit a vLLM-port artifact?

## Question (user, 2026-11-09)

Ambitious v4_2ph scores 0.63-0.65 on Winogrande (0-shot) at every checkpoint from 5k to 23k. The JUPITER
oellm_32b b1consTW_300k scores 0.706 and q35c_gqa_59600 0.680. Is this an eval problem?

## Comparability (FLAG CSVs + lm-eval samples)

- The task configs are identical (lm-eval 0.4.13 on MN5, 0.4.12 on JUPITER).
- The request strings are byte-identical to the JUPITER run: Winogrande 1267/1267 docs, CoQA 500/500.
- Both tokenizers come from the same openeurollm 256k base and both prepend BOS. agenttok only adds special tokens.
- OpenBookQA is on par: acc_norm 0.446-0.450 for ambitious 14.9k / 23k and for the 32B.

## Finding: a closer-mention bias (scripts/closer_bias.py)

Split by the option mentioned last before the blank. The gold answer is the closer option in 49 % of docs.

| model | picks closer | acc, gold = closer | acc, gold = farther |
|---|---:|---:|---:|
| ambitious, all checkpoints 2000-23000 (hd128 + hd256, ctx2048 run too) | 0.66-0.74 | 0.80-0.84 | 0.35-0.48 |
| q35c_gqa 32000 / 59600 | 0.50-0.52 | 0.68-0.70 | 0.66-0.68 |
| oellm_32b b1consTW_300k | 0.51 | 0.72 | 0.69 |

## HF re-score (scripts/score_mc_hf.py)

- Config: config/experiments/korbi/production/winogrande_hf_check_mn5.yaml, p264.0296 on host003, snapshot autoexp_wghf, 2026-11-09 23:02, 1 node, ~40 min.
- Inputs: the vLLM run's exact (context, continuation) pairs from its lm-eval samples, encoded as lm-eval does, at batch 1.
- Paths: HF fast on all requests; HF reference (`OELLM_HF_IMPL=reference`, no custom kernels) on the first 150 docs of each task.
- Token ids on the cluster equal a local transformers 5.2 tokenization for 3080/3080 requests. The container's "incorrect regex pattern" tokenizer warning is spurious.

ambitious_v4_2ph_hd256_fp8_23000 (results/mn5_23000):

| | docs | acc | same pick as vLLM | mean abs dLL / token vs vLLM | picks closer | acc gold = farther |
|---|---:|---:|---:|---:|---:|---:|
| Winogrande vLLM | 1267 | 0.640 | -- | -- | 0.685 | 0.458 |
| Winogrande HF fast | 1267 | 0.636 | 0.977 | 0.031 | 0.690 | 0.448 |
| Winogrande vLLM, first 150 | 150 | 0.727 | -- | -- | 0.633 | 0.597 |
| Winogrande HF reference, first 150 | 150 | 0.713 | 0.987 | 0.031 | 0.647 | 0.569 |
| WSC273 vLLM | 273 | 0.780 | -- | -- | | |
| WSC273 HF fast | 273 | 0.784 | 0.967 | 0.033 | | |
| WSC273 HF reference, first 150 | 150 | 0.807 (vLLM 0.807) | 0.973 | 0.028 | | |

## Result

**Not a vLLM-port artifact.**
- Both HF paths reproduce the vLLM picks (97-99 % identical) and the closer-mention bias.
- The log-likelihood differences (0.03 nats / token) are the known HF-vs-vLLM bf16 noise (experiment 013: mean |dlogp| 0.025).

The bias is in the exported model. Not excluded: a Megatron -> HF conversion error shared by both ports. There is no paired Megatron-vs-HF check of a v4 export yet; conservative was exact to 5e-6. The step-2000 export already shows the bias, so if it is a model property it formed early.

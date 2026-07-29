# 0.1B 7:1 Full Attention vs mLSTM vs GDN

## Objective

Run a minimal, directly comparable multilingual 0.1B experiment with full
attention, 7:1 mLSTM:full attention, and 7:1 GDN:full attention.

## Hypothesis

At the completed 0.1B low baseline cell, the two 7:1 hybrid variants can be
compared fairly against a fresh full-attention reproduction when data, seed,
batch size, optimizer, and WSD schedule are held fixed.

## Setup

- Config: `config/experiments/architecture_scaling_variants/multilingual/fullattn_mlstm7_gdn7_0.1B_50BT.yaml`
- Models: 0.1B Qwen3-style, 16 layers, 512 hidden size, 8 attention heads.
- Shared recipe: 50B tokens, global batch 128, learning rate 2e-3, 2,000-step
  warmup, 20% linear WSD cooldown, seed 1234, multilingual 1TT-option-4 mix,
  and 99/1/0 split.
- Baseline reference: full-attention end-of-decay validation loss 2.757 for
  the same low-ladder 50B cell.

## Run Log

- 2026-07-29: Added the three-point direct-comparison sweep. It uses the
  validated JUPITER linear-attention image plus explicit CUDA driver binds.
- 2026-07-29: Dry-run validation expanded exactly three jobs. Each renders
  95,368 train iterations and 19,073 WSD decay iterations; all share gbs 128,
  mbs 4, split 99/1/0, and the same multilingual data-path blend. Only mLSTM
  and GDN render an experimental-attention flag, each with frequency 8.

## Results

- Pending submission.

## Interpretation

The sweep deliberately avoids branch-and-cooldown reuse: each variant receives
one complete 50B-token WSD trajectory, making the final loss comparison 1:1.

## Related

- Previous: `experiment/001_tiny_mlstm_gdn_hybrid_smoke/`
- Reference ladder: `config/experiments/architecture_scaling_variants/multilingual/swa5_0.1B_low.yaml`

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
- Shared training recipe: 50B tokens, global batch 128, learning rate 2e-3,
  2,000-step warmup, 20% linear WSD cooldown, seed 1234, multilingual
  1TT-option-4 mix, and 100/0/0 split.
- External evaluation: after each final checkpoint, an eval-only one-node job
  evaluates 204,800 samples from the separately staged `datamix4-val` blend.
- Historical reference: the published full-attention 2.757 loss was on the
  baseline's 1% train-mix slice, so it is retained as context only and is not
  a strict numerical target for this updated protocol.

## Run Log

- 2026-07-29: Added the three-point direct-comparison sweep. It uses the
  validated JUPITER linear-attention image plus explicit CUDA driver binds.
- 2026-07-29: Dry-run validation expanded exactly three jobs. Each renders
  95,368 train iterations and 19,073 WSD decay iterations; all share gbs 128,
  mbs 4, split 99/1/0, and the same multilingual data-path blend. Only mLSTM
  and GDN render an experimental-attention flag, each with frequency 8.
- 2026-07-29: Updated the protocol to train with split 100/0/0 and added an
  `eval50BT` post-annealing stage per architecture. Each eval job waits for
  its matching `final50BT` checkpoint, loads it without optimizer/RNG state,
  and evaluates 204,800 samples from `datamix4-val` with W&B offline; its
  metric will be collected from the Megatron job log.
- 2026-07-29: Local dry-run passed: six jobs, three dependency edges, and
  each eval resolves to its matching `iter_0095368` checkpoint with 155
  weighted validation prefixes. JUPITER dry-run is blocked before planning:
  its Megatron submodule is at `055f7defc`, which lacks the mLSTM/GDN
  arguments. The local validated source is `bb72650a8`; advance the remote
  runtime deliberately before submission rather than mixing schemas and code.
- 2026-07-29: Correction: the preceding remote check used the obsolete
  `oellm-autoexp` checkout. `sync_to_jupiter.sh` targets
  `~/work/Projects/oellm-autoexp-hybrid`, which correctly matches
  `hybrid_exp` at `fa8143d` with Megatron `bb72650a8`. A dry-run there passed,
  rendered all six jobs, and confirmed that the staged validation data exists.

## Results

- Pending submission.

## Interpretation

The sweep deliberately avoids branch-and-cooldown reuse: each variant receives
one complete 50B-token WSD trajectory. The external validation stage gives all
three final models the same held-out metric, while avoiding validation work or
data-order changes during training.

## Related

- Previous: `experiment/001_tiny_mlstm_gdn_hybrid_smoke/`
- Reference ladder: `config/experiments/architecture_scaling_variants/multilingual/swa5_0.1B_low.yaml`

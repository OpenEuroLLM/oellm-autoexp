# From document masking to cross-document attention: what can go wrong, and how to switch

## Short answer

**Recommendation:** do not flip the main run from `doc` to `none` in one step without a measurement.
- Semantic packing plus `none` is what the literature supports: Shi et al., Zhao et al., SPLiCe
  ([references.md](references.md)).
- Plain unmasked random packing is what it warns against: ProLong §6.1, Zhao et al.
- v4b is a hybrid. Most of its layers are recurrent (CKDA), and Engram and the short convolutions
  reset at document boundaries. That raises the risk beyond the attention-only models of those papers.

**Proposal:**
1. A short paired branch test from a main-run checkpoint (see "Checks"), arms `group` and `none`
   against the main run's own losses.
2. If clean: the main run takes plans with `group` for a few hundred steps, then `none`.
3. If the recurrent layers suffer (F2): implement the hybrid boundary mode (attention unmasked, local
   operators reset per document), sketched at the end.

## What the model learned under document masking

Under `--dataloader-inter-document-masking`, every document started from a clean slate in every
operator (from the Megatron code, see [README.md](README.md)):

| operator | under document masking | under `none` |
| --- | --- | --- |
| softmax attention (6 + 6 kv-shared V layers, head dim 256) | sees its own document only; RoPE restarts at 0 | sees everything before it in the 64k sequence; positions run on |
| CKDA (most layers, recurrent) | state reset to zero at each document (cu_seqlens to fla) | state carries over from the previous document |
| Engram n-gram lookup | history before the document start = filler; its depthwise conv drops taps across the boundary | the first n−1 tokens hash n-grams through the previous document's tail and its EOD |
| ShortConv | masked by in-document position | taps cross the boundary |
| BOS | each document's first token, the sink of its attention | a BOS in the middle of the sequence, preceded by context |

`group` mode keeps all of these boundaries, but at group (leaf) level: the reset happens between
semantic groups, not between documents.

## Failure cases

**F1. Distraction by unrelated context.**
- **What:** attention spends mass on earlier, unrelated documents. Zhao et al. measure the downstream
  damage for random packing; ProLong measures it for long-context continued training.
- **Mitigation:** semantic packing, so the earlier documents are mostly related; `group` mode as the
  intermediate step.
- **Signal:** loss on short documents (per-position-in-document loss) above the doc-masked control.

**F2. Recurrent state contamination (the hybrid-specific risk).**
- **What:** CKDA never needed to forget at a document start: the state was zero. Under `none` its state
  enters each new document full of the previous one. The decay gates have to learn to flush on
  BOS / EOD. Until they do, the first tokens of every document are predicted from a polluted state.
- **Size:** about 55 documents per 64k sequence on average, so this touches a lot of positions.
  Attention can route around bad context; a recurrent state cannot ignore what it has absorbed.
- **Mitigation:** `group` first, so contamination is between related documents. The hybrid boundary
  mode below if it persists.
- **Signal:** loss in the first 64–256 tokens of documents (position-in-document curve) and per-layer
  CKDA state norms at document starts.

**F3. Attention sinks move.**
- **What:** with masking, each document's BOS (at relative position 0) was its sink (Xiao et al.).
  Under `none`, the sequence's first token, often a document continuation without BOS, sits at
  position 0, and the BOS tokens of later documents are in the middle.
- **Expectation:** heads that dumped mass on "the first token" now have one true first token and
  several BOS tokens. Usually a quick adaptation, but it shows up as a loss / grad-norm bump in the
  first steps.
- **Watch:** the kv-shared V layers copy their source layers, so their sinks shift together with the
  source's.

**F4. Copy shortcut.**
- **What:** a document next to its copy or near copy predicts itself. The loss falls without learning,
  and the model learns to copy.
- **In the plan:** exact repeats (another epoch inside the window) and near duplicates (cosine ≥ 0.97)
  are deferred, then a repair pass guarantees that no sequence holds two members of a conflict class.
  `stats.json`: `near_duplicate_pairs`, `conflict_moves`.
- **Residual risk:** paraphrases below 0.97 and boilerplate-heavy sites. Watch the training loss of
  small, repeated datasets against their validation loss.

**F5. Training loss is no longer comparable.**
- **What:** with related documents in context, the training loss drops by itself. That is the point,
  not progress.
- **Rule:** the fixed validation sets (document-masked, unchanged) are the only comparable curve
  across the switch.
- **Action:** health watches with training-loss thresholds need re-baselining at the switch (a
  downward step is expected, an upward one is F1–F3).

**F6. Train / inference mismatch.**
- **What:** at inference a prompt is one document, which matches document-masked training exactly.
  Cross-document training teaches the model to use earlier context (ICL, RAG and multi-document QA
  gain in Shi et al.); it does not hurt the single-document case in those papers.
- **Watch:** the recall suite (SQuAD F1, CoQA) and long_64k validation, before and after.

**F7. Engram noise at boundaries.**
- **What:** n-grams across a document boundary (the previous tail plus EOD plus BOS) are meaningless
  keys that collect gradient.
- **Size:** about 3 positions per document, roughly 0.25 % of positions. Small, but the table had a
  stability history (row-norm cap). `group` limits it to group boundaries.

**F8. Throughput.**
- **What:** under `none` every rank runs full 64k causal attention. The step time is the slowest
  rank's, and with 1024 sequences per step some rank almost always holds one long document, so the
  step should cost about what it costs now (the premise of the change).
- **Speed paths:**
  - the packed short-document fast paths (page-sparse, flex) stop helping;
  - CKDA (linear) and the experts are unchanged;
  - activation memory with flash attention is linear in length, so no memory change.
- **Check:** step time in the branch test. The plan stats report the solo-slot fraction; a solo slot
  is a sequence that already costs the full 64k.

**F9. The plan is tied to the run's data configuration.**
- **What breaks it:** a plan replaces specific blended sample indices. It is valid for one blend
  (checked: `blend_md5`), one sequence length (checked), and one sampler geometry: gbs 1024,
  reference layout 512 × 1, phase start 23691, train_iters 236733. The geometry is **not** checked at
  load time.
- **If the geometry changes:** the replaced indices are served at other steps. Still conserved
  overall, but the windows no longer match the steps.
- **Rule:** any change of these settings means re-extracting.
- **Coverage:** past the last planned step the run silently returns to regular (document-masked)
  samples. Plans have to be built ahead of the run (pipeline throughput, see README).

**F10. Mixture and order.**
- **What is preserved:** documents move only within a window (500 steps by default), so the mixture
  over time is kept, and each batch is a random draw of the window's sequences.
- **What changes:** a sequence now mixes datasets (sources) where Megatron's sequences were
  single-dataset. That is intended (semantic neighbours across sources), but it changes
  per-sequence statistics, e.g. the length distribution of documents in one sequence.

## The transition

1. **Plans ahead.** Extract and plan at least one full segment ahead of the run, checked with
   `verify_stream.py` and the plan `stats.json` (cosine before / after, conflict moves).
2. **Branch test** (below). Decide the mode from it.
3. **Main run.** Re-enqueue at a checkpoint with `--packing-plan-path` and `--packing-plan-boundaries
   group`, then `none` after a few hundred steps if `group` was clean. Both are just a re-enqueue
   with a different flag: the sampler position, optimizer and data order are untouched, so switching
   (or reverting to `doc`) is free at any checkpoint.
4. **Watch for the first 500 steps:**
   - validation (unchanged sets) at every eval;
   - grad norm, z-loss, the Engram row norms;
   - step time;
   - the per-position-in-document loss curve, if logged.

## Checks (proposed, not run)

A paired branch from the main run, like t21 (sparse branch from `ckpt_eval 46000`, paired with v4b's
logged losses): the branch reads the main run's exact steps, so arm differences are pairing-clean
([data-order rule](../campaign_3b_hybrid_midtraining.md)).

| arm | data | boundaries | question |
| --- | --- | --- | --- |
| control | main run (logged) | doc | baseline |
| P-doc | plan | doc | does semantic packing alone change anything? (should be ~0: masked) |
| P-group | plan | group | the transition mode |
| P-none | plan | none | the target |
| R-none | regular (no plan) | none | ProLong's negative case (random packing, unmasked) |

- **Metrics:** fixed validation (all sets incl. long_64k), recall evals at the end, step time,
  per-position-in-document loss.
- **Size:** 200–300 steps on 64 nodes (~20 s per step at 256 nodes, ~4× slower at 64 nodes with the
  reference layout): about 300–450 GPU-h per arm.
- **Minimum useful set:** P-group, P-none and R-none (the control is free).
- **Prerequisite:** plans for the branch's steps (any `aux.from_ckpt`, e.g. 46000).
- **Cost:** about 1–1.4k GPU-h. Needs the user's go.

## The hybrid boundary mode (option, not implemented)

The cleanest answer to F2 / F7 for this architecture is to unmask **attention only**:

- softmax attention takes `cu_seqlens_attn` (`none` or `group`);
- CKDA, Engram, ShortConv keep `cu_seqlens_doc` (per document).

Implementation sketch:
- the dataset returns both;
- `PackedSeqParams` carries the second set;
- the mixers pick theirs.

Same speed benefit (attention is the straggler, CKDA is linear), no recurrent contamination, the
Engram filler kept. Cost: a model-code change in the THD paths (attention, CKDA, Engram, ShortConv)
plus the HF export staying document-local for its local operators. Worth building if the branch test
shows P-none's loss excess concentrated at document starts.

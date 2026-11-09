# Grad-norm spikes of v4_2ph: batch content (2026-11-09)

Question: are the growing grad-norm spikes of the ambitious v4_2ph main run (steps ~20k-22.9k; led by sublayer 0
= first attention and the Engram tables, see per-layer diag/grad_norm) caused by unusual batches?

Method: `scripts/korbi/campaign/spike_batches.py` (CPU only) rebuilds the global batch (8192 x 8193 tokens) of
37 spike steps (grad-norm > 1.0) and 40 calm steps (< median 0.261), steps 21500-22854 (`steps_v4_21500_22854.csv`),
exactly as the dataloader did: Megatron's MegatronPretrainingReferenceLayoutSampler (ref layout 512 x 8), the
phase-1 opt81 blend cache b2ac5f22bb793c504e870e109893983d (size 194,076,928 = len(dataset) of phase 1; its
weights equal datamix_v8ml_opt81_8k64k_stage1_mn5.txt to 3e-17), the component GPTDataset indices and the
IndexedDataset with the virtual BOS. Logged step N = consumed (N - 1) x 8192. Job: kernel_tests_mn5
aux.name=spikebatch (acc_debug 48834934, snapshot autoexp_spikes, autoexp a176b2e8).

Result (`contrast_v4_21500_22854.txt`, `per_step_features_v4_21500_22854.csv`): spike and calm batches do not
differ. Per-batch means / maxima of documents per sample, longest single-token run, unique-token fraction,
most frequent token share, 4- / 16-gram repetition, and the source shares (within 0.15 percentage points) are
the same; no feature correlates with the grad norm (|Spearman| <= 0.19 over 77 steps, ~0.22 would be p 0.05).
Batch content at this granularity does not explain the spikes; extreme sequences are in EVERY batch.

Side finding (data hygiene): every batch holds ~9 samples with a single-token run >= 512 and ~40 with >= 90 %
repeated 16-grams, e.g. 8193 x "0" or "7" (mostly megamath-0.0.0, v8-decon-agenttok/hftok, vtok; hftok doc
370981 alone spans several steps); ~990 samples per batch (12 %) have < 2 % unique tokens.

Not ruled out: a few sequences per batch driving the spike through their interaction with the current weights
(e.g. rare Engram rows, layer-0 attention saturation). That needs a GPU replay of spike steps from a checkpoint
with per-token losses (--diag-consumed-train-samples, --diag-token-loss-dir), ~64 nodes for ~1 h.

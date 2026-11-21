# Cross-document sequences for the v4b main run

Goal: from a checkpoint on (49,000 if ready, else 50,000 or later; nothing is tied to one number), train
the v4b main run on 64k sequences whose documents are **related** (semantic packing), and stop masking
attention between them. The data stays the same: the same documents, the same mixture schedule, the
same token count per window of steps. Only which documents share a sequence changes.

Why: at 64k with document masking, a step's time is set by the slowest rank, the rank whose sequence
holds one long document ([`v4-64k-attention-imbalance`](../campaign_3b_hybrid_midtraining.md)). Every
other rank waits. Attention across documents fills that idle time, and with related documents it is
context worth attending to.

Status (2026-11-21): the pipeline is built and tested. It runs as pool jobs (c136); see
[Runs](#runs). **Nothing is applied to the running training.** A plan only acts when a future
enqueue passes `--packing-plan-path`.

Documents in this folder:

| file | content |
| --- | --- |
| [README.md](README.md) | the workflow, the steps, how to run each and how to switch the start step |
| [references.md](references.md) | the literature behind it |
| [adaptation.md](adaptation.md) | removing document masking from a masked model: failure cases, safeguards, the proposed transition and checks |
| [metadata.md](metadata.md) | what document metadata exists, where, and what it is worth inside a window |

## The pipeline

```
 run's datamix + data cache               (nothing rebuilt; the run's own index files)
        │
 1 extract_doc_stream.py  ──>  stream/ck<a>_<b>/   per step x batch position: the sample the
        │                                         sampler serves and its document pieces
 1b verify_stream.py          random slots == Megatron's own GPTDataset (production scale)
        │
 2 embed_docs.py          ──>  emb/<name>/ck<a>_<b>/part_*.npz   one vector per document
        │
 4 build_plan.py          ──>  plan/<name>/ck<w0>_<w1>/          per window: which pieces form
        │                                                        each 64k sequence
 Megatron --packing-plan-path <plan dir> [--packing-plan-boundaries none|group|doc]
```

Code: `scripts/korbi/cross_doc_seqs/`, Megatron `megatron/core/datasets/packing_plan.py`
(commit 4cbb542b3), job `config/experiments/korbi/cross_doc_seqs/cds_pipeline_mn5.yaml`, test
`tests/unit/test_cross_doc_seqs.py`.

### Step 1: which documents does each step read? (`extract_doc_stream.py`)

Megatron decides this in three layers. All three are replayed from the run's cache files and
nothing is rebuilt.

1. **Sampler.** `--dataloader-type cyclic`, data sharding, reference layout 512 × 1, gbs 1024. A step
   taken from checkpoint `c` sits at phase position `(c − 23691) × 1024`. The reference-layout sampler
   turns that into 1024 blended sample indices, with the same set for any node count. The replay
   (`cds_data.StepSampler`) is vectorized, and at every chunk start it is checked against Megatron's
   own `MegatronPretrainingReferenceLayoutSampler` (`--check-megatron`).
2. **Blend.** The BlendedDataset `dataset_index` / `dataset_sample_index` map a blended sample to
   (dataset, sample). The entry is found by its description: same 531 paths, sequence length and
   weights, and the md5 is re-verified. For the main run this is `09a2a1c43379c8fa39237a9bdc9d3115`
   (218,155,276 samples; phase 2 needs (236733 − 23691) × 1024 = 218,155,008).
3. **GPTDataset.** `shuffle_index` → `sample_index` → `document_index` give the document pieces. They
   are in Megatron's virtual coordinates: with `--dataloader-add-bos`, offset 0 is the BOS added at
   read time.

A slot's pieces cover its 65,536 **owned** tokens. The extra label token is the first token of
another sample, so the pieces of a set of slots partition their tokens. That is what lets a repacking
conserve the tokens exactly.

Checks:
- the end-to-end test compares every slot's tokens with Megatron's `BlendedDataset` on a real blend;
- `verify_stream.py` compares random slots of the production stream with `GPTDataset` objects built
  from the run's cache. They are built through a symlink directory, so the shared cache is never
  written. A cache miss counts as a failure.

### Step 2: embeddings (`embed_docs.py`)

- **What is embedded:** one vector per document (dataset, document id) of the non-solo slots. A solo
  slot is one long document filling the whole sequence and is never repacked.
- **Text:** the document's first 640 training tokens, decoded with the training tokenizer and
  re-tokenized for the embedding model, truncated at 512.
- **Model:** [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), Apache 2.0.
  It is multilingual (the mix is 20 % non-English) and covers code (stack-v2 alone is 19.5 %).
  Pooling is on the end token. The tokenizer appends `<|endoftext|>` itself; that was checked on MN5.
  Vectors are Matryoshka-truncated to 256 dimensions and stored fp16.
- **Fallback:** multilingual-e5-small (`--pooling mean --prefix "passage: "`), about 5× cheaper.
- **Where the models are:** both are in `/gpfs/projects/ehpc533/users/korbip/models/`, downloaded on
  JUPITER and moved with hpctransfer.

### Step 3: metadata

See [metadata.md](metadata.md). In short:
- MN5 holds only token bins.
- URLs / repos / ids exist in the release jsonl on LUMI; stack-v2-full's repo and path are on JUPITER.
- The training document → source row mapping can be inverted step by step.
- Inside a window, though, two documents of the same repo or site rarely meet: in a 9.3k-file sample
  of stack-v2 there were 8,620 repos.

So metadata is wired in as an optional input (`build_plan.py --group-keys`: groups kept together inside
a leaf), but it is not on the critical path.

### Step 4: grouping and packing (`build_plan.py`)

The plan is built per **window** of `--window` steps (default 500, about 34B tokens):

1. **Units.** A unit is one occurrence of a document: its pieces in the window, merged in offset
   order, so a document cut at a sequence boundary comes back together. A document that the window
   reads twice (another epoch of a small dataset) becomes two units. Solo slots stay as they are.
2. **Order.** The order is hierarchical, so it scales to tens of millions of units per window. This is
   in-context pretraining's nearest-neighbour traversal (Shi et al.) applied to centroids:
   - coarse spherical k-means, about 4096 units per cluster;
   - per cluster, a fine k-means into **leaves** of about 32k tokens, i.e. half a sequence;
   - leaves chained greedily by centroid similarity, starting next to the previous cluster's last
     leaf;
   - coarse clusters chained greedily.
3. **Copies never meet.** Exact repeats and near duplicates (cosine ≥ 0.97 within a coarse cluster) go
   to later pools, each ordered on its own. A repair pass then moves any unit that still shares a
   sequence with a member of its conflict class (same document, or a linked near-duplicate) to a
   random place, and repeats until no sequence holds two members of one class. A copy in context is
   a free lunch for the loss: it teaches copying, not language.
4. **Cut.** The units' pieces are concatenated in that order and cut into 65,536 + 1 tokens, exactly
   like Megatron's own packing. Tokens are conserved exactly; the test checks this.
5. **Shuffle.** The window's sequences are shuffled over its non-solo slots. Every global batch stays
   a random mix of the window; only the inside of a sequence is coherent.

Documents move only inside their window. The mixture over time, the token count per window and the
run's validation therefore stay as they were.

`stats.json` per window records:
- units, solo slots, near-duplicate pairs and repair moves;
- datasets per sequence, before and after;
- **mean cosine between the documents of a sequence**, before (Megatron's random packing) and after.

That cosine is the quality number for the packing.

### Megatron side (`--packing-plan-path`)

`PackingPlanDataset` wraps the training BlendedDataset:
- a sample index listed in a plan is served as the plan sequence, through `GPTDataset._sample_from_text`,
  the same code path as every sample;
- every other index is served unchanged.

The sampler, consumed samples, checkpoints and resumes are untouched, and outside the planned steps
everything is bit-identical. The plan's `blend_md5` and sequence length must match the run's or
loading fails.

`--packing-plan-boundaries` sets the cu_seqlens of plan sequences:

| mode | cu_seqlens | use |
| --- | --- | --- |
| `doc` | one segment per document piece | semantic packing with document masking (control) |
| `group` | one segment per run of pieces of the same leaf | related documents see each other, unrelated ones don't (transition) |
| `none` | the whole sequence | full cross-document attention (target) |

All three keep `--dataloader-inter-document-masking` on, so the model takes the same THD code path.
Only the segment boundaries change. See [adaptation.md](adaptation.md) for why `group` should come
before `none`.

## How to run

The job is one pool run, with one task per node and 4 GPU processes per node. Stages are skipped where
their files exist, so a rerun resumes and a new range extends the same directory:

```bash
# from a fresh snapshot (./sync_to_marenostrum.sh <name>), on the MN5 login node in $R/autoexp_<name>
scripts/korbi/mn5_autoexp.sh scripts/run_autoexp.py --no-monitor --monitor-state-dir $R/pool_state \
    --pool /gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_c136 \
    --config-name experiments/korbi/cross_doc_seqs/cds_pipeline_mn5 \
    aux.from_ckpt=49000 aux.steps=4000 slurm.sbatch.nodes=4 slurm.sbatch.time=08:00:00 \
    "aux.stages='extract,verify,embed,plan'" slurm.sbatch.account=cns143 slurm.sbatch.qos=acc_xlarge --yes
```

Outputs go to `/gpfs/projects/ehpc533/users/korbip/data/cross_doc_seqs/v4b_p2/{stream,emb,plan,logs}`.

**Switching the start step** (49k → 50k, or anything else): `aux.from_ckpt=50000`. The stream of
49000–50000 is simply not used, and windows are counted from `from_ckpt`. Plans for an already
extracted range can be rebuilt with other parameters under a new name, e.g.
`aux.plan_name=q3e256_w1000 aux.window=1000` (extra `build_plan.py` options go comma-separated, without spaces,
into `aux.plan_args`, e.g. `--dup-cos=0.98`).

**Applying it later** (not done, needs the user's go): re-enqueue the main run from the start
checkpoint with
`++backend.megatron.packing_plan_path=[/gpfs/projects/.../plan/q3e256]` and
`++backend.megatron.packing_plan_boundaries=group`, from a snapshot whose Megatron contains
4cbb542b3. The plan directory holds one subdirectory per window, and all of them are loaded. The
plans must cover every step the segment will run: past the last window the run silently falls back
to the regular stream.

## Readiness for the main run (2026-11-21)

Prepared, **not launched**. The user decides the switch, its timing and its first mode.

| item | state |
| --- | --- |
| stream replay == Megatron | ✅ sampler replay == `MegatronPretrainingReferenceLayoutSampler` (production geometry, MN5); 64/64 random sequences == Megatron's `GPTDataset` (53 datasets) |
| plans, logically | ✅ tests (`tests/unit/test_cross_doc_seqs.py`, 7; Megatron `tests/unit_tests/data/test_packing_plan.py`, 6): token conservation, copy separation, all boundary modes, coverage + geometry guards |
| plans through Megatron's loader on the **real** blend | `check_plan.py` (pipeline stage `checkplan`): BlendedDataset rebuilt read-only from the cache (hash == plan), `wrap_with_packing_plans` with the run's geometry, per-window token conservation, served samples == plan segments in doc / group / none, unplanned samples unchanged. Runs at the end of the coverage job |
| plan coverage 48,500 → 53,000 | pool job `cds_q3e256_ck48500_4500` (c136, 8 nodes; ~225 GPU-h), plans in `plan/q3e256_w500` (windows of 500) |
| guards | Megatron 88efc4279: the plan's sampler geometry must match the run; the run refuses to start on an unplanned step and saves + exits at the end of the planned range (`--packing-plan-at-end exit`, default) |
| main-run code | Megatron HEAD 88efc4279 differs from the running 4a0fab4a1 on v4b's code path only by the packing-plan commits. The other commits are page-sparse / rowmask modules, imported only with `page_sparse_selector`, which v4b does not set. The superproject's rendered v4b arguments == the live step script's, except three committed v4b config changes that apply at any re-enqueue: `--load` own tree, `--save-interval 1000`, `--save-extra-steps` |
| switch config | `config/experiments/korbi/cross_doc_seqs/v4b_learned_cds_mn5.yaml`: renders to v4b's arguments + `--packing-plan-path …/plan/q3e256_w500 --packing-plan-boundaries group` |

**Switch procedure** (only with the user's go, AGENTS.md):
1. Plans cover the resume checkpoint and at least one daily segment; `checkplan` passed.
2. Fresh snapshot (`./sync_to_marenostrum.sh <name>`, Megatron ≥ 88efc4279).
3. `pool_ctl cancel` the running v4b entry. This is a graceful save, and the run resumes from it.
4. Enqueue `experiments/korbi/cross_doc_seqs/v4b_learned_cds_mn5` into p264, the same way the main run is enqueued now.
5. Check the start log for `> packing plans: … planned from <ckpt> to <end> (at the end: exit)`. Then watch validation, grad norm and the Engram row norms for the first few hundred steps ([adaptation.md](adaptation.md)).

## Runs

| date | pool run | range | result |
| --- | --- | --- | --- |
| 11-21 14:56 | c136.0167 | 49000 + 100 | failed at once: `--out` missing in run_pipeline.sh (fixed 857b70af) |
| 11-21 15:00 | c136.0168 | 49000 + 100 | cancelled after 30 min in one chunk: random page faults into ~120 GB of memory-mapped index files on GPFS, CPU idle (fixed: bulk preload, ec07a715) |
| 11-21 15:35 | c136.0169 | 49000 + 100 | **stream + verify OK**: preload 119 GB in 27 s, 102,400 slots / 5,446,398 pieces (53.2 documents per 64k sequence), sampler == Megatron's, **64/64 slots == Megatron's GPTDataset** (53 datasets). Embedding bench 229 docs/s/GPU, loader-bound (3 serial random reads per document); one GPU OOM in masked SDPA. Cancelled for the fixes 1e2317a4, e2fb4e89 |
| 11-21 16:03 | c136.0170 | 49000 + 100 | stream reused; bench **314 docs/s/GPU, 98k tok/s, GPU-bound** (loader wait 4 s of 128 s, flash_attention_2); full embedding + first plan running |

Measured costs (Qwen3-Embedding-0.6B, 512 tokens, mean 313 tokens per document):

- **Stream:** seconds per 100 steps after a 27 s preload; free in practice.
- **Embedding:** 100 steps hold ~5.4M distinct documents, so ~4,500 s per node = ~5 GPU-h per 100 steps.
  - That is <1 % of the training compute of those steps (~570 GPU-h).
  - 4,000 steps ≈ 200 GPU-h (50 node-hours); keeping pace with the main run (~4,400 steps/day) needs ~2.3 nodes.
  - multilingual-e5-small would be ~5× cheaper, at lower multilingual/code quality.

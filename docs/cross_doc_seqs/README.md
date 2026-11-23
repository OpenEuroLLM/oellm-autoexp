# Cross-document sequences for the v4b main run

Goal: from a checkpoint on (49,000 if ready, else 50,000 or later; nothing is tied to one number), train
the v4b main run on 64k sequences whose documents are **related** (semantic packing), and stop masking
attention between them. The data stays the same: the same documents, the same mixture schedule, the
same token count per window of steps. Only which documents share a sequence changes.

Why: at 64k with document masking, a step's time is set by the slowest rank, the rank whose sequence
holds one long document ([`v4-64k-attention-imbalance`](../campaign_3b_hybrid_midtraining.md)). Every
other rank waits. Attention across documents fills that idle time, and with related documents it is
context worth attending to.

Status (2026-11-22): **live in the main run since checkpoint 51,000** (user: "Go ahead with the switch at 51k in
group mode"). Plan `q3e256_g51000` covers 51,000 → 61,000 with `boundaries=group`; the run saves and exits at
61,000, and the next window's plan must be built and checked before it is re-enqueued. See [Runs](#runs).

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
`aux.plan_name=q3e256_w1000 aux.window=1000` (extra `build_plan.py` options go
into `aux.plan_args` joined by `+`, e.g. `--dup-cos=0.98+--coarse=8192`; the container's `--env` splits values at
`,` and cannot carry spaces).

**Applying it later** (not done, needs the user's go): re-enqueue the main run from the start
checkpoint with
`++backend.megatron.packing_plan_path=[/gpfs/projects/.../plan/q3e256]` and
`++backend.megatron.packing_plan_boundaries=group`, from a snapshot whose Megatron contains
4cbb542b3. The plan directory holds one subdirectory per window, and all of them are loaded. The
plans must cover every step the segment will run: past the last window the run silently falls back
to the regular stream.

## Readiness for the main run (2026-11-21)

Switched on 2026-11-22 at checkpoint 51,000 (see the end of [Runs](#runs)). The table records the state before the switch.

| item | state |
| --- | --- |
| stream replay == Megatron | ✅ sampler replay == `MegatronPretrainingReferenceLayoutSampler` (production geometry, MN5); 64/64 random sequences == Megatron's `GPTDataset` (53 datasets) |
| plans, logically | ✅ tests (`tests/unit/test_cross_doc_seqs.py`, 7; Megatron `tests/unit_tests/data/test_packing_plan.py`, 6): token conservation, copy separation, all boundary modes, coverage + geometry guards |
| plans through Megatron's loader on the **real** blend | ✅ `check_plan.py` (pipeline stage `checkplan`): BlendedDataset rebuilt read-only from the cache (hash == plan), `wrap_with_packing_plans` with the run's geometry, per-window token conservation, served samples == plan segments in doc / group / none, unplanned samples unchanged |
| plan coverage 48,500 → 53,000 | ✅ `plan/q3e256_w500` (9 windows of 500 steps), `checkplan` 0 failures (2026-11-21 22:03) |
| **single window 51,000 → 61,000** (user 2026-11-22) | ✅ `plan/q3e256_g51000` (`build_plan_global.py`), `checkplan` 0 failures (2026-11-22 08:29) -- the plan the switch config uses |
| guards | Megatron 88efc4279: the plan's sampler geometry must match the run; the run refuses to start on an unplanned step and saves + exits at the end of the planned range (`--packing-plan-at-end exit`, default) |
| main-run code | Megatron HEAD 88efc4279 differs from the running 4a0fab4a1 on v4b's code path only by the packing-plan commits. The other commits are page-sparse / rowmask modules, imported only with `page_sparse_selector`, which v4b does not set. The superproject's rendered v4b arguments == the live step script's, except three committed v4b config changes that apply at any re-enqueue: `--load` own tree, `--save-interval 1000`, `--save-extra-steps` |
| switch config | `config/experiments/korbi/cross_doc_seqs/v4b_learned_cds_mn5.yaml`: renders to v4b's arguments + `--packing-plan-path …/plan/q3e256_g51000 --packing-plan-boundaries group --packing-plan-at-end exit` |

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

| 11-21 18:04 | c136.0196 | 48500 + 4500 (8 nodes) | stream 4,500 steps in 3 min; verify 64/64 (60 datasets); HF embedding at ~240 docs/s/GPU; 9 of 45 chunks done, then cancelled for the faster engine (chunks kept) |
| 11-21 19:19 – 20:00 | c136.0199 / .0202 / .0204 | bench | embedding speed (below) |
| 11-21 20:05 | c136.0205, 24 nodes | 48500 + 4500 | vLLM embedding of the remaining 36 chunks in 88 min (~460 docs/s/GPU under shared GPFS load); plan stage failed: the repair rounds could not separate a class of ~94k near-identical documents (replaced by `place_copies`, 04c2fa56) |
| 11-21 21:48 | c136.0215, 3 nodes | 48500 + 4500 | **9 plans built (~2.5 min each, 7 GB) and `checkplan` passed: 0 failures** |

**Plans `q3e256_w500`, steps 48,501–53,000** (per window of 500 steps):
- 26.7–26.8M documents and ~427.7k repacked sequences;
- ~84k solo sequences (16 %) left as they are;
- ~491k near duplicates and ~40k repeats; the largest conflict class has ~94k members;
- **0 copies forced, 0 units sharing a sequence with a copy**;
- mean cosine between the documents of a sequence: **0.22 → 0.53** (0.49 with 100-step windows);
- ~22.5 dataset shards per sequence.

`checkplan` through Megatron's loader on the real blend:
- BlendedDataset hash `09a2a1c4…` matches;
- `wrap_with_packing_plans`: planned from 48,500 to 53,000, and a plan for another global batch size is refused;
- per window: slots == the window's non-solo slots, and exactly its ~28.0B owned tokens;
- served samples == the plan segments in doc / group / none, and unplanned samples are unchanged.

| 11-22 05:05 – 08:29 | c136.0528 / .0529 / .0530 / .0821 | **single window 51000 + 10000** | e-means centroids 116 s (16,312 coarse clusters, 534.5M documents, 16M sample, mean cosine 0.688); assign 2 min and order ~5 min on 64 GPUs (one bucket OOMed on a ~830k-document boilerplate cluster -> adaptive tiles, f6b236b5); assemble 15.5 min (after replacing per-row vector reads, b671c585); `checkplan` **0 failures** |

**Plan `q3e256_g51000` (51,000 → 61,000, one window):**
- 534.1M documents in 8,556,280 sequences;
- 58.2M copies (42.8M near-duplicate pairs + repeats), **0 forced, 0 units sharing a sequence with a copy**;
- coherence 0.219 → **0.520**. That is slightly below the 500-step windows (0.533): the big window finds ~11 % near duplicates (vs. ~2 %), and copies are spread over it non-semantically. Placing copies next to related non-duplicates is a possible refinement;
- `checkplan`: exactly the window's 560,744,366,080 owned tokens, the 8,556,280 non-solo slots, every sequence once, all boundary modes.

**Main-run switch (2026-11-22, user's go: 51k, group mode):**
- 10:35 the 51,000 save completed; `pool_ctl cancel mn5-pool-p264.2081` (graceful, stop save at 51,005);
- the stop save moved to `ckpt_set_aside_switch51k/iter_0051005` (kept), tracker reset to 51000 (rolling save at 50,900 is older);
- 10:39 enqueued `v4b_learned_cds_mn5` from snapshot `autoexp_v4bcds` (superproject 4e8bb2be, Megatron 14772f454) → p264.2083 on host009;
- loaded `iter_0051000` (6 min), log: `> packing plans: 1 plans covering checkpoints [(51000, 61000)], 8556280 replaced samples, boundaries=group; planned from 51000 to 61000 (at the end: exit)`;
- first steps: 51,004 / 51,006 at 21.0 / 20.9 s per step (3,114 / 3,142 tok/s/GPU, as before the switch: 20.9 s, 3,140), lm loss 1.065 / 1.089 (before: 1.04–1.14), grad norm 0.24 (before: 0.22–0.33), no skipped / NaN steps;
- the run exits at 61,000 by itself; a later window's plan (embeddings 63k+ are `q3e128`) must exist and pass `checkplan` before re-enqueueing.

**Restart without the slow node (2026-11-22, user's go):**
- the step time had been 20.8 s since the move to host009 on 11-20 (19.4 s on host008 with identical Megatron arguments); a read-only probe of the allocation (`srun --overlap`: GPU clocks / power / PCIe link, IB rate / state / error counters) found `as02r2b25` GPU 3 at PCIe x8 (all others x16) -- a node of host009's allocation only;
- 12:03 graceful cancel of p264.2083 (stop save 51,058, resumed from it: the plan covers every step of 51,000-61,000), `pool_ctl exclude as02r2b25 as01r2b12 as02r1b08` (the latter two: idle spares with a GPU power read error / 5 IB link drops; the pool has `refresh_queued_excludes` off, so no queued job was edited), re-enqueued with `aux.node_check=light` → p264.2084;
- node check pass (all GPU groups >= 0.75 x median, 4 min); **19.65 s per step** from 51,062 (was 21.0); the remaining +0.25 s vs host008 matches group mode's ~1 %;
- restart cost: 51 min to the first step (the 51k switch: 64 min, earlier restarts 23-25 min). The first launch from a new snapshot was cold (library setup 5 min, first step 20 min vs 1 min / 6.8 min warm). Dataset loading is the remaining big phase: rank 0 alone (97-471 s, GPFS load) then all 1,023 other ranks open the ~2,100 cache files of the 531 datasets at once (12-15 min).

**Faster restarts: dataset index bundle (2026-11-22, user: "implement these first two fixes"):**
- Megatron `--dataloader-index-bundle` (f2f6b0835, b49d49844, ab06e114e): the ~3,200 cached index files the run's datasets open per rank are bundled byte for byte into one file (named by their cache file names); a rank maps it, a bundled index is a cache hit without file checks, no rank-0-alone pass, no `safe_numpy_load` lock. Missing indices load from their files. Megatron's own `--dataloader-fast-cache-load` refuses a `--data-args-path` blend (and skips the empty-split handling), so it is not used.
- Built on one node by `cds_index_bundle_mn5` (stage `bundle`: the run's own launch arguments from its step.sh -> `tools/prepare_cache.py --write-index-bundle`, which now builds the datasets as the training does, fixed validation sets included): `data_cache_bundles/v4b_p2_09a2a1c4`, 3,211 indices, 71.9 GiB. Checked: byte equality per array, all 8 splits built from the bundle alone == built from the files (lengths, blend indices, sample tokens), training blend 09a2a1c4 (218,155,276 samples). Building the whole blend from the bundle: 3.6 s in one process (rank 0 alone from the files: 97-471 s in the run).
- `v4b_learned_cds_mn5` passes it from the next (re-)enqueue on; the data order is unchanged.
- First use at scale: t24 pmp_tok (128 nodes, c136.1736, 2026-11-22 16:03): datasets built in **37 s** (model built 16:03:13 -> dataloaders 16:03:50), 3,211 indices from the bundle and 0 from files on rank 0 and rank 511, plan line as expected (51,500 -> 61,000, group). Paired control, the same branch without the bundle (t24 pmp_sb16_rm, 128 nodes, 17:30, GPFS equally quiet -- no embedding jobs running): **86 s** (rank 0 alone 6 s, own pass 53 s). So ~2.3x at 128 nodes on a quiet GPFS; the main run's 7-25 min (256 nodes) had rank 0 alone at 97-471 s, i.e. mostly GPFS contention (the 136 embedding jobs read the same data) on twice the ranks. Pausing the embedding feeder around a main-run restart is the other half of the fix.

**Plan `q3e128_g61000` (61,000 → 71,000, one window; 2026-11-22, c136.1980, 16 nodes, 49 min incl. checkplan):** embeddings `q3e128mix` (symlink view: q3e256 chunks 61k-63k + q3e128 63k+, read with `--dim=128` = Matryoshka truncation + renormalization); 16,318 coarse clusters (mean cosine 0.713 at 128 dims); 534.3M documents in 8,554,091 sequences; 59.6M copies, **0 forced, 0 sharing a sequence with a copy**; coherence 0.226 → **0.539**; `checkplan` **0 failures** (exactly 560,600,907,776 owned tokens, all modes). `v4b_learned_cds_mn5` (and the sparse config inheriting it) now pass both plans: 51,000-71,000 contiguous, from the next (re-)enqueue.

**Sparse transition, crashes and dense fallback (2026-11-22/23):**
- 52,000: the sparse agent switched the main run to page_maxpool per token (p264.2085, plan + group + bundle kept). From ~52,420 grad-norm waves (median 0.3 -> 2.5, up to 82 % of steps clipped, loss flat ~1.074); rolling 52,400 / 52,500 hard-linked into `ckpt_protected/`.
- 01:11: the sparse agent restarted from the stop save 52,989 with per-token fixes (Megatron 4720b9057, snapshot autoexp_v4bsparse4, p264.2097): grad norms calm, 17.8 s/step.
- 09:13 and 09:44: two crashes, `CUDA error: an illegal memory access` in the NCCL watchdog, on two different nodes (as03r2b17 rank 393 at 54,493; as02r2b08 rank 178 at 54,460 after the automatic restart p264.2106) -> a software fault of the new per-token code, not a node.
- 09:46: restarted on the last user-approved configuration, dense + plans 51k-71k + group + bundle (v4b_learned_cds_mn5 from autoexp_cdsbundle, p264.2108) from rolling 54,400: datasets from the bundle, `packing plans: 2 plans ... planned from 54400 to 71000`, 19.4 s/step. The pool's own retry of the sparse job (p264.2107, never assigned) was closed by hand (stop + rc), and a precautionary exclusion of as03r2b17 was undone: it left host009 with 255 usable nodes for the 256-node run.

**Embeddings 63k → end** (`q3e128`, 128-dim, projects quota):
- **complete 2026-11-22 20:59**: all 1,738 chunks 63,000-236,733 (0 failed); with `q3e256` 48,500-63,000 the whole rest of the run is embedded. One-node chunk jobs on c136 host005 (up to 136 at a time, ~55 min per chunk), paused 15:47-18:20 for the t24 branch.
- one-node pool job per 100-step chunk (`cds_embed_chunk_mn5.yaml`, batches generated by `gen_embed_chunks.py`);
- extraction of the range as one 16-node run (10 min).

**Embedding speed** (Qwen3-Embedding-0.6B, 20k documents per GPU, H100). Every variant was checked
against the HF vectors of the same documents.

| engine / variant | docs/s per GPU | speedup | cosine vs. HF |
| --- | --- | --- | --- |
| HF, padded batches (first version) | 310 | 1.00× | — |
| HF, padding-free (varlen flash attention), 128k / 256k tokens | 355 / 370 | 1.19× | ≥ 0.9997 (packed vs. padded) |
| HF, `torch.compile` (dynamic shapes) | 226 | 0.73× | recompiles dominate |
| HF, packed, 256-token truncation | ~515 | 1.66× | (less text per document) |
| HF, packed + Liger (fused RMSNorm / RoPE / SwiGLU) | 555 | 1.79× | ≥ 0.9990, mean 0.99984 |
| **vLLM pooling runner, 128k tokens per step** (default) | **~790** | **2.55×** | ≥ 0.9990, mean 0.99982 |

The profile of the HF forward explains the gap:
- GEMMs were 16 % of GPU time and flash attention 6 %;
- the rest was unfused fp32 elementwise work: RMSNorm on hidden states and on per-head q / k, RoPE copies, SwiGLU.

vLLM and Liger fuse exactly that. With vLLM the loader becomes the limit at ~20 % wait: detokenizing with the
training tokenizer and re-tokenizing for Qwen.

Measured costs (first version: HF padded, Qwen3-Embedding-0.6B, 512 tokens, mean 313 tokens per document;
divide the embedding numbers by ~2.5 for the vLLM default):

- **Stream:** seconds per 100 steps after a 27 s preload; free in practice.
- **Embedding:** 100 steps hold ~5.4M distinct documents, so ~4,500 s per node = ~5 GPU-h per 100 steps.
  - That is <1 % of the training compute of those steps (~570 GPU-h).
  - 4,000 steps ≈ 200 GPU-h (50 node-hours); keeping pace with the main run (~4,400 steps/day) needs ~2.3 nodes.
  - multilingual-e5-small would be ~5× cheaper, at lower multilingual/code quality.

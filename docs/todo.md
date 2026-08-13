# Node-local data prefetch — status & TODO

Goal: on stall-prone shared filesystems (GPFS on MareNostrum), train Megatron from a
node-local copy of the data without copying the whole (>1 TB) dataset up front — a
*partial, continuous, read-through* prefetch.

Full history, measurements and gotchas live in the agent memory file
`memory/marenostrum_data_staging.md`. This file tracks what's done and what's left.

## What works today (proven on the real 1.2 TB dataset)

- **Read-through mirror** (`oellm_autoexp/data_staging/mirror.py`): a node-local sparse
  mirror of the `.bin` + a per-1 MiB-block presence bitmap. The patched reader
  (`megatron_patch.py` → `LocalMirrorBinReader`) serves a read from the mirror iff every
  block it spans is present, else from the source FS. Byte-identical at any fill state
  (validated); zeros are never served.
- **Windowed (locality-preserving) shuffle** (`windowed_shuffle.py`): replaces Megatron's
  global shuffle with a K=256-lane transpose + local block-shuffle. Keeps consumption
  locality (so prefetch is sequential) while staying source-mixed.
- **Lane prefetcher** (`scripts/prefetch_warm.py --mode lanes`): stages each lane's
  contiguous file region (exact `sample_index` boundaries, block-aligned), round-robin by
  block, with a **pre-stage barrier** (training's first read blocks until block 0 of all
  lanes is staged). Parallel `pread`/`pwrite` (~500 MiB/s in-situ; GPFS itself does
  1.7–5.8 GiB/s, so we're not FS-bound).
- **Multi-epoch / chunk-aware reshuffle**: each epoch re-shuffles within lanes (seed+e),
  same lanes/file-regions → the staged mirror is reused every epoch.
- **Result: 100% node-local hits**, zero GPFS reads in the training loop, both single- and
  multi-epoch. Recipe in `run_mirror_test.sh` + `config/slurm/marenostrum_mirror.yaml`.
- **REQUIRED**: `dataloader_type=single` — `cyclic` re-shuffles on top of our windowed
  index and destroys locality.

---

## ✅ DONE — Consumption cursor (replaces the `OELLM_MAX_LANE_BLOCKS` cap)

Built and validated (job 41695739): training writes `consumed_train_samples` to a node-local
cursor file each iter (`megatron_patch.apply_cursor` → patches `training_log`); the lanes
prefetcher reads it, keeps rounds `[cursor−retain, cursor+lookahead]` resident, **evicts behind**,
and stages within budget. Confirmed: `cursor_round` advanced 0→3, window slid `[1..4]→[2..5]`,
rounds 0,1 evicted, **resident bounded at 0.3 GiB constant**, **100% local hits throughout**.
Env: `OELLM_CURSOR_FILE`, `OELLM_LOOKAHEAD_BLOCKS=3`, `OELLM_RETAIN_BLOCKS=1`.
**Requirement**: `OELLM_SHUFFLE_BLOCK ≪ per-lane samples (Me/K)` — else the within-lane shuffle goes
global and the `cursor_round→doc` mapping breaks (real configs with K=256 over a large epoch satisfy
this trivially; only tiny-lane test configs violate it).

### (original analysis kept for reference)
### TODO 1 — Consumption cursor (replace the `OELLM_MAX_LANE_BLOCKS` cap)

### The problem
The lanes prefetcher is **blind to where training is**. It stages rounds (block 0 of all
lanes, then block 1, …) as fast as it can — ~100× faster than the GPU consumes — until it
hits `--budget-gb`, then **FIFO-evicts the oldest-staged block**.

Three failure modes without a cursor:
1. **Evicts data training still needs.** Oldest-staged == earliest in consumption order ==
   what training is reading *now* (the prefetcher raced ahead and is deleting block 0 while
   the GPU is still on block 0). Training falls back to source — correct, but the locality
   benefit and GPFS-offload are lost.
2. **Fills the disk.** `budget > free` → `ENOSPC` crash (observed at 192 GiB).
3. **Current band-aid**: `OELLM_MAX_LANE_BLOCKS=3` just stops after 3 blocks/lane (~48 GiB).
   Works only because short/≤1-epoch runs never advance past block 0. Not general.

### When it actually bites
Only when the **per-node consumed set exceeds local disk** (~436 GiB on MN5). Per epoch a
node reads ≈ `dataset_size / num_nodes`:
- 1.2 TB on ~64 nodes → ~19 GiB/node/epoch → fits → **no cursor needed** (even multi-epoch).
- 1.2 TB on 1–4 nodes → 300+ GiB/node/epoch → exceeds disk → **cursor required**.

So: needed iff `dataset_size / num_nodes > local_free_disk`.

### The fix
A **cursor** = training's consumption position (`consumed_train_samples` → column → round).
Turns blind FIFO into a sliding window:
- **stage ahead, bounded**: keep rounds `[cursor, cursor+margin]` staged (~tens of GiB),
- **evict behind, safely**: drop rounds `< cursor` (already consumed, not re-read until next
  epoch),
- **constant disk footprint** == the margin, independent of run length.

### Implementation sketch
1. Training-side hook (1 line in the Megatron loop or in `megatron_patch`): write
   `consumed_train_samples` to a node-local cursor file every N iters (the loop has the
   global count; dataloader workers don't).
2. Prefetcher reads the cursor, maps `samples → column → round`, gates "stage round c" on
   `c ≤ cursor_round + margin`, and evicts by round instead of pure FIFO. `run_lanes`
   already has the round (`c`) + FIFO eviction structure to hook into.
This replaces `OELLM_MAX_LANE_BLOCKS` and is correct for any run length / disk size.

---

## ✅ DONE — Multi-file / blended datasets (the prefetcher)

Built and validated on a real 2-file blend (jobs 41701650 / 41702431, `blendA`+`blendB`,
`dataloader_type=single`). The lanes prefetcher now handles a comma-separated `--prefix`:
- **Per-sub-dataset cache discovery** (`prefetch_order.discover_hashes_by_prefix`): maps each
  prefix → its `*-GPTDataset-train-*` hash by matching the prefix path inside `description.txt`
  (a blend cache holds one GPTDataset per prefix + the `BlendedDataset` index).
- **Blend weights** (`load_blend_weights`): per-file share from the `BlendedDataset-*-dataset_index`
  (bincount). The global cursor maps to each file by way of `file_consumed = global_consumed * weight`.
- **One mirror + lane set per file** (`run_lanes_multi` + `_LaneState`): each `IndexedDataset` already
  gets its own `LocalMirrorBinReader` (`<name>.bin`/`.bitmap`/`.window0_ready`), so the writer side
  just runs one `_LaneState` per file; budget is split across files; round-0 + sentinel of EVERY file
  is primed first so all per-file barriers release.
- **Discovery race fix**: the prefetcher retries discovery until ALL sub-dataset caches + the
  `BlendedDataset` weight cache exist (waiting on just the first prefix's `shuffle_index` raced).
- **Driving it**: `data_path` (list) + `PREFETCH_PREFIX` (comma string) must live in a YAML config —
  commas/lists on the hydra CLI trip sweep-override validation. See
  `config/experiments/megatron_marenostrum_blend_test.yaml`; run by way of `run_mirror_test.sh -` with
  `CONFIG_NAME=...`. Confirmed: both files mirror-active on all ranks, distinct hashes, 0.5/0.5
  weights, both staged, resident bounded, 1500/1500 iters clean.
- **Gotcha found**: `chunk_0000.idx` in `cerebrase-SlimPajama-627B/train` is malformed
  (`document_indices[-1] != seq_count`) → Megatron `_IndexReader` assertion. Use valid prefixes;
  test blend lives in `/gpfs/scratch/ehpc390/data/blend_test/{blendA,blendB}` (copies of `small`).

Still open (refinements, not blockers): weight-proportional budget *and* per-file lane-count `K`
scaling (currently equal split / same K), and per-file barrier ordering. Captured below.

## TODO 2 (orig analysis) — Multi-file / blended datasets (incl. single files of 100s of GB)

### Background: what "multi-file" means in Megatron
`data_path` can be a **blend**: a list of prefixes (optionally weighted). Each prefix is a
separate `.bin/.idx` = a separate `IndexedDataset`/`GPTDataset`, combined by `BlendedDataset`.
A "single huge file" is just one prefix that happens to be 100s of GB.

### Key fact (verified in `blended_dataset.py` + `helpers.build_blending_indices`)
The blend interleaves **which** sub-dataset each position draws from (weighted), but
**each sub-dataset is consumed in its own order** (its `dataset_sample_index` increments
0,1,2,…). So with `dataloader_type=single`, every file is read in *its own windowed order*
→ **per-file locality is preserved**. Think of the files as an extra outer layer of "lanes":
training interleaves across files just like it interleaves across lanes within a file.

### What already generalizes (no work)
- **Read-through reader**: the monkeypatch fires per `IndexedDataset`, so each prefix gets
  its **own** `LocalMirrorBinReader` → its own `mirror_dir/<name>.bin` + `<name>.bitmap`.
  Reads to file A hit A's mirror, file B hit B's. Already correct.
- **Huge single files**: the mirror is **sparse** (`ftruncate` to logical size, physical
  only when written), and the bitmap is tiny (300 GB / 1 MiB = ~37 KB). So size alone is a
  non-issue; only the *consumed* portion is staged.
- **Windowed shuffle**: the patch hits each sub-dataset `GPTDataset` independently.
- **Per-file barrier**: `window0_sentinel(mirror_dir, prefix)` is already keyed by prefix.

### What needs building
1. **Prefetcher over multiple prefixes.** `prefetch_warm.py` currently takes one `--prefix`.
   Extend it to take the blend's prefix list (from config) and stage all of them. Either one
   process iterating files, or one thread-group per file. Each file → its own `MirrorWriter`
   + lanes computed from *that file's* `sample_index`.
2. **Per-sub-dataset cache discovery.** A blend's cache dir holds **multiple** train hashes
   (one GPTDataset per prefix) **plus** the BlendedDataset's `dataset_index` cache.
   `discover_desc_hash` currently assumes exactly one — must map prefix → its hash (for example by
   matching the `description.txt`, or by passing hashes explicitly).
3. **Budget allocation across files.** Local disk is shared. Training reads from all files
   concurrently (interleaved), so the prefetcher must keep the leading window of **every**
   file staged at once — split the budget across files **∝ blend weight × per-file epoch
   share** (a file read at higher weight / a bigger file needs a bigger resident window).
   This is exactly TODO 1's sliding window, but now per-file with a shared budget pool.
4. **Lane count per file.** A 300 GB file wants many lanes (it spans most of the order); a
   2 GB file few. Scale `K` (and the staged block size) per file by its size/weight so each
   file's leading window is a sensible fraction.
5. **Barrier semantics.** Block training's first read of file f until file f's window-0 is
   staged. Already per-file by way of the sentinel; just need all files' prefetchers running.

### Single huge file (one 100s-of-GB prefix) — special case of the above
Already handled by the single-file path *except* that one epoch's per-node share of that one
file may exceed local disk → it needs **TODO 1 (the cursor)**. No new structure, just the
sliding window. (`Me`-detection, lanes, sparse mirror all already scale to 100s of GB.)

### Suggested order of work
1. Land **TODO 1 (cursor)** first — multi-file budget sharing *is* per-file sliding windows,
   so the cursor is the prerequisite.
2. Then **multi-prefix prefetcher** + per-sub-dataset hash discovery + weight-proportional
   budget split.

---

## Minor TODOs / cleanup
- **Framework bugs found** (affect the stock MareNostrum path too, not just prefetch):
  - `--job-name` lacks the sweep `_0` suffix the submit validator wants
    (`generator.py:30` vs `validator.py:37`); worked around with `slurm.sbatch.job_name=null`.
  - `marenostrum.yaml` exports `SLURM_CPUS_PER_TASK=12` while sbatch sets `--cpus-per-task=20`
    → modern SLURM aborts srun; worked around with `~slurm.env.SLURM_CPUS_PER_TASK`.
- **Container env**: `os.copy_file_range` is unavailable in `scope_env` python *and* cross-fs
  (GPFS→/scratch) EXDEV — do not use; plain `pread`/`pwrite` does ~1.6 GiB/s.
- **`single` sampler requirement** should be enforced/validated when prefetch is enabled (warn
  if `cyclic`), since `cyclic` silently destroys the locality.
- Per-job `/scratch` is wiped, so the mirror is rebuilt each job (no cross-job reuse). If MN5
  ever offers persistent node-local scratch, a shared mirror across the many concurrent runs
  would be a large additional win (read once for the whole fleet).

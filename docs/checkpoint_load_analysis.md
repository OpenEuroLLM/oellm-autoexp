# Checkpoint load and restart cost of the v4 main run (2026-11-09)

The question (user): a resume took "10+ min" while a save takes 1–2 min. Is the load sequential and going through
host RAM instead of loading in parallel straight into GPU memory?

**Short answer:**
- The load itself is now ~1.5–5 min at 256 nodes (`fsdp_dtensor`, packed optimizer scalars since Megatron ac1020efc).
  The 23–40 min loads were before the packing (11-01 to 11-03: 40–46 min) and one special restart (11-05, head-dim
  widening: 23 min).
- Each rank does read its own file sequentially, in one thread, through host memory. But that is ~1.1 GB per rank,
  i.e. ~1–3 s, so it is not the bottleneck.
- The rest of a ~20 min restart is Megatron init, the pool gap, model build, dataset build and the first steps.

All numbers below are from the rank-0 `datetime` lines of `slurm-mn5-pool-*.log` in the run dir, read-only.

## 1. Where a restart goes

**p264.0022 → p264.0133 (11-09, 256 nodes):** last step 14:30:31, steady 15 s steps from 14:51:07, i.e. 20.6 min.

| phase | time |
|---|---|
| exit save + teardown | 41 s + 49 s |
| pool gap until the new run starts | 2:56 |
| launch + library setup | 28 s + 1:49 |
| megatron init | 3:43 (98 s at 128n) |
| model build + checkpoint load | 6:17, of which ~4–5 min is the load |
| datasets | 1:29 |
| first two steps above steady | ~2.4 min |

**init → "model, optimizer, and learning rate scheduler are built":**

| run | nodes | time |
|---|---|---|
| p264.0133 (new snapshot) | 256 | 377 s |
| p264.0022 | 256 | 183 s |
| p264.0019 (512 → 1024 ranks reshard) | 256 | 311 s |
| p134c.0074 | 128 | 171 s |
| p134c.0015 (widened checkpoint, new snapshot) | 128 | 1384 s |
| speed runs, no checkpoint | 260 | 59–120 s |

**Saves at 256n:**
- The blocking part is 18.5 s.
- All 1024 shard files land within ~9 s (1.16 TB, likely buffered by GPFS).
- "Successfully saved" comes 30–50 s after "saving checkpoint".
- Nothing is logged between "loading checkpoint" and "successfully loaded", so the split inside the load is estimated,
  not measured.

## 2. The load path (Megatron 3d8071635, container torch 2.14.0a0)

- **`.metadata` (322 MB at 256n: 318k keys, 1.14M storage items) is read and unpickled 3 times per rank, on all 1024
  ranks:**
  1. `_load_base_checkpoint(rank0=True)` → `dcp.load({'args','iteration','checkpoint_version'})`
     (`checkpointing.py:2193`, a collective load with a central plan for three small items);
  2. `FileSystemReader(...).read_metadata()` to check for `rerun_state_machine` (`checkpointing.py:2796–2798`);
  3. `_MetadataOnceReader` (`checkpointing.py:2203`).

  One unpickle takes 11.2 s on the MN5 Xeon 8460Y+ (3–4.5 GB RSS). That is ~34 s CPU per rank, plus ~1 PB of logical
  reads against one file.
- **Tensor read:** torch `FileSystemReader.read_data`.
  - One thread per rank, files in sequence.
  - Each item is `torch.load(map_location="cpu")`, then narrowed, then copied from pageable memory to the GPU.
  - No thread pool, no pinned memory, no overlap.
  - With an unchanged layout each rank reads essentially its own file (582–2559 items, ~1.1 GB). Measured: 0.65 s per
    real 1.1 GB shard from warm cache; 5 GB/s single-stream `dd` from GPFS.
  - Throughput with 1024 concurrent readers is unverified.
- **Planning:** `FSDP_DCP_LOCAL_LOAD_PLAN=1` gives `no_dist=True`. Planning is local (chunks per tensor: median 2, max
  1024), followed by one global barrier.
- **Other load-only work:**
  - a full dummy optimizer step to create the state slots (`distrib_optimizer.py:1455–1469`);
  - `preprocess_fsdp_dtensor_state_dict` (≤ ~18 s);
  - `print_diff_in_state_dicts` after `sleep(rank × 1 ms)`;
  - the optimizer state loads in place in the same DCP pass.
- **Why saves are fast:** the nvrx writer reuses the cached global plan (`--ckpt-assume-constant-structure`), never
  reads `.metadata`, and writes per rank in a background process.

## 3. Causes

1. **Cold start per new code snapshot** (strong correlation; the cache responsible is unverified). The first job on
   a fresh snapshot is 2–5× slower in build + load and ~80 s slower in library setup:

   | snapshot | first job | next job |
   |---|---|---|
   | v4valfix | o260c.0016: 411 s | 172 s |
   | v4move | 882 s | 160–442 s |
   | v4fc | p134c.0009: 750 s | – |

   Suspects: empty `__pycache__` (1024 ranks compiling `.pyc` on GPFS), Inductor / Triton cache misses.
2. **Metadata read 3×:** ~34 s CPU per rank (measured) and a hot spot on one GPFS file.
3. **Checkpoint-surgery restarts:** `scripts/korbi/production/widen_attention_head_dim.py` writes the rewritten tensors
   unsharded into one extra file (`no_dist=True`). All ranks then read the full tensors from that file and narrow them,
   ~512× read amplification. This explains most of 11-05's 23 min.
4. **The sequential, host-RAM tensor read:** real, but ~1–3 s per rank. It matters only under contention or for
   resharded / unsharded items.

## 4. Fixes, ranked by gain / risk

| # | fix | saves | risk | test |
|---|---|---|---|---|
| 1 | load-phase timing (like `MEGATRON_DEBUG_SAVE_TIMING`): args, metadata, dummy step, preprocess, plan, `read_data`, barrier wait min/max, `load_state_dict` | – (measures the rest) | none | 2-node, then next approved restart |
| 2 | warm a new snapshot: `python -m compileall`, path-independent Inductor / Triton cache keys, `TRITON_CACHE_AUTOTUNING=1` | 2–12 min per new snapshot (+80 s setup) | low | 2-node first-of-snapshot run vs second run |
| 3 | read the metadata once per process (cache keyed by path + mtime for the 3 call sites); args with `no_dist=True`; optionally one reader per node | ~22 s CPU per rank + the duplicate GPFS reads | low (read-only path) | amb_rs resume test |
| 4 | surgery scripts write rewritten tensors sharded like the source | ~20 min per widen / re-seed-type restart | low | 2-node widen → resume |
| 5 | prefetch the metadata and the rank's own file into pinned memory during init | ~15–40 s | medium (+1.1 GB pinned per rank) | 2-node |
| 6 | parallel / pinned reader (offset-sorted reads, 4–8 threads, non-blocking H2D) | ~1–3 s per rank, more on reshard | low | shard microbenchmark |
| 7 | smaller metadata (pack the ≤ 4 KB items) | ~half the unpickle | medium (format change, one-way) | multi-rank resume test |

Outside the load but larger in total: megatron init (228 s at 256n vs 98 s at 128n), the ~3 min pool gap, the
dataset build (~90 s) and the slower first steps. Together these are ~10 of the 20.6 min and need the same
instrumentation.

**Not verified:** the split inside the load window; GPFS throughput with 1024 readers; which cache causes the
first-of-snapshot slowdown; unpickle time under 4-rank contention on a compute node.

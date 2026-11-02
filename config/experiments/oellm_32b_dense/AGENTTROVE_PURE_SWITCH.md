# Decontaminated agenttrove in the B1 continuation, data order kept

Status as of **2026-11-02**. Applies to `b1_conservative60k_100k_tensorwise_continue.yaml`
(the B1 32B run in the e-ext pool), from **iteration 326513** on. Code references are to the
run's Megatron (`submodules/Megatron-LM`, `c9cf7e84c`). Tool:
`scripts/korbi/agenttrove_pure_shuffle.py`. Config commit: `94d6aad6`.

## In one sentence

The run still reads the original `agenttrove-0.0` files; it only skips every training window
that touches one of the 4192 documents missing from `agenttrove-0.0-pure`, and every other
sample of the run, including the order of the remaining agenttrove windows, is unchanged.

## What agenttrove-pure is

`flag/agenttrove-0.0-pure/megatron-lm/openeurollm-256k-without-bos/data/shard_0000`, compared
document by document (full md5 of the token ids) with
`flag/agenttrove-0.0/megatron-lm/data/shard_0000_text_document`:

| | original | pure |
|---|---|---|
| documents | 1,568,412 | 1,564,220 |
| distinct documents | 1,330,717 | 1,327,005 |
| tokens | 20,222,264,518 | 20,169,866,228 |

- pure = original minus **3712 distinct documents (4192 copies)**; nothing else removed,
  no copy count changed, every kept document token-identical.
- The **file order differs** (only ~70% of adjacent pairs keep their order).
- The sibling `openeurollm-256k-reason-without-bos/` is a different tokenizer variant
  (~2% fewer tokens) and must not be used for this run.
- The original is itself heavily duplicated (168k documents appear more than once); the pure
  version keeps those duplicates.

## Why not just swap the path in the mix file

The blend tables (which dataset every global sample comes from, and its sample number in that
dataset) depend only on the weights, the number of datasets and the total size
(`blended_dataset.py`, `build_blending_indices`), so a path swap with the same weight would keep
every other dataset exactly. But agenttrove itself would change completely:

- Megatron shuffles a dataset by **file position** (`document_index` over doc ids, then windows,
  then `shuffle_index`). A file of different length and order gets an unrelated shuffle.
- The 998/2/0 train/valid split is taken by file position too: ~3000 documents that are
  agenttrove *validation* documents today would become training documents, and vice versa.

## What was done instead

Megatron reads agenttrove sample `j` as window `shuffle_index[j]`; window `w` covers
`document_index` positions `sample_index[w,0] .. sample_index[w+1,0]` inclusive (the end position
always contributes at least the extra label token) (`gpt_dataset.py`,
`_query_document_sample_shuffle_indices`). The blend draws agenttrove samples 0, 1, 2, ... in
order, so the samples already trained are exactly `j < J0`.

The train `shuffle_index` of agenttrove was replaced by

```
new[:J0] = old[:J0]                               # drawn before the switch: untouched
new[J0:] = [w for w in old[J0:] if not bad[w]]    # same windows, same order, minus every
           + padding from the same list            #   window touching a removed document
```

with `J0 = 7,196,168` (the agenttrove samples drawn before iteration 326513). 60,107 of the
remaining windows are dropped (0.34%). Each removed document copy sits in about 4 windows per
pass; a dropped window can also hold the edge of a neighbouring clean document, so the cut
removes slightly more than the contaminated tokens, never less. The index was built with spare
samples (24.6M windows for 19.8M requested), so the shorter list still covers the run to 894k.

The file lives in a new cache directory,
`/e/scratch/e-ext-2025e02-108/oellm_32b_dense_b1/data_cache_agenttrove_pure/`: a symlink tree of
the read-only `jj1/data_cache_link` (13,047 entries) with that one file real. Megatron counts a
cache hit when the four index files exist, so it loads everything as before and builds nothing.
`build.json` there records the switch iteration and the counts; `README.txt` says what it is.

## Effect on the run

- **All other datasets** (99.46% of samples): the same samples at the same positions.
- **agenttrove**: the windows the run would have trained, in the same order, minus the dropped
  ones. Same share of training (weight 0.00538), the removed documents are never trained again.
- **Validation**: unchanged. Only the train index was filtered; about 1 of the removed documents
  lies in agenttrove's 0.2% validation split and still counts in the validation loss.
- **Before the switch**: the 7.2M agenttrove samples trained up to iteration 326512 included the
  contaminated documents. That cannot be undone.
- The mix file still **names** `agenttrove-0.0`; the trained tokens are the pure set.

## How it was checked

Offline, at the resume iteration (`verify --iteration 326512 --megatron`, all passed):

- the first J0 entries are identical to the old file; the rest equal the old order minus exactly
  the marked windows; dtype and length unchanged;
- no window drawn after the switch (all draws up to 19.8M) touches a removed document;
  independently, 20,000 random draws decoded through the document index and 2,000 through
  Megatron's own `GPTDataset._query_document_sample_shuffle_indices`: none contain one;
- the blend tables, the agenttrove document/sample index and every other entry are the old files;
  no 0-byte files.

At the restart (job `oellm_32b_dense_b1_conservative60k_100k_tensorwise_1842bc`, host008 =
SLURM 2650475): the log shows the new `data_cache_path`, 906 index loads and 0
"Build and save"; training resumed at 326513, loss 1.350-1.358 against 1.349-1.364 before the
stop, ~1920 tok/s/GPU.

## Operating notes

- The config now sets `backend.megatron.data_cache_path` to the new directory; a plain
  re-enqueue renders it (no CLI override needed).
- **Do not rebuild the index for a later iteration.** It already holds the switch point;
  `build` refuses while `build.json` exists unless `--force` is given.
- Run `verify` on a JUPITER login node with the venv python (`source ~/work/venv/bin/activate`),
  not inside the container: the container's python imports Transformer Engine, which needs
  `libcuda`, which login nodes do not have.
- The same procedure applies to any other dataset in the mix that gets a "minus some documents"
  version: `setup` / `mask` / `build --iteration X` / `verify --iteration X --megatron`, with the
  dataset paths at the top of the script adjusted.

# Document metadata for grouping

Question: besides embeddings, can documents be grouped by metadata (same GitHub repo, same website,
same book, same Wikipedia language edition)? Findings of 2026-11-21:

## What exists, and where

| source (share of the stage-2 mix) | metadata in the release records | where |
| --- | --- | --- |
| common-pile stack-v2 full (19.5 %) | `metadata.repo_name`, `path`, `url`, `language`, `blob_id`, stars, ... | JUPITER `/e/scratch/e-ext-2025e02-108/poeppel1/v8/flag_src/common-pile-stackv2-0.1/sample/documents/*.jsonl.zst` (764 files; scratch, may be purged) |
| hplt-4.0 (16.9 %) | `u` (URL), `crawl_id`, language ids, web register, ... | LUMI `/scratch/project_465002530/training/collection/flag/<ds>/release` |
| dclm (8.0 %), finepdfs | probably `url` (HF versions; no counts on MN5) | LUMI |
| nemotron-cc (7.8 %), -opus | `url`, `language` | LUMI |
| megamath (8.5 %) | optional `url`, `domain`, `repo_info` (code subset) | LUMI |
| finewiki (0.19 %) | `title`, `page_id`, `wikiname`, `url` | LUMI |
| swallow-code | `repo_name`, `path` | LUMI |
| tower+ / vtok (SWE trajectories, SFT sets) | none useful | — |

**MareNostrum holds no source text or metadata**, only the Megatron bins. JUPITER's mirror
(`.../oellm/container/data/training/collection/flag`) holds only two mixture-vitae directories.

## Mapping a training document back to its record

Possible for every layer, deterministic, no stored id files needed:

1. **Stage view (s3) → full shard.** `filter_index` keeps the original `pointers`; `searchsorted` in the
   parent `.idx` gives the parent document. Checked on 200k documents: 100 % found.
2. **Decontaminated → original index.** `<prefix>.removed.npz` (`doc`), plus
   `.removed_external.npy` where present: re-insert the gaps.
3. **Agent re-tokenization:** order and boundaries unchanged.
4. **Flag shard → release file:** document counts match `release.json` exactly (shard k = release
   file k, in order). Order preservation is Megatron preprocessing's normal behaviour, not verified
   row by row.
5. **GigaToken sources (stack-v2-full):** files sorted, rows in order, `shard_NNNN.info.json` gives the
   unit range. Skipped rows are deterministic (`fmt_doc`).

## What metadata is worth inside a window

The plan regroups only the documents a window of steps reads anyway (that keeps mixture, order and
token count). Metadata groups help only if members of a group **meet inside one window**.

- **Code.** A 9,323-row sample of stack-v2-full held 8,620 different repos: rows are not grouped by
  repo, and repos are sparse. A 500-step window reads ~1M stack-v2 documents out of 191M (0.5 %).
  Two files of the same repo in one window are therefore rare except for huge repos (cdnjs-like),
  which the embedding already groups.
- **Web.** Large domains (Wikipedia, news sites, forums) do recur within a window. A domain key could
  matter there; the embedding captures most of the topical part.

**Conclusion.** Within the window design, metadata is a refinement, not the main signal.
- It is wired in as `build_plan.py --group-keys <npz {keys, group}>`: members of a group stay
  contiguous inside a leaf.
- Building the key files means extracting (row → key hash) per release file on LUMI (hplt / dclm /
  nemotron URLs → registered domain) or JUPITER (stack-v2 repo), then joining through the mapping
  above.

**What would make metadata powerful** is pulling a document's siblings from **outside** the window,
e.g. repository-level samples as in DeepSeek-Coder. That changes which data is trained, and how often:
a repo pulled together is consumed in one go and its files leave their scheduled windows. It is a data
selection decision, not a packing one, so it is not part of this pipeline.

Not done: the key extraction itself. It needs LUMI access, which this workstation does not have.

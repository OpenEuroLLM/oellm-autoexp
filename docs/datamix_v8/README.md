# datamix v8 (OpenEuroLLM, 12T) on MN5

`datamix_v8.tsv`: the v8 table (60 entries); `inventory.tsv` (`scripts/datamix_v8/inventory.py`): where each source
exists (MN5 flag / JUPITER / LUMI / HF only).

## Collection (2026-10-31)

| part | where from | how | MN5 target (`/gpfs/projects/ehpc533/users/korbip/data/`) |
|---|---|---|---|
| 21 sources | MN5 `flag/` | already there | (flag-decon-v2 trees) |
| 16 mixture-vitae-1.0 subsets + stackexchange (782 GB) | JUPITER `.../mixture-vitae-1.0-subsets-reasoning-instruct/<s>/megatron-lm` | hpctransfer (`--jobs 64 --split-size 1G`) | `v8_sources/jsc_mv1_subsets/` |
| 15 LUMI-only sets (16 files, ~1.27 TB) | LUMI `vitiugin/data_tok` (`<bos> doc <eos>`, our tokenizer) | hpctransfer | `v8_sources/lumi_vtok/` (strip `<bos>` on MN5) |
| 7 HF-only sets (~1.07 TB raw) | HF -> JUPITER scratch `poeppel1/v8/hf` | GigaToken, 8 nodes | `v8_sources/jsc_tok/` |
| common-pile stackv2, full filtered (2,839B) | LUMI flag `common-pile-stackv2-0.1/sample` -> JUPITER | GigaToken | `v8_sources/jsc_tok/` |

hpctransfer: `--stats` cannot be combined with `--split-size`; without `--split-size` one big file is one rsync
stream (~35 MB/s); a killed transfer can leave a partial file under the final name (check sizes / delete).

## Size differences vs the table's "available"

From flag's own counts (`counts/.../{source,sample,release}.json`) and `metadata.yaml`:

| source | table avail | flag source | flag sample (filter) | flag release (= ours) | conclusion |
|---|---:|---:|---:|---:|---|
| common-pile-stackv2-0.1 | 2,929B | 2,929B | 2,839B (propella filter, 97 %) | 710B (**random 25 %**) | budget cut only: take the full filtered 2,839B (v8 budget 2,050B = 0.72 epochs instead of 2.9) |
| hplt-4.0 eng_Latn ccc | 955B | 955B (cc crawls >= CC-MAIN-2024-31, `etc/ccc.slurm`) | 350B (**bsc-edu >= 1.5**) | 349B (+ PII masks, xml/md scrub) | quality filter, not a sample: the other 605B are low-edu web; kept at 349B (2.3 epochs at budget 799B) unless decided otherwise |
| hplt-4.0 eng_Latn ia | -- | 659B | 176B (bsc-edu >= 1.5) | 175B | same |
| dolma-3-math | 22B | -- | -- | 11.8B | the table adds ingredient1 + ingredient2 of dolmino-mix-100B-1125, both drawn from the same 10.7B dolmino-math pool; ours (ingredient2) is effectively the full pool |

## Tokenization (`scripts/datamix_v8/tokenize_gt.py`, `tokenize_gt.sbatch`)

GigaToken (`pip install gigatoken`, aarch64 wheels) with `Tokenizer.from_sentencepiece(tokenizer-256k/tokenizer.model)`:
identical ids to SentencePiece (0 mismatches: 4k stackexchange documents + edge cases, and sampled formatted documents of
every chat format), ~47 MB/s per core single-threaded, ~680 MB/s on a 72-core login node (HF tokenizers: 53 MB/s).
Output as flag: ids + `<eos>`, no `<bos>`, int32, one document per sequence; shards <= 50B tokens.

Chat sets -> ChatML (`<|im_start|>role\n...<|im_end|>`, "\n"-joined); `reasoning_content` -> `<think>\n...\n</think>\n\n`
before the answer; tool calls `<tool_call>` JSON, tool results `<tool_response>`, tool schemas a `<tools>` block in the
system turn (Qwen style). marin open-thoughts-4: user = `instruction_seed`, assistant = `generated_text` (Qwen3-32B,
starts with `<think>`; some are cut at the generation limit, kept as published). The agent-delimiter re-tokenization
(tokenizer-256k-agent) runs on MN5 after decontamination, as for flag-decon-v2.

Sources: Nemotron-Cascade-2-SFT-Data (593 GB jsonl, all 8 domains), Nemotron-SFT-Science-v2 (53 GB), Nemotron-SFT-Math-v4
(5.5 GB parquet), marin open-thoughts-4 code / math / science qwen3-32b annotated (21 / 50 / 3.3 GB),
Nemotron-Pretraining-Specialized-v1.2 (54 GB, 4 subsets, text).

## English vs multilingual: v8 and the earlier OpenEuroLLM mixes (2026-10-31)

The mix files on LUMI (`/scratch/project_465002530/users/vitiugin/datamix/`, copied for reference where noted):

| file | weight | content |
|---|---:|---|
| `flag-option4.txt` / `flag-option5.txt` | 0.80 | English part (27 / 29 sources) |
| `flag-ml.txt` (= `docs/datamix_v8/flag-ml_lumi.txt`) | 0.20 | multilingual part: 286 shards, 42 languages |
| `flag-option4-shards.txt` / `flag-option5-shards.txt` | 1.00 | the two combined per shard; option5-shards = the flag mix trained on MN5 so far (452 shards) |
| `flag-option5-project2530.txt` | 1.00 | English only (the 12T table) -- option5's English part renormalized |
| v8 (`datamix_v8.tsv`) | 1.00 | English only |

English part, % of the English weight (path-matched; mixture-vitae counted as one source):

| source | opt4 | opt5 | project2530 | v8 |
|---|---:|---:|---:|---:|
| common-pile-stackv2 | 23.2 | 23.1 | 24.4 | 17.1 |
| dclm | 0 | 11.3 | 11.3 | 11.8 |
| megamath | 10.6 | 10.6 | 10.6 | 10.6 |
| mixture-vitae (full) | 14.3 | 14.3 | 14.3 | 0 |
| mixture-vitae, 17 instruct / reasoning subsets | 0 | 0 | 0 | 6.7 |
| new agent / reasoning SFT sets (SWE-ZERO, Nemotron-Cascade, ...) | 0 | 0 | 0 | 13.7 |
| finephrase | 16.3 | 8.1 | 8.1 | 4.1 |
| hplt-4.0 eng ccc | 8.0 | 7.9 | 8.0 | 6.7 |
| Nemotron-CC high / medium-high | 9.3 / 0 | 4.6 / 1.3 | 4.6 / 1.3 | 7.0 / 6.4 |
| nemotron-pretraining-specialized 1.0 | 5.7 | 5.7 | 5.7 | 5.3 |
| Nemotron-MIND | 0 | 1.1 | 1.1 | 0 |
| openthoughts-3 | 0.7 | 0 | 0 | 0 |
| stackv2-edu / dolma-3-math / other dolmino | 1.3 / 0.4 / 0.9 | 1.3 / 0.4 / 0.9 | 0.7 / 0.7 / 0 | 0.7 / 0.7 / 0 |

The rest (starcoder, finemath, swallow-math / -code, openwebmath, finepdfs-edu eng, hplt ab / ia, agenttrove,
specialized-1.1) is unchanged across the four. So v8 = project2530 with the full mixture-vitae replaced by its
reasoning / instruct subsets plus new agent / reasoning SFT data, less stackv2 and finephrase, more Nemotron-CC --
and, like project2530, **no multilingual data**, whereas every full mix so far was 80 % English + 20 % `flag-ml`.

**flag-ml** (identical in option4 and option5): hplt-4.0 58.6 %, nemotron-cc-tower+ 14.0 %, finepdfs-edu 13.2 %,
finepdfs 4.8 %, nemotron-cc-opus 4.4 %, fineopus 3.2 %, finewiki 1.0 %, dochplt 0.7 % (of the 20 %); 42 languages, top
spa 15.6 / deu 13.9 / fra 13.8 / por 5.4 / ita 4.9 % (eng-X parallel pairs 3.9 %). All 286 shards are on MareNostrum in
`flag-decon-v2-agenttok` (v6-decontaminated, agent tokenizer): 4,963B tokens -> at 2.4T (20 % of 12T) 0.48 epochs of
hplt, 0.62 tower+, 2.25 finepdfs-edu, <= 1 for the rest; a few eng-X pair sets ~3.

**Proposal (v8 + ML):** 80 % v8 (its table weights x 0.8) + 20 % flag-ml (its own weights), as option4 / option5 and
the current training mix -- `build_datamix.py --ml docs/datamix_v8/flag-ml_lumi.txt --en-share 0.8`. The English
sources' epochs drop by a fifth (the instruct / reasoning sets ~4 -> ~3.2); no new tokenization or decontamination.

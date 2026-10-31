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

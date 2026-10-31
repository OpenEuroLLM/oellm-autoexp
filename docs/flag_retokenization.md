# Delimiter re-tokenization of the flag mix (tokenizer-256k-agent)

Goal (user, 2026-10-29): chat / agent delimiters such as `<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`
are not tokens of our tokenizer; the data holds them as 3-11 ordinary pieces. Extend the tokenizer and
re-tokenize the mix, on top of the decontaminated `flag-decon-v2`; never write `flag/`.

Location (user, 2026-10-29): new data trees live under **`/gpfs/projects/ehpc533/users/korbip/data/`**
(`flag-decon-v2`, `flag-agenttok`, `flag-decon-v2-agenttok`); `flag-decon-v1` stays in
`/gpfs/projects/ehpc533/training-data-collection/`, next to the original `flag/`.

## Tokenizer

`openeurollm/tokenizer-256k` (SentencePiece BPE, 262,144 pieces) reserves ids 0..112 as special pieces:
`<unk> <bos> <eos>`, `<start_of_turn> <end_of_turn>`, three image tokens, `<fim_prefix> <fim_middle>
<fim_suffix>`, `<tool_call> </tool_call>` (ids 0..12) and `<unused_0..99>` (ids 13..112, `USER_DEFINED`:
SentencePiece matches such a piece in raw text as one token). `openeurollm/tokenizer-256k-reason` renames
ids 3 / 4 (ChatML) and 13-18. **tokenizer-256k-agent** (`scripts/flag_retok/make_tokenizer.py`,
`delimiters.py`; `/gpfs/projects/ehpc533/users/korbip/data/tokenizers/tokenizer-256k-agent`) renames free slots only:

- ids 0..12 unchanged (the data may hold them: agenttrove uses `<tool_call>` / `</tool_call>` already);
- 13..18 as tokenizer-256k-reason: `<tools> </tools> <tool_response> </tool_response> </think> <think>`;
- 19..71: ChatML (`<|im_start|> <|im_end|> <|endoftext|>`), Qwen-Coder FIM / repo (6), Llama 3 (7),
  gpt-oss Harmony (7), Phi / GLM / Kimi (8), DeepSeek (9, full-width bars), Mistral (11), `<|think|>`,
  `<|/think|>`; 41 slots left free.

Vocabulary size, ordinary pieces and merges are unchanged: text without delimiters encodes identically
(checked, SentencePiece and the HF fast tokenizer alike). Delimiter costs before: `<|im_start|>` 7 tokens,
`<think>` 3, Llama 3 headers 9, DeepSeek 9-11.

## Method

Source text is not on MN5 (only `megatron-lm/` shards), and decoding + re-encoding everything would be
needless churn. Probe (10,000 sequences of agenttrove, dclm, stack-v2, megamath, finepdfs): `encode(decode(seq))
== seq` for all, so re-encoding a sequence with the agent tokenizer is exactly what tokenizing its text
with it would give. `scripts/flag_retok/retok.py`:

1. `find`: decode every sequence of the mix (base model, ~17M tok/s per process), find the delimiters;
   `$WORK/find/<shard>.part<i>.npz` = ids of the sequences that hold one + counts per delimiter.
2. `write` -> **flag-agenttok/**: per shard with hits a new `.bin` / `.idx` in which those sequences are
   `encode_agent(decode_base(seq))` (trailing `<bos>` / `<eos>` kept; sequences with ids outside the
   vocabulary left as they are) and all others are copied byte for byte; document boundaries and order
   unchanged; `info.json` with sizes / md5 and a `retokenization` entry. Shards without hits: symlinks.
3. `decon` -> **flag-decon-v2-agenttok/**: flag-decon-v2's dropped documents (`.removed.npz`, same
   document ids) applied to flag-agenttok. The decontamination decisions carry over: the same documents,
   only the delimiter spans shorter.
4. `verify`: per shard, document / sequence counts and doc_idx equal; untouched sequences token-identical;
   re-encoded ones decode to the same text; `.bin` size = 4 x tokens.

Test: `scripts/flag_retok/test_retok.py` (tiny SentencePiece model with the same reserved layout,
find -> write -> decon -> verify). Runs: `config/experiments/korbi/decontam/flag_retok_mn5.yaml`
(`scripts/flag_retok/run_step.sh <step>`), work `/gpfs/scratch/ehpc533/users/korbip/flag_retok`.

## Results (2026-10-29)

`find` (16 nodes of pool p129, 13.5T tokens, ~410M tok/s per node, 38 min): 51.3M sequences hold a
delimiter, ~190M occurrences (`</think>` 48.2M, `<think>` 45.7M, `<|im_start|>` 42.0M, `<|im_end|>` 42.0M,
the three `<|fim_*|>` 7.5M each, `<|fim_pad|>` 0.8M, `<|endoftext|>` 68k, `[INST]` 13k, ...). By source:
mixture-vitae 40.7M sequences (12.3 %), dolmino 7.9M (13.4 %, FIM), agenttrove 1.57M (100 %),
nemotron-pretraining-specialized 1.17M (1.9 %); 18 more sources only mention delimiters as text
(~25k sequences: code, docs, PDFs), but in 246 shards (11.6T tokens).

**Scope (user: option B):** only the four sources above (`scripts/flag_retok/retok_sources.txt`); the
literal mentions elsewhere stay text -- re-tokenizing them would mean copying 49 TB for 0.0003 % of
their sequences (one changed sequence rewrites a whole .bin).

`write` (re-encode, 8 nodes; the first single-node try was CPU-bound and was resumed from its part
files) + `assemble` (8 nodes): **14 shards rewritten, 0.79B tokens fewer** (agenttrove 20.22B ->
19.88B, -1.7 %; mixture-vitae shard 3 -284M; dolmino stack_edu-fim -129M); 6 of ~800k sampled
re-encoded sequences were not `encode(decode(seq))` under the base tokenizer (their text is unchanged;
only the split is the standard one now). `decon`: flag-decon-v2's dropped documents applied (same
document and sequence counts as flag-decon-v2 per shard). `verify` (4 nodes): 452 shards, 92,889
sequences checked, 0 mismatches.

Trees (under `/gpfs/projects/ehpc533/users/korbip/data/`): `flag-agenttok` (full mix, 14 new shards,
symlinks for the rest), **`flag-decon-v2-agenttok`** (training: `data_flag_decon_v2_agenttok_mn5`,
datamix `$HYBRID_ENV/data/flag_decon_v2_agenttok_datamix_mn5.txt`, same weights as v2), tokenizer
`tokenizers/tokenizer-256k-agent`. Lessons: write the `find` results as one file per node, not per
shard (16 nodes creating ~6k files in one GPFS directory took 6 min); assembly must check only the
node's own shards and write info.json before deleting the parts.

### openthoughts-3 and the -pure removals ("double decontamination", user 2026-10-29)

- **openthoughts-3 added** (not in flag/datamix.txt; `scripts/flag_retok/extra_openthoughts.txt`, weight
  **0.0054** = agenttrove's: ~0.54 % of training, ~3.6 epochs, as the collection's usage 4.0 for the
  synthetic reasoning / math / agent sets). Our v2 rule on its v6 hits (`finalize.py --datamix`, same
  exclusions): 16,410 documents dropped -> `flag-decon-v2/openthoughts-3` (+ datamix line); re-tokenized
  (`retok_sources.txt` + `--extra-rels`).
- **The OpenEuroLLM -pure removals on top** (`retok.py decon --extra-drop`, their decision from
  compare_pure's exact document match; their reason-tokenized bins are not used -- tokenizer-256k-reason
  puts ChatML at ids 3 / 4, ours at 19 / 20), recorded as `external_decontamination` in info.json and
  `<prefix>.removed_external.npy`:

| | documents | v2 drops | they drop | only they drop | kept |
|---|---:|---:|---:|---:|---:|
| agenttrove-0.0 | 1,568,412 | 10,148 | 4,192 | 3,604 | 1,554,660 |
| openthoughts-3 | 1,200,000 | 16,410 | 5,013 | 4,645 | 1,178,945 |

- Verify: 453 shards, 92,889 sequences, 0 mismatches. Training datamix files regenerated (453 shards):
  `$HYBRID_ENV/data/flag_decon_v2_datamix_mn5.txt` (old tokenizer, v2 only) and
  `flag_decon_v2_agenttok_datamix_mn5.txt` (**the final clean dataset**: v2 + openthoughts-3 + -pure
  removals + agent tokens; config `data_flag_decon_v2_agenttok_mn5`).

### openthoughts-3 is inside mixture-vitae (2026-10-31)

Feedback: openthoughts-3 is part of mixture-vitae. Checked with `scripts/flag_decontam/doc_overlap.py`
(5,000 random openthoughts-3 documents as references, the decontamination matcher -- word level,
formatting-independent -- over the 5 mixture-vitae shards; work `flag_decon_ovl_ot_mv`): **92.1 %** of
the documents are >= 70 % contained in one verbatim run (89.4 % >= 90 %), 98.3 % share some run; 1.7 %
none. mixture-vitae holds them reformatted (`Q:`, `Teacher`, `### User:` instead of ChatML) -- and it
was re-tokenized here too (4 of 5 shards; its own <think> / ChatML spans).

So the separate openthoughts-3 entry was a second copy. New datamix files without it
(`flag_datamix_mn5.py --exclude openthoughts-3`): `$HYBRID_ENV/data/flag_decon_v2_agenttok_without_ot_datamix_mn5.txt`
and `flag_decon_v2_without_ot_datamix_mn5.txt` (452 shards); the configs `data_flag_decon_v2_agenttok_mn5` /
`data_flag_decon_v2_mn5` point at them. The files WITH openthoughts-3 stay unchanged: the running
ambitious_v2_128n (kept as a stability check, user) and runs enqueued before read them. The
openthoughts-3 shard stays in the trees (unused by the new datamix). The new blend needs its own
dataset index (prebuild configs `data_cache_prebuild_*_mn5`).

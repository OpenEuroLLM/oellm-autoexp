# Delimiter re-tokenization of the flag mix (tokenizer-256k-agent)

Goal (user, 2026-10-29): chat / agent delimiters such as `<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`
are not tokens of our tokenizer; the data holds them as 3-11 ordinary pieces. Extend the tokenizer and
re-tokenize the mix, on top of the decontaminated `flag-decon-v2`; never write `flag/`.

## Tokenizer

`openeurollm/tokenizer-256k` (SentencePiece BPE, 262,144 pieces) reserves ids 0..112 as special pieces:
`<unk> <bos> <eos>`, `<start_of_turn> <end_of_turn>`, three image tokens, `<fim_prefix> <fim_middle>
<fim_suffix>`, `<tool_call> </tool_call>` (ids 0..12) and `<unused_0..99>` (ids 13..112, `USER_DEFINED`:
SentencePiece matches such a piece in raw text as one token). `openeurollm/tokenizer-256k-reason` renames
ids 3 / 4 (ChatML) and 13-18. **tokenizer-256k-agent** (`scripts/flag_retok/make_tokenizer.py`,
`delimiters.py`; `/gpfs/projects/ehpc533/tokenizers/tokenizer-256k-agent`) renames free slots only:

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

## Results

(filled in as the steps finish)

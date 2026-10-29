"""Chat / agent / FIM delimiter tokens of tokenizer-256k-agent (docs/flag_retokenization.md).

The base tokenizer (openeurollm/tokenizer-256k) reserves ids 0..112 as special pieces: <unk>, <bos>, <eos>,
the Gemma turn tokens, image tokens, <fim_*>, <tool_call>, </tool_call> (ids 0..12, all kept: the data may
hold them already) and <unused_0..99> (ids 13..112, USER_DEFINED: SentencePiece matches a renamed slot in
raw text as one piece). The agent tokenizer renames free slots; the vocabulary size (262,144), every
ordinary piece and every merge stay as they are. Ids 13..18 follow openeurollm/tokenizer-256k-reason.
"""

FIRST_FREE, LAST_FREE = 13, 112          # <unused_0> .. <unused_99>

AGENT_TOKENS = [
    # openeurollm/tokenizer-256k-reason (same ids)
    (13, "<tools>"), (14, "</tools>"), (15, "<tool_response>"), (16, "</tool_response>"), (17, "</think>"),
    (18, "<think>"),
    # ChatML / Qwen
    (19, "<|im_start|>"), (20, "<|im_end|>"), (21, "<|endoftext|>"),
    # Qwen-Coder FIM / repository context
    (22, "<|fim_prefix|>"), (23, "<|fim_middle|>"), (24, "<|fim_suffix|>"), (25, "<|fim_pad|>"),
    (26, "<|repo_name|>"), (27, "<|file_sep|>"),
    # Llama 3
    (28, "<|begin_of_text|>"), (29, "<|end_of_text|>"), (30, "<|start_header_id|>"), (31, "<|end_header_id|>"),
    (32, "<|eot_id|>"), (33, "<|eom_id|>"), (34, "<|python_tag|>"),
    # gpt-oss (Harmony)
    (35, "<|start|>"), (36, "<|end|>"), (37, "<|message|>"), (38, "<|channel|>"), (39, "<|constrain|>"),
    (40, "<|return|>"), (41, "<|call|>"),
    # Phi / GLM / Kimi
    (42, "<|system|>"), (43, "<|user|>"), (44, "<|assistant|>"), (45, "<|observation|>"), (46, "<|im_system|>"),
    (47, "<|im_user|>"), (48, "<|im_assistant|>"), (49, "<|im_middle|>"),
    # DeepSeek (full-width bars, U+2581 inside)
    (50, "<｜begin▁of▁sentence｜>"), (51, "<｜end▁of▁sentence｜>"), (52, "<｜User｜>"), (53, "<｜Assistant｜>"),
    (54, "<｜tool▁calls▁begin｜>"), (55, "<｜tool▁calls▁end｜>"), (56, "<｜tool▁call▁begin｜>"),
    (57, "<｜tool▁call▁end｜>"), (58, "<｜tool▁sep｜>"),
    # Mistral
    (59, "[INST]"), (60, "[/INST]"), (61, "[TOOL_CALLS]"), (62, "[AVAILABLE_TOOLS]"), (63, "[/AVAILABLE_TOOLS]"),
    (64, "[TOOL_RESULTS]"), (65, "[/TOOL_RESULTS]"), (66, "[THINK]"), (67, "[/THINK]"), (68, "[SYSTEM_PROMPT]"),
    (69, "[/SYSTEM_PROMPT]"),
    # pipe forms of think
    (70, "<|think|>"), (71, "<|/think|>"),
]

KEPT_BASE = [(3, "<start_of_turn>"), (4, "<end_of_turn>"), (8, "<fim_prefix>"), (9, "<fim_middle>"),
             (10, "<fim_suffix>"), (11, "<tool_call>"), (12, "</tool_call>")]


def check():
    ids = [i for i, _ in AGENT_TOKENS]
    assert len(set(ids)) == len(ids) and all(FIRST_FREE <= i <= LAST_FREE for i in ids)
    names = [t for _, t in AGENT_TOKENS] + [t for _, t in KEPT_BASE]
    assert len(set(names)) == len(names)
    return dict(AGENT_TOKENS)


if __name__ == "__main__":
    m = check()
    print(f"{len(m)} agent tokens in ids {min(m)}..{max(m)}; free slots left: {LAST_FREE - max(m)}")

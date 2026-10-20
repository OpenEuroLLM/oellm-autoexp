"""Engram's tokenizer compression (Cheng et al. 2026, DeepSeek's engram_demo_v1.py
CompressedTokenizer) as a token map for Megatron's ``engram_token_map``:

    uv run --no-project --with transformers --with sentencepiece --with tokenizers \
        --with numpy --with protobuf python scripts/korbi/campaign/engram_token_map.py \
        <tokenizer dir> <out.npy>

Every token id is decoded alone and normalized (NFKC, NFD, strip accents, lowercase, runs of
whitespace -> one space, strip; a token that is only whitespace stays one space); tokens with
the same normalized text share a canonical id. Tokens that do not decode alone (byte-fallback
pieces, "�") keep their own raw piece. The map is int32, the tokenizer size rounded up to a
multiple of 128 (ids past the tokenizer -- the embedding's padding -- get distinct ids after the
canonical ones). Prints the compression ratio (OELLM 256k: 262,145 -> 175,231 ids, 66.8 %).
"""

import sys

import numpy as np
from tokenizers import Regex, normalizers
from transformers import AutoTokenizer

SENTINEL = ""
NORMALIZER = normalizers.Sequence([
    normalizers.NFKC(),
    normalizers.NFD(),
    normalizers.StripAccents(),
    normalizers.Lowercase(),
    normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
    normalizers.Replace(Regex(r"^ $"), SENTINEL),
    normalizers.Strip(),
    normalizers.Replace(SENTINEL, " "),
])


def build(tokenizer) -> np.ndarray:
    key_to_new: dict[str, int] = {}
    n = len(tokenizer)
    old_to_new = np.empty(n, dtype=np.int64)
    for tid in range(n):
        text = tokenizer.decode([tid], skip_special_tokens=False)
        if "�" in text:
            key = tokenizer.convert_ids_to_tokens(tid)
        else:
            norm = NORMALIZER.normalize_str(text)
            key = norm if norm else text
        old_to_new[tid] = key_to_new.setdefault(key, len(key_to_new))
    padded = -(-n // 128) * 128
    extra = np.arange(len(key_to_new), len(key_to_new) + padded - n, dtype=np.int64)
    return np.concatenate([old_to_new, extra]).astype(np.int32), len(key_to_new)


def main():
    tok_dir, out = sys.argv[1], sys.argv[2]
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    table, unique = build(tokenizer)
    np.save(out, table)
    print(f"{len(tokenizer)} tokens -> {unique} canonical ids "
          f"({unique / len(tokenizer):.1%}); map of {len(table)} written to {out}")
    for text in ("The Great", " the great", "THE", "Über", "uber"):
        ids = tokenizer(text, add_special_tokens=False).input_ids
        print(f"  {text!r}: {ids} -> {table[ids].tolist()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Emit a FAST (tokenizer.json) copy of the OELLM 256k SentencePiece tokenizer.

The flagship tokenizer at
``/e/data1/datasets/playground/mmlaion/oellm/oellm_tokenizer_256k`` is a slow
``LlamaTokenizer``: a 4.8 MB SentencePiece ``tokenizer.model`` with no
``tokenizer.json``. Megatron loads it fine, but the Megatron-Bridge container's newer
transformers cannot: converting slow -> fast it routes a SentencePiece model through the
TIKTOKEN converter and dies with

    ValueError: Error parsing line b'\\x0e' in .../oellm_tokenizer_256k/tokenizer.model

which killed jobs 1735753/54/55 at the bridge's tokenizer probe, before any weight was
touched. Handing the bridge a directory that already contains ``tokenizer.json`` skips the
conversion entirely.

Run this in the TRAINING container, whose transformers can still do the conversion::

    apptainer exec /e/project1/e-sta-openeurollm/container/MegatronTraining-JUPITER-te218-fa3_aarch64_202608280932.sif \\
        /opt/venv/bin/python scripts/korbi/make_fast_tokenizer.py \\
        /e/data1/datasets/playground/mmlaion/oellm/oellm_tokenizer_256k \\
        /e/project1/e-sta-openeurollm/poeppel1/tokenizers/oellm_tokenizer_256k_fast

Then verify in the BRIDGE container that the result loads, which is the whole point::

    apptainer exec <bridge.sif> python -c "from transformers import AutoTokenizer; \\
        t=AutoTokenizer.from_pretrained('<dst>'); print(t.vocab_size, len(t))"

Expect ``262144 262145`` -- 256k padded to 2**18, plus one added token. A mismatch there
means the conversion changed the vocabulary and the export would be wrong.
"""

import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", help="slow tokenizer directory (SentencePiece)")
    ap.add_argument("dst", help="directory to write the fast tokenizer into")
    ap.add_argument(
        "--expect-vocab",
        type=int,
        default=262144,
        help="fail if vocab_size differs (0 disables the check)",
    )
    args = ap.parse_args()

    from transformers import AutoTokenizer

    slow = AutoTokenizer.from_pretrained(args.src, use_fast=False, trust_remote_code=True)
    fast = AutoTokenizer.from_pretrained(args.src, use_fast=True, trust_remote_code=True)

    if not getattr(fast, "is_fast", False):
        print("FATAL: transformers returned a slow tokenizer; no tokenizer.json to emit.")
        return 1

    # The conversion must not change the vocabulary, or every exported model is wrong.
    if (slow.vocab_size, len(slow)) != (fast.vocab_size, len(fast)):
        print(
            f"FATAL: vocabulary changed in conversion: "
            f"slow {slow.vocab_size}/{len(slow)} vs fast {fast.vocab_size}/{len(fast)}"
        )
        return 1
    if args.expect_vocab and fast.vocab_size != args.expect_vocab:
        print(f"FATAL: vocab_size {fast.vocab_size}, expected {args.expect_vocab}")
        return 1

    fast.save_pretrained(args.dst)
    print(f"[fast-tokenizer] {args.dst}")
    print(f"[fast-tokenizer]   class {type(fast).__name__}, is_fast=True")
    print(f"[fast-tokenizer]   vocab_size {fast.vocab_size}, len {len(fast)} (unchanged)")
    print(f"[fast-tokenizer]   files: {sorted(os.listdir(args.dst))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

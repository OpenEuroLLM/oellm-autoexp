"""Patch a HuggingFace export to use a custom tokenizer.

Replaces the tokenizer files written by ``convert_checkpoints.py`` and
updates ``config.json`` so ``vocab_size`` and special-token IDs match
the custom tokenizer.

Ported from ``OpenEuroLLM/Megatron-Bridge-utils::export_custom_tokenizer_standalone.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def patch_config_and_tokenizer(hf_path: Path, tokenizer_path: str) -> None:
    from transformers import AutoTokenizer

    print(f"Loading custom tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    print(f"  vocab size  : {len(tokenizer)}")
    print(f"  type        : {type(tokenizer).__name__}")
    print(f"  bos_token_id: {tokenizer.bos_token_id}")
    print(f"  eos_token_id: {tokenizer.eos_token_id}")
    print(f"  pad_token_id: {tokenizer.pad_token_id}")

    print(f"Saving tokenizer into: {hf_path}")
    tokenizer.save_pretrained(str(hf_path))

    config_file = hf_path / "config.json"
    if not config_file.exists():
        print(f"WARNING: {config_file} not found — skipping config patch")
        return

    config = json.loads(config_file.read_text())
    changed: list[str] = []

    # NOTE: intentionally do NOT touch `vocab_size`. The conversion stage
    # (write_hf_config_dir → Bridge.save_hf_pretrained) sets it to the
    # Megatron-padded value (rounded up to make_vocab_size_divisible_by) so
    # the embed-table shape matches the trained checkpoint. Rewriting it to
    # `len(tokenizer)` would shrink the recorded vocab below the actual
    # embedding rows and break loading with `ignore_mismatched_sizes=False`.
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
        tok_val = getattr(tokenizer, attr, None)
        if tok_val is not None and config.get(attr) != tok_val:
            changed.append(f"  {attr}: {config.get(attr)} -> {tok_val}")
            config[attr] = tok_val

    if changed:
        print("Patching config.json:")
        for line in changed:
            print(line)
        config_file.write_text(json.dumps(config, indent=2) + "\n")
    else:
        print("config.json already matches tokenizer; nothing to patch")


def _parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-path", required=True, type=Path)
    ap.add_argument("--tokenizer-path", required=True)
    return ap.parse_args()


def main() -> int:
    args = _parse()
    if not args.hf_path.exists():
        print(f"ERROR: hf-path does not exist: {args.hf_path}", file=sys.stderr)
        return 1
    patch_config_and_tokenizer(hf_path=args.hf_path, tokenizer_path=args.tokenizer_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

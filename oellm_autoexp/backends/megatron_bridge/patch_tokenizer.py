"""Patch a HuggingFace export to use a custom tokenizer.

Replaces the tokenizer files written by ``convert_checkpoints.py`` and
updates ``config.json`` so ``vocab_size`` and special-token IDs match
the custom tokenizer.

Ported from ``OpenEuroLLM/Megatron-Bridge-utils::export_custom_tokenizer_standalone.py``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# Tokenizer files copied verbatim from a local tokenizer dir. Copying (instead
# of AutoTokenizer.save_pretrained) preserves metadata that re-serialization can
# silently drop under newer ``transformers`` — notably ``add_bos_token`` /
# ``add_eos_token``, ``additional_special_tokens``, ``added_tokens_decoder`` and
# chat templates — which downstream eval harnesses (lm_eval/lighteval) rely on.
_CANONICAL_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "chat_template.json",
)


def _reconcile_tokenizer_with_embedding(
    hf_path: Path, vocab_size: int, pad_token: str
) -> list[str]:
    """Drop tokenizer entries the embedding cannot represent, and repoint
    ``pad_token``.

    A tokenizer can define more ids than the trained embedding has rows: the
    OpenEuroLLM 256k tokenizer carries ``<pad>`` at 262144 while the 32B runs pin
    ``padded_vocab_size: 262144`` (rows 0..262143), so ``<pad>`` has no row and
    tokenizing a padded batch emits an id the model cannot embed. Training never
    hit it because documents are packed and ``<eos>``-separated.

    Publishing that unchanged hands the problem to anyone fine-tuning the export.
    So any added token with ``id >= vocab_size`` is removed, and if ``pad_token``
    was one of them it is repointed at ``pad_token`` (default ``<unused_0>``, a
    real allocated row that never occurs in training data). ``<eos>`` would also
    work, but the usual SFT recipe masks labels wherever ``input_ids ==
    pad_token_id``, which would then also mask every genuine end-of-sequence
    token and teach the model never to stop.

    No-op when the tokenizer already fits (e.g. the 9B, padded to 262272).
    """
    notes: list[str] = []

    def _load(name: str):
        f = hf_path / name
        return (json.loads(f.read_text()), f) if f.exists() else (None, f)

    added, added_f = _load("added_tokens.json")
    over = sorted(t for t, i in (added or {}).items() if i >= vocab_size)
    if not over:
        return notes
    notes.append(f"dropped {over} (id >= vocab_size {vocab_size})")

    added = {t: i for t, i in added.items() if i < vocab_size}
    added_f.write_text(json.dumps(added, indent=2) + "\n")

    cfg, cfg_f = _load("tokenizer_config.json")
    if cfg is not None:
        decoder = cfg.get("added_tokens_decoder") or {}
        cfg["added_tokens_decoder"] = {k: v for k, v in decoder.items() if int(k) < vocab_size}
        if cfg.get("pad_token") in over:
            cfg["pad_token"] = pad_token
            notes.append(f"pad_token -> {pad_token}")
        cfg_f.write_text(json.dumps(cfg, indent=2) + "\n")

    stm, stm_f = _load("special_tokens_map.json")
    if stm is not None and stm.get("pad_token") in over:
        stm["pad_token"] = pad_token
        stm_f.write_text(json.dumps(stm, indent=2) + "\n")

    tj, tj_f = _load("tokenizer.json")
    if tj is not None and isinstance(tj.get("added_tokens"), list):
        kept = [a for a in tj["added_tokens"] if a.get("id", 0) < vocab_size]
        if len(kept) != len(tj["added_tokens"]):
            tj["added_tokens"] = kept
            tj_f.write_text(json.dumps(tj, ensure_ascii=False) + "\n")

    return notes


def patch_config_and_tokenizer(
    hf_path: Path, tokenizer_path: str, pad_token: str = "<unused_0>"
) -> None:
    from transformers import AutoTokenizer

    print(f"Loading custom tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    print(f"  vocab size  : {len(tokenizer)}")
    print(f"  type        : {type(tokenizer).__name__}")
    print(f"  bos_token_id: {tokenizer.bos_token_id}")
    print(f"  eos_token_id: {tokenizer.eos_token_id}")
    print(f"  pad_token_id: {tokenizer.pad_token_id}")

    src = Path(tokenizer_path)
    if src.is_dir():
        # Local tokenizer dir: copy files verbatim so ALL metadata survives.
        copied = []
        for fname in _CANONICAL_TOKENIZER_FILES:
            f = src / fname
            if f.exists():
                shutil.copy2(f, hf_path / fname)
                copied.append(fname)
        print(f"Copied tokenizer files verbatim into {hf_path}: {copied}")
    else:
        # Remote HF Hub id (no local files): fall back to re-serialization.
        print(f"Saving tokenizer (HF id) into: {hf_path}")
        tokenizer.save_pretrained(str(hf_path))

    config_file = hf_path / "config.json"
    if not config_file.exists():
        print(f"WARNING: {config_file} not found — skipping config patch")
        return

    config = json.loads(config_file.read_text())
    changed: list[str] = []

    # The embedding, not the source tokenizer, is the authority on how many ids
    # exist. Reconcile before reading special-token ids back.
    vocab_size = config.get("vocab_size")
    if isinstance(vocab_size, int):
        for note in _reconcile_tokenizer_with_embedding(hf_path, vocab_size, pad_token):
            print(f"  tokenizer: {note}")
        tokenizer = AutoTokenizer.from_pretrained(str(hf_path), trust_remote_code=True)
        print(f"  reconciled vocab: len(tokenizer)={len(tokenizer)} vs vocab_size={vocab_size}")

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

    # Normalise RoPE base to a version-agnostic top-level `rope_theta`.
    # transformers >= 5 records the base under `rope_parameters` and the
    # top-level `rope_theta` is left unset — so the Qwen3Config class default
    # (10000) fills in. transformers 4.x Qwen3 reads that top-level `rope_theta`
    # and ignores `rope_parameters`, silently using 10000 even when the model was
    # trained with a different base (e.g. 100000). Mirror the real value from
    # `rope_parameters` up to the top level so every transformers version agrees.
    rope_params = config.get("rope_parameters")
    if isinstance(rope_params, dict) and rope_params.get("rope_theta") is not None:
        real_theta = rope_params["rope_theta"]
        if config.get("rope_theta") != real_theta:
            changed.append(f"  rope_theta: {config.get('rope_theta')} -> {real_theta}")
            config["rope_theta"] = real_theta

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
    ap.add_argument(
        "--pad-token",
        default="<unused_0>",
        help="Token to use as pad when the tokenizer's own pad id exceeds the embedding",
    )
    return ap.parse_args()


def main() -> int:
    args = _parse()
    if not args.hf_path.exists():
        print(f"ERROR: hf-path does not exist: {args.hf_path}", file=sys.stderr)
        return 1
    patch_config_and_tokenizer(
        hf_path=args.hf_path, tokenizer_path=args.tokenizer_path, pad_token=args.pad_token
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

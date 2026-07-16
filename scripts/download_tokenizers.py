#!/usr/bin/env python3
"""Populate the megatron_bridge tokenizers/ tree with heavy files from the HF
Hub.

The resources/megatron_bridge/tokenizers/ tree ships with small metadata
(``tokenizer_config.json`` etc.) but skips ``tokenizer.json``, ``tokenizer.model``,
``merges.txt``, and ``vocab.json`` (see the .gitignore in that directory). This
script fetches those files from the HF Hub for every tokenizer the bridge backend
knows about. Run it after cloning the repo, before invoking ``MegatronBridgeBackend``.

By default it downloads every tokenizer whose directory exists under the
resources tree. Use ``--tokenizers`` to limit the set.

Examples
--------
    # Download all tracked tokenizers (skips any already complete).
    python scripts/download_tokenizers.py

    # Just one.
    python scripts/download_tokenizers.py --tokenizers Qwen/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

LOGGER = logging.getLogger(Path(__file__).stem)

# Heavy files we skipped in git. If a tokenizer dir is missing any of these AND
# the file exists on the Hub, fetch it. Some tokenizers don't have all four
# (e.g. SentencePiece-only ones), so 404s for individual files are not fatal.
HEAVY_FILES = ("tokenizer.json", "tokenizer.model", "merges.txt", "vocab.json")

DEFAULT_RESOURCES = (
    Path(__file__).resolve().parent.parent
    / "oellm_autoexp"
    / "postprocess"
    / "resources"
    / "megatron_bridge"
)


def _iter_tokenizer_dirs(tokenizers_root: Path) -> list[tuple[str, Path]]:
    """Yield (hf_repo_id, local_dir) for every real tokenizer dir under the
    resources.

    Symlinks (e.g. ``openeurollm/Qwen3-0.1B-ne -> tokenizer-256k``) are skipped — we
    download to the symlink target only.
    """
    out: list[tuple[str, Path]] = []
    for path in sorted(tokenizers_root.rglob("*")):
        if not path.is_dir() or path.is_symlink():
            continue
        # Only leaf dirs (no subdirs) qualify as a tokenizer
        if any(p.is_dir() for p in path.iterdir()):
            continue
        # repo id is the path relative to tokenizers_root
        repo_id = str(path.relative_to(tokenizers_root))
        out.append((repo_id, path))
    return out


def _download_one(repo_id: str, local_dir: Path, force: bool) -> tuple[int, int]:
    """Download missing heavy files from the Hub into local_dir.

    Returns (downloaded, skipped) counts.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    downloaded = 0
    skipped = 0
    for fname in HEAVY_FILES:
        target = local_dir / fname
        if target.exists() and not force:
            skipped += 1
            continue
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=fname,
                local_dir=str(local_dir),
            )
            downloaded += 1
            LOGGER.info("  %s: %s", repo_id, fname)
        except EntryNotFoundError:
            # File doesn't exist for this tokenizer — fine
            LOGGER.debug("  %s: no %s on the Hub", repo_id, fname)
        except RepositoryNotFoundError:
            LOGGER.warning("  %s: repo not found on the Hub; skipping", repo_id)
            return 0, len(HEAVY_FILES)
    return downloaded, skipped


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--resources",
        type=Path,
        default=DEFAULT_RESOURCES,
        help="Root of the bridge resources tree (default: %(default)s)",
    )
    parser.add_argument(
        "--tokenizers",
        nargs="+",
        default=None,
        help="Limit to these HF repo ids (default: every tokenizer dir found)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download heavy files even if present",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    tokenizers_root = args.resources / "tokenizers"
    if not tokenizers_root.exists():
        print(f"ERROR: tokenizers root not found: {tokenizers_root}", file=sys.stderr)
        return 1

    all_dirs = _iter_tokenizer_dirs(tokenizers_root)
    if args.tokenizers:
        wanted = set(args.tokenizers)
        all_dirs = [(rid, p) for rid, p in all_dirs if rid in wanted]
        unknown = wanted - {rid for rid, _ in all_dirs}
        if unknown:
            print(f"ERROR: unknown tokenizers: {sorted(unknown)}", file=sys.stderr)
            return 1

    LOGGER.info("Downloading heavy files for %d tokenizer(s)", len(all_dirs))
    total_dl = 0
    total_skip = 0
    for repo_id, local_dir in all_dirs:
        LOGGER.info("%s -> %s", repo_id, local_dir.relative_to(args.resources))
        dl, sk = _download_one(repo_id, local_dir, args.force)
        total_dl += dl
        total_skip += sk
    LOGGER.info("Done: %d files downloaded, %d already present", total_dl, total_skip)
    return 0


if __name__ == "__main__":
    sys.exit(main())

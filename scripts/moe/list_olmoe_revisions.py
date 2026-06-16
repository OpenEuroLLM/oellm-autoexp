#!/usr/bin/env python3
"""List allenai/OLMoE-1B-7B-0924 branches sorted by training step and print
every-Nth (default 10th) so the driver bash can pass each as --revision.

Usage:
  python scripts/moe/list_olmoe_revisions.py            # every 10th
  python scripts/moe/list_olmoe_revisions.py --every 5  # every 5th
  python scripts/moe/list_olmoe_revisions.py --all      # all of them
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

STEP_RE = re.compile(r"step(\d+)-tokens(\d+)([KMB])")


def _parse_step(name: str) -> int | None:
    m = STEP_RE.match(name)
    if not m:
        return None
    return int(m.group(1))


def _list_from_cache(model: str) -> List[Tuple[int, str]]:
    """Read step branches from the local HF cache `refs/` dir (offline-safe)."""
    hf_home = os.environ.get(
        "HF_HOME", "/leonardo_scratch/large/userexternal/ajha0001/HF_CACHE"
    )
    repo_dir = "models--" + model.replace("/", "--")
    refs = Path(hf_home) / "hub" / repo_dir / "refs"
    out: List[Tuple[int, str]] = []
    if not refs.is_dir():
        print(f"cache refs dir not found: {refs}", file=sys.stderr)
        return out
    for ref in refs.iterdir():
        s = _parse_step(ref.name)
        if s is not None:
            out.append((s, ref.name))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    p.add_argument("--every", type=int, default=10)
    p.add_argument("--all", action="store_true")
    p.add_argument("--include-main", action="store_true", default=True)
    p.add_argument("--from-cache", action="store_true",
                   help="List branches from the local HF cache refs/ dir "
                        "(offline-safe) instead of querying the Hub API.")
    args = p.parse_args()

    if args.from_cache:
        step_branches = _list_from_cache(args.model)
    else:
        try:
            from huggingface_hub import HfApi
        except ImportError:
            print("huggingface_hub not installed; pip install huggingface_hub", file=sys.stderr)
            return 1

        api = HfApi()
        refs = api.list_repo_refs(args.model)
        step_branches = []
        for b in refs.branches:
            s = _parse_step(b.name)
            if s is not None:
                step_branches.append((s, b.name))
    step_branches.sort()

    if args.all:
        picked = [n for _, n in step_branches]
    else:
        picked = [n for i, (_, n) in enumerate(step_branches) if i % args.every == 0]
        # Always include the final step
        if step_branches and step_branches[-1][1] not in picked:
            picked.append(step_branches[-1][1])

    if args.include_main:
        picked.append("main")

    for r in picked:
        print(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())

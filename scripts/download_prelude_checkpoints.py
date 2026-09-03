#!/usr/bin/env python3
"""Download an evenly spaced sample of openeurollm/prelude HF branches."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_REPO = "openeurollm/prelude"
DEFAULT_OUTPUT = Path(
    os.environ.get(
        "PRELUDE_CHECKPOINT_DIR",
        "/e/home/jusers/luukkonen1/jupiter/e-sta-workdir/"
        "oellm-autoexp/hf_checkpoints/prelude",
    )
)


@dataclass(frozen=True)
class CheckpointRef:
    iteration: int
    revision: str
    commit: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--samples",
        type=int,
        default=6,
        help="Number of evenly spaced revisions to download (default: 6).",
    )
    parser.add_argument(
        "--iterations",
        help="Comma-separated iterations to download instead of even sampling.",
    )
    parser.add_argument("--min-iteration", type=int)
    parser.add_argument("--max-iteration", type=int)
    parser.add_argument("--save-interval", type=int, default=2400)
    parser.add_argument("--global-batch-size", type=int, default=2048)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument(
        "--no-cadence-filter",
        action="store_true",
        help="Keep numeric branches that are not multiples of --save-interval.",
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def iteration_from_branch(name: str) -> int | None:
    explicit_patterns = (
        r"(?:^|[/_-])(?:iter(?:ation)?|step|global[_-]?step|ckpt|checkpoint)[_-]?(\d+)(?:$|[/_-])",
        r"^(\d+)$",
    )
    for pattern in explicit_patterns:
        match = re.search(pattern, name, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    numeric_parts = re.findall(r"(?:^|[/_-])(\d+)(?=$|[/_-])", name)
    if numeric_parts:
        return int(numeric_parts[-1])
    return None


def discover_checkpoints(api: object, repo_id: str) -> list[CheckpointRef]:
    refs = api.list_repo_refs(repo_id=repo_id, repo_type="model")
    by_iteration: dict[int, CheckpointRef] = {}
    for branch in refs.branches:
        iteration = iteration_from_branch(branch.name)
        if iteration is None:
            continue
        checkpoint = CheckpointRef(
            iteration=iteration,
            revision=branch.name,
            commit=getattr(branch, "target_commit", None),
        )
        previous = by_iteration.setdefault(iteration, checkpoint)
        if previous.revision != checkpoint.revision:
            print(
                f"warning: multiple branches map to iteration {iteration}; "
                f"using {previous.revision!r}, ignoring {checkpoint.revision!r}",
                file=sys.stderr,
            )
    return sorted(by_iteration.values(), key=lambda checkpoint: checkpoint.iteration)


def evenly_spaced(items: list[CheckpointRef], count: int) -> list[CheckpointRef]:
    if count <= 0:
        raise ValueError("--samples must be positive")
    if count >= len(items):
        return items
    if count == 1:
        return [items[-1]]
    indices = {
        round(sample_index * (len(items) - 1) / (count - 1))
        for sample_index in range(count)
    }
    return [items[index] for index in sorted(indices)]


def select_checkpoints(
    checkpoints: list[CheckpointRef], args: argparse.Namespace
) -> list[CheckpointRef]:
    candidates = [
        checkpoint
        for checkpoint in checkpoints
        if (args.min_iteration is None or checkpoint.iteration >= args.min_iteration)
        and (args.max_iteration is None or checkpoint.iteration <= args.max_iteration)
        and (
            args.no_cadence_filter
            or args.save_interval <= 0
            or checkpoint.iteration % args.save_interval == 0
        )
    ]
    if not candidates:
        raise RuntimeError("No numeric checkpoint branches matched the requested filters")

    if args.iterations:
        requested = {int(value.strip()) for value in args.iterations.split(",") if value.strip()}
        available = {checkpoint.iteration: checkpoint for checkpoint in candidates}
        missing = sorted(requested - available.keys())
        if missing:
            raise RuntimeError(f"Requested iterations are unavailable: {missing}")
        return [available[iteration] for iteration in sorted(requested)]

    return evenly_spaced(candidates, args.samples)


def checkpoint_record(checkpoint: CheckpointRef, args: argparse.Namespace) -> dict[str, object]:
    tokens = checkpoint.iteration * args.global_batch_size * args.sequence_length
    return {
        "iteration": checkpoint.iteration,
        "revision": checkpoint.revision,
        "commit": checkpoint.commit,
        "tokens": tokens,
        "tokens_billions": tokens / 1e9,
        "local_dir": str(args.output_dir / f"iter_{checkpoint.iteration:07d}"),
    }


def main() -> None:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as error:
        raise SystemExit(
            "huggingface_hub is required; install it with: pip install huggingface_hub"
        ) from error

    args = parse_args()
    checkpoints = discover_checkpoints(HfApi(), args.repo_id)
    selected = select_checkpoints(checkpoints, args)
    records = [checkpoint_record(checkpoint, args) for checkpoint in selected]

    print(f"repo: {args.repo_id}")
    print(
        f"tokens/iteration: {args.global_batch_size * args.sequence_length:,}; "
        f"tokens/{args.save_interval}-step interval: "
        f"{args.save_interval * args.global_batch_size * args.sequence_length / 1e9:.3f}B"
    )
    for record in records:
        print(
            f"iter {record['iteration']:>7}  {record['tokens_billions']:>9.3f}B tokens  "
            f"revision={record['revision']}"
        )

    if args.dry_run:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "repo_id": args.repo_id,
        "save_interval": args.save_interval,
        "global_batch_size": args.global_batch_size,
        "sequence_length": args.sequence_length,
        "checkpoints": records,
    }
    (args.output_dir / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    for checkpoint, record in zip(selected, records, strict=True):
        local_dir = Path(str(record["local_dir"]))
        print(f"downloading {checkpoint.revision} -> {local_dir}", flush=True)
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="model",
            revision=checkpoint.revision,
            local_dir=local_dir,
            cache_dir=args.cache_dir,
            max_workers=args.max_workers,
        )

    print(f"downloaded {len(selected)} checkpoints under {args.output_dir}")


if __name__ == "__main__":
    main()

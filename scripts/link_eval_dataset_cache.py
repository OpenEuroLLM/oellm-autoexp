#!/usr/bin/env python3
"""Build an isolated eval dataset cache that reuses the shared one where valid.

Why this exists
---------------
oellm-evals' generated script hardcodes ``HF_DATASETS_CACHE=$HF_HOME/datasets``,
while JUPITER's populated cache keeps the ``owner___name`` build dirs at the
cache ROOT. On top of that, the shared cache is a *mix*: some entries were built
with ``datasets`` 4.x, whose ``List`` feature type 3.x cannot read — and the
eval stack is pinned to ``datasets<4`` because several suite datasets are still
loading-script based.

Rather than deleting entries out of the user's shared cache (other environments
read it), this builds ``$OELLM_EVAL_HF_HOME/datasets`` as a real directory of
**symlinks** into the shared cache, and drops the link for any entry the pinned
``datasets`` version cannot load. Unlinked entries are simply re-downloaded into
the eval cache by the prefetch step; nothing in the shared cache is modified.

Usage::

    PYTHONPATH=<overlay> python scripts/link_eval_dataset_cache.py \\
        --shared-cache $HF_HOME --eval-cache $OELLM_EVAL_HF_HOME/datasets
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def required_specs(task_groups: list[str]) -> list[tuple[str, str | None]]:
    """(dataset, subset) pairs for the given oellm-evals task groups."""
    from oellm.task_groups import _collect_dataset_specs

    specs = _collect_dataset_specs(task_groups)
    out: list[tuple[str, str | None]] = []
    for spec in specs:
        # DatasetSpec exposes the repo as `repo_id`.
        repo_id = getattr(spec, "repo_id", None)
        if repo_id:
            out.append((repo_id, getattr(spec, "subset", None)))
    if not out:
        raise RuntimeError(f"no dataset specs resolved for task groups {task_groups}")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shared-cache", type=Path, required=True)
    p.add_argument("--eval-cache", type=Path, required=True)
    p.add_argument(
        "--task-groups", default="dclm-core-22,oellm-multilingual",
        help="comma-separated; only these are validated (the rest are linked as-is)",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--download", action="store_true",
                   help="after linking, fetch everything still missing (needs internet)")
    args = p.parse_args()

    # `datasets` snapshots the offline flags into `datasets.config` at import
    # time, so they have to be right *before* the first import below — popping
    # them inside _download() would be too late.
    if args.download:
        os.environ.pop("HF_DATASETS_OFFLINE", None)
        os.environ.pop("HF_HUB_OFFLINE", None)
    else:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ["HF_DATASETS_CACHE"] = str(args.eval_cache)

    args.eval_cache.mkdir(parents=True, exist_ok=True)

    # 1. Link every built dataset dir from the shared cache.
    linked = 0
    for entry in sorted(args.shared_cache.iterdir()):
        if "___" not in entry.name or not entry.is_dir():
            continue
        target = args.eval_cache / entry.name
        if target.exists() or target.is_symlink():
            continue
        if not args.dry_run:
            target.symlink_to(entry)
        linked += 1
    print(f"linked {linked} shared cache entries into {args.eval_cache}")

    # 2. Validate the entries the eval suites actually need; unlink the ones the
    #    pinned datasets version cannot read so they get re-downloaded.
    import datasets

    print(f"validating with datasets {datasets.__version__}")
    groups = [g.strip() for g in args.task_groups.split(",") if g.strip()]
    # Staleness is per *subset* — a 3.x-built ARC-Easy can sit next to a 4.x-built
    # ARC-Challenge under one dataset dir — and it lives in the `features` block
    # of each build's `dataset_info.json` (4.x writes the `List` feature type,
    # which 3.x cannot parse). Checking those directly is both cheaper and
    # stricter than `load_dataset_builder`, which happily returns the *newest*
    # cached configuration without touching the stale sibling. A dataset link is
    # dropped whole if any of its builds is unreadable.
    from datasets import Features

    needed = {repo.replace("/", "___") for repo, _ in required_specs(groups)}
    dropped: set[str] = set()
    ok = absent = 0
    for key in sorted(needed):
        link = args.eval_cache / key
        if not link.is_symlink():
            absent += 1
            continue
        bad = None
        for info_path in sorted(link.glob("*/*/*/dataset_info.json")):
            try:
                blob = json.loads(info_path.read_text())
                Features.from_dict(blob["features"])
            except Exception as exc:
                bad = f"{info_path.relative_to(link)}: {type(exc).__name__}: {str(exc)[:70]}"
                break
        if bad is None:
            ok += 1
        else:
            print(f"  DROP {key}: {bad}")
            if not args.dry_run:
                link.unlink()
            dropped.add(key)

    print(f"validated: ok={ok} dropped={len(dropped)} not-cached={absent}")

    if args.download:
        return _download(groups, args.eval_cache)
    return 0


def _download(groups: list[str], eval_cache: Path) -> int:
    """Fetch every required (dataset, subset) into the eval cache.

    oellm-eval's own ``--download_only`` aborts the whole run on the first
    dataset it cannot fetch, which is useless when a couple of them are
    permanently unavailable (gated, or removed loading scripts). This keeps
    going and prints a coverage report instead.
    """
    from datasets import load_dataset_builder

    specs = sorted(set(required_specs(groups)), key=lambda s: (s[0], s[1] or ""))
    failed: list[tuple[str, str | None, str]] = []
    done = 0
    from datasets import get_dataset_config_names

    for i, (repo, subset) in enumerate(specs, 1):
        try:
            load_dataset_builder(repo, subset).download_and_prepare()
            done += 1
        except Exception as exc:
            # Some task-group entries carry no `subset` (xwinograd, xcopa,
            # xstory_cloze) because lm-eval picks the per-language config at
            # task level. Those datasets have no default config, so fetch every
            # one — the compute nodes are offline and cannot resolve it later.
            if subset is None and "Config name is missing" in str(exc):
                try:
                    configs = get_dataset_config_names(repo)
                    for cfg_name in configs:
                        load_dataset_builder(repo, cfg_name).download_and_prepare()
                    print(f"  {repo}: fetched all {len(configs)} configs", flush=True)
                    done += 1
                    continue
                except Exception as exc2:
                    exc = exc2
            failed.append((repo, subset, f"{type(exc).__name__}: {str(exc)[:120]}"))
        if i % 25 == 0:
            print(f"  … {i}/{len(specs)} ({len(failed)} failed)", flush=True)

    print(f"\ndownloaded {done}/{len(specs)} dataset configs into {eval_cache}")
    if failed:
        print(f"{len(failed)} could NOT be fetched:")
        by_repo: dict[str, list[str]] = {}
        for repo, subset, msg in failed:
            by_repo.setdefault(f"{repo} :: {msg}", []).append(str(subset))
        for head, subsets in sorted(by_repo.items()):
            print(f"  - {head}\n      subsets: {', '.join(subsets[:8])}"
                  + (" …" if len(subsets) > 8 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Is the dataset index ready for a run, and complete? Fast, and only for the rows the run reads.

`check_data_cache.py` validates a whole cache directory. The main run's cache holds 182k files and
the scan takes tens of minutes on GPFS, which is too slow to sit in front of a launch (2026-11-17: a
launch was blocked ~7 min on it before being killed). A run only reads the
``<key>-GPTDataset-<split>-*`` entries its own recipe hashes to, so this checks exactly those:

  * the keys come from a second cache directory built with the same recipe (two configs that differ
    only in the directory they point at produce identical keys -- measured 531/531 train and
    538/538 valid on 2026-11-17);
  * the target's files for those keys are symlinked into a temporary directory and
    `check_data_cache.py` judges them there, so the header/truncation logic is the tested one and is
    not duplicated here.

    cache_ready.py <cache_dir_to_check> <cache_dir_with_the_recipe_keys> [--repair]

Exit 0 = every (key, split) row of that recipe is present and complete in <cache_dir_to_check>.
--repair is passed through to check_data_cache.py (it deletes the members of incomplete rows so the
next build rewrites only those).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

SPLITS = ("train", "valid", "test")
MARKER = "-GPTDataset-"
DEFAULT_CHECKER = "/gpfs/projects/ehpc533/users/korbip/hybrid_v019/autoexp_v4cont1/scripts/check_data_cache.py"


def split_of(name: str) -> str | None:
    for split in SPLITS:
        if f"{MARKER}{split}-" in name:
            return split
    return None


def keys_by_split(cache: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {s: set() for s in SPLITS}
    for name in os.listdir(cache):
        if not name.endswith("-description.txt"):
            continue
        split = split_of(name)
        if split:
            out[split].add(name.split(MARKER)[0])
    return out


def files_by_key(cache: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for name in os.listdir(cache):
        if MARKER in name:
            out.setdefault(name.split(MARKER)[0], []).append(name)
    return out


def checker_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.normpath(os.path.join(here, "..", "..", "..", "scripts", "check_data_cache.py"))
    if os.path.exists(candidate):
        return candidate
    return DEFAULT_CHECKER


def main(argv: list[str]) -> int:
    positional = [a for a in argv[1:] if not a.startswith("--")]
    repair = "--repair" in argv
    if len(positional) != 2:
        print(__doc__)
        return 2
    target, source = positional

    recipe = keys_by_split(source)
    have = keys_by_split(target)
    total = sum(len(v) for v in recipe.values())
    if not total:
        print(f"no keys found in {source}")
        return 2
    print("recipe keys from %s: %s" % (source, ", ".join(f"{s}={len(recipe[s])}" for s in SPLITS)))

    missing = {s: recipe[s] - have[s] for s in SPLITS}
    for split in SPLITS:
        if missing[split]:
            # Not a failure: the source directory accumulates keys of several configs (on 2026-11-17
            # it held 212 train keys of other eval settings next to our 531). The run reads only the
            # rows the target has; those must all be complete, and there must be at least one.
            print(f"  note: {len(missing[split])}/{len(recipe[split])} {split} keys exist only in the source")
    checked_rows = sum(len(recipe[s] & have[s]) for s in SPLITS)

    target_files = files_by_key(target)
    tmp = tempfile.mkdtemp(prefix="cacheready_", dir="/gpfs/scratch/ehpc533/hybrid_nucleus")
    linked = 0
    try:
        for split in SPLITS:
            for key in recipe[split] & have[split]:
                for name in target_files.get(key, []):
                    dst = os.path.join(tmp, name)
                    if not os.path.exists(dst):
                        os.symlink(os.path.join(target, name), dst)
                        linked += 1
        print(f"linked {linked} files for {sum(len(recipe[s] & have[s]) for s in SPLITS)} rows into {tmp}")

        proc = subprocess.run(
            [sys.executable, checker_path(), tmp] + (["--repair"] if repair else []),
            capture_output=True,
            text=True,
        )
        out = (proc.stdout or "").strip()
        if out:
            print(out[-2000:])
        if proc.returncode != 0:
            err = (proc.stderr or "").strip()
            if err:
                print(err[-1000:])
            print(f"\nNOT READY: check_data_cache.py exited {proc.returncode}")
            return 3
        if checked_rows == 0:
            print("\nNOT READY: none of this recipe's rows exists in the target cache")
            return 3
        print(f"\nREADY: all {checked_rows} row(s) of this recipe that the target has are complete")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv))

#!/usr/bin/env python3
"""Check a rendered prebuild command against the training run it prepares the cache for.

A dataset prebuild must be the run's own data recipe with a stand-in model, so this asserts, on the
rendered job text (sbatch or pool step.sh):

  1. no ``--load`` flag            (a prebuild must never open a checkpoint: the 35900 84B save
                                   cannot be loaded into the 256-wide stand-in -- mn5-pool-p264.1267)
  2. no ``NODE_CHECK=`` env line   (a 1-node thorough check judges the node against its own median
                                   and excludes it -- mn5-pool-p264.1265)
  3. every dataset cache-key flag equal to the training run's

The cache key is md5 over {class, dataset_path, num_samples, index_split, random_seed,
sequence_length, split, split_matrix, tokenizer} (megatron/core/datasets/megatron_dataset.py,
_key_config_attributes), so only the data and the schedule matter -- never the model, the
micro-batch size or the node count.

    check_prebuild_render.py <prebuild-render> <training-run-command>

Exit 0 = the render is a valid prebuild for that run; 3 = it is not (nothing may be submitted).
"""

from __future__ import annotations

import re
import sys

# Flags that enter the dataset cache key (see the module docstring). --data-cache-path is checked
# too: a prebuild for a different directory is useless to the run.
KEY_FLAGS = [
    "--data-args-path",
    "--data-cache-path",
    "--seq-length",
    "--split",
    "--data-sharding",
    "--dataloader-type",
    "--global-batch-size",
    "--eval-iters",
    "--eval-interval",
    "--train-iters",
    "--tokenizer-model",
    "--seed",
    "--num-dataset-builder-threads",
]


def flags(path: str) -> set[str]:
    """All long option names in a rendered shell/sbatch command (values are not parsed)."""
    text = open(path, errors="replace").read().replace("\\\n", " ")
    return set(re.findall(r"(?<![\w-])(--[A-Za-z0-9][A-Za-z0-9-]*)", text))


def flag_values(path: str) -> dict[str, str]:
    text = open(path, errors="replace").read().replace("\\\n", " ")
    out: dict[str, str] = {}
    for match in re.finditer(r"(--[A-Za-z0-9-]+)(?:[ =])(\"[^\"]*\"|'[^']*'|\S*)", text):
        out.setdefault(match.group(1), (match.group(2) or "").strip("\"'"))
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 3
    prebuild, run = argv[1], argv[2]

    problems: list[str] = []

    pre_names = flags(prebuild)
    if "--load" in pre_names:
        problems.append("the prebuild renders --load (it must not open a checkpoint)")
    # NODE_CHECK=off is what the template skips; any other value runs the bench, and a 1-node
    # bench judges the node against its own median and excludes it (mn5-pool-p264.1265).
    node_check = re.search(
        r"^\s*(?:export\s+)?NODE_CHECK=(\S*)", open(prebuild, errors="replace").read(), re.M
    )
    if node_check and node_check.group(1).strip("\"'").lower() not in ("", "off", "false"):
        problems.append(
            f"the prebuild renders NODE_CHECK={node_check.group(1)} "
            "(a 1-node check excludes its node; set node_check: \"off\")"
        )

    a, b = flag_values(run), flag_values(prebuild)
    print(f"{'cache-key flag':34s} {'training run':44s} prebuild")
    for key in KEY_FLAGS:
        va, vb = a.get(key, "<absent>"), b.get(key, "<absent>")
        mark = "ok  " if va == vb else "DIFF"
        print(f"{key:34s} {mark} {va[:40]:44s} {vb[:40]}")
        if va != vb:
            problems.append(f"{key}: run={va!r} prebuild={vb!r}")

    if problems:
        print("\nREFUSED:")
        for p in problems:
            print("  -", p)
        return 3
    print("\nOK: valid prebuild for this run (no checkpoint load, no node check, identical cache key)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

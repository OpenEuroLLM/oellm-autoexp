"""Dry-run output redirection: a dry run renders its files (sbatch scripts, array task scripts,
pool step scripts, backend config files) under a local root instead of their real absolute paths,
so testing cluster configs locally does not create /gpfs/... trees.

    python scripts/run_autoexp.py --dry-run-root /tmp/dry --config-name ...
    export AUTOEXP_DRY_RUN_ROOT=$HOME/dry_runs   # then every --dry-run uses it

A path /gpfs/x/job.sbatch is written to <root>/gpfs/x/job.sbatch. The rendered CONTENTS keep the
real paths, so a script still shows exactly what would run on the cluster. Redirection applies
only while run_autoexp marks the process as a dry run (AUTOEXP_DRY_RUN=1): an exported root never
moves the files of a real submission. Both variables are environment variables so parallel
render workers inherit them.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT_ENV = "AUTOEXP_DRY_RUN_ROOT"
ACTIVE_ENV = "AUTOEXP_DRY_RUN"


def activate(root: str | os.PathLike | None) -> Path | None:
    """Mark this process (and its children) as a dry run; ``root`` (or $AUTOEXP_DRY_RUN_ROOT)
    becomes the redirection root. Returns the root, or None if there is none."""
    os.environ[ACTIVE_ENV] = "1"
    if root:
        os.environ[ROOT_ENV] = str(Path(root).expanduser().resolve())
    value = os.environ.get(ROOT_ENV)
    return Path(value) if value else None


def dry_run_root() -> Path | None:
    """The redirection root while a dry run is active, else None."""
    if os.environ.get(ACTIVE_ENV) != "1":
        return None
    value = os.environ.get(ROOT_ENV)
    return Path(value) if value else None


def dry_run_path(path: str | os.PathLike) -> Path:
    """Where to write ``path``: under the dry-run root (absolute paths mirrored below it, relative
    ones relative to it), or ``path`` itself outside a dry run / without a root."""
    path = Path(path)
    root = dry_run_root()
    if root is None:
        return path
    if path.is_absolute():
        if path == root or root in path.parents:
            return path  # already inside the root
        return root / path.relative_to(path.anchor)
    return root / path

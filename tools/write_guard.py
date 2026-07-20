"""Hard guard against writing outside the calling user's own directories.

These tools read from shared project trees that belong to other users — the
multilingual_scaling results directories, other people's monitor_state, and so
on.  Several of them also write derived artifacts (throughput caches,
gpu_hours.csv) and historically resolved those write paths *relative to the
directory being read*, which silently put files into other users' trees.

This module makes that a hard error instead of an intention.  Call
:func:`guard_write` immediately before any write and it raises unless the
resolved destination sits under an allowed root.

Allowed roots default to the current user's home and their per-user scratch
directory, and can be overridden with the ``OELLM_WRITE_ROOTS`` environment
variable (colon-separated).  Set ``OELLM_WRITE_GUARD=off`` to disable, which
should only ever be done deliberately.

Usage::

    from write_guard import guard_write
    guard_write(csv_path)
    with csv_path.open("w") as f:
        ...
"""

from __future__ import annotations

import getpass
import os
from pathlib import Path

__all__ = ["guard_write", "allowed_roots", "is_write_allowed", "WriteGuardError"]


class WriteGuardError(PermissionError):
    """Raised when a tool attempts to write outside the allowed roots."""


def _default_roots() -> list[Path]:
    """Home plus any per-user scratch directory belonging to this user."""
    user = getpass.getuser()
    roots: list[Path] = []

    home = Path.home()
    if home.is_dir():
        roots.append(home)

    # Project scratch is laid out as /scratch/<project>/users/<user>.  Discover
    # rather than hardcode, so this keeps working across projects and clusters.
    # Errors must be caught per-entry: these bases hold ~2000 project dirs and
    # many are unreadable, so a loop-level handler would abort discovery on the
    # first permission error and silently leave the user's own scratch out.
    for base in (Path("/scratch"), Path("/pfs/lustrep3/scratch"), Path("/pfs/lustrep4/scratch")):
        try:
            projects = list(base.iterdir())
        except OSError:
            continue
        for project in projects:
            candidate = project / "users" / user
            try:
                if candidate.is_dir():
                    roots.append(candidate)
            except OSError:
                continue

    return roots


def allowed_roots() -> list[Path]:
    """Return the write-allowed roots, resolved."""
    env = os.environ.get("OELLM_WRITE_ROOTS")
    raw = [Path(p) for p in env.split(":") if p] if env else _default_roots()
    out: list[Path] = []
    for p in raw:
        try:
            out.append(p.resolve())
        except OSError:
            continue
    return out


def _enabled() -> bool:
    return os.environ.get("OELLM_WRITE_GUARD", "on").strip().lower() not in {"off", "0", "false"}


def is_write_allowed(path: str | os.PathLike) -> bool:
    """Return True if *path* may be written to."""
    if not _enabled():
        return True
    # Resolve through symlinks and ".." so the check cannot be side-stepped.
    # The file itself need not exist yet; its nearest existing parent is what
    # determines the real location.
    target = Path(path).expanduser().absolute()
    probe = target
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        resolved = probe.resolve() / target.relative_to(probe)
    except (OSError, ValueError):
        resolved = target
    return any(resolved == r or r in resolved.parents for r in allowed_roots())


def guard_write(path: str | os.PathLike) -> Path:
    """Raise :class:`WriteGuardError` unless *path* is writable by policy.

    Returns the path unchanged so it can be used inline.
    """
    if not is_write_allowed(path):
        roots = "\n  ".join(str(r) for r in allowed_roots()) or "(none discovered)"
        raise WriteGuardError(
            f"refusing to write outside your own directories:\n"
            f"  {Path(path).absolute()}\n"
            f"allowed roots:\n  {roots}\n"
            f"Use --cache-dir (or point --csv/--md into your own tree). "
            f"Override deliberately with OELLM_WRITE_ROOTS or OELLM_WRITE_GUARD=off."
        )
    return Path(path)

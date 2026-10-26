#!/usr/bin/env python3
"""Code provenance: which code a job was planned with, and which code it ran.

STDLIB ONLY, and runnable as a plain script: the pool supervisor calls it on a
compute node, in the batch shell, outside any container and without the
project's environment::

    python3 oellm_autoexp/utils/provenance.py --repo . --out launch.json [--compare run.json]

``code_provenance(repo)`` returns::

    {"captured": "2026-10-26T14:20:11",
     "repo": {"commit": "a335d1e2...", "branch": "feat/sub_slurm_jobs",
              "dirty": {"path": "<git blob sha1>" | "deleted", ...}},
     "submodules": {"submodules/Megatron-LM": {"commit": ..., "branch": ..., "dirty": {...}}, ...}}

``dirty`` is every path ``git status`` lists -- modified (staged or not),
deleted, renamed (new path, with ``"<old> -> "`` noted under ``renamed``) and
UNTRACKED files -- except gitignored ones. Untracked files are in on purpose:
experiment configs are often untracked, and they are exactly what a run uses.
Values are git blob hashes (what ``git hash-object <file>`` prints), so any
file can be checked later against a commit or a copy. Submodules are recorded
on their own, so the parent repo's status ignores them.

Failures never raise: a field that cannot be determined says why
(``{"error": ...}``), because provenance must never block a submission or a run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

TIMEOUT_S = 30.0


def _git(repo: Path, *args: str, input_text: str | None = None) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        input=input_text,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"git {args[0]} failed")
    return proc.stdout


def git_state(repo: str | Path) -> dict:
    """Commit, branch and dirty-file map of one git work tree."""
    repo = Path(repo)
    try:
        commit = _git(repo, "rev-parse", "HEAD").strip()
    except Exception as exc:  # not a git work tree, git missing, timeout
        return {"error": f"no git state: {exc}"}
    state: dict = {"commit": commit}
    try:
        state["branch"] = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip()
    except Exception:
        state["branch"] = None
    try:
        raw = _git(
            repo,
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--ignore-submodules=all",
        )
    except Exception as exc:
        state["dirty"] = {"error": str(exc)}
        return state
    entries = raw.split("\0")
    dirty: dict[str, str] = {}
    renamed: dict[str, str] = {}
    to_hash: list[str] = []
    i = 0
    while i < len(entries):
        entry = entries[i]
        i += 1
        if len(entry) < 4:
            continue
        code, path = entry[:2], entry[3:]
        if "R" in code or "C" in code:
            # -z puts the ORIGINAL path in the next field
            renamed[path] = entries[i] if i < len(entries) else ""
            i += 1
        if "D" in code and not (repo / path).exists():
            dirty[path] = "deleted"
        elif (repo / path).is_file():
            to_hash.append(path)
        else:
            dirty[path] = "not-a-file"
    if to_hash:
        try:
            hashes = _git(
                repo, "hash-object", "--stdin-paths", input_text="\n".join(to_hash) + "\n"
            )
            for path, sha in zip(to_hash, hashes.split()):
                dirty[path] = sha
        except Exception as exc:
            for path in to_hash:
                dirty[path] = f"unhashed: {exc}"
    state["dirty"] = dict(sorted(dirty.items()))
    if renamed:
        state["renamed"] = renamed
    return state


def _submodules(repo: Path) -> list[Path]:
    root = repo / "submodules"
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if (p / ".git").exists())


def code_provenance(repo: str | Path, **extra) -> dict:
    """Provenance of a repo checkout and its git submodules (plus ``extra``
    keys)."""
    repo = Path(repo).absolute()
    out: dict = {"captured": time.strftime("%Y-%m-%dT%H:%M:%S"), **extra}
    out["repo"] = git_state(repo)
    subs = _submodules(repo)
    out["submodules"] = {str(p.relative_to(repo)): git_state(p) for p in subs}
    # a nested repo that is not a registered submodule shows up in the parent's
    # status as an untracked directory; it is recorded on its own above
    names = {str(p.relative_to(repo)) for p in subs}
    inside = tuple(f"{n}/" for n in names)
    dirty = out["repo"].get("dirty")
    if names and isinstance(dirty, dict) and "error" not in dirty:
        out["repo"]["dirty"] = {
            k: v
            for k, v in dirty.items()
            if k.rstrip("/") not in names and not k.startswith(inside)
        }
    return out


def code_changes(enqueued: dict, launched: dict) -> list[str]:
    """Human-readable differences between two ``code_provenance`` results."""
    changes: list[str] = []
    trees = [("repo", enqueued.get("repo") or {}, launched.get("repo") or {})]
    for name in sorted(
        set(enqueued.get("submodules") or {}) | set(launched.get("submodules") or {})
    ):
        trees.append(
            (
                name,
                (enqueued.get("submodules") or {}).get(name) or {},
                (launched.get("submodules") or {}).get(name) or {},
            )
        )
    for name, before, after in trees:
        if "error" in before or "error" in after:
            if before.get("error") != after.get("error"):
                changes.append(
                    f"{name}: {before.get('error') or 'ok'} -> {after.get('error') or 'ok'}"
                )
            continue
        if before.get("commit") != after.get("commit"):
            changes.append(
                f"{name}: commit {str(before.get('commit'))[:10]} -> {str(after.get('commit'))[:10]}"
            )
        b, a = before.get("dirty") or {}, after.get("dirty") or {}
        for path in sorted(set(b) | set(a)):
            if b.get(path) != a.get(path):
                changes.append(
                    f"{name}: {path} {str(b.get(path, 'clean'))[:10]} -> {str(a.get(path, 'clean'))[:10]}"
                )
    return changes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo", default=".")
    parser.add_argument("--out", required=True, help="JSON file to write")
    parser.add_argument(
        "--compare",
        default=None,
        help="JSON file whose 'provenance' key (or top level) is the enqueue-time state to compare with",
    )
    args = parser.parse_args(argv)
    try:
        launched = code_provenance(args.repo, host=os.uname().nodename)
        changes: list[str] | None = None
        if args.compare:
            ref = json.loads(Path(args.compare).read_text())
            ref = ref.get("provenance", ref)
            if ref:
                changes = code_changes(ref, launched)
        launched["changed_since_enqueue"] = changes
        tmp = Path(args.out).with_name(f".{Path(args.out).name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(launched, indent=2))
        os.replace(tmp, args.out)
    except Exception as exc:  # never block a run
        print(f"provenance: unavailable ({exc})")
        return 0
    if changes is None:
        print("provenance: recorded (no enqueue state to compare)")
    elif changes:
        print(
            f"provenance: CODE CHANGED SINCE ENQUEUE ({len(changes)}): " + "; ".join(changes[:10])
        )
    else:
        print("provenance: code unchanged since enqueue")
    return 0


if __name__ == "__main__":
    sys.exit(main())

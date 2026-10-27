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

WITHOUT A ``git`` BINARY (JUPITER compute nodes, most containers) the state is
read from the ``.git`` files directly: HEAD, loose and packed refs, ``gitdir:``
links of submodules and worktrees. The record says ``"git": "unavailable"``
and has ``dirty: null``, because the working-tree status needs git's index.
Given the enqueue-time record (``--compare``), the files that were dirty
THEN are hashed again in pure Python (``dirty_checked``), so a sync or an edit
of them is still caught, and so is a commit change. What is not seen without
git: files that were clean at enqueue and changed later, and new untracked
files. A check that could not be made is never reported as a change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
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


def _have_git() -> bool:
    return shutil.which("git") is not None


def blob_hash(path: Path) -> str:
    """The git blob id of a file's content (what ``git hash-object``
    prints)."""
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _git_dirs(repo: Path) -> tuple[Path, Path] | None:
    """(git dir with HEAD, common dir with refs), following ``gitdir:``
    files."""
    dot = repo / ".git"
    if dot.is_dir():
        gitdir = dot
    elif dot.is_file():
        text = dot.read_text().strip()
        if not text.startswith("gitdir:"):
            return None
        gitdir = Path(text.split(":", 1)[1].strip())
        if not gitdir.is_absolute():
            gitdir = (repo / gitdir).resolve()
    else:
        return None
    common = gitdir
    if (gitdir / "commondir").is_file():  # a worktree: refs live in the main repo
        common = (gitdir / (gitdir / "commondir").read_text().strip()).resolve()
    return gitdir, common


def read_head(repo: str | Path) -> tuple[str, str | None]:
    """(commit, branch) from the .git files alone; raises when unreadable."""
    dirs = _git_dirs(Path(repo))
    if dirs is None:
        raise RuntimeError("not a git work tree")
    gitdir, common = dirs
    head = (gitdir / "HEAD").read_text().strip()
    if not head.startswith("ref:"):
        return head, None  # detached
    ref = head.split(":", 1)[1].strip()
    branch = ref.removeprefix("refs/heads/")
    for base in (gitdir, common):
        loose = base / ref
        if loose.is_file():
            return loose.read_text().strip(), branch
    packed = common / "packed-refs"
    if packed.is_file():
        for line in packed.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1] == ref:
                return parts[0], branch
    raise RuntimeError(f"cannot resolve {ref}")


def _state_without_git(repo: Path, check: dict | None) -> dict:
    try:
        commit, branch = read_head(repo)
    except Exception as exc:
        return {"error": f"no git state: {exc}", "git": "unavailable"}
    state: dict = {"commit": commit, "branch": branch, "dirty": None, "git": "unavailable"}
    if check:
        checked: dict[str, str] = {}
        for path in check:
            full = repo / path
            if not full.exists():
                checked[path] = "deleted"
            elif full.is_file():
                try:
                    checked[path] = blob_hash(full)
                except OSError as exc:
                    checked[path] = f"unhashed: {exc}"
            else:
                checked[path] = "not-a-file"
        state["dirty_checked"] = checked
    return state


def git_state(repo: str | Path, *, check: dict | None = None) -> dict:
    """Commit, branch and dirty-file map of one git work tree.

    ``check``: the enqueue-time dirty map; only used without a git binary,
    to hash those files again (see the module doc).
    """
    repo = Path(repo)
    if not _have_git():
        return _state_without_git(repo, check)
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


def code_provenance(repo: str | Path, *, reference: dict | None = None, **extra) -> dict:
    """Provenance of a repo checkout and its git submodules (plus ``extra``
    keys).

    ``reference``: an earlier result (the enqueue state); without git, the
    files dirty in it are hashed again.
    """
    repo = Path(repo).absolute()
    ref = reference or {}

    def check_of(tree: dict | None) -> dict | None:
        dirty = (tree or {}).get("dirty")
        return dirty if isinstance(dirty, dict) and "error" not in dirty else None

    out: dict = {"captured": time.strftime("%Y-%m-%dT%H:%M:%S"), **extra}
    out["repo"] = git_state(repo, check=check_of(ref.get("repo")))
    subs = _submodules(repo)
    out["submodules"] = {
        str(p.relative_to(repo)): git_state(
            p, check=check_of((ref.get("submodules") or {}).get(str(p.relative_to(repo))))
        )
        for p in subs
    }
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
        # a side that could not be read is UNKNOWN, never a change (a compute node
        # without git once reported "repo: ok -> no git state" as a code change)
        if not before or not after or "error" in before or "error" in after:
            continue
        if before.get("commit") and after.get("commit") and before["commit"] != after["commit"]:
            changes.append(
                f"{name}: commit {str(before.get('commit'))[:10]} -> {str(after.get('commit'))[:10]}"
            )
        b = before.get("dirty")
        if not isinstance(b, dict) or "error" in b:
            continue  # no enqueue dirty map to compare against
        a = after.get("dirty")
        if isinstance(a, dict) and "error" not in a:
            paths = set(b) | set(a)
        elif isinstance(after.get("dirty_checked"), dict):
            # without git: only the files dirty at enqueue were hashed again
            a, paths = after["dirty_checked"], set(b)
        else:
            continue
        for path in sorted(paths):
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
        ref = None
        if args.compare:
            ref = json.loads(Path(args.compare).read_text())
            ref = ref.get("provenance", ref)
        launched = code_provenance(args.repo, reference=ref, host=os.uname().nodename)
        changes: list[str] | None = code_changes(ref, launched) if ref else None
        launched["changed_since_enqueue"] = changes
        tmp = Path(args.out).with_name(f".{Path(args.out).name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(launched, indent=2))
        os.replace(tmp, args.out)
    except Exception as exc:  # never block a run
        print(f"provenance: unavailable ({exc})")
        return 0
    partial = (
        " (no git here: commit + files dirty at enqueue checked; clean or new files not)"
        if (launched.get("repo") or {}).get("git") == "unavailable"
        else ""
    )
    if changes is None:
        print(f"provenance: recorded (no enqueue state to compare){partial}")
    elif changes:
        print(
            f"provenance: CODE CHANGED SINCE ENQUEUE ({len(changes)}): "
            + "; ".join(changes[:10])
            + partial
        )
    else:
        print(f"provenance: code unchanged since enqueue{partial}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

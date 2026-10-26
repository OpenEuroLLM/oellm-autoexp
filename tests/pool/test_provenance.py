"""Tests of oellm_autoexp/utils/provenance.py on a throwaway git repo."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from oellm_autoexp.utils.provenance import code_changes, code_provenance, git_state

pytestmark = pytest.mark.skipif(not shutil.which("git"), reason="needs git")

SCRIPT = Path(__file__).resolve().parents[2] / "oellm_autoexp" / "utils" / "provenance.py"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


def _blob(repo: Path, path: str) -> str:
    return _git(repo, "hash-object", path).strip()


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _git(r, "config", "user.email", "t@t")
    _git(r, "config", "user.name", "t")
    for name in ("modified.py", "deleted.py", "renamed_old.py", "clean.py"):
        (r / name).write_text(f"{name}\n")
    (r / ".gitignore").write_text("ignored.log\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "init")
    (r / "modified.py").write_text("changed\n")
    (r / "deleted.py").unlink()
    _git(r, "mv", "renamed_old.py", "renamed_new.py")
    (r / "configs").mkdir()
    (r / "configs" / "untracked.yaml").write_text("a: 1\n")
    (r / "ignored.log").write_text("noise\n")
    return r


def test_dirty_covers_modified_deleted_renamed_untracked_but_not_ignored(repo):
    state = git_state(repo)
    assert state["commit"] == _git(repo, "rev-parse", "HEAD").strip()
    assert state["branch"] == "main"
    dirty = state["dirty"]
    assert dirty["modified.py"] == _blob(repo, "modified.py")
    assert dirty["deleted.py"] == "deleted"
    assert dirty["renamed_new.py"] == _blob(repo, "renamed_new.py")
    assert state["renamed"] == {"renamed_new.py": "renamed_old.py"}
    assert dirty["configs/untracked.yaml"] == _blob(repo, "configs/untracked.yaml")
    assert "ignored.log" not in dirty and "clean.py" not in dirty


def test_submodules_are_recorded_separately(repo):
    sub = repo / "submodules" / "Megatron-LM"
    sub.mkdir(parents=True)
    _git(sub, "init", "-q", "-b", "dev")
    _git(sub, "config", "user.email", "t@t")
    _git(sub, "config", "user.name", "t")
    (sub / "x.py").write_text("x\n")
    _git(sub, "add", "-A")
    _git(sub, "commit", "-q", "-m", "x")
    (sub / "x.py").write_text("y\n")
    prov = code_provenance(repo, manifest="m.json")
    assert prov["manifest"] == "m.json"
    assert prov["submodules"]["submodules/Megatron-LM"]["dirty"] == {"x.py": _blob(sub, "x.py")}
    assert not any(p.startswith("submodules/") for p in prov["repo"]["dirty"])


def test_not_a_repo_never_raises(tmp_path):
    assert "error" in git_state(tmp_path)


def test_code_changes(repo):
    before = code_provenance(repo)
    assert code_changes(before, code_provenance(repo)) == []
    (repo / "configs" / "untracked.yaml").write_text("a: 2\n")
    (repo / "new.py").write_text("n\n")
    changes = code_changes(before, code_provenance(repo))
    assert any("configs/untracked.yaml" in c for c in changes)
    assert any("new.py clean ->" in c for c in changes)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "next")
    assert any(c.startswith("repo: commit ") for c in code_changes(before, code_provenance(repo)))


def test_script_compares_with_the_enqueue_state(repo, tmp_path):
    run_json = tmp_path / "run.json"
    run_json.write_text(json.dumps({"rid": "p.0001", "provenance": code_provenance(repo)}))
    out = tmp_path / "launch.json"
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--repo",
        str(repo),
        "--out",
        str(out),
        "--compare",
        str(run_json),
    ]
    same = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert same.stdout.strip() == "provenance: code unchanged since enqueue"
    assert json.loads(out.read_text())["changed_since_enqueue"] == []
    (repo / "modified.py").write_text("changed again\n")
    diff = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "CODE CHANGED SINCE ENQUEUE (1)" in diff.stdout and "modified.py" in diff.stdout
    # no enqueue state, unreadable reference, not a repo: a line, never a failure
    run_json.write_text(json.dumps({"rid": "p.0001", "provenance": None}))
    assert "no enqueue state" in subprocess.run(cmd, capture_output=True, text=True).stdout
    bad = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo",
            str(repo),
            "--out",
            str(out),
            "--compare",
            str(tmp_path / "nope"),
        ],
        capture_output=True,
        text=True,
    )
    assert bad.returncode == 0 and "unavailable" in bad.stdout

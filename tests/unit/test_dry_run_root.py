"""oellm_autoexp/utils/dry_run.py and run_autoexp --dry-run-root: a dry run writes its rendered files
below a local root (mirroring their absolute paths) and never creates the real output tree; the
contents keep the real paths; an exported root never redirects outside a dry run."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from oellm_autoexp.utils import dry_run

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(dry_run.ROOT_ENV, raising=False)
    monkeypatch.delenv(dry_run.ACTIVE_ENV, raising=False)
    yield
    # activate() sets os.environ directly (for render workers): never leak a dry run into later
    # tests, whose renders would then go below this test's root
    for name in (dry_run.ROOT_ENV, dry_run.ACTIVE_ENV):
        os.environ.pop(name, None)


def test_no_dry_run_no_redirect(tmp_path, monkeypatch):
    # an exported root alone (no active dry run) never moves a real submission's files
    monkeypatch.setenv(dry_run.ROOT_ENV, str(tmp_path))
    assert dry_run.dry_run_root() is None
    assert dry_run.dry_run_path("/gpfs/x/job.sbatch") == Path("/gpfs/x/job.sbatch")


def test_dry_run_without_root_keeps_paths():
    assert dry_run.activate(None) is None
    assert dry_run.dry_run_path("/gpfs/x/job.sbatch") == Path("/gpfs/x/job.sbatch")


def test_absolute_paths_are_mirrored_below_the_root(tmp_path):
    root = dry_run.activate(tmp_path)
    assert root == tmp_path.resolve()
    assert dry_run.dry_run_path("/gpfs/x/job.sbatch") == root / "gpfs/x/job.sbatch"
    assert dry_run.dry_run_path("rel/job.sbatch") == root / "rel/job.sbatch"
    inside = root / "already" / "here.sbatch"
    assert dry_run.dry_run_path(inside) == inside


def test_exported_root_applies_to_every_dry_run(tmp_path, monkeypatch):
    monkeypatch.setenv(dry_run.ROOT_ENV, str(tmp_path))
    assert dry_run.activate(None) == tmp_path
    assert dry_run.dry_run_path("/a/b") == tmp_path / "a/b"


def test_run_autoexp_dry_run_root_end_to_end(tmp_path):
    """A config whose output root is an absolute path the user cannot create (/autoexp_dry_*):
    without the redirect the render would fail; with it the script lands below the root, keeps
    the real paths inside, and nothing is created outside it."""
    real_root = Path(f"/autoexp_dry_run_test_{os.getpid()}")
    assert not real_root.exists()
    root = tmp_path / "dry"
    env = {**os.environ, "POOL_TEST_ROOT": str(real_root), "PYTHONPATH": str(REPO)}
    env.pop(dry_run.ROOT_ENV, None)
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_autoexp.py",
            "--dry-run-root",
            str(root),
            "--monitor-state-dir",
            str(tmp_path / "state"),
            "--config-name",
            "experiments/tests/pool_local_dummy",
        ],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    scripts = list(root.rglob("*.sbatch"))
    assert scripts, proc.stdout[-3000:]
    mirrored = root / real_root.relative_to("/")
    assert all(mirrored in s.parents for s in scripts), scripts
    assert str(real_root) in scripts[0].read_text(), "the content keeps the real paths"
    assert not real_root.exists()
    assert "below " + str(root.resolve()) in proc.stdout

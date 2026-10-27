"""The pool adoption hook (scripts/pool/adopt_sitecustomize.py, installed by `pool_ctl adopt`)
must fail CLOSED: in a listed job, nothing of the job's own Python code may ever run -- held
until released, or leaving with rc 0 when no supervisor attaches. Everything else runs as
before, with the original sitecustomize chained in."""

from __future__ import annotations

import importlib.util
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _pool_ctl():
    spec = importlib.util.spec_from_file_location("pool_ctl", REPO / "scripts" / "pool_ctl.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def workdir(tmp_path):
    """A job WorkDir with the hook as its sitecustomize, and an 'image' sitecustomize later on
    the path that the hook must chain in."""
    wd = tmp_path / "snapshot"
    image = tmp_path / "image"
    wd.mkdir()
    image.mkdir()
    (image / "sitecustomize.py").write_text(
        "import sys\nsys.stderr.write('[image sitecustomize]\\n')\n"
    )
    adopt = tmp_path / "adopt"
    text = _pool_ctl().render_hook({"4711"}, adopt, hold_max_s=3)
    (wd / "sitecustomize.py").write_text(text)
    return wd, image, adopt


def _python(workdir, image, *, job="", run_id="", code="print('JOB CODE RAN')", timeout=30):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "AUTOEXP_"))}
    env["PYTHONPATH"] = f".{os.pathsep}{image}"
    if job:
        env["SLURM_JOB_ID"] = job
    if run_id:
        env["AUTOEXP_RUN_ID"] = run_id
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_a_listed_job_never_runs_its_own_code(workdir):
    wd, image, adopt = workdir
    t0 = time.time()
    proc = _python(wd, image, job="4711")
    assert "JOB CODE RAN" not in proc.stdout
    assert proc.returncode == 0, proc.stderr
    assert "no pool supervisor attached" in proc.stderr
    assert time.time() - t0 >= 3
    assert (adopt / "4711.started").exists() and (adopt / "4711.abandoned").exists()


def test_a_held_job_leaves_when_released(workdir):
    wd, image, adopt = workdir
    adopt.mkdir()
    (adopt / "4711.attached").write_text("x")
    (adopt / "4711.release").write_text("x")
    proc = _python(wd, image, job="4711")
    assert "JOB CODE RAN" not in proc.stdout and proc.returncode == 0
    assert "released" in proc.stderr


def test_a_held_job_leaves_on_sigterm(workdir):
    wd, image, adopt = workdir
    adopt.mkdir()
    (adopt / "4711.attached").write_text("x")  # attached: it would hold for good
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "AUTOEXP_"))}
    env.update(PYTHONPATH=f".{os.pathsep}{image}", SLURM_JOB_ID="4711")
    proc = subprocess.Popen(
        [sys.executable, "-c", "print('JOB CODE RAN')"],
        cwd=wd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.time() + 20
    while not (adopt / "4711.started").exists() and time.time() < deadline:
        time.sleep(0.1)
    time.sleep(0.5)
    assert proc.poll() is None, "an attached hold must not end on its own"
    proc.send_signal(signal.SIGTERM)
    out, err = proc.communicate(timeout=20)
    assert "JOB CODE RAN" not in out and proc.returncode == 0, err


@pytest.mark.parametrize("job,run_id", [("", ""), ("999", ""), ("4711", "pool.0001")])
def test_everything_else_runs_normally_with_the_image_sitecustomize(workdir, job, run_id):
    wd, image, _ = workdir
    proc = _python(wd, image, job=job, run_id=run_id)
    assert proc.returncode == 0 and "JOB CODE RAN" in proc.stdout, proc.stderr
    assert "[image sitecustomize]" in proc.stderr
    assert "[pool-adopt]" not in proc.stderr


def test_hook_ids_round_trip_and_refuse_foreign_files(tmp_path):
    pc = _pool_ctl()
    hook = tmp_path / "sitecustomize.py"
    hook.write_text(pc.render_hook({"1", "2"}, tmp_path, 60))
    assert pc._hook_ids(hook) == {"1", "2"}
    hook.write_text(pc.render_hook(set(), tmp_path, 60))
    assert pc._hook_ids(hook) == set()
    hook.write_text("print('someone else')\n")
    assert pc._hook_ids(hook) is None

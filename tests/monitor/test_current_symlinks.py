"""Tests for per-job log resolution and RUNNING-triggered current.* symlinks.

The monitor must read each job's OWN log (``slurm-<jobid>.log``), never the
shared ``current`` symlink - otherwise, in a dependency chain where all jobs
share one base_output_dir, every job would read whichever job is currently
running. The ``current.*`` symlinks are a tailing convenience that the monitor
re-points to the running job when it enters RUNNING.
"""

from __future__ import annotations

from pathlib import Path

from oellm_autoexp.monitor.loop import JobFileStore, JobRecord, JobRuntime, MonitorLoop
from oellm_autoexp.monitor.submission import SlurmJobConfig
from oellm_autoexp.slurm_gen.schema import SlurmConfig


def _slurm_job(tmp_path: Path, *, job_id: str, runtime_id: str, current: bool) -> JobRecord:
    base = str(tmp_path)
    definition = SlurmJobConfig(
        name=job_id,
        log_path=f"{base}/slurm-%j.log",
        log_path_current=f"{base}/current.log" if current else None,
        config_path=f"{base}/config-%j.yaml",
        config_path_current=f"{base}/current.yaml" if current else None,
        slurm=SlurmConfig(
            name=job_id, template_path="templates/base.sbatch", script_dir=base, log_dir=base
        ),
    )
    runtime = JobRuntime(submitted=True, runtime_job_id=runtime_id, attempts=1)
    return JobRecord(job_id=job_id, definition=definition, runtime=runtime)


def _loop(tmp_path: Path) -> MonitorLoop:
    return MonitorLoop(JobFileStore(tmp_path / "state"))


def test_resolve_log_path_ignores_current_symlink(tmp_path: Path):
    """Even with log_path_current set, the monitor reads the per-job log."""
    loop = _loop(tmp_path)
    job = _slurm_job(tmp_path, job_id="r1", runtime_id="48800001", current=True)
    resolved = loop._resolve_log_path(job)
    assert resolved == Path(f"{tmp_path}/slurm-48800001.log")
    assert "current.log" not in str(resolved)


def test_update_current_symlinks_points_at_running_job(tmp_path: Path):
    loop = _loop(tmp_path)
    # Two chain siblings sharing one base_output_dir (tmp_path) and one current.*.
    (tmp_path / "slurm-48800001.log").write_text("r1 output")
    (tmp_path / "config-48800001.yaml").write_text("r1 config")
    (tmp_path / "slurm-48800002.log").write_text("r2 output")
    (tmp_path / "config-48800002.yaml").write_text("r2 config")

    r1 = _slurm_job(tmp_path, job_id="r1", runtime_id="48800001", current=True)
    loop._update_current_symlinks(r1)
    assert (tmp_path / "current.log").resolve() == (tmp_path / "slurm-48800001.log").resolve()
    assert (tmp_path / "current.yaml").resolve() == (tmp_path / "config-48800001.yaml").resolve()

    # When the next sibling runs, the shared symlink re-points to it.
    r2 = _slurm_job(tmp_path, job_id="r2", runtime_id="48800002", current=True)
    loop._update_current_symlinks(r2)
    assert (tmp_path / "current.log").resolve() == (tmp_path / "slurm-48800002.log").resolve()
    assert (tmp_path / "current.yaml").resolve() == (tmp_path / "config-48800002.yaml").resolve()


def test_update_current_symlinks_noop_without_current(tmp_path: Path):
    loop = _loop(tmp_path)
    job = _slurm_job(tmp_path, job_id="r1", runtime_id="48800001", current=False)
    loop._update_current_symlinks(job)  # must not raise
    assert not (tmp_path / "current.log").exists()


def test_observe_once_updates_symlink_on_running_transition(tmp_path: Path):
    """A PENDING->RUNNING transition observed by the loop re-points
    current.*."""

    class FakeClient:
        def __init__(self, statuses):
            self.statuses = statuses

        def squeue(self):
            return dict(self.statuses)

        def remove(self, job_id):  # pragma: no cover - unused here
            pass

    store = JobFileStore(tmp_path / "state")
    job = _slurm_job(tmp_path, job_id="r1", runtime_id="48800001", current=True)
    job.runtime.last_status = "PENDING"
    store.upsert(job)
    (tmp_path / "slurm-48800001.log").write_text("running output")
    (tmp_path / "config-48800001.yaml").write_text("cfg")

    loop = MonitorLoop(store, slurm_client=FakeClient({"48800001": "RUNNING"}))
    loop.observe_once()

    assert (tmp_path / "current.log").resolve() == (tmp_path / "slurm-48800001.log").resolve()

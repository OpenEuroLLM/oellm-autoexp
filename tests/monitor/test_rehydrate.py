"""Tests for re-adopting already-submitted jobs when a monitor re-attaches.

``scripts/monitor_autoexp.py --session <id>`` is the documented way to resume
monitoring after the terminal, the ssh connection or the login node dies. But
the job clients only ever learn about a job through ``submit()``, and
``SlurmClient.squeue()`` short-circuits to ``{}`` while it tracks nothing — so a
resumed monitor used to stay blind to SLURM for its entire lifetime: every job
sat at ``last_status=None``, the terminal-state branch in ``observe_once`` never
fired, jobs were never marked finished and the loop never exited. Only the
log-event path kept working (it reads files), which is what made the failure so
quiet.

``MonitorLoop.rehydrate()`` closes that gap; these tests pin it down.
"""

from __future__ import annotations

from pathlib import Path

from oellm_autoexp.monitor.loop import (
    JobFileStore,
    JobRecord,
    JobRuntime,
    MonitorLoop,
)
from oellm_autoexp.monitor.local_client import (
    LocalCommandClient,
    LocalCommandClientConfig,
)
from oellm_autoexp.monitor.slurm_client import (
    SlurmClient as MonitorSlurmClient,
    SlurmClientConfig,
)
from oellm_autoexp.monitor.submission import LocalJobConfig, SlurmJobConfig
from oellm_autoexp.slurm_gen import SlurmConfig
from oellm_autoexp.slurm_gen.client import FakeSlurmClientConfig


def _slurm_job(
    tmp_path: Path,
    *,
    job_id: str,
    runtime_id: str | None,
    submitted: bool = True,
    last_status: str | None = "RUNNING",
    final_state: str | None = None,
) -> JobRecord:
    definition = SlurmJobConfig(
        name=job_id,
        log_path=str(tmp_path / f"{job_id}.log"),
        slurm=SlurmConfig(
            name=job_id,
            template_path="templates/base.sbatch",
            script_dir=str(tmp_path),
            log_dir=str(tmp_path),
        ),
    )
    runtime = JobRuntime(
        submitted=submitted,
        runtime_job_id=runtime_id,
        attempts=1,
        last_status=last_status,
        final_state=final_state,
    )
    return JobRecord(job_id=job_id, definition=definition, runtime=runtime)


def _local_job(tmp_path: Path, *, job_id: str, runtime_id: str) -> JobRecord:
    definition = LocalJobConfig(
        name=job_id,
        command=["true"],
        log_path=str(tmp_path / f"{job_id}.log"),
    )
    runtime = JobRuntime(
        submitted=True,
        runtime_job_id=runtime_id,
        attempts=1,
        last_status="RUNNING",
    )
    return JobRecord(job_id=job_id, definition=definition, runtime=runtime)


def _monitor_client() -> MonitorSlurmClient:
    return MonitorSlurmClient(SlurmClientConfig(base_client=FakeSlurmClientConfig()))


def test_fresh_client_is_blind_until_rehydrated(tmp_path: Path):
    """The regression itself: attaching without rehydrate sees no statuses."""
    store = JobFileStore(tmp_path / "state")
    store.upsert(_slurm_job(tmp_path, job_id="j", runtime_id="1524558"))

    client = _monitor_client()
    MonitorLoop(store, slurm_client=client)

    assert client.squeue() == {}


def test_rehydrate_registers_submitted_jobs(tmp_path: Path):
    store = JobFileStore(tmp_path / "state")
    store.upsert(_slurm_job(tmp_path, job_id="j1", runtime_id="1524558"))
    store.upsert(_slurm_job(tmp_path, job_id="j2", runtime_id="1524559", last_status="PENDING"))

    client = _monitor_client()
    loop = MonitorLoop(store, slurm_client=client)

    assert loop.rehydrate() == 2
    # The last known status is carried over so the first poll does not invent a
    # spurious transition before squeue has spoken.
    assert client.squeue() == {"1524558": "RUNNING", "1524559": "PENDING"}


def test_rehydrate_carries_no_status_as_pending(tmp_path: Path):
    """A job submitted but never yet polled has last_status=None."""
    store = JobFileStore(tmp_path / "state")
    store.upsert(_slurm_job(tmp_path, job_id="j", runtime_id="1524558", last_status=None))

    client = _monitor_client()
    MonitorLoop(store, slurm_client=client).rehydrate()

    assert client.squeue() == {"1524558": "PENDING"}


def test_rehydrate_handles_array_task_ids(tmp_path: Path):
    store = JobFileStore(tmp_path / "state")
    store.upsert(_slurm_job(tmp_path, job_id="j_0", runtime_id="1524558_0"))
    store.upsert(_slurm_job(tmp_path, job_id="j_1", runtime_id="1524558_1"))

    client = _monitor_client()
    assert MonitorLoop(store, slurm_client=client).rehydrate() == 2
    assert set(client.squeue()) == {"1524558_0", "1524558_1"}


def test_rehydrate_skips_unsubmitted_finished_and_idless(tmp_path: Path):
    """Only live, already-submitted jobs are adopted.

    An unsubmitted job must stay unsubmitted (the first poll submits
    it); a finished job must not be resurrected into the queue; and a
    record with no runtime id has nothing to adopt.
    """
    store = JobFileStore(tmp_path / "state")
    store.upsert(_slurm_job(tmp_path, job_id="live", runtime_id="1"))
    store.upsert(
        _slurm_job(tmp_path, job_id="pending", runtime_id=None, submitted=False, last_status=None)
    )
    store.upsert(_slurm_job(tmp_path, job_id="done", runtime_id="3", final_state="finished"))
    store.upsert(_slurm_job(tmp_path, job_id="idless", runtime_id=None))

    client = _monitor_client()
    loop = MonitorLoop(store, slurm_client=client)

    assert loop.rehydrate() == 1
    assert client.squeue() == {"1": "RUNNING"}


def test_rehydrate_local_job_is_a_noop(tmp_path: Path):
    """Local processes cannot be adopted; it must not raise either."""
    store = JobFileStore(tmp_path / "state")
    store.upsert(_local_job(tmp_path, job_id="loc", runtime_id="7"))

    local = LocalCommandClient(LocalCommandClientConfig())
    loop = MonitorLoop(store, local_client=local)

    # Counted as visited, but the client deliberately tracks nothing.
    assert loop.rehydrate() == 1
    assert local.squeue() == {}


def test_rehydrate_without_a_client_does_not_raise(tmp_path: Path):
    """A local-only monitor holding a SLURM record must not blow up."""
    store = JobFileStore(tmp_path / "state")
    store.upsert(_slurm_job(tmp_path, job_id="j", runtime_id="1"))

    assert MonitorLoop(store, slurm_client=None).rehydrate() == 0

"""job.concurrency_group / job.max_active: at most N jobs of a group in flight."""

from __future__ import annotations

from pathlib import Path

from compoconf import parse_config

from oellm_autoexp.monitor.loop import JobFileStore, JobRecord, JobRuntime, MonitorLoop
from oellm_autoexp.monitor.slurm_client import SlurmClient, SlurmClientConfig
from oellm_autoexp.monitor.submission import SlurmJobConfig
from oellm_autoexp.slurm_gen import SlurmConfig
from oellm_autoexp.slurm_gen.client import FakeSlurmClientConfig


def _job(
    tmp_path: Path, index: int, group: str | None = "debugq", limit: int | None = 1, qos=False
):
    out = tmp_path / "out" / f"job_{index}"
    slurm = parse_config(
        SlurmConfig,
        {
            "name": f"job_{index}",
            "template_path": "templates/base.sbatch",
            "script_dir": str(out),
            "log_dir": str(out),
            "command": [f"echo {index}"],
            "sbatch": {"time": "00:10:00", "nodes": 1, "output": str(out / "slurm-%j.log")},
        },
    )
    definition = SlurmJobConfig(
        name=f"job_{index}", log_path=str(out / "slurm-%j.log"), slurm=slurm,
        concurrency_group=group, max_active=limit, concurrency_qos=qos,
    )
    return JobRecord(job_id=f"job_{index}_x", definition=definition, runtime=JobRuntime())


def _loop(tmp_path: Path, jobs):
    store = JobFileStore(tmp_path / "state")
    for job in jobs:
        store.upsert(job)
    client = SlurmClient(SlurmClientConfig(base_client=FakeSlurmClientConfig()))
    return store, client, MonitorLoop(store, slurm_client=client)


def _submitted(store):
    return sorted(r.job_id for r in store.load_all(include_finished=True) if r.runtime.submitted)


def test_one_at_a_time(tmp_path: Path):
    store, client, loop = _loop(tmp_path, [_job(tmp_path, i) for i in range(3)])
    loop.observe_once()
    assert len(_submitted(store)) == 1
    loop.observe_once()  # still running: nothing new
    assert len(_submitted(store)) == 1

    first = next(r for r in store.load_all() if r.runtime.submitted)
    client._client.set_state(first.runtime.runtime_job_id, "COMPLETED")
    loop.observe_once()  # finishes the first ...
    loop.observe_once()  # ... and the next takes its slot
    assert len(_submitted(store)) == 2
    assert sum(r.runtime.submitted for r in store.load_all()) == 1


def test_limit_two_and_unthrottled_jobs(tmp_path: Path):
    jobs = [_job(tmp_path, i, limit=2) for i in range(4)] + [_job(tmp_path, 9, group=None)]
    store, _, loop = _loop(tmp_path, jobs)
    loop.observe_once()
    assert set(_submitted(store)) >= {"job_9_x"}
    assert len([j for j in _submitted(store) if j != "job_9_x"]) == 2


def test_qos_group_counts_other_sessions_jobs(tmp_path: Path, monkeypatch):
    """concurrency_qos: a job already in the QOS from another monitor blocks the slot."""
    store, client, loop = _loop(tmp_path, [_job(tmp_path, i, qos=True) for i in range(2)])
    elsewhere = {"n": 1}
    monkeypatch.setattr(client._client, "count_in_qos", lambda qos: elsewhere["n"], raising=False)
    loop.observe_once()
    assert _submitted(store) == []
    elsewhere["n"] = 0
    loop.observe_once()
    assert len(_submitted(store)) == 1


def test_qos_unknown_falls_back_to_the_session_count(tmp_path: Path):
    store, _, loop = _loop(tmp_path, [_job(tmp_path, i, qos=True) for i in range(2)])
    loop.observe_once()  # the fake client cannot count: session count only
    assert len(_submitted(store)) == 1


def test_rejected_submit_in_a_throttled_group_is_retried(tmp_path: Path, monkeypatch):
    """A QOS rejection (cap reached by another session / a race) keeps the job waiting and
    holds the rest of its group for this poll, instead of cancelling every job."""
    store, client, loop = _loop(tmp_path, [_job(tmp_path, i, qos=True) for i in range(3)])
    real_submit = client._client.submit
    calls = {"n": 0}

    def flaky(cfg):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("sbatch: error: QOSMaxSubmitJobPerUserLimit")
        return real_submit(cfg)

    monkeypatch.setattr(client._client, "submit", flaky)
    loop.observe_once()
    assert calls["n"] == 1  # one attempt, the others held back
    assert _submitted(store) == [] and len(store.load_all()) == 3
    loop.observe_once()
    assert len(_submitted(store)) == 1

from pathlib import Path

from monitor.loop import MonitorLoop, JobFileStore, JobRecordConfig, JobRuntimeConfig
from monitor.submission import JobRegistration
from monitor.actions import LogEventConfig
from oellm_autoexp.monitor.adapter import SlurmClientAdapter
from oellm_autoexp.monitor.actions import RestartActionConfig
from oellm_autoexp.slurm.client import FakeSlurmClient, FakeSlurmClientConfig


def test_monitor_loop_restart(tmp_path: Path):
    slurm = FakeSlurmClient(FakeSlurmClientConfig())
    adapter = SlurmClientAdapter(slurm)
    store = JobFileStore(tmp_path / "store")
    loop = MonitorLoop(store, adapter, poll_interval_seconds=0.1)

    script = tmp_path / "script.sh"
    script.write_text("#!/bin/bash\n")
    log = tmp_path / "log.txt"
    log.write_text("start\n")

    # Submit job manually (as orchestrator does)
    job_id = slurm.submit("demo", str(script), str(log))

    # Register job
    reg = JobRegistration(
        name="demo",
        command=[str(script)],
        log_path=str(log),
        log_events=[
            LogEventConfig(
                name="stall",
                pattern="stall detected",
                action=RestartActionConfig(reason="restart stalled job"),
            )
        ],
    )

    record = JobRecordConfig(
        job_id=job_id,
        registration=reg,
        runtime=JobRuntimeConfig(submitted=True, runtime_job_id=job_id),
    )
    store.upsert(record)

    # First observe - running
    slurm.set_state(job_id, "RUNNING")
    loop.observe_once()

    # Append stall to log
    log.write_text("start\nstall detected\n")

    # Second observe - should detect stall and restart
    loop.observe_once()

    # Verify restart
    updated = store.load(job_id)
    # MonitorLoop _restart_job calls _start_job which calls client.submit
    # And updates runtime.runtime_job_id
    assert updated.runtime.runtime_job_id != job_id
    # Old job should be removed (FakeSlurmClient remove pops it)
    assert job_id not in slurm._jobs

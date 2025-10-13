from pathlib import Path

import pytest

import oellm_autoexp.slurm.client
from oellm_autoexp.slurm.client import (
    FakeSlurmClient,
    FakeSlurmClientConfig,
    SlurmClient,
    SlurmClientConfig,
)
from oellm_autoexp.slurm.validator import SlurmValidationError, validate_job_script
from oellm_autoexp.config.schema import SlurmConfig


def test_validate_job_script_missing_job_name():
    with pytest.raises(SlurmValidationError):
        validate_job_script("#!/bin/bash", "demo")


def test_validate_job_script_placeholder():
    script = "#SBATCH --job-name=demo\nvalue {placeholder}\n"
    with pytest.raises(SlurmValidationError):
        validate_job_script(script, "demo")


def test_validate_job_script_required_tokens():
    script = "#SBATCH --job-name=demo\n"
    with pytest.raises(SlurmValidationError):
        validate_job_script(script, "demo", required_tokens=["module load"])


def test_fake_slurm_state_transitions(tmp_path: Path) -> None:
    slurm = FakeSlurmClient(FakeSlurmClientConfig())
    job_id = slurm.submit("demo", tmp_path / "job.sbatch", tmp_path / "log.txt")
    slurm.set_state(job_id, "RUNNING")
    assert slurm.squeue()[job_id] == "RUNNING"
    slurm.set_state(job_id, "COMPLETED", return_code=0)
    job = slurm.get_job(job_id)
    assert job.state == "COMPLETED"
    assert job.return_code == 0


def test_fake_slurm_submit_array(tmp_path: Path) -> None:
    """Test that FakeSlurmClient can submit job arrays."""
    slurm = FakeSlurmClient(FakeSlurmClientConfig())
    script_path = tmp_path / "array.sbatch"
    log_paths = [tmp_path / f"log_{i}.txt" for i in range(3)]
    task_names = ["task0", "task1", "task2"]

    job_ids = slurm.submit_array("test_array", script_path, log_paths, task_names)

    assert len(job_ids) == 3
    assert all(isinstance(jid, str) for jid in job_ids)

    # Check all jobs are tracked
    for job_id in job_ids:
        job = slurm.get_job(job_id)
        assert job.state == "PENDING"
        assert job.script_path == script_path


def test_real_slurm_submit_array(tmp_path: Path, monkeypatch) -> None:
    """Test that SlurmClient.submit_array correctly calls sbatch with --array."""

    calls = []

    class MockResult:
        returncode = 0
        stdout = "Submitted batch job 10000"
        stderr = ""

    def mock_subprocess_run(cmd, **kwargs):
        calls.append(cmd)
        return MockResult()

    monkeypatch.setattr(oellm_autoexp.slurm.client, "run_command", mock_subprocess_run)

    client = SlurmClient(SlurmClientConfig())
    slurm_config = SlurmConfig(
        template_path="templates/base.sbatch",
        script_dir=str(tmp_path / "scripts"),
        log_dir=str(tmp_path / "logs"),
        submit_cmd="sbatch",
    )
    client.configure(slurm_config)

    script_path = tmp_path / "array.sbatch"
    log_paths = [tmp_path / f"log_{i}.txt" for i in range(3)]
    task_names = ["task0", "task1", "task2"]

    job_ids = client.submit_array("test_array", script_path, log_paths, task_names)

    # Verify sbatch was called with --array flag
    assert len(calls) == 1
    assert calls[0][0] == "sbatch"
    assert "--array=0-2" in calls[0]
    assert str(script_path) in calls[0]

    # Verify job IDs were generated
    assert len(job_ids) == 3
    assert job_ids == ["10000_0", "10000_1", "10000_2"]

    # Verify jobs are tracked
    for idx, job_id in enumerate(job_ids):
        job = client.get_job(job_id)
        assert f"task{idx}" in job.name
        assert job.script_path == script_path

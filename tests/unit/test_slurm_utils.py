from pathlib import Path

import pytest

from oellm_autoexp.slurm.fake_sbatch import FakeSlurm
from oellm_autoexp.slurm.validator import SlurmValidationError, validate_job_script


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
    slurm = FakeSlurm()
    job_id = slurm.submit("demo", tmp_path / "job.sbatch", tmp_path / "log.txt")
    slurm.set_state(job_id, "RUNNING")
    assert slurm.squeue()[job_id] == "RUNNING"
    slurm.set_state(job_id, "COMPLETED", return_code=0)
    job = slurm.get_job(job_id)
    assert job.state == "COMPLETED"
    assert job.return_code == 0

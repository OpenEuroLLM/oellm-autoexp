"""Tests for SLURM client implementations."""

from pathlib import Path
from dataclasses import dataclass
from collections.abc import Callable

import pytest

from oellm_autoexp.slurm_gen import SlurmConfig
from oellm_autoexp.slurm_gen.client import (
    FakeSlurmClient,
    FakeSlurmClientConfig,
    SlurmClient,
    SlurmClientConfig,
    SlurmJob,
)
import oellm_autoexp.slurm_gen.client


@dataclass
class MockResult:
    """Mock result for run_command calls."""

    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


def make_mock_run(responses: dict[str, MockResult]) -> Callable:
    """Create a mock_run function that returns different responses based on
    command.

    Args:
        responses: Dict mapping command name (e.g., "sbatch") to MockResult.
    """
    calls = []

    def mock_run(cmd, **kwargs):
        calls.append(cmd)
        return responses.get(cmd[0], MockResult())

    mock_run.calls = calls
    return mock_run


def make_job_config(
    tmp_path: Path,
    *,
    name: str = "job",
    script_name: str = "job.sh",
    log_name: str = "job.log",
) -> SlurmConfig:
    config = SlurmConfig(
        template_path="templates/base.sbatch",
        script_dir=str(tmp_path / "scripts"),
        log_dir=str(tmp_path / "logs"),
    )
    config.name = name
    config.script_path = str(tmp_path / script_name)
    config.log_path = str(tmp_path / log_name)
    return config


@pytest.fixture
def slurm_config(tmp_path: Path) -> SlurmConfig:
    """Standard SlurmConfig for tests."""
    return make_job_config(tmp_path, name="default")


@pytest.fixture
def slurm_client_config() -> SlurmClientConfig:
    config = SlurmClientConfig(
        submit_cmd="sbatch",
        squeue_cmd="squeue",
        scancel_cmd="scancel",
        sacct_cmd="sacct",
    )
    return config


@pytest.fixture
def configured_client(tmp_path: Path, slurm_config: SlurmConfig) -> SlurmClient:
    """Pre-configured SlurmClient for tests."""
    return SlurmClient(SlurmClientConfig())


class TestSlurmJob:
    """Tests for SlurmJob dataclass."""

    def test_job_creation(self, tmp_path: Path):
        """Test basic job creation."""
        config = make_job_config(tmp_path, name="test_job")
        job = SlurmJob(job_id="12345", config=config)
        assert job.job_id == "12345"
        assert job.config == config
        assert job.state == "PENDING"
        assert job.return_code is None

    def test_job_with_state(self, tmp_path: Path):
        """Test job creation with custom state."""
        config = make_job_config(tmp_path, name="test_job")
        job = SlurmJob(job_id="12345", config=config, state="RUNNING", return_code=None)
        assert job.state == "RUNNING"


class TestFakeSlurmClient:
    """Tests for FakeSlurmClient."""

    def test_submit_job(self, tmp_path: Path):
        """Test submitting a single job."""
        client = FakeSlurmClient(FakeSlurmClientConfig())
        config = make_job_config(
            tmp_path, name="test_job", script_name="job.sbatch", log_name="log.txt"
        )
        job_id = client.submit(config)
        assert job_id == "1"
        assert client.squeue()[job_id] == "PENDING"

    def test_submit_multiple_jobs(self, tmp_path: Path):
        """Test submitting multiple jobs."""
        client = FakeSlurmClient(FakeSlurmClientConfig())
        job1 = client.submit(
            make_job_config(tmp_path, name="job1", script_name="j1.sh", log_name="l1.txt")
        )
        job2 = client.submit(
            make_job_config(tmp_path, name="job2", script_name="j2.sh", log_name="l2.txt")
        )
        assert job1 == "1"
        assert job2 == "2"
        assert len(client.squeue()) == 2

    def test_state_transitions(self, tmp_path: Path):
        """Test job state transitions."""
        client = FakeSlurmClient(FakeSlurmClientConfig())
        job_id = client.submit(
            make_job_config(tmp_path, name="demo", script_name="job.sbatch", log_name="log.txt")
        )

        client.set_state(job_id, "RUNNING")
        assert client.squeue()[job_id] == "RUNNING"

        client.set_state(job_id, "COMPLETED", return_code=0)
        job = client.get_job(job_id)
        assert job.state == "COMPLETED"
        assert job.return_code == 0

    def test_cancel_job(self, tmp_path: Path):
        """Test cancelling a job."""
        client = FakeSlurmClient(FakeSlurmClientConfig())
        job_id = client.submit(
            make_job_config(tmp_path, name="test", script_name="job.sh", log_name="log.txt")
        )
        client.cancel(job_id)
        assert client.squeue()[job_id] == "CANCELLED"

    def test_remove_job(self, tmp_path: Path):
        """Test removing a job from tracking."""
        client = FakeSlurmClient(FakeSlurmClientConfig())
        job_id = client.submit(
            make_job_config(tmp_path, name="test", script_name="job.sh", log_name="log.txt")
        )
        assert job_id in client.squeue()
        client.remove(job_id)
        assert job_id not in client.squeue()

    def test_job_ids_by_name(self, tmp_path: Path):
        """Test finding jobs by name."""
        client = FakeSlurmClient(FakeSlurmClientConfig())
        client.submit(make_job_config(tmp_path, name="job_a", script_name="a.sh", log_name="a.txt"))
        client.submit(make_job_config(tmp_path, name="job_b", script_name="b.sh", log_name="b.txt"))
        client.submit(
            make_job_config(tmp_path, name="job_a", script_name="a2.sh", log_name="a2.txt")
        )

        job_a_ids = client.job_ids_by_name("job_a")
        assert len(job_a_ids) == 2

    def test_submit_array(self, tmp_path: Path):
        """Test submitting a job array."""
        client = FakeSlurmClient(FakeSlurmClientConfig())
        config = make_job_config(
            tmp_path, name="test_array", script_name="array.sbatch", log_name="logs/array.log"
        )
        indices = [0, 1, 3]
        job_ids = client.submit_array(config, indices)

        assert len(job_ids) == 3
        assert all(isinstance(jid, str) for jid in job_ids)
        # Array job IDs should be like "1_0", "1_1", "1_3"
        assert job_ids == ["1_0", "1_1", "1_3"]

        # Check all jobs are tracked
        for job_id in job_ids:
            job = client.get_job(job_id)
            assert job.state == "PENDING"
            assert job.config.script_path == config.script_path

    def test_submit_array_with_start_index(self, tmp_path: Path):
        """Test submitting an array job with custom start index."""
        client = FakeSlurmClient(FakeSlurmClientConfig())
        config = make_job_config(
            tmp_path, name="test_array", script_name="array.sbatch", log_name="logs/array.log"
        )
        indices = [5, 6]
        job_ids = client.submit_array(config, indices)

        assert job_ids == ["1_5", "1_6"]

    def test_submit_array_with_non_contiguous_index(self, tmp_path: Path):
        """Test submitting an array job with custom start index."""
        client = FakeSlurmClient(FakeSlurmClientConfig())
        config = make_job_config(
            tmp_path, name="test_array", script_name="array.sbatch", log_name="logs/array.log"
        )
        indices = [4, 6]
        job_ids = client.submit_array(config, indices)

        assert job_ids == ["1_4", "1_6"]

    def test_submit_array_with_no_index(self, tmp_path: Path):
        """Test submitting an array job with custom start index."""
        client = FakeSlurmClient(FakeSlurmClientConfig())
        config = make_job_config(
            tmp_path, name="test_array", script_name="array.sbatch", log_name="logs/array.log"
        )
        with pytest.raises(ValueError):
            client.submit_array(config, [])

    def test_register_job(self, tmp_path: Path):
        """Test registering an external job."""
        client = FakeSlurmClient(FakeSlurmClientConfig())
        config = make_job_config(
            tmp_path,
            name="external_job",
            script_name="external.sh",
            log_name="external.log",
        )
        job_id = client.register_job("99999", config, state="RUNNING")
        assert job_id == "99999"
        job = client.get_job(job_id)
        assert job.state == "RUNNING"
        assert job.config.name == "external_job"

    def test_register_job_with_non_numeric_id(self, tmp_path: Path):
        """Test registering a job with a non-numeric ID."""
        client = FakeSlurmClient(FakeSlurmClientConfig())
        config = make_job_config(
            tmp_path,
            name="external_job",
            script_name="external.sh",
            log_name="external.log",
        )
        job_id = client.register_job("abc_xyz", config)
        assert job_id == "abc_xyz"
        # Next job should use _next_id (not crash on ValueError)
        next_job_id = client.submit(
            make_job_config(tmp_path, name="new_job", script_name="j.sh", log_name="l.txt")
        )
        assert next_job_id is not None


class TestSlurmClient:
    """Tests for SlurmClient (mocked)."""

    def test_submit_parses_job_id(self, configured_client, tmp_path, monkeypatch):
        """Test that submit correctly parses job ID from sbatch output."""
        mock_run = make_mock_run({"sbatch": MockResult(stdout="Submitted batch job 12345")})
        monkeypatch.setattr(oellm_autoexp.slurm_gen.client, "run_command", mock_run)

        config = make_job_config(tmp_path, name="test", script_name="job.sh", log_name="log.txt")
        job_id = configured_client.submit(config)
        assert job_id == "12345"

    def test_submit_handles_failure(self, configured_client, tmp_path, monkeypatch):
        """Test that submit raises on sbatch failure."""
        mock_run = make_mock_run(
            {"sbatch": MockResult(returncode=1, stderr="sbatch: error: invalid option")}
        )
        monkeypatch.setattr(oellm_autoexp.slurm_gen.client, "run_command", mock_run)

        with pytest.raises(RuntimeError, match="sbatch failed"):
            configured_client.submit(
                make_job_config(tmp_path, name="test", script_name="job.sh", log_name="log.txt")
            )

    def test_submit_array(self, configured_client, tmp_path, monkeypatch):
        """Test that submit_array correctly calls sbatch with --array."""
        mock_run = make_mock_run({"sbatch": MockResult(stdout="Submitted batch job 10000")})
        monkeypatch.setattr(oellm_autoexp.slurm_gen.client, "run_command", mock_run)

        config = make_job_config(
            tmp_path, name="test_array", script_name="array.sbatch", log_name="logs/array.log"
        )
        indices = [0, 1, 2]
        job_ids = configured_client.submit_array(config, indices)

        # Verify sbatch was called with --array flag
        assert len(mock_run.calls) == 1
        assert mock_run.calls[0][0] == "sbatch"
        assert "--array=0-2" in mock_run.calls[0]
        assert config.script_path in mock_run.calls[0]

        # Verify job IDs were generated
        assert job_ids == ["10000_0", "10000_1", "10000_2"]

        # Verify jobs are tracked
        for idx, job_id in enumerate(job_ids):
            job = configured_client.get_job(job_id)
            assert job.config.script_path == config.script_path

    def test_submit_noncontiguous_array(self, configured_client, tmp_path, monkeypatch):
        """Test that submit_array correctly calls sbatch with --array."""
        mock_run = make_mock_run({"sbatch": MockResult(stdout="Submitted batch job 10000")})
        monkeypatch.setattr(oellm_autoexp.slurm_gen.client, "run_command", mock_run)

        config = make_job_config(
            tmp_path, name="test_array", script_name="array.sbatch", log_name="logs/array.log"
        )
        indices = [0, 2]
        job_ids = configured_client.submit_array(config, indices)

        # Verify sbatch was called with --array flag
        assert len(mock_run.calls) == 1
        assert mock_run.calls[0][0] == "sbatch"
        assert "--array=0,2" in mock_run.calls[0]
        assert config.script_path in mock_run.calls[0]

        # Verify job IDs were generated
        assert job_ids == ["10000_0", "10000_2"]

    def test_cancel(self, configured_client, monkeypatch):
        """Test cancelling a job."""
        mock_run = make_mock_run({})
        monkeypatch.setattr(oellm_autoexp.slurm_gen.client, "run_command", mock_run)

        configured_client.cancel("12345")
        assert mock_run.calls[-1] == ["scancel", "12345"]

    def test_parse_job_id_variations(self):
        """Test parsing various sbatch output formats."""
        assert SlurmClient._parse_job_id("Submitted batch job 12345") == "12345"
        assert SlurmClient._parse_job_id("12345") == "12345"
        assert SlurmClient._parse_job_id("Job 12345_0 submitted") == "12345_0"
        assert SlurmClient._parse_job_id("No job id here") is None

    def test_submit_unable_to_parse_job_id(self, configured_client, tmp_path, monkeypatch):
        """Test that submit raises when job ID can't be parsed."""
        mock_run = make_mock_run({"sbatch": MockResult(stdout="Some output without job id")})
        monkeypatch.setattr(oellm_autoexp.slurm_gen.client, "run_command", mock_run)

        with pytest.raises(RuntimeError, match="Unable to parse job id"):
            configured_client.submit(
                make_job_config(tmp_path, name="test", script_name="job.sh", log_name="log.txt")
            )

    def test_submit_array_sbatch_failure(self, configured_client, tmp_path, monkeypatch):
        """Test that submit_array raises on sbatch failure."""
        mock_run = make_mock_run(
            {"sbatch": MockResult(returncode=1, stderr="sbatch: error: invalid option")}
        )
        monkeypatch.setattr(oellm_autoexp.slurm_gen.client, "run_command", mock_run)

        with pytest.raises(RuntimeError, match="sbatch failed for array"):
            configured_client.submit_array(
                make_job_config(
                    tmp_path, name="test_array", script_name="array.sbatch", log_name="array.log"
                ),
                [0],
            )

    def test_submit_array_parse_failure(self, configured_client, tmp_path, monkeypatch):
        """Test that submit_array raises when job ID can't be parsed."""
        mock_run = make_mock_run({})
        monkeypatch.setattr(oellm_autoexp.slurm_gen.client, "run_command", mock_run)

        with pytest.raises(ValueError, match="submit_array requires at least one task"):
            configured_client.submit_array(
                make_job_config(
                    tmp_path, name="test_array", script_name="array.sbatch", log_name="array.log"
                ),
                [],
            )

    def test_submit_array_index_failure(self, configured_client, tmp_path, monkeypatch):
        """Test that submit_array raises when job ID can't be parsed."""
        mock_run = make_mock_run({"sbatch": MockResult(stdout="No valid job id here")})
        monkeypatch.setattr(oellm_autoexp.slurm_gen.client, "run_command", mock_run)

        with pytest.raises(RuntimeError, match="Unable to parse job id"):
            configured_client.submit_array(
                make_job_config(
                    tmp_path, name="test_array", script_name="array.sbatch", log_name="array.log"
                ),
                [0],
            )

    def test_remove(self, configured_client, tmp_path, monkeypatch):
        """Test removing a job from tracking."""
        mock_run = make_mock_run({"sbatch": MockResult(stdout="Submitted batch job 12345")})
        monkeypatch.setattr(oellm_autoexp.slurm_gen.client, "run_command", mock_run)

        job_id = configured_client.submit(
            make_job_config(tmp_path, name="test", script_name="job.sh", log_name="log.txt")
        )
        assert job_id == "12345"

        configured_client.remove(job_id)
        with pytest.raises(KeyError):
            configured_client.get_job(job_id)

    def test_job_ids_by_name(self, configured_client, tmp_path, monkeypatch):
        """Test finding jobs by name."""
        call_count = [0]

        def mock_run(cmd, **kwargs):
            call_count[0] += 1
            return MockResult(stdout=f"Submitted batch job {10000 + call_count[0]}")

        monkeypatch.setattr(oellm_autoexp.slurm_gen.client, "run_command", mock_run)

        configured_client.submit(
            make_job_config(tmp_path, name="job_a", script_name="a.sh", log_name="a.txt")
        )
        configured_client.submit(
            make_job_config(tmp_path, name="job_b", script_name="b.sh", log_name="b.txt")
        )
        configured_client.submit(
            make_job_config(tmp_path, name="job_a", script_name="a2.sh", log_name="a2.txt")
        )

        job_a_ids = configured_client.job_ids_by_name("job_a")
        assert len(job_a_ids) == 2

    def test_register_job(self, configured_client, tmp_path):
        """Test registering an external job."""
        config = make_job_config(
            tmp_path,
            name="external_job",
            script_name="external.sh",
            log_name="external.log",
        )
        job_id = configured_client.register_job("99999", config, state="RUNNING")

        assert job_id == "99999"
        job = configured_client.get_job(job_id)
        assert job.state == "RUNNING"
        assert job.config.name == "external_job"

    def test_squeue_no_jobs(self, configured_client):
        """Test squeue with no tracked jobs."""
        statuses = configured_client.squeue()
        assert statuses == {}

    def test_squeue_with_jobs(self, configured_client, tmp_path, monkeypatch):
        """Test squeue with tracked jobs."""
        mock_run = make_mock_run(
            {
                "sbatch": MockResult(stdout="Submitted batch job 12345"),
                "squeue": MockResult(stdout="12345 RUNNING"),
            }
        )
        monkeypatch.setattr(oellm_autoexp.slurm_gen.client, "run_command", mock_run)

        job_id = configured_client.submit(
            make_job_config(tmp_path, name="test", script_name="job.sh", log_name="log.txt")
        )
        statuses = configured_client.squeue()

        assert job_id in statuses
        assert statuses[job_id] == "RUNNING"

    def test_squeue_failed_falls_back_to_sacct(self, configured_client, tmp_path, monkeypatch):
        """Test squeue falls back to sacct when squeue fails."""
        mock_run = make_mock_run(
            {
                "sbatch": MockResult(stdout="Submitted batch job 12345"),
                "squeue": MockResult(returncode=1, stderr="Invalid job id"),
                "sacct": MockResult(stdout="12345|COMPLETED"),
            }
        )
        monkeypatch.setattr(oellm_autoexp.slurm_gen.client, "run_command", mock_run)

        job_id = configured_client.submit(
            make_job_config(tmp_path, name="test", script_name="job.sh", log_name="log.txt")
        )
        statuses = configured_client.squeue()

        assert job_id in statuses
        assert statuses[job_id] == "COMPLETED"

    def test_squeue_jobs_missing_from_queue_check_sacct(
        self, configured_client, tmp_path, monkeypatch
    ):
        """Test squeue checks sacct for jobs not in queue."""
        mock_run = make_mock_run(
            {
                "sbatch": MockResult(stdout="Submitted batch job 12345"),
                "squeue": MockResult(stdout=""),
                "sacct": MockResult(stdout="12345|FAILED"),
            }
        )
        monkeypatch.setattr(oellm_autoexp.slurm_gen.client, "run_command", mock_run)

        job_id = configured_client.submit(
            make_job_config(tmp_path, name="test", script_name="job.sh", log_name="log.txt")
        )
        statuses = configured_client.squeue()

        assert job_id in statuses
        assert statuses[job_id] == "FAILED"

    def test_squeue_sacct_returns_various_states(self, configured_client, tmp_path, monkeypatch):
        """Test sacct parsing for various job states."""
        mock_run = make_mock_run(
            {
                "sbatch": MockResult(stdout="Submitted batch job 10000"),
                "squeue": MockResult(stdout=""),
                "sacct": MockResult(
                    stdout="10000_0|CANCELLED by 12345\n10000_1|COMPLETED\n10000_2|FAILED\n10000_3|TIMEOUT"
                ),
            }
        )
        monkeypatch.setattr(oellm_autoexp.slurm_gen.client, "run_command", mock_run)

        config = make_job_config(
            tmp_path, name="test_array", script_name="array.sbatch", log_name="array.log"
        )
        indices = [0, 1, 2, 3]
        configured_client.submit_array(config, indices)

        statuses = configured_client.squeue()

        assert statuses.get("10000_0") == "CANCELLED"
        assert statuses.get("10000_1") == "COMPLETED"
        assert statuses.get("10000_2") == "FAILED"
        assert statuses.get("10000_3") == "TIMEOUT"

    def test_squeue_sacct_failure(self, configured_client, tmp_path, monkeypatch):
        """Test squeue when sacct also fails."""
        mock_run = make_mock_run(
            {
                "sbatch": MockResult(stdout="Submitted batch job 12345"),
                "squeue": MockResult(returncode=1, stderr="Error"),
                "sacct": MockResult(returncode=1, stderr="sacct error"),
            }
        )
        monkeypatch.setattr(oellm_autoexp.slurm_gen.client, "run_command", mock_run)

        configured_client.submit(
            make_job_config(tmp_path, name="test", script_name="job.sh", log_name="log.txt")
        )
        statuses = configured_client.squeue()

        assert statuses == {}

    def test_check_sacct_for_missing_jobs_empty_list(self, configured_client):
        """Test _check_sacct_for_missing_jobs with empty list."""
        result = configured_client._check_sacct_for_missing_jobs([], {})
        assert result == {}

    def test_squeue_with_empty_and_malformed_lines(self, configured_client, tmp_path, monkeypatch):
        """Test squeue handles empty lines and lines without state."""
        mock_run = make_mock_run(
            {
                "sbatch": MockResult(stdout="Submitted batch job 12345"),
                "squeue": MockResult(stdout="12345 RUNNING\n\n99999\n"),
            }
        )
        monkeypatch.setattr(oellm_autoexp.slurm_gen.client, "run_command", mock_run)

        job_id = configured_client.submit(
            make_job_config(tmp_path, name="test", script_name="job.sh", log_name="log.txt")
        )
        statuses = configured_client.squeue()

        assert statuses[job_id] == "RUNNING"

    def test_squeue_unknown_job_id_logged(self, configured_client, tmp_path, monkeypatch):
        """Test squeue logs warning for unknown job IDs."""
        mock_run = make_mock_run(
            {
                "sbatch": MockResult(stdout="Submitted batch job 12345"),
                "squeue": MockResult(stdout="12345 RUNNING\n99999 PENDING"),
            }
        )
        monkeypatch.setattr(oellm_autoexp.slurm_gen.client, "run_command", mock_run)

        job_id = configured_client.submit(
            make_job_config(tmp_path, name="test", script_name="job.sh", log_name="log.txt")
        )
        statuses = configured_client.squeue()

        assert job_id in statuses
        assert "99999" not in statuses

    def test_sacct_with_empty_and_malformed_lines(self, configured_client, tmp_path, monkeypatch):
        """Test sacct handles empty lines and lines with less than 2 parts."""
        mock_run = make_mock_run(
            {
                "sbatch": MockResult(stdout="Submitted batch job 12345"),
                "squeue": MockResult(stdout=""),
                "sacct": MockResult(
                    stdout="12345|COMPLETED\n\nmalformed_line_no_pipe\n99999.batch|COMPLETED"
                ),
            }
        )
        monkeypatch.setattr(oellm_autoexp.slurm_gen.client, "run_command", mock_run)

        job_id = configured_client.submit(
            make_job_config(tmp_path, name="test", script_name="job.sh", log_name="log.txt")
        )
        statuses = configured_client.squeue()

        assert job_id in statuses
        assert statuses[job_id] == "COMPLETED"


class TestSlurmClientConfiguration:
    """Tests for SLURM client configuration."""

    def test_client_has_config(self):
        """Test that client exposes its config."""
        cfg = SlurmClientConfig()
        client = SlurmClient(cfg)
        assert client.config is cfg

    def test_fake_client_has_config(self):
        """Test that fake client exposes its config."""
        cfg = FakeSlurmClientConfig()
        client = FakeSlurmClient(cfg)
        assert client.config is cfg

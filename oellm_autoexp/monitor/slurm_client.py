"""SLURM job client adapter that uses the external slurm_gen library."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from compoconf import ConfigInterface, register
from oellm_autoexp.slurm_gen import generate_script, validate_job_script
from oellm_autoexp.slurm_gen.client import (
    BaseSlurmClient,
    SlurmClientConfig as SGClientConfig,
)

from oellm_autoexp.monitor.job_client_protocol import JobClientInterface
from oellm_autoexp.monitor.submission import SlurmJobConfig
from oellm_autoexp.monitor.utils.paths import expand_log_path, update_log_symlink

LOGGER = logging.getLogger(__name__)


@dataclass(kw_only=True)
class SlurmClientConfig(ConfigInterface):
    base_client: BaseSlurmClient.cfgtype = field(default_factory=SGClientConfig)


@register
class SlurmClient(JobClientInterface):
    """Execute and monitor local bash commands as background processes.

    This client implements the JobClientProtocol to allow monitor to track
    local scripts/commands without requiring SLURM. Jobs are started as
    background processes with output redirected to log files.

    Example:
        >>> client = LocalCommandClient()
        >>> job_id = client.submit("my-job", "./train.sh", "./train.log")
        >>> statuses = client.squeue()
        >>> print(statuses[job_id])  # "RUNNING", "COMPLETED", etc.

    Note:
        - Processes are started in a new session (detached from terminal)
        - Job IDs are simple incrementing integers
        - No array job support for start_index parameter
    """

    config: SlurmClientConfig

    def __init__(self, config: SlurmClientConfig | None = None) -> None:
        self.config = config or SlurmClientConfig()
        self._client = config.base_client.instantiate(BaseSlurmClient)

    def submit(
        self,
        job: SlurmJobConfig,
    ) -> str:
        """Submit a local script as a background process.

        Args:
            name: Human-readable job name
            command: Command to execute
            log_path: Path where stdout/stderr will be written

        Returns:
            job_id: String identifier for this job

        Raises:
            FileNotFoundError: If command[0] doesn't exist
            OSError: If process creation fails
        """
        generate_script(job.slurm)
        validate_job_script(
            job.slurm.script_path,
            job.slurm.name,
        )
        job_id = self._client.submit(job.slurm)
        if job.log_path_current:
            update_log_symlink(expand_log_path(job.log_path, job_id), Path(job.log_path_current))
        return job_id

    def submit_array(
        self,
        job: SlurmJobConfig,
        indices: list[int],
    ) -> list[str]:
        """Submit multiple instances of a script.

        Each task is submitted as an independent process. The script receives
        environment variables TASK_ID (0-indexed).

        Args:
            array_name: Base name for the job array
            command: Command to execute for each task
            log_paths: Log paths, one per task

        Returns:
            List of job_ids for submitted tasks
        """
        generate_script(job.slurm)
        validate_job_script(job.slurm.name)
        validate_job_script()
        job_ids = self._client.submit_array(job.slurm, indices)
        if job.log_path_current:
            for job_id in job_ids:
                update_log_symlink(
                    expand_log_path(job.log_path, job_id),
                    Path(job.log_path_current.replace("%a", job_id.split("_")[-1])),
                )
        return job_ids

    def cancel(self, job_id: str) -> None:
        """Terminate a running process.

        Sends SIGTERM, waits up to 5 seconds, then sends SIGKILL if needed.

        Args:
            job_id: Job to cancel
        """
        return self._client.cancel(job_id)

    def remove(self, job_id: str) -> None:
        """Remove a job from tracking.

        Args:
            job_id: Job to remove
        """
        return

    def squeue(self) -> dict[str, str]:
        """Get current status of all tracked jobs.

        Returns:
            Dictionary of job_id -> status
            Statuses: "RUNNING", "COMPLETED", "FAILED", "CANCELLED"
        """
        return self._client.squeue()


__all__ = ["BaseSlurmClient", "SlurmClient", "SlurmClientConfig"]

"""Protocol defining the job client interface required by monitor.

This protocol allows monitor to work with different job execution backends:
- SLURM (via slurm_gen.client and monitor's SLURM adapter)
- Local processes (via LocalCommandClient)
- Other batch systems (PBS, LSF, etc.)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from compoconf import RegistrableConfigInterface, register_interface
from oellm_autoexp.monitor.submission import BaseJob


@register_interface
class JobClientInterface(RegistrableConfigInterface):
    """Registrable interface for job execution backends."""


@runtime_checkable
class JobClientProtocol(Protocol):
    """Protocol for job submission and management.

    Any class implementing these methods can be used with MonitorLoop,
    allowing monitor to work with SLURM, local processes, or other batch
    systems.
    """

    def submit(
        self,
        job: BaseJob,
    ) -> str:  # pragma: no cover
        """Submit a single job.

        Args:
            name: Human-readable job name
        command: Command to execute (first element is the executable or script)
            log_path: Path where job output should be logged
            log_path_current: Optional stable log path or symlink target
            slurm: Optional slurm_gen configuration payload for script generation

        Returns:
            job_id: Unique identifier for the submitted job

        Raises:
            RuntimeError: If submission fails
        """
        ...

    def submit_array(
        self,
        job: BaseJob,
        indices: list[int],
    ) -> list[str]:  # pragma: no cover
        """Submit an array of jobs (multiple instances of same script).

        Args:
            array_name: Base name for the job array
        command: Command to execute for each task
            log_paths: List of log paths, one per task
            task_names: List of task names, one per task
            log_path_current: Optional stable log path or symlink target
            slurm: Optional slurm_gen configuration payload for script generation

        Returns:
            List of job_ids, one per submitted task

        Raises:
            RuntimeError: If submission fails
        """
        ...

    def cancel(self, job_id: str) -> None:  # pragma: no cover
        """Cancel/terminate a running or pending job.

        Args:
            job_id: Job identifier returned by submit()
        """
        ...

    def remove(self, job_id: str) -> None:  # pragma: no cover
        """Remove a job from tracking.

        This is used to clean up finished jobs from the client's internal state.

        Args:
            job_id: Job identifier to remove
        """
        ...

    def squeue(self) -> dict[str, str]:  # pragma: no cover
        """Query the status of all tracked jobs.

        Returns:
            Dictionary mapping job_id -> status string
            Common statuses: "PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"
        """
        ...


__all__ = ["JobClientProtocol", "JobClientInterface"]

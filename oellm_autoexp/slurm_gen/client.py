"""SLURM client implementations for job submission and management."""

from __future__ import annotations

import logging
import re
import shlex
import time
from dataclasses import MISSING, dataclass, field

from compoconf import ConfigInterface, register, register_interface, RegistrableConfigInterface

from oellm_autoexp.slurm_gen.schema import SlurmClientInterface, SlurmConfig
from oellm_autoexp.slurm_gen.shell import run_command

LOGGER = logging.getLogger(__name__)


@dataclass(kw_only=True)
class SlurmJob:
    """Metadata tracked for submitted SLURM jobs.

    Attributes:
        job_id: SLURM job ID.
        name: Job name.
        script_path: Path to the job script.
        log_path: Path to the log file.
        state: Current job state.
        return_code: Exit code if completed.
        submitted_at: Timestamp when submitted.
    """

    job_id: str = field(default_factory=MISSING)
    config: SlurmConfig = field(default_factory=MISSING)
    state: str = "PENDING"
    return_code: int | None = None
    submitted_at: float = field(default_factory=time.time)


@register_interface
class BaseSlurmClient(SlurmClientInterface, RegistrableConfigInterface):
    """Base functionality shared by SLURM client implementations.

    Subclasses must implement:
        - submit(): Submit a single job
        - submit_array(): Submit an array job
        - cancel(): Cancel a job
        - remove(): Remove job from tracking
        - squeue(): Query job statuses
        - job_ids_by_name(): Find jobs by name
        - get_job(): Get job metadata
    """

    config: ConfigInterface

    def __init__(self, config: ConfigInterface) -> None:
        self.config = config

    def submit(self, slurm_config: SlurmConfig) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def submit_array(
        self,
        slurm_config: SlurmConfig,
        indices: list[int],
    ) -> list[str]:  # pragma: no cover - interface
        raise NotImplementedError

    def cancel(self, job_id: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def remove(self, job_id: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def squeue(self) -> dict[str, str]:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass(kw_only=True)
class BaseSlurmClientConfig(ConfigInterface):
    """Shared configuration fields for SLURM clients."""


@dataclass(kw_only=True)
class FakeSlurmClientConfig(BaseSlurmClientConfig):
    """Configuration for the fake SLURM client."""


@register
class FakeSlurmClient(BaseSlurmClient):
    """In-memory SLURM simulator for testing.

    This client simulates SLURM behavior without actually submitting
    jobs. Useful for unit tests and development without a SLURM cluster.
    """

    config: FakeSlurmClientConfig

    def __init__(self, config: FakeSlurmClientConfig) -> None:
        super().__init__(config)
        self._jobs: dict[str, SlurmJob] = {}
        self._next_id = 1

    def submit(self, slurm_config: SlurmConfig) -> str:
        job_id = str(self._next_id)
        self._next_id += 1
        job = SlurmJob(job_id=job_id, config=slurm_config)
        job.state = "PENDING"
        self._jobs[job_id] = job

        return job_id

    def submit_array(
        self,
        slurm_config: SlurmConfig,
        indices: list[int],
    ) -> list[str]:
        job_ids: list[str] = []
        if not indices:
            raise ValueError("submit_array requires at least one task")
        base_job_id = str(self._next_id)
        self._next_id += 1

        for array_idx in indices:
            task_job_id = f"{base_job_id}_{array_idx}"

            job = SlurmJob(
                job_id=task_job_id,
                config=slurm_config,
            )
            job.state = "PENDING"
            self._jobs[task_job_id] = job

            job_ids.append(task_job_id)

        return job_ids

    def cancel(self, job_id: str) -> None:
        if job_id in self._jobs:
            self._jobs[job_id].state = "CANCELLED"

    def remove(self, job_id: str) -> None:
        self._jobs.pop(job_id, None)

    def squeue(self) -> dict[str, str]:
        return {job_id: job.state for job_id, job in self._jobs.items()}

    def get_job(self, job_id: str) -> SlurmJob:
        return self._jobs[job_id]

    def job_ids_by_name(self, name: str) -> list[str]:
        return [job_id for job_id, job in self._jobs.items() if job.config.name == name]

    def set_state(self, job_id: str, state: str, return_code: int | None = None) -> None:
        """Set the state of a job (for testing)."""
        job = self._jobs[job_id]
        job.state = state
        if return_code is not None:
            job.return_code = return_code

    def register_job(
        self,
        job_id: str,
        slurm_config: SlurmConfig,
        state: str = "PENDING",
    ) -> str:
        """Register an externally submitted job for tracking."""
        job = SlurmJob(
            job_id=job_id,
            config=slurm_config,
            state=state,
        )
        self._jobs[job_id] = job
        try:
            base_id = int(str(job_id).split("_")[0])
        except ValueError:
            base_id = self._next_id
        self._next_id = max(self._next_id, base_id + 1)
        return job_id


@dataclass(kw_only=True)
class SlurmClientConfig(BaseSlurmClientConfig):
    """Configuration for the real SLURM client."""

    class_name: str = "SlurmClient"
    submit_cmd: str = "sbatch"
    squeue_cmd: str = "squeue"
    scancel_cmd: str = "scancel"
    sacct_cmd: str = "sacct"


@register
class SlurmClient(BaseSlurmClient):
    """SLURM client that executes real SLURM commands.

    This client shells out to sbatch, squeue, scancel, and sacct to
    interact with a real SLURM cluster.
    """

    config: SlurmClientConfig

    def __init__(self, config: SlurmClientConfig) -> None:
        super().__init__(config)
        self._jobs: dict[str, SlurmJob] = {}

    def submit(self, slurm_config: SlurmConfig) -> str:
        submit_cmd = shlex.split(self.config.submit_cmd)
        proc = run_command([*submit_cmd, str(slurm_config.script_path)])
        if proc.returncode != 0:
            raise RuntimeError(
                f"sbatch failed for {slurm_config.script_path}: {proc.stderr.strip()}"
            )
        job_id = self._parse_job_id(proc.stdout)
        if job_id is None:
            raise RuntimeError(f"Unable to parse job id from sbatch output: {proc.stdout.strip()}")
        self._jobs[job_id] = SlurmJob(job_id=job_id, config=slurm_config)
        return job_id

    def submit_array(
        self,
        slurm_config: SlurmConfig,
        indices: list[int],
    ) -> list[str]:
        """Submit a SLURM job array.

        Args:
            slurm_config: Job configuration for the array submission.
            indices: Array indices to submit.

        Returns:
            List of job IDs (one per task).
        """

        submit_cmd = shlex.split(self.config.submit_cmd)
        if not indices:
            raise ValueError("submit_array requires at least one task")
        sorted_indices = sorted(set(indices))
        if sorted_indices == list(range(sorted_indices[0], sorted_indices[-1] + 1)):
            array_range = f"{sorted_indices[0]}-{sorted_indices[-1]}"
        else:
            array_range = ",".join(str(idx) for idx in sorted_indices)
        proc = run_command([*submit_cmd, f"--array={array_range}", str(slurm_config.script_path)])

        if proc.returncode != 0:
            raise RuntimeError(
                f"sbatch failed for array {slurm_config.script_path}: {proc.stderr.strip()}"
            )

        base_job_id = self._parse_job_id(proc.stdout)
        if base_job_id is None:
            raise RuntimeError(f"Unable to parse job id from sbatch output: {proc.stdout.strip()}")

        LOGGER.info(
            "submit_array: submitted array job with base_id=%s, num_tasks=%d, indices=%s",
            base_job_id,
            len(indices),
            indices,
        )

        job_ids: list[str] = []
        for array_idx in indices:
            task_job_id = f"{base_job_id}_{array_idx}"

            job = SlurmJob(
                job_id=task_job_id,
                config=slurm_config,
            )
            self._jobs[task_job_id] = job
            job_ids.append(task_job_id)
            LOGGER.debug(
                "submit_array: registered array_idx=%s: synthetic_id=%s, real_id=%s_%s",
                array_idx,
                task_job_id,
                base_job_id,
                array_idx,
            )

        return job_ids

    def cancel(self, job_id: str) -> None:
        cmd = shlex.split(self.config.scancel_cmd)
        LOGGER.debug(f"cancel: cancelling job_id={job_id}")
        run_command([*cmd, str(job_id)])

    def remove(self, job_id: str) -> None:
        self._jobs.pop(job_id, None)

    def squeue(self) -> dict[str, str]:
        if not self._jobs:
            LOGGER.debug("squeue: no jobs tracked, returning empty")
            return {}

        LOGGER.info(f"squeue: tracking {len(self._jobs)} jobs: {list(self._jobs.keys())}")

        cmd = shlex.split(self.config.squeue_cmd)
        format_arg = ["--noheader", "--format", "%i %T"]

        job_ids = list(self._jobs.keys())
        job_id_to_key = {str(jid): jid for jid in job_ids}

        job_ids_str = ",".join(str(jid) for jid in job_ids)
        full_cmd = [*cmd, "--jobs", job_ids_str, *format_arg]
        LOGGER.info(f"squeue: querying jobs {job_ids_str} with command: {' '.join(full_cmd)}")

        proc = run_command(full_cmd)
        if proc.returncode != 0:
            LOGGER.warning(
                f"squeue: command failed with rc={proc.returncode}, stderr={proc.stderr}"
            )
            return self._check_sacct_for_missing_jobs(job_ids, job_id_to_key)

        LOGGER.debug(f"squeue: output: {proc.stdout.strip()}")
        statuses: dict[str, str] = {}

        for line in proc.stdout.strip().splitlines():
            parts = line.strip().split(None, 1)
            if not parts:
                continue

            slurm_id = parts[0]
            state = parts[1] if len(parts) > 1 else "UNKNOWN"

            if slurm_id in job_id_to_key:
                statuses[job_id_to_key[slurm_id]] = state
            else:
                LOGGER.warning(f"squeue: received unknown job ID {slurm_id}")

        missing_jobs = [jid for jid in job_ids if jid not in statuses]
        if missing_jobs:
            LOGGER.info(
                f"squeue: {len(missing_jobs)} jobs not in queue, checking sacct for recent completion"
            )
            missing_id_to_key = {str(jid): jid for jid in missing_jobs}
            sacct_statuses = self._check_sacct_for_missing_jobs(missing_jobs, missing_id_to_key)
            statuses.update(sacct_statuses)

        LOGGER.debug(f"squeue: parsed statuses: {statuses}")
        return statuses

    def job_ids_by_name(self, name: str) -> list[str]:
        return [job_id for job_id, job in self._jobs.items() if job.config.name == name]

    def _check_sacct_for_missing_jobs(
        self, job_ids: list[str], job_id_to_key: dict[str, str]
    ) -> dict[str, str]:
        """Check sacct for jobs that are no longer in squeue."""
        if not job_ids:
            return {}

        sacct_cmd = shlex.split(self.config.sacct_cmd)

        job_ids_str = ",".join(str(jid) for jid in job_ids)
        full_cmd = [
            *sacct_cmd,
            "--jobs",
            job_ids_str,
            "--noheader",
            "--format",
            "JobID,State",
            "--parsable2",
        ]

        LOGGER.info(f"sacct: checking for {len(job_ids)} missing jobs: {' '.join(full_cmd)}")

        proc = run_command(full_cmd)
        if proc.returncode != 0:
            LOGGER.warning(f"sacct: command failed with rc={proc.returncode}, stderr={proc.stderr}")
            return {}

        LOGGER.debug(f"sacct: output: {proc.stdout.strip()}")
        statuses: dict[str, str] = {}

        for line in proc.stdout.strip().splitlines():
            if not line.strip():
                continue

            parts = line.strip().split("|")
            if len(parts) < 2:
                continue

            slurm_id = parts[0].strip()
            state = parts[1].strip()

            if "CANCELLED" in state:
                state = "CANCELLED"
            elif "COMPLETED" in state:
                state = "COMPLETED"
            elif "FAILED" in state:
                state = "FAILED"
            elif "TIMEOUT" in state:
                state = "TIMEOUT"

            if slurm_id in job_id_to_key:
                statuses[job_id_to_key[slurm_id]] = state
                LOGGER.info(f"sacct: found job {slurm_id} in state {state}")
            else:
                LOGGER.debug(f"sacct: ignoring irrelevant job {slurm_id}")

        return statuses

    def get_job(self, job_id: str) -> SlurmJob:
        return self._jobs[job_id]

    def register_job(
        self,
        job_id: str,
        slurm_config: SlurmConfig,
        state: str = "PENDING",
    ) -> str:
        """Register an externally submitted job for tracking."""
        job = SlurmJob(
            job_id=job_id,
            config=slurm_config,
            state=state,
        )
        self._jobs[job_id] = job
        return job_id

    @staticmethod
    def _parse_job_id(output: str) -> str | None:
        """Parse job ID from sbatch output."""
        for token in output.split():
            if re.match(r"\d+(?:_\d+)?", token):
                return token
        return None


# Backward compatibility alias
FakeSlurm = FakeSlurmClient


__all__ = [
    "BaseSlurmClient",
    "BaseSlurmClientConfig",
    "FakeSlurmClient",
    "FakeSlurmClientConfig",
    "SlurmClient",
    "SlurmClientConfig",
    "SlurmJob",
    "FakeSlurm",
]

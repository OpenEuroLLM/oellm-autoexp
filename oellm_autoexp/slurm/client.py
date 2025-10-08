"""SLURM client implementations used by oellm_autoexp."""

from __future__ import annotations

import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from compoconf import ConfigInterface, register

from oellm_autoexp.config.schema import SlurmClientInterface, SlurmConfig
from oellm_autoexp.utils.shell import run_command


@dataclass
class SlurmJob:
    """Metadata tracked for submitted SLURM jobs."""

    job_id: str
    name: str
    script_path: Path
    log_path: Path
    state: str = "PENDING"
    return_code: Optional[int] = None
    submitted_at: float = field(default_factory=time.time)


class BaseSlurmClient(SlurmClientInterface):
    """Base functionality shared by SLURM client implementations."""

    config: ConfigInterface
    supports_array: bool = False

    def __init__(self, config: ConfigInterface) -> None:
        self.config = config
        self._slurm_config: Optional[SlurmConfig] = None

    def configure(self, slurm_config: SlurmConfig) -> None:
        self._slurm_config = slurm_config

    def submit(self, name: str, script_path: Path, log_path: Path) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    def submit_array(
        self,
        array_name: str,
        script_path: Path,
        log_paths: List[Path],
        task_names: List[str],
    ) -> List[int]:  # pragma: no cover - interface
        raise NotImplementedError

    def cancel(self, job_id: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def remove(self, job_id: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def squeue(self) -> Dict[int, str]:  # pragma: no cover - interface
        raise NotImplementedError

    def job_ids_by_name(self, name: str) -> List[int]:  # pragma: no cover - interface
        raise NotImplementedError

    def get_job(self, job_id: str) -> SlurmJob:  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def slurm_config(self) -> SlurmConfig:
        if self._slurm_config is None:
            raise RuntimeError("SLURM client has not been configured with SlurmConfig")
        return self._slurm_config


@dataclass
class BaseSlurmClientConfig(ConfigInterface):
    """Shared configuration fields for SLURM clients."""

    class_name: str
    persist_artifacts: bool = False


@dataclass
class FakeSlurmClientConfig(BaseSlurmClientConfig):
    class_name: str = "FakeSlurmClient"
    persist_artifacts: bool = False


@register
class FakeSlurmClient(BaseSlurmClient):
    """In-memory SLURM simulator closely mirroring real client behaviour."""

    config: FakeSlurmClientConfig
    supports_array = True

    def __init__(self, config: FakeSlurmClientConfig) -> None:
        super().__init__(config)
        self._jobs: Dict[int, SlurmJob] = {}
        self._next_id = 1

    def submit(self, name: str, script_path: Path, log_path: Path) -> str:
        job_id = str(self._next_id)
        self._next_id += 1
        job = SlurmJob(job_id=job_id, name=name, script_path=Path(script_path), log_path=Path(log_path))
        job.state = "PENDING"
        self._jobs[job_id] = job
        if self.config.persist_artifacts:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.touch(exist_ok=True)
        return job_id

    def submit_array(
        self,
        array_name: str,
        script_path: Path,
        log_paths: List[Path],
        task_names: List[str],
        start_index: int = 0,
    ) -> List[str]:
        job_ids: List[str] = []
        base_id = str(self._next_id)
        self._next_id += 1

        for offset, (log_path, task_name) in enumerate(zip(log_paths, task_names)):
            array_idx = start_index + offset
            task_job_id = f"{base_id}_{array_idx}"
            job_name = f"{array_name}_{task_name}"

            job = SlurmJob(
                job_id=task_job_id,
                name=job_name,
                script_path=Path(script_path),
                log_path=Path(log_path),
            )
            job.state = "PENDING"
            self._jobs[task_job_id] = job

            if self.config.persist_artifacts:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.touch(exist_ok=True)

            job_ids.append(task_job_id)

        return job_ids

    def cancel(self, job_id: str) -> None:
        if job_id in self._jobs:
            self._jobs[job_id].state = "CANCELLED"

    def remove(self, job_id: str) -> None:
        self._jobs.pop(job_id, None)

    def squeue(self) -> Dict[str, str]:
        return {job_id: job.state for job_id, job in self._jobs.items()}

    def job_ids_by_name(self, name: str) -> List[int]:
        return [job_id for job_id, job in self._jobs.items() if job.name == name]

    def get_job(self, job_id: str) -> SlurmJob:
        return self._jobs[job_id]

    def set_state(self, job_id: str, state: str, return_code: Optional[int] = None) -> None:
        job = self._jobs[job_id]
        job.state = state
        if return_code is not None:
            job.return_code = return_code

    def register_job(
        self,
        job_id: str,
        name: str,
        script_path: Path,
        log_path: Path,
        state: str = "PENDING",
    ) -> int:
        job = SlurmJob(job_id=job_id, name=name, script_path=Path(script_path), log_path=Path(log_path), state=state)
        self._jobs[job_id] = job
        self._next_id = max(self._next_id, int(job_id.split("_")[0]) + 1)
        return job_id


@dataclass
class SlurmClientConfig(BaseSlurmClientConfig):
    class_name: str = "SlurmClient"


@register
class SlurmClient(BaseSlurmClient):
    """SLURM client that shells out to the system commands."""

    config: SlurmClientConfig
    supports_array = True

    def __init__(self, config: SlurmClientConfig) -> None:
        super().__init__(config)
        self._jobs: Dict[int, SlurmJob] = {}

    def submit(self, name: str, script_path: Path, log_path: Path) -> int:
        slurm_conf = self.slurm_config
        submit_cmd = shlex.split(slurm_conf.submit_cmd)
        proc = run_command([*submit_cmd, str(script_path)])
        if proc.returncode != 0:
            raise RuntimeError(f"sbatch failed for {script_path}: {proc.stderr.strip()}")
        job_id = self._parse_job_id(proc.stdout)
        if job_id is None:
            raise RuntimeError(f"Unable to parse job id from sbatch output: {proc.stdout.strip()}")
        if self.config.persist_artifacts:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.touch(exist_ok=True)
        self._jobs[job_id] = SlurmJob(job_id=job_id, name=name, script_path=Path(script_path), log_path=Path(log_path))
        return job_id

    def submit_array(
        self,
        array_name: str,
        script_path: Path,
        log_paths: List[Path],
        task_names: List[str],
        start_index: int = 0,
    ) -> List[int]:
        """Submit a SLURM job array.

        Args:
            array_name: Base name for the array job
            script_path: Path to the array script
            log_paths: List of log paths for each task
            task_names: List of task names
            start_index: Starting array index (default 0, used for restarting specific tasks)

        Returns:
            List of job IDs (one per task) - these are synthetic IDs for tracking
        """
        import logging

        logger = logging.getLogger(__name__)

        slurm_conf = self.slurm_config
        num_tasks = len(task_names)

        # Submit the array job with --array=start_index-(start_index+num_tasks-1)
        submit_cmd = shlex.split(slurm_conf.submit_cmd)
        array_range = f"{start_index}-{start_index + num_tasks - 1}"
        proc = run_command([*submit_cmd, f"--array={array_range}", str(script_path)])

        if proc.returncode != 0:
            raise RuntimeError(f"sbatch failed for array {script_path}: {proc.stderr.strip()}")

        # Parse the job ID from output like "Submitted batch job 12345"
        base_job_id = self._parse_job_id(proc.stdout)
        if base_job_id is None:
            raise RuntimeError(f"Unable to parse job id from sbatch output: {proc.stdout.strip()}")

        logger.info(
            f"submit_array: submitted array job with base_id={base_job_id}, num_tasks={num_tasks}, "
            f"start_index={start_index}"
        )

        # SLURM array jobs have IDs like 12345_0, 12345_1
        # The array index corresponds to start_index + position in the list
        job_ids: List[str] = []
        for offset, (log_path, task_name) in enumerate(zip(log_paths, task_names)):
            array_idx = start_index + offset
            task_job_id = f"{base_job_id}_{array_idx}"
            job_name = f"{array_name}_{task_name}"

            if self.config.persist_artifacts:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.touch(exist_ok=True)

            job = SlurmJob(
                job_id=task_job_id,
                name=job_name,
                script_path=Path(script_path),
                log_path=Path(log_path),
            )
            self._jobs[task_job_id] = job
            job_ids.append(task_job_id)
            logger.debug(
                f"submit_array: registered task offset={offset}, array_idx={array_idx}: "
                f"synthetic_id={task_job_id}, real_id={base_job_id}_{array_idx}"
            )

        return job_ids

    def cancel(self, job_id: str) -> None:
        import logging

        logger = logging.getLogger(__name__)

        slurm_conf = self.slurm_config
        cmd = shlex.split(getattr(slurm_conf, "cancel_cmd", "scancel"))

        # Check if this is an array job and convert to real SLURM ID
        if hasattr(self, "_array_jobs") and job_id in self._array_jobs:
            base_id, array_idx = self._array_jobs[job_id]
            real_job_id = f"{base_id}_{array_idx}"
            logger.debug(f"cancel: converting synthetic_id={job_id} to real_id={real_job_id}")
            run_command([*cmd, real_job_id])
        else:
            logger.debug(f"cancel: using job_id={job_id} as-is")
            run_command([*cmd, str(job_id)])

    def remove(self, job_id: str) -> None:
        self._jobs.pop(job_id, None)

    def squeue(self) -> Dict[int, str]:
        import logging

        logger = logging.getLogger(__name__)

        if not self._jobs:
            logger.debug("squeue: no jobs tracked, returning empty")
            return {}

        logger.info(f"squeue: tracking {len(self._jobs)} jobs: {list(self._jobs.keys())}")
        if hasattr(self, "_array_jobs"):
            logger.info(f"squeue: array job mappings: {self._array_jobs}")

        slurm_conf = self.slurm_config
        cmd = shlex.split(getattr(slurm_conf, "squeue_cmd", "squeue"))
        format_arg = ["--noheader", "--format", "%i %T"]

        # Build job IDs to query - convert synthetic IDs to real SLURM IDs
        real_job_ids = []
        synthetic_to_real = {}  # Map real SLURM ID back to our synthetic ID

        for synthetic_id in self._jobs.keys():
            if hasattr(self, "_array_jobs") and synthetic_id in self._array_jobs:
                # Array job: query as base_id_array_index
                base_id, array_idx = self._array_jobs[synthetic_id]
                real_id = f"{base_id}_{array_idx}"
                real_job_ids.append(real_id)
                synthetic_to_real[real_id] = synthetic_id
                logger.debug(f"squeue: mapped synthetic_id={synthetic_id} to real_id={real_id}")
            else:
                # Regular job: use as-is
                real_job_ids.append(str(synthetic_id))
                synthetic_to_real[str(synthetic_id)] = synthetic_id
                logger.debug(f"squeue: using job_id={synthetic_id} as-is")

        job_ids_str = ",".join(real_job_ids)
        full_cmd = [*cmd, "--jobs", job_ids_str, *format_arg]
        logger.info(f"squeue: querying jobs {job_ids_str} with command: {' '.join(full_cmd)}")

        proc = run_command(full_cmd)
        if proc.returncode != 0:
            logger.warning(f"squeue: command failed with rc={proc.returncode}, stderr={proc.stderr}")
            # squeue failure doesn't mean jobs don't exist, try sacct as fallback
            return self._check_sacct_for_missing_jobs(real_job_ids, synthetic_to_real, logger)

        logger.debug(f"squeue: output: {proc.stdout.strip()}")
        statuses: Dict[int, str] = {}

        for line in proc.stdout.strip().splitlines():
            parts = line.strip().split(None, 1)
            if not parts:
                continue

            real_slurm_id = parts[0]  # e.g., "12345" or "12345_0"
            state = parts[1] if len(parts) > 1 else "UNKNOWN"

            # Map back to synthetic ID
            if real_slurm_id in synthetic_to_real:
                synthetic_id = synthetic_to_real[real_slurm_id]
                statuses[synthetic_id] = state
            else:
                logger.warning(f"squeue: received unknown job ID {real_slurm_id}")

        # For jobs not found in squeue, check sacct (they may have completed/cancelled recently)
        missing_jobs = [jid for jid in real_job_ids if synthetic_to_real.get(jid) not in statuses]
        if missing_jobs:
            logger.info(f"squeue: {len(missing_jobs)} jobs not in queue, checking sacct for recent completion")
            sacct_statuses = self._check_sacct_for_missing_jobs(missing_jobs, synthetic_to_real, logger)
            statuses.update(sacct_statuses)

        logger.debug(f"squeue: parsed statuses: {statuses}")
        return statuses

    def _check_sacct_for_missing_jobs(
        self, real_job_ids: List[str], synthetic_to_real: Dict[str, str], logger
    ) -> Dict[str, str]:
        """Check sacct for jobs that are no longer in squeue.

        This catches jobs that have recently completed/cancelled/failed.
        """
        if not real_job_ids:
            return {}

        slurm_conf = self.slurm_config
        sacct_cmd = shlex.split(getattr(slurm_conf, "sacct_cmd", "sacct"))

        # Use --brief format for simple output: JobID|State
        job_ids_str = ",".join(real_job_ids)
        full_cmd = [
            *sacct_cmd,
            "--jobs",
            job_ids_str,
            "--noheader",
            "--format",
            "JobID,State",
            "--parsable2",  # Use | as delimiter
        ]

        logger.info(f"sacct: checking for {len(real_job_ids)} missing jobs: {' '.join(full_cmd)}")

        proc = run_command(full_cmd)
        if proc.returncode != 0:
            logger.warning(f"sacct: command failed with rc={proc.returncode}, stderr={proc.stderr}")
            return {}

        logger.debug(f"sacct: output: {proc.stdout.strip()}")
        statuses: Dict[str, str] = {}

        for line in proc.stdout.strip().splitlines():
            if not line.strip():
                continue

            parts = line.strip().split("|")
            if len(parts) < 2:
                continue

            real_slurm_id = parts[0].strip()
            state = parts[1].strip()

            # Map sacct states to squeue-equivalent states
            # sacct uses more detailed states like "CANCELLED by <user>"
            if "CANCELLED" in state:
                state = "CANCELLED"
            elif "COMPLETED" in state:
                state = "COMPLETED"
            elif "FAILED" in state:
                state = "FAILED"
            elif "TIMEOUT" in state:
                state = "TIMEOUT"

            # Map back to synthetic ID
            if real_slurm_id in synthetic_to_real:
                synthetic_id = synthetic_to_real[real_slurm_id]
                statuses[synthetic_id] = state
                logger.info(f"sacct: found job {real_slurm_id} in state {state}")
            else:
                logger.debug(f"sacct: ignoring irrelevant job {real_slurm_id}")

        return statuses

    def job_ids_by_name(self, name: str) -> List[int]:
        return [job_id for job_id, job in self._jobs.items() if job.name == name]

    def get_job(self, job_id: str) -> SlurmJob:
        return self._jobs[job_id]

    def register_job(
        self,
        job_id: str,
        name: str,
        script_path: Path,
        log_path: Path,
        state: str = "PENDING",
    ) -> int:
        """Register an externally submitted job for tracking.

        This is useful when jobs are submitted outside the orchestrator
        but need to be monitored (e.g., for testing or resume scenarios).
        """
        job = SlurmJob(job_id=job_id, name=name, script_path=Path(script_path), log_path=Path(log_path), state=state)
        self._jobs[job_id] = job
        return job_id

    @staticmethod
    def _parse_job_id(output: str) -> Optional[int]:
        for token in output.split():
            if token.isdigit():
                return int(token)
        return None


FakeSlurm = FakeSlurmClient


__all__ = [
    "BaseSlurmClient",
    "FakeSlurmClient",
    "SlurmClient",
    "FakeSlurmClientConfig",
    "SlurmClientConfig",
    "SlurmJob",
    "FakeSlurm",
]

"""SLURM client implementations used by oellm_autoexp."""

from __future__ import annotations

import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from compoconf import ConfigInterface, register

from oellm_autoexp.config.schema import SlurmClientInterface, SlurmConfig
from oellm_autoexp.utils.shell import run_command


@dataclass
class SlurmJob:
    """Metadata tracked for submitted SLURM jobs."""

    job_id: int
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

    def cancel(self, job_id: int) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def remove(self, job_id: int) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def squeue(self) -> Dict[int, str]:  # pragma: no cover - interface
        raise NotImplementedError

    def job_ids_by_name(self, name: str) -> List[int]:  # pragma: no cover - interface
        raise NotImplementedError

    def get_job(self, job_id: int) -> SlurmJob:  # pragma: no cover - interface
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

    def submit(self, name: str, script_path: Path, log_path: Path) -> int:
        job_id = self._next_id
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
    ) -> List[int]:
        job_ids: List[int] = []
        for log_path, task_name in zip(log_paths, task_names):
            job_name = f"{array_name}_{task_name}"
            job_ids.append(self.submit(job_name, script_path, log_path))
        return job_ids

    def cancel(self, job_id: int) -> None:
        if job_id in self._jobs:
            self._jobs[job_id].state = "CANCELLED"

    def remove(self, job_id: int) -> None:
        self._jobs.pop(job_id, None)

    def squeue(self) -> Dict[int, str]:
        return {job_id: job.state for job_id, job in self._jobs.items()}

    def job_ids_by_name(self, name: str) -> List[int]:
        return [job_id for job_id, job in self._jobs.items() if job.name == name]

    def get_job(self, job_id: int) -> SlurmJob:
        return self._jobs[job_id]

    def set_state(self, job_id: int, state: str, return_code: Optional[int] = None) -> None:
        job = self._jobs[job_id]
        job.state = state
        if return_code is not None:
            job.return_code = return_code

    def register_job(
        self,
        job_id: int,
        name: str,
        script_path: Path,
        log_path: Path,
        state: str = "PENDING",
    ) -> int:
        job = SlurmJob(job_id=job_id, name=name, script_path=Path(script_path), log_path=Path(log_path), state=state)
        self._jobs[job_id] = job
        self._next_id = max(self._next_id, job_id + 1)
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
    ) -> List[int]:
        """Submit a SLURM job array.

        Args:
            array_name: Base name for the array job
            script_path: Path to the array script
            log_paths: List of log paths for each task
            task_names: List of task names

        Returns:
            List of job IDs (one per task)
        """
        slurm_conf = self.slurm_config
        num_tasks = len(task_names)

        # Submit the array job with --array=0-N
        submit_cmd = shlex.split(slurm_conf.submit_cmd)
        array_range = f"0-{num_tasks - 1}"
        proc = run_command([*submit_cmd, f"--array={array_range}", str(script_path)])

        if proc.returncode != 0:
            raise RuntimeError(f"sbatch failed for array {script_path}: {proc.stderr.strip()}")

        # Parse the job ID from output like "Submitted batch job 12345"
        base_job_id = self._parse_job_id(proc.stdout)
        if base_job_id is None:
            raise RuntimeError(f"Unable to parse job id from sbatch output: {proc.stdout.strip()}")

        # Create job entries for each task in the array
        # SLURM array jobs have IDs like 12345_0, 12345_1, etc., but we track them individually
        job_ids: List[int] = []
        for idx, (log_path, task_name) in enumerate(zip(log_paths, task_names)):
            # For tracking purposes, we use unique IDs (base_job_id + idx offset)
            # Note: In reality, SLURM uses base_job_id_taskid format, but for our tracking
            # we'll generate sequential IDs
            task_job_id = base_job_id + idx
            job_name = f"{array_name}_{task_name}"

            if self.config.persist_artifacts:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.touch(exist_ok=True)

            self._jobs[task_job_id] = SlurmJob(
                job_id=task_job_id,
                name=job_name,
                script_path=Path(script_path),
                log_path=Path(log_path),
            )
            job_ids.append(task_job_id)

        return job_ids

    def cancel(self, job_id: int) -> None:
        slurm_conf = self.slurm_config
        cmd = shlex.split(getattr(slurm_conf, "cancel_cmd", "scancel"))
        run_command([*cmd, str(job_id)])

    def remove(self, job_id: int) -> None:
        self._jobs.pop(job_id, None)

    def squeue(self) -> Dict[int, str]:
        if not self._jobs:
            return {}
        slurm_conf = self.slurm_config
        job_ids = ",".join(str(job_id) for job_id in self._jobs)
        cmd = shlex.split(getattr(slurm_conf, "squeue_cmd", "squeue"))
        format_arg = ["--noheader", "--format", "%i %T"]
        proc = run_command([*cmd, "--jobs", job_ids, *format_arg])
        if proc.returncode != 0:
            return {}
        statuses: Dict[int, str] = {}
        for line in proc.stdout.strip().splitlines():
            parts = line.strip().split(None, 1)
            if not parts:
                continue
            try:
                job_id = int(parts[0])
            except ValueError:
                continue
            state = parts[1] if len(parts) > 1 else "UNKNOWN"
            statuses[job_id] = state
        return statuses

    def job_ids_by_name(self, name: str) -> List[int]:
        return [job_id for job_id, job in self._jobs.items() if job.name == name]

    def get_job(self, job_id: int) -> SlurmJob:
        return self._jobs[job_id]

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

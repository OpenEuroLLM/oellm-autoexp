"""Orchestration utilities for resolving sweeps and running monitor loops."""

from __future__ import annotations

import logging
import shlex
import time
from dataclasses import dataclass, field, MISSING, replace
import hashlib
import json
from pathlib import Path

from compoconf import asdict

from oellm_autoexp.hydra_staged_sweep import expand_sweep, resolve_sweep_with_dag
from oellm_autoexp.hydra_staged_sweep.expander import SweepPoint
from oellm_autoexp.hydra_staged_sweep.planner import JobPlan

from oellm_autoexp.monitor.loop import MonitorLoop, JobFileStore, JobRecordConfig, JobRuntimeConfig
from oellm_autoexp.monitor.slurm_client import SlurmClient, SlurmClientConfig
from oellm_autoexp.monitor.local_client import LocalCommandClient, LocalCommandClientConfig
from oellm_autoexp.monitor.submission import SlurmJobConfig, LocalJobConfig

import oellm_autoexp.backends.megatron_backend  # noqa  - register
from oellm_autoexp.config.schema import RootConfig, ConfigSetup, BackendInterface


LOGGER = logging.getLogger(__name__)


def stable_hash_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass(kw_only=True)
class ExecutionPlan:
    config: RootConfig = field(default_factory=MISSING)
    config_setup: ConfigSetup = field(default_factory=MISSING)
    sweep_points: dict[int, SweepPoint] = field(default_factory=dict)
    jobs: list[JobPlan] = field(default_factory=list)


@dataclass(kw_only=True)
class SubmissionResult:
    """Return value capturing monitoring setup."""

    loop: MonitorLoop = field(default_factory=MISSING)
    state_store: JobFileStore = field(default_factory=MISSING)
    session_id: str = ""
    submitted_job_ids: list[str] = field(default_factory=list)

    @property
    def submitted_jobs(self) -> list[str]:
        return list(self.submitted_job_ids)


def build_execution_plan(
    config: RootConfig,
    config_setup: ConfigSetup,
    subset_indices: set[int] | None = None,
) -> ExecutionPlan:
    root = config

    points = expand_sweep(root.sweep)
    points_by_idx = {point.index: point for point in points}
    if subset_indices:
        points_by_idx = {
            idx: point for idx, point in points_by_idx.items() if idx in subset_indices
        }
        if not points_by_idx:
            raise ValueError(f"No sweep points match indices: {sorted(subset_indices)}")

    jobs = resolve_sweep_with_dag(
        root,
        points_by_idx,
        config_setup=config_setup,
        config_class=RootConfig,
    )

    return ExecutionPlan(
        config=root, config_setup=config_setup, sweep_points=points_by_idx, jobs=jobs
    )


def submit_jobs(
    plan: ExecutionPlan,
    *,
    slurm_client: SlurmClient | None = None,
    session_id: str | None = None,
    no_error_catching: bool = False,
    local_mode: bool = False,
) -> SubmissionResult:
    store, session_id = _ensure_state_store(plan, session_id=session_id)
    client = slurm_client or SlurmClient(SlurmClientConfig())
    local_client = LocalCommandClient(LocalCommandClientConfig())
    loop = MonitorLoop(
        store, slurm_client=client, local_client=local_client, no_error_catching=no_error_catching
    )

    submitted_job_ids: list[str] = []
    for job in plan.jobs:
        record = _build_job_record(plan, job, session_id, local_mode=local_mode)
        store.upsert(record)
        submitted_job_ids.append(record.job_id)

    return SubmissionResult(
        loop=loop,
        state_store=store,
        session_id=session_id,
        submitted_job_ids=submitted_job_ids,
    )


def load_monitor_controller(
    plan: ExecutionPlan,
    *,
    slurm_client: SlurmClient | None = None,
    session_id: str | None = None,
) -> SubmissionResult:
    if not session_id:
        raise ValueError("session_id required to load existing monitor session")

    store, _ = _ensure_state_store(plan, session_id=session_id)
    client = slurm_client or SlurmClient(SlurmClientConfig())
    local_client = LocalCommandClient(LocalCommandClientConfig())
    loop = MonitorLoop(store, slurm_client=client, local_client=local_client)

    return SubmissionResult(
        loop=loop, state_store=store, session_id=session_id, submitted_job_ids=[]
    )


def run_loop(controller: MonitorLoop) -> None:
    loop = controller
    while True:
        active_jobs = list(loop._store.load_all())
        if not active_jobs:
            LOGGER.info("All jobs finished.")
            break
        loop.observe_once()
        time.sleep(loop.poll_interval_seconds)


def run_loop_sync(controller: MonitorLoop) -> None:
    run_loop(controller)


def _ensure_state_store(
    plan: ExecutionPlan, *, session_id: str | None = None
) -> tuple[JobFileStore, str]:
    monitor_state_dir = Path(plan.config_setup.monitor_state_dir)
    if not session_id:
        import time

        session_id = str(int(time.time()))

    session_dir = monitor_state_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    store = JobFileStore(session_dir)
    LOGGER.info("Created monitoring session directory: %s", session_dir)
    return store, session_id


def _build_job_record(
    plan: ExecutionPlan, job: JobPlan, session_id: str, *, local_mode: bool = False
) -> JobRecordConfig:
    if not isinstance(job.config, RootConfig):
        raise ValueError("JobPlan.config must be RootConfig")

    job_name = _resolve_job_name(job.config)

    job_hash = stable_hash_hex(json.dumps(asdict(job.config)))[:6]
    job_id = f"{job_name}_{job_hash}"

    backend = job.config.backend.instantiate(BackendInterface)
    launch_cmd = backend.build_launch_command()
    script_path = (
        job.config.slurm.script_path or Path(job.config.slurm.script_dir) / f"{job_name}.sbatch"
    )
    slurm_config = replace(
        job.config.slurm,
        name=job_name,
        script_path=str(script_path),
        log_path=str(job.config.job.log_path),
        command=[launch_cmd],
        env=job.config.slurm.env,
    )

    base_job = job.config.job

    if local_mode:
        merged_env = {
            **{k: str(v) for k, v in job.config.slurm.env.items()},
            **{k: str(v) for k, v in job.config.backend.env.items()},
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(job.config.slurm.env.get("MASTER_PORT", "29500")),
            "LOCAL_ADDR": "localhost",
            "SLURM_NODEID": "0",
        }
        env_str = " ".join(f"export {k}={shlex.quote(str(v))};" for k, v in merged_env.items())
        definition = LocalJobConfig(
            name=job_name,
            command=["bash", "-c", f"{env_str} {launch_cmd}"],
            log_path=str(job.config.job.log_path),
            log_path_current=str(job.config.job.log_path_current),
            config_path=str(job.config.job.config_path),
            config_path_current=str(job.config.job.config_path_current),
            log_events=list(base_job.log_events),
            state_events=list(base_job.state_events),
            start_condition=base_job.start_condition,
            cancel_condition=base_job.cancel_condition,
            finish_condition=base_job.finish_condition,
            metadata={
                **dict(base_job.metadata),
                "session_id": session_id,
                "sweep_index": getattr(job.config, "index", None),
                "stage": getattr(job.config, "stage", ""),
            },
            base_config=job,
        )
        return JobRecordConfig(
            job_id=job_id,
            definition=definition,
            runtime=JobRuntimeConfig(submitted=False),
        )

    definition = SlurmJobConfig(
        name=job_name,
        log_path=str(job.config.job.log_path),
        log_path_current=str(job.config.job.log_path_current),
        config_path=str(job.config.job.config_path),
        config_path_current=str(job.config.job.config_path_current),
        log_events=list(base_job.log_events),
        state_events=list(base_job.state_events),
        start_condition=base_job.start_condition,
        cancel_condition=base_job.cancel_condition,
        finish_condition=base_job.finish_condition,
        metadata={
            **dict(base_job.metadata),
            "session_id": session_id,
            "sweep_index": getattr(job.config, "index", None),
            "stage": getattr(job.config, "stage", ""),
        },
        slurm=slurm_config,
        base_config=job,
    )

    return JobRecordConfig(
        job_id=job_id,
        definition=definition,
        runtime=JobRuntimeConfig(submitted=False),
    )


def _resolve_job_name(config: RootConfig) -> str:
    base_name = str(config.job.name or "job")
    index = getattr(config, "index", None)
    if index is None:
        return base_name
    index_str = str(index)
    if index_str in base_name:
        return base_name
    if "%a" in base_name:
        return base_name.replace("%a", index_str)
    else:
        return f"{base_name}_{index_str}"


__all__ = [
    "ExecutionPlan",
    "SubmissionResult",
    "build_execution_plan",
    "submit_jobs",
    "load_monitor_controller",
    "run_loop",
    "run_loop_sync",
]

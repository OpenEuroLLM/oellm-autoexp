"""Host-side helpers to consume plan manifests."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field, MISSING
from importlib import import_module
from pathlib import Path
from typing import Any

from compoconf import parse_config

from oellm_autoexp.monitor.controller import JobRegistration, MonitorController, MonitorRecord
from oellm_autoexp.persistence.state_store import MonitorStateStore, StoredJob, _serialize_for_json
from oellm_autoexp.slurm.client import FakeSlurmClient, FakeSlurmClientConfig, BaseSlurmClient
from oellm_autoexp.workflow.manifest import PlanJobSpec, PlanManifest
from oellm_autoexp.monitor.watcher import BaseMonitor
from oellm_autoexp.utils.start_condition import (
    resolve_start_condition_interval,
    wait_for_start_condition,
)
from oellm_autoexp.config.schema import SlurmConfig


def _import_object(module: str, name: str) -> Any:
    mod = import_module(module)
    return getattr(mod, name)


@dataclass(kw_only=True)
class HostRuntime:
    manifest: PlanManifest = field(default_factory=MISSING)
    monitor: BaseMonitor = field(default_factory=MISSING)
    slurm_client: BaseSlurmClient = field(default_factory=MISSING)
    state_store: MonitorStateStore = field(default_factory=MISSING)
    action_queue_dir: Path = field(default_factory=MISSING)


async def _monitor_loop(
    controller: MonitorController,
    monitor: BaseMonitor,
    action_queue_dir: Path,
) -> None:
    interval = getattr(
        monitor.config,
        "check_interval_seconds",
        getattr(monitor.config, "poll_interval_seconds", 60),
    )
    sleep_seconds = max(1, int(interval))

    while list(controller.jobs()):
        await asyncio.sleep(sleep_seconds)
        cycle = await controller.observe_once()
        records = controller.drain_events()
        if not records and cycle.events:
            records = list(cycle.events)
        actionable = [record for record in records if record.action]
        if actionable:
            _record_monitor_events(actionable, action_queue_dir)

    controller.clear_state()


def _monitor_loop_sync(
    controller: MonitorController,
    monitor: BaseMonitor,
    action_queue_dir: Path,
) -> None:
    interval = getattr(
        monitor.config,
        "check_interval_seconds",
        getattr(monitor.config, "poll_interval_seconds", 60),
    )
    sleep_seconds = max(1, int(interval))

    while list(controller.jobs()):
        time.sleep(sleep_seconds)
        cycle = controller.observe_once_sync()
        records = controller.drain_events()
        if not records and cycle.events:
            records = list(cycle.events)
        actionable = [record for record in records if record.action]
        if actionable:
            _record_monitor_events(actionable, action_queue_dir)

    controller.clear_state()


def _record_monitor_events(records: list[MonitorRecord], action_queue_dir: Path) -> None:
    action_queue_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    for idx, record in enumerate(records):
        if not record.action:
            continue
        payload = {
            "job_id": record.job_id,
            "job_name": record.job_name,
            "event": record.event,
            "state": record.state,
            "action": record.action,
            "payload": _serialize_for_json(record.payload),
            "metadata": _serialize_for_json(record.metadata),
            "created_at": timestamp,
        }
        name = f"{timestamp}_{record.job_id}_{idx}_{record.action}.json"
        path = action_queue_dir / name
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            f"[actions] wrote {path} for job {record.job_id} ({record.job_name}) -> {record.action}"
        )


def build_host_runtime(
    manifest: PlanManifest,
    *,
    use_fake_slurm: bool = False,
    manifest_path: Path | None = None,
) -> HostRuntime:
    monitor = manifest.monitor.instantiate()

    if use_fake_slurm:
        slurm_client: BaseSlurmClient = FakeSlurmClient(FakeSlurmClientConfig())
    else:
        slurm_client = manifest.slurm_client.instantiate()

    slurm_config_obj = parse_config(SlurmConfig, manifest.slurm_config)
    slurm_client.configure(slurm_config_obj)

    state_store = MonitorStateStore(manifest.monitoring_state_dir, session_id=manifest.plan_id)
    if not state_store.session_path.exists():
        state_store.save_session(manifest.config, manifest.project_name)
        if manifest_path is not None:
            payload = {
                "manifest_path": str(manifest_path),
            }
            text = state_store.session_path.read_text(encoding="utf-8")
            data = json.loads(text)
            data.update(payload)
            state_store.session_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    action_queue_dir = Path(
        manifest.action_queue_dir or state_store.session_path.parent / "actions"
    )

    return HostRuntime(
        manifest=manifest,
        monitor=monitor,
        slurm_client=slurm_client,
        state_store=state_store,
        action_queue_dir=action_queue_dir,
    )


def instantiate_controller(runtime: HostRuntime, *, quiet: bool = False) -> MonitorController:
    """Create a monitor controller and restore any persisted jobs."""

    controller = MonitorController(
        runtime.monitor,
        runtime.slurm_client,
        state_store=runtime.state_store,
    )
    restored = restore_jobs(runtime, controller)
    if restored and not quiet:
        print(f"Restored {len(restored)} job(s): {', '.join(sorted(restored))}")
    return controller


def restore_jobs(
    runtime: HostRuntime,
    controller: MonitorController,
) -> set[str]:
    saved_jobs = runtime.state_store.load()
    restored_names: set[str] = set()

    for saved in saved_jobs.values():
        registration = JobRegistration(
            name=saved.name,
            script_path=saved.script_path,
            log_path=saved.log_path,
            metadata=dict(saved.metadata),
            termination_string=saved.termination_string,
            termination_command=saved.termination_command,
            inactivity_threshold_seconds=saved.inactivity_threshold_seconds,
            output_paths=list(saved.output_paths),
            start_condition_cmd=saved.start_condition_cmd,
            start_condition_interval_seconds=saved.start_condition_interval_seconds,
        )
        if hasattr(runtime.slurm_client, "register_job"):
            log_path_for_client = saved.resolved_log_path or saved.log_path
            try:
                runtime.slurm_client.register_job(  # type: ignore[attr-defined]
                    saved.job_id,
                    saved.name,
                    saved.script_path,
                    str(log_path_for_client),
                    state=saved.last_slurm_state or "PENDING",
                )
            except TypeError:
                runtime.slurm_client.register_job(  # type: ignore[attr-defined]
                    saved.job_id,
                    saved.name,
                    saved.script_path,
                    saved.log_path,
                )
        controller.register_job(saved.job_id, registration, attempts=saved.attempts)
        restored_names.add(saved.name)
    return restored_names


def _register_job(
    controller: MonitorController,
    job_id: str,
    job: PlanJobSpec,
    attempts: int = 1,
) -> None:
    metadata = {"parameters": dict(job.parameters), "output_dir": job.output_dir}
    registration = JobRegistration(
        name=job.name,
        script_path=job.script_path,
        log_path=job.log_path,
        metadata=metadata,
        output_paths=list(job.output_paths),
        start_condition_cmd=job.start_condition_cmd,
        start_condition_interval_seconds=job.start_condition_interval_seconds,
        termination_string=job.termination_string,
        termination_command=job.termination_command,
        inactivity_threshold_seconds=job.inactivity_threshold_seconds,
    )
    controller.register_job(job_id, registration, attempts=attempts)


def submit_pending_jobs(
    runtime: HostRuntime,
    controller: MonitorController,
    *,
    dry_run: bool = False,
) -> list[str]:
    slurm_client = runtime.slurm_client
    restored_names = restore_jobs(runtime, controller)
    pending_jobs = [job for job in runtime.manifest.jobs if job.name not in restored_names]
    submitted_job_ids: list[str] = []

    if not pending_jobs:
        return submitted_job_ids

    monitor_config = runtime.monitor.config

    if runtime.manifest.rendered.array and getattr(slurm_client, "supports_array", False):
        array = runtime.manifest.rendered.array
        log_paths = [job.log_path for job in pending_jobs]
        task_names = [job.name for job in pending_jobs]

        if dry_run:
            print(
                f"[dry-run] would submit array job {array.job_name} using script {array.script_path}"
            )
            for job in pending_jobs:
                print(f"[dry-run]   task {job.name} -> log {job.log_path}")
        else:
            job_ids = slurm_client.submit_array(
                array.job_name, array.script_path, log_paths, task_names
            )
            if len(job_ids) != len(pending_jobs):
                raise RuntimeError("SLURM client returned mismatched job ids for array submission")
            for job_id, job in zip(job_ids, pending_jobs):
                _register_job(controller, job_id, job)
                submitted_job_ids.append(job_id)
        return submitted_job_ids

    for job in pending_jobs:
        if job.start_condition_cmd:
            interval = resolve_start_condition_interval(
                job.start_condition_interval_seconds, monitor_config
            )
            if not dry_run:
                wait_for_start_condition(job.start_condition_cmd, interval_seconds=interval)
            else:
                print(
                    "[dry-run] would wait for start condition command "
                    f"{job.start_condition_cmd!r} (interval {interval}s)"
                )

        if dry_run:
            print(
                f"[dry-run] would submit job {job.name} using script {job.script_path} -> log {job.log_path}"
            )
            continue

        job_id = slurm_client.submit(job.name, job.script_path, job.log_path)
        _register_job(controller, job_id, job)
        submitted_job_ids.append(job_id)

    return submitted_job_ids


def run_monitoring(runtime: HostRuntime, controller: MonitorController) -> None:
    if runtime.monitor.config.debug_sync:
        _monitor_loop_sync(controller, runtime.monitor, runtime.action_queue_dir)
    else:
        asyncio.run(_monitor_loop(controller, runtime.monitor, runtime.action_queue_dir))


def snapshot_runtime(runtime: HostRuntime, controller: MonitorController) -> None:
    """Persist controller state for crash recovery."""
    jobs = []
    for state in controller.jobs():
        resolved_log_path = controller._expand_log_path(  # type: ignore[attr-defined]
            state.job_id,
            state.registration.log_path,
        )
        stored = StoredJob.from_registration(
            state.job_id,
            state.attempts,
            state.registration,
            resolved_log_path=str(resolved_log_path),
            monitor_state=state.state.key if state.state else None,
            slurm_state=state.last_slurm_state,
        )
        jobs.append(stored)
    runtime.state_store.save_jobs(jobs)


__all__ = [
    "HostRuntime",
    "build_host_runtime",
    "instantiate_controller",
    "run_monitoring",
    "snapshot_runtime",
    "restore_jobs",
    "submit_pending_jobs",
]

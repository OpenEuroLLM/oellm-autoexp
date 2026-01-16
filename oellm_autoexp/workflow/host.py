"""Host-side helpers to consume plan manifests."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field, MISSING
from importlib import import_module
from pathlib import Path
from typing import Any

from compoconf import parse_config

from oellm_autoexp.monitor.controller import JobRegistration, MonitorController, MonitorRecord
from oellm_autoexp.monitor.action_queue import ActionQueue
from oellm_autoexp.monitor.actions import ActionContext, BaseMonitorAction
from oellm_autoexp.persistence.state_store import MonitorStateStore, StoredJob, _serialize_for_json
from oellm_autoexp.slurm.client import FakeSlurmClient, FakeSlurmClientConfig, BaseSlurmClient
from oellm_autoexp.workflow.manifest import PlanJobSpec, PlanManifest
from oellm_autoexp.monitor.watcher import BaseMonitor

# Note: check_start_condition is imported locally where needed
from oellm_autoexp.config.schema import SlurmConfig
from oellm_autoexp.monitor.conditions import (
    ConditionContext,
    MonitorConditionInterface,
)
from oellm_autoexp.monitor.events import EventRecord


LOGGER = logging.getLogger(__name__)


def _check_start_conditions(job: PlanJobSpec, context: ConditionContext) -> bool:
    """Check if all start_conditions for a job are satisfied.

    Returns:
        True if all conditions pass, False otherwise.
    """
    conditions = job.start_conditions
    if not conditions:
        return True

    for cond_dict in conditions:
        try:
            cond_cfg = parse_config(MonitorConditionInterface.cfgtype, cond_dict)
            condition = cond_cfg.instantiate(MonitorConditionInterface)
            result = condition.check(context)
            if not result.passed:
                LOGGER.debug(
                    f"[job {job.name}] start condition '{cond_dict.get('class_name', 'unknown')}' "
                    f"not met: {result.message}"
                )
                return False
        except Exception as exc:
            LOGGER.warning(f"[job {job.name}] failed to evaluate start condition: {exc}")
            return False

    LOGGER.info(f"[job {job.name}] all start conditions met")
    return True


def _import_object(module: str, name: str) -> Any:
    mod = import_module(module)
    return getattr(mod, name)


def _flatten_config(config: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten nested config dict so {project.name} template syntax works.

    Example:
        {"project": {"name": "test"}} -> {"project.name": "test"}
    """
    items: list[tuple[str, Any]] = []
    for k, v in config.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _check_and_submit_pending_jobs(
    runtime: HostRuntime,
    controller: MonitorController,
) -> list[str]:
    """Check for jobs with unmet start conditions and submit them if now ready.

    This function is called periodically from the monitor loop to handle jobs
    whose start_conditions weren't met at initial submission time.

    Returns:
        List of newly submitted job IDs.
    """
    # Check if manifest has jobs attribute (may not in tests)
    manifest_jobs = getattr(runtime.manifest, "jobs", None)
    if not manifest_jobs:
        return []

    # Get names of jobs already registered/submitted
    registered_names = {getattr(state, "name", None) for state in controller.jobs()}

    # Find jobs from manifest that haven't been submitted yet
    pending_jobs = [job for job in manifest_jobs if job.name not in registered_names]

    if not pending_jobs:
        return []

    submitted_job_ids: list[str] = []

    for job in pending_jobs:
        # Create context for condition evaluation
        dummy_event = EventRecord(
            event_id=f"start_condition:{job.name}",
            name="start_condition_check",
            source="host",
            payload={},
            metadata={"job_name": job.name},
        )
        job_metadata: dict[str, Any] = {"job_name": job.name, "output_dir": job.output_dir}
        if runtime.manifest.config:
            job_metadata.update(_flatten_config(runtime.manifest.config))
        context = ConditionContext(
            event=dummy_event,
            job_metadata=job_metadata,
            attempts=0,
        )

        # Check old-style command-based condition
        if job.start_condition_cmd:
            from oellm_autoexp.utils.start_condition import check_start_condition

            result = check_start_condition(job.start_condition_cmd)
            if not result.success:
                LOGGER.debug(f"[job {job.name}] start_condition_cmd still not met")
                continue

        # Check new-style list-based conditions
        if job.start_conditions:
            if not _check_start_conditions(job, context):
                LOGGER.debug(f"[job {job.name}] start_conditions still not met")
                continue

        # All conditions met - submit the job
        LOGGER.info(f"[job {job.name}] start conditions now met, submitting")
        job_id = runtime.slurm_client.submit(job.name, job.script_path, job.log_path)
        _register_job(
            controller,
            job_id,
            job,
            config=runtime.manifest.config,
            session_id=runtime.state_store.session_id,
        )
        _create_current_log_symlink(job_id, job.log_path, job.log_path_current)
        submitted_job_ids.append(job_id)
        print(f"[monitor] Submitted pending job {job.name} (job_id={job_id})", flush=True)

    return submitted_job_ids


def _expand_log_path_for_job(job_id: str, log_path: str) -> Path:
    log_str = str(log_path)
    if "_" in job_id:
        base_id, array_idx = job_id.split("_")
        log_str = log_str.replace("%A", str(base_id))
        log_str = log_str.replace("%a", str(array_idx))
    log_str = log_str.replace("%j", str(job_id))
    return Path(log_str)


def _create_current_log_symlink(job_id: str, log_path: str, log_path_current: str | None) -> None:
    if not log_path_current:
        return
    resolved_log_path = _expand_log_path_for_job(job_id, log_path)
    current_path = Path(log_path_current)
    try:
        current_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        LOGGER.warning("Unable to create log symlink parent %s: %s", current_path.parent, exc)
        return
    try:
        if current_path.exists() or current_path.is_symlink():
            current_path.unlink()
        current_path.symlink_to(resolved_log_path)
    except OSError as exc:
        LOGGER.warning(
            "Unable to create log symlink %s -> %s: %s",
            current_path,
            resolved_log_path,
            exc,
        )


@dataclass(kw_only=True)
class HostRuntime:
    manifest: PlanManifest = field(default_factory=MISSING)
    monitor: BaseMonitor = field(default_factory=MISSING)
    slurm_client: BaseSlurmClient = field(default_factory=MISSING)
    state_store: MonitorStateStore = field(default_factory=MISSING)
    action_queue_dir: Path = field(default_factory=MISSING)


def _process_action_queue(
    state_store: MonitorStateStore,
    controller: MonitorController,
    runtime: HostRuntime,
) -> int:
    """Process all pending actions in the queue and return count of processed
    actions."""
    queue = ActionQueue(state_store.session_path.with_suffix(".actions"))
    processed = 0

    # Track existing jobs before processing actions
    existing_job_ids = {state.job_id for state in controller.jobs()}

    while True:
        record = queue.claim_next()
        if record is None:
            break

        print(
            f"[worker] Processing {record.action_class} (queue_id={record.queue_id}, event_id={record.event_id})",
            flush=True,
        )

        events = state_store.load_events()
        event_record = events.get(record.event_id)
        if event_record is None:
            error_msg = f"Event record not found for event_id={record.event_id}"
            print(f"[worker] ERROR: {error_msg}", flush=True)
            queue.mark_done(
                record.queue_id,
                status="failed",
                result={"error": error_msg},
            )
            continue

        cfg_dict = dict(record.config)
        cfg_dict.setdefault("class_name", record.action_class)

        try:
            action_cfg = parse_config(BaseMonitorAction.cfgtype, cfg_dict)
            action = action_cfg.instantiate(BaseMonitorAction)
        except Exception as exc:
            queue.mark_done(
                record.queue_id,
                status="failed",
                result={"error": f"failed to instantiate action: {exc}"},
            )
            continue

        context = ActionContext(
            event=event_record,
            job_metadata=record.metadata.get("job", {}),
            workspace=None,
            env={},
        )

        try:
            result = action.execute(context)
        except Exception as exc:
            queue.mark_done(
                record.queue_id,
                status="failed",
                result={"error": str(exc)},
            )
            continue

        action.update_event(event_record, result)
        state_store.upsert_event(event_record)
        queue.mark_done(
            record.queue_id,
            status="done" if result.status == "success" else "failed",
            result={"message": result.message, "metadata": result.metadata},
        )
        processed += 1

    # After processing all actions, check if new jobs were submitted to this session
    restored = restore_jobs(runtime, controller)
    new_jobs = [
        name
        for name in restored
        if name
        and not any(
            state.name == name for state in controller.jobs() if state.job_id in existing_job_ids
        )
    ]
    if new_jobs:
        print(
            f"[worker] Registered {len(new_jobs)} new job(s) from actions: {', '.join(new_jobs)}",
            flush=True,
        )

    return processed


def _has_pending_manifest_jobs(runtime: HostRuntime, controller: MonitorController) -> bool:
    """Check if there are manifest jobs not yet submitted (waiting for start
    conditions)."""
    manifest_jobs = getattr(runtime.manifest, "jobs", None)
    if not manifest_jobs:
        return False
    registered_names = {getattr(state, "name", None) for state in controller.jobs()}
    return any(job.name not in registered_names for job in manifest_jobs)


async def _monitor_loop(
    controller: MonitorController,
    monitor: BaseMonitor,
    action_queue_dir: Path,
    state_store: MonitorStateStore,
    runtime: HostRuntime,
) -> None:
    interval = getattr(
        monitor.config,
        "check_interval_seconds",
        getattr(monitor.config, "poll_interval_seconds", 60),
    )
    sleep_seconds = max(1, int(interval))

    # Continue while there are active jobs OR pending manifest jobs waiting for start conditions
    while list(controller.jobs()) or _has_pending_manifest_jobs(runtime, controller):
        await asyncio.sleep(sleep_seconds)

        # Check and submit any pending jobs whose start conditions are now met
        newly_submitted = _check_and_submit_pending_jobs(runtime, controller)
        if newly_submitted:
            LOGGER.info(f"Submitted {len(newly_submitted)} pending job(s)")

        cycle = await controller.observe_once()
        records = controller.drain_events()
        if not records and cycle.events:
            records = list(cycle.events)
        actionable = [record for record in records if record.action]
        if actionable:
            _record_monitor_events(actionable, action_queue_dir)

        # Process queued actions automatically
        processed = _process_action_queue(state_store, controller, runtime)
        if processed > 0:
            print(f"Processed {processed} queued action(s).", flush=True)
        else:
            # Debug: check if there are any queued actions
            queue = ActionQueue(state_store.session_path.with_suffix(".actions"))
            pending = [r for r in queue.list() if r.status == "pending"]
            if pending:
                print(f"WARNING: {len(pending)} actions queued but not processed!", flush=True)
                for r in pending[:3]:  # Show first 3
                    print(
                        f"  - {r.action_class} (event_id={r.event_id}, status={r.status})",
                        flush=True,
                    )

    controller.clear_state()


def _monitor_loop_sync(
    controller: MonitorController,
    monitor: BaseMonitor,
    action_queue_dir: Path,
    state_store: MonitorStateStore,
    runtime: HostRuntime,
) -> None:
    interval = getattr(
        monitor.config,
        "check_interval_seconds",
        getattr(monitor.config, "poll_interval_seconds", 60),
    )
    sleep_seconds = max(1, int(interval))

    # Continue while there are active jobs OR pending manifest jobs waiting for start conditions
    while list(controller.jobs()) or _has_pending_manifest_jobs(runtime, controller):
        time.sleep(sleep_seconds)

        # Check and submit any pending jobs whose start conditions are now met
        newly_submitted = _check_and_submit_pending_jobs(runtime, controller)
        if newly_submitted:
            LOGGER.info(f"Submitted {len(newly_submitted)} pending job(s)")

        cycle = controller.observe_once_sync()
        records = controller.drain_events()
        if not records and cycle.events:
            records = list(cycle.events)
        actionable = [record for record in records if record.action]
        if actionable:
            _record_monitor_events(actionable, action_queue_dir)

        # Process queued actions automatically
        processed = _process_action_queue(state_store, controller, runtime)
        if processed > 0:
            print(f"Processed {processed} queued action(s).", flush=True)
        else:
            # Debug: check if there are any queued actions
            queue = ActionQueue(state_store.session_path.with_suffix(".actions"))
            pending = [r for r in queue.list() if r.status == "pending"]
            if pending:
                print(f"WARNING: {len(pending)} actions queued but not processed!", flush=True)
                for r in pending[:3]:  # Show first 3
                    print(
                        f"  - {r.action_class} (event_id={r.event_id}, status={r.status})",
                        flush=True,
                    )

    controller.clear_state()


def _record_monitor_events(records: list[MonitorRecord], action_queue_dir: Path) -> None:
    action_queue_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time() * 1000)  # Millisecond precision to avoid overwrites
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
        # Include checkpoint_iteration in filename if present to avoid overwrites
        checkpoint_iter = record.metadata.get("checkpoint_iteration", "")
        if checkpoint_iter:
            name = f"{timestamp}_{record.job_id}_iter{checkpoint_iter}_{idx}_{record.action}.json"
        else:
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
        metadata = dict(saved.metadata)

        # Add flattened config to metadata so templates like {project.name} work
        if runtime.manifest.config:
            flattened = _flatten_config(runtime.manifest.config)
            metadata.update(flattened)

        registration = JobRegistration(
            name=saved.name,
            script_path=saved.script_path,
            log_path=saved.log_path,
            metadata=metadata,
            termination_string=saved.termination_string,
            termination_command=saved.termination_command,
            inactivity_threshold_seconds=saved.inactivity_threshold_seconds,
            output_paths=list(saved.output_paths),
            start_condition_cmd=saved.start_condition_cmd,
            start_condition_interval_seconds=saved.start_condition_interval_seconds,
            start_conditions=list(saved.start_conditions),
            cancel_conditions=list(saved.cancel_conditions),
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
        controller.register_job(
            saved.job_id,
            registration,
            attempts=saved.attempts,
            last_slurm_state=saved.last_slurm_state,
        )
        restored_names.add(saved.name)
    return restored_names


def _register_job(
    controller: MonitorController,
    job_id: str,
    job: PlanJobSpec,
    config: dict[str, Any] | None = None,
    attempts: int = 1,
    session_id: str | None = None,
) -> None:
    metadata = {"parameters": list(job.parameters), "output_dir": job.output_dir}

    # Add session_id to metadata so actions can reuse it for nested jobs
    if session_id:
        metadata["session_id"] = session_id

    # Add flattened config to metadata so templates like {project.name} work
    if config:
        flattened = _flatten_config(config)
        metadata.update(flattened)
        # Debug: show a sample of flattened config keys
        sample_keys = list(flattened.keys())[:5]
        LOGGER.debug(f"Added {len(flattened)} config keys to job metadata (sample: {sample_keys})")

    registration = JobRegistration(
        name=job.name,
        script_path=job.script_path,
        log_path=job.log_path,
        metadata=metadata,
        output_paths=list(job.output_paths),
        start_condition_cmd=job.start_condition_cmd,
        start_condition_interval_seconds=job.start_condition_interval_seconds,
        start_conditions=list(job.start_conditions),
        cancel_conditions=list(job.cancel_conditions),
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
                _register_job(
                    controller,
                    job_id,
                    job,
                    config=runtime.manifest.config,
                    session_id=runtime.state_store.session_id,
                )
                _create_current_log_symlink(job_id, job.log_path, job.log_path_current)
                submitted_job_ids.append(job_id)
        return submitted_job_ids

    for job in pending_jobs:
        # Check start conditions (non-blocking)
        if job.start_condition_cmd or job.start_conditions:
            # Create context for condition evaluation
            dummy_event = EventRecord(
                event_id=f"start_condition:{job.name}",
                name="start_condition_check",
                source="host",
                payload={},
                metadata={"job_name": job.name},
            )
            job_metadata: dict[str, Any] = {"job_name": job.name, "output_dir": job.output_dir}
            if runtime.manifest.config:
                job_metadata.update(_flatten_config(runtime.manifest.config))
            context = ConditionContext(
                event=dummy_event,
                job_metadata=job_metadata,
                attempts=0,
            )

            # Handle old-style command-based condition (single check, non-blocking)
            if job.start_condition_cmd:
                from oellm_autoexp.utils.start_condition import check_start_condition

                result = check_start_condition(job.start_condition_cmd)
                if not result.success:
                    LOGGER.debug(f"[job {job.name}] start_condition_cmd not met, skipping for now")
                    if dry_run:
                        print(f"[dry-run] job {job.name} start_condition_cmd not met, would skip")
                    continue

            # Handle new-style list-based conditions (single check, non-blocking)
            if job.start_conditions:
                if not _check_start_conditions(job, context):
                    LOGGER.debug(f"[job {job.name}] start_conditions not met, skipping for now")
                    if dry_run:
                        print(f"[dry-run] job {job.name} start_conditions not met, would skip")
                    continue

        if dry_run:
            print(
                f"[dry-run] would submit job {job.name} using script {job.script_path} -> log {job.log_path}"
            )
            continue

        job_id = slurm_client.submit(job.name, job.script_path, job.log_path)
        _register_job(
            controller,
            job_id,
            job,
            config=runtime.manifest.config,
            session_id=runtime.state_store.session_id,
        )
        _create_current_log_symlink(job_id, job.log_path, job.log_path_current)
        submitted_job_ids.append(job_id)

    return submitted_job_ids


def run_monitoring(runtime: HostRuntime, controller: MonitorController) -> None:
    if runtime.monitor.config.debug_sync:
        _monitor_loop_sync(
            controller, runtime.monitor, runtime.action_queue_dir, runtime.state_store, runtime
        )
    else:
        asyncio.run(
            _monitor_loop(
                controller, runtime.monitor, runtime.action_queue_dir, runtime.state_store, runtime
            )
        )


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

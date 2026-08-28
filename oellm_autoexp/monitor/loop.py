"""Simplified synchronous monitor loop."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, MISSING
from pathlib import Path
from typing import Any
from collections.abc import Iterable

from compoconf import ConfigInterface, parse_config, asdict

from oellm_autoexp.monitor.conditions import (
    ConditionContext,
    ConditionResult,
    MonitorConditionInterface,
)
from oellm_autoexp.monitor.actions import (
    EventRecord,
    build_event_id,
    event_key,
    LogEventConfig,
    LogEvent,
    StateEventConfig,
    StateEvent,
    ActionResult,
    BaseMonitorAction,
    NewJobActionConfig,
    UpdateChainExcludesActionConfig,
)
from oellm_autoexp.hydra_staged_sweep.config.resolvers import oc_exclude_nodes
from oellm_autoexp.monitor.job_client_protocol import JobClientProtocol
from oellm_autoexp.monitor.submission import JobInterface, SlurmJobConfig, LocalJobConfig
from oellm_autoexp.monitor.utils.paths import (
    resolve_log_path,
    expand_log_path,
    update_log_symlink,
)

LOGGER = logging.getLogger(__name__)


SCHEMA_VERSION = 1


@dataclass
class JobRuntime:
    class_name: str = "JobRuntime"
    submitted: bool = False
    attempts: int = 0
    runtime_job_id: str | None = None
    start_ts: float | None = None
    end_ts: float | None = None
    log_cursor: int = 0
    condition_state: dict[str, Any] = field(default_factory=dict)
    action_state: dict[str, Any] = field(default_factory=dict)
    # Persisted EventRecords keyed by stable event_key, surviving across polls.
    # Used to accumulate inactivity streaks and no-progress streaks (count +
    # timestamps, plus the last observed counter for progress events).
    events: dict[str, Any] = field(default_factory=dict)
    last_status: str | None = None
    final_state: str | None = None  # "finished", "cancelled", or None for active jobs


@dataclass(kw_only=True)
class JobRecord:
    class_name: str = "JobRecord"
    job_id: str = ""
    definition: JobInterface.cfgtype = field(default_factory=MISSING)
    runtime: JobRuntime = field(default_factory=JobRuntime)
    schema_version: int = SCHEMA_VERSION
    array_idx: int | None = None


class JobFileStore:
    """Store job records as files in a state directory."""

    def __init__(self, state_dir: str | Path) -> None:
        self.root = Path(state_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def list_paths(self) -> Iterable[Path]:
        return self.root.glob("*.job.json")

    def load_all(self, *, include_finished: bool = False) -> list[JobRecord]:
        """Load job records, optionally excluding finished/cancelled jobs."""
        jobs: list[JobRecord] = []
        for path in self.list_paths():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                _import_registry()
                job = parse_config(JobRecord, payload)
                # Parse nested event configs once
                _normalize_job_definition(job)
                # Skip jobs that are finished/cancelled unless requested
                if not include_finished and job.runtime.final_state is not None:
                    continue
                jobs.append(job)
            except (OSError, json.JSONDecodeError, ValueError, KeyError):
                continue
        return jobs

    def upsert(self, record: JobRecord) -> None:
        path = self.path_for(record.job_id)
        payload = asdict(record)
        payload.setdefault("schema_version", SCHEMA_VERSION)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def mark_finished(self, job_id: str, final_state: str) -> None:
        """Mark a job as finished or cancelled without deleting it."""
        job = self.load(job_id)
        if job is None:
            return
        job.runtime.final_state = final_state
        job.runtime.end_ts = time.time()
        self.upsert(job)

    def remove(self, job_id: str) -> None:
        """Actually delete a job file (use with caution - prefer mark_finished)."""
        path = self.path_for(job_id)
        if path.exists():
            path.unlink()

    def load(self, job_id: str, include_finished=False) -> JobRecord | None:
        path = self.path_for(job_id)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            _import_registry()
            job = parse_config(JobRecord, payload)
            # Parse nested event configs once
            _normalize_job_definition(job)
            if not include_finished and job.runtime.final_state is not None:
                return None
            return job
        except (OSError, json.JSONDecodeError, ValueError, KeyError):
            return None

    def path_for(self, job_id: str) -> Path:
        return self.root / f"{job_id}.job.json"


class MonitorLoop:
    """Synchronous monitor loop that evaluates jobs and actions inline."""

    def __init__(
        self,
        store: JobFileStore,
        *,
        slurm_client: JobClientProtocol | None = None,
        local_client: JobClientProtocol | None = None,
        poll_interval_seconds: float = 60.0,
        show_poll_state: bool = True,
        no_error_catching: bool = False,
    ) -> None:
        _import_registry()
        self._store = store
        self._slurm_client = slurm_client
        self._local_client = local_client
        self.poll_interval_seconds = poll_interval_seconds
        self.show_poll_state = show_poll_state
        self.no_error_catching = no_error_catching

    def _get_client(self, job: JobRecord) -> JobClientProtocol:
        """Get the appropriate client based on job configuration."""
        if job.definition is None:
            raise ValueError("Job definition is None")

        if isinstance(job.definition, SlurmJobConfig):
            return self._slurm_client
        elif isinstance(job.definition, LocalJobConfig):
            return self._local_client

    def observe_once(self) -> None:
        # Query both clients and merge statuses
        statuses: dict[str, str] = {}
        if self._slurm_client:
            statuses.update(self._slurm_client.squeue())
        if self._local_client:
            statuses.update(self._local_client.squeue())

        for job in self._store.load_all():
            if job.definition is None:
                continue
            runtime = job.runtime
            runtime_id = runtime.runtime_job_id
            new_status = statuses.get(runtime_id) if runtime_id else None
            # On entering RUNNING, re-point the current.* symlinks at this job so
            # a shared 'current' symlink (dependency chain) tracks the running job.
            if runtime_id and new_status == "RUNNING" and runtime.last_status != "RUNNING":
                self._update_current_symlinks(job)
            status_effect = self._status_action(job, runtime.last_status, new_status)
            runtime.last_status = new_status
            if self._apply_effect(job, status_effect, runtime_id):
                continue
            if not runtime.submitted:
                if self._check_cancel(job):
                    self._store.mark_finished(job.job_id, "cancelled")
                    continue
                if self._check_finish(job):
                    self._store.mark_finished(job.job_id, "finished")
                    continue
                if self._check_start(job):
                    self._start_jobs(job)

                    if job.runtime.runtime_job_id is None:
                        LOGGER.warning(f"Failed to start job: {job}")
                        self._store.mark_finished(job.job_id, "cancelled")
                else:
                    self._store.upsert(job)
                continue

            if self._check_cancel(job):
                if runtime_id:
                    client = self._get_client(job)
                    client.cancel(runtime_id)
                    client.remove(runtime_id)
                self._store.mark_finished(job.job_id, "cancelled")
                continue
            if self._check_finish(job):
                if runtime_id:
                    client = self._get_client(job)
                    client.remove(runtime_id)
                self._store.mark_finished(job.job_id, "finished")
                continue

            # Process log events
            effect = self._process_log_events(job)
            if self._apply_effect(job, effect, runtime_id):
                continue

            # Check if job completed naturally
            if runtime.last_status in {"COMPLETED", "FAILED", "CANCELLED"}:
                if runtime_id:
                    client = self._get_client(job)
                    client.remove(runtime_id)
                terminal = "finished" if runtime.last_status == "COMPLETED" else "cancelled"
                self._store.mark_finished(job.job_id, terminal)
                continue

            self._store.upsert(job)

        if self.show_poll_state:
            statuses: dict[str, str] = {}
            if self._slurm_client:
                statuses.update(self._slurm_client.squeue())
            if self._local_client:
                statuses.update(self._local_client.squeue())

            poll_state = {}
            for job in self._store.load_all():
                if self._check_finish(job):
                    continue
                runtime = job.runtime
                if not runtime.submitted:
                    poll_state[job.runtime.runtime_job_id] = {
                        "state": "pending",
                        "start_condition": job.definition.start_condition,
                        "runtime": job.runtime,
                    }
                else:
                    poll_state[job.runtime.runtime_job_id] = {
                        "state": statuses.get(runtime.runtime_job_id),
                        "cancel_condition": job.definition.cancel_condition,
                        "runtime": job.runtime,
                    }

            LOGGER.info(f"[{time.time():0.6f}]" + f"Monitor Polling: {poll_state}")

    def _check_start(self, job: JobRecord) -> bool:
        condition = job.definition.start_condition
        if condition is None:
            return True
        return self._evaluate_condition(job, condition, label="start").passed

    def _check_cancel(self, job: JobRecord) -> bool:
        condition = job.definition.cancel_condition
        if condition is None:
            return False
        return self._evaluate_condition(job, condition, label="cancel").passed

    def _check_finish(self, job: JobRecord) -> bool:
        condition = job.definition.finish_condition
        if condition is None:
            return False
        return self._evaluate_condition(job, condition, label="finish").passed

    def _start_job(self, job: JobRecord) -> None:
        runtime = job.runtime
        runtime.attempts += 1
        runtime.start_ts = time.time()
        client = self._get_client(job)
        if job.array_idx is not None:
            job_ids = client.submit_array(job.definition, indices=[job.array_idx])
            if not isinstance(job_ids, list):
                runtime_job_id = job_ids
            elif not job_ids:
                raise ValueError("submit_array returned no job ids")
            else:
                runtime_job_id = job_ids[0]
        else:
            if self.no_error_catching:
                runtime_job_id = client.submit(job.definition)
            else:
                try:
                    runtime_job_id = client.submit(job.definition)
                except Exception as e:
                    LOGGER.error(f"Unable to submit {job.definition.name}: {e}")
                    runtime_job_id = None
        if runtime_job_id is not None:
            runtime.runtime_job_id = runtime_job_id
            runtime.submitted = True
            runtime.log_cursor = 0

    def _start_jobs(self, job: JobRecord, indices: list[int] | None = None) -> None:
        definition = job.definition
        if definition is None:
            return
        array_len = int(getattr(definition, "array_len", 1) or 1)
        if array_len <= 1:
            self._start_job(job)
            self._store.upsert(job)
            return

        runtime = job.runtime
        runtime.attempts += 1
        runtime.start_ts = time.time()
        client = self._get_client(job)
        indices = indices or list(range(array_len))
        try:
            job_ids = client.submit_array(definition, indices)
        except Exception as e:
            LOGGER.error(f"Unable to submit {job.definition.name}: {e}")
            job_ids = []

        for idx, runtime_job_id in zip(indices, job_ids):
            task_runtime = JobRuntime(
                submitted=True,
                attempts=runtime.attempts,
                runtime_job_id=runtime_job_id,
                start_ts=runtime.start_ts,
                log_cursor=0,
            )
            task_record = JobRecord(
                job_id=f"{job.job_id}_{idx}",
                definition=definition,
                runtime=task_runtime,
                array_idx=idx,
            )
            self._store.upsert(task_record)
        if job_ids:
            self._store.remove(job.job_id)

    def _process_log_events(self, job: JobRecord) -> str:
        """Process log events by checking patterns in new log content.

        Returns: "continue", "finished", "cancelled", or "restart"
        """
        runtime = job.runtime
        definition = job.definition
        log_path = self._resolve_log_path(job)
        if not log_path.exists():
            return "continue"

        try:
            with log_path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(runtime.log_cursor)
                new_text = handle.read()
                runtime.log_cursor = handle.tell()
        except OSError:
            return "continue"

        now = time.time()
        had_activity = bool(new_text)

        # Process each log event configuration (already parsed). Inactivity and
        # progress events must be evaluated every poll (including when there is
        # no new text — for progress, an empty slice IS the no-movement signal);
        # pattern events only when there is new text to scan.
        log_events = getattr(definition, "log_events", None) or []
        for idx, event_cfg in enumerate(log_events):
            if event_cfg.action is None:
                continue

            log_event = LogEvent(event_cfg)

            if event_cfg.pattern_type == "inactivity":
                effect = self._process_inactivity_event(
                    job, log_event, idx, had_activity=had_activity, now=now
                )
                if effect in ("finished", "cancelled", "restart"):
                    return effect
                continue

            if event_cfg.pattern_type == "progress":
                effect = self._process_progress_event(
                    job, log_event, idx, new_text=new_text, now=now
                )
                if effect in ("finished", "cancelled", "restart"):
                    return effect
                continue

            if not had_activity:
                continue

            # Check if event triggers
            triggers = log_event.check_triggers(new_text)

            for metadata in triggers:
                event_id = build_event_id(job.job_id, event_cfg.name, metadata)
                action_id = f"log:{event_cfg.name}:{idx}"
                action_state = runtime.action_state.get(action_id, {})

                event = EventRecord(
                    event_id=event_id,
                    name=event_cfg.name,
                    source="log",
                    payload=metadata,
                    metadata={
                        "job_id": job.job_id,
                        "job_name": definition.name,
                        "last_action_ts": float(action_state.get("last_action_ts", 0.0)),
                    },
                )

                # Instantiate and execute action
                action = event_cfg.action.instantiate(BaseMonitorAction)
                if self._evaluate_event_condition(job, event, event_cfg.condition, action_id):
                    result = action.execute(self._action_context(event, job))
                    self._update_action_state(job, action_id, result)
                    effect = self._handle_action_result(job, result)
                    if effect in ("finished", "cancelled", "restart"):
                        return effect

        return "continue"

    def _process_inactivity_event(
        self,
        job: JobRecord,
        log_event: LogEvent,
        idx: int,
        *,
        had_activity: bool,
        now: float,
    ) -> str:
        """Accumulate an inactivity streak and fire its action once it
        qualifies.

        The streak is stored as a persistent EventRecord (keyed by the stable
        event_key) in ``runtime.events``: ``count`` counts consecutive inactive
        polls and ``first_seen_ts``/``last_seen_ts`` bound the real elapsed time.
        Any poll with new log output breaks (clears) the streak.
        """
        runtime = job.runtime
        definition = job.definition
        event_cfg = log_event.config
        metadata = log_event.inactivity_metadata()
        key = ":".join(event_key(job.job_id, event_cfg.name, metadata))

        if had_activity:
            # Activity resets the streak.
            runtime.events.pop(key, None)
            return "continue"

        stored = runtime.events.get(key)
        if stored is None:
            action_state = runtime.action_state.get(f"log:{event_cfg.name}:{idx}", {})
            record = EventRecord(
                event_id=build_event_id(job.job_id, event_cfg.name, metadata),
                name=event_cfg.name,
                source="log",
                count=1,
                first_seen_ts=now,
                last_seen_ts=now,
                payload=metadata,
                metadata={
                    "job_id": job.job_id,
                    "job_name": definition.name,
                    "last_action_ts": float(action_state.get("last_action_ts", 0.0)),
                },
            )
        else:
            record = EventRecord(**stored)
            record.touch()
            record.last_seen_ts = now
        runtime.events[key] = asdict(record)

        elapsed_s = record.last_seen_ts - record.first_seen_ts
        if not log_event.inactivity_qualifies(count=record.count, elapsed_s=elapsed_s):
            return "continue"

        action_id = f"log:{event_cfg.name}:{idx}"
        action = event_cfg.action.instantiate(BaseMonitorAction)
        if not self._evaluate_event_condition(job, record, event_cfg.condition, action_id):
            return "continue"

        result = action.execute(self._action_context(record, job))
        self._update_action_state(job, action_id, result)
        # Clear the streak so it must re-accumulate before firing again.
        runtime.events.pop(key, None)
        return self._handle_action_result(job, result)

    def _process_progress_event(
        self,
        job: JobRecord,
        log_event: LogEvent,
        idx: int,
        *,
        new_text: str,
        now: float,
    ) -> str:
        """Accumulate a FORWARD-PROGRESS stall streak and fire its action.

        The inactivity check only asks whether the log grew, so it cannot see a
        job that is dead but noisy: an ft_launcher restart loop emits fresh
        setup banners forever (job 1375720 wrote 251 MB over 4 h while pinned at
        iteration 100), and a job whose ranks never finish startup keeps
        printing per-rank banners with zero iterations (jobs 1392564, 1392777).
        This parses a counter out of the log instead and asks whether TRAINING
        advanced.

        Streak bookkeeping mirrors :meth:`_process_inactivity_event`: a
        persistent EventRecord keyed by the stable ``progress_metadata`` key,
        ``count`` counting consecutive polls without movement and
        ``first_seen_ts``/``last_seen_ts`` bounding the real elapsed time. The
        observed counter lives in ``payload`` (never in the key). A poll in
        which the counter moved resets the streak in place — the record is kept
        so the tracked value survives.
        """
        runtime = job.runtime
        definition = job.definition
        event_cfg = log_event.config
        metadata = log_event.progress_metadata()
        key = ":".join(event_key(job.job_id, event_cfg.name, metadata))

        last_value, max_value, raw = log_event.observe_progress(new_text)

        stored = runtime.events.get(key)
        if stored is None:
            action_state = runtime.action_state.get(f"log:{event_cfg.name}:{idx}", {})
            record = EventRecord(
                event_id=build_event_id(job.job_id, event_cfg.name, metadata),
                name=event_cfg.name,
                source="log",
                count=0,
                first_seen_ts=now,
                last_seen_ts=now,
                payload=dict(metadata),
                metadata={
                    "job_id": job.job_id,
                    "job_name": definition.name,
                    "last_action_ts": float(action_state.get("last_action_ts", 0.0)),
                },
            )
        else:
            record = EventRecord(**stored)
            # A record carried across a restart by _restart_job keeps its
            # high-water mark and its banked no-progress time, but its clock
            # spans a queue wait in which no poll ran. Slide first_seen_ts
            # forward so elapsed_s resumes from the ACTIVE time already banked
            # and the PENDING gap contributes nothing: elapsed_s then measures
            # time the job spent RUNNING without net progress, summed across
            # restarts, which is what catches a restart loop.
            # `count` needs no adjustment — it only ever advanced on polls that
            # read a log.
            if record.payload.pop("progress_reanchor", False):
                banked = max(0.0, record.last_seen_ts - record.first_seen_ts)
                record.first_seen_ts = now - banked

        advanced = _progress_advanced(event_cfg, record, last_value, max_value)
        if advanced:
            # Movement: restart the streak from this poll, keeping the record so
            # the tracked counter is not lost.
            record.count = 0
            record.first_seen_ts = now
        else:
            record.count += 1
        record.last_seen_ts = now
        if raw is not None:
            record.payload["progress_raw"] = raw
        runtime.events[key] = asdict(record)

        if advanced:
            return "continue"

        elapsed_s = record.last_seen_ts - record.first_seen_ts
        if not log_event.progress_qualifies(count=record.count, elapsed_s=elapsed_s):
            return "continue"

        action_id = f"log:{event_cfg.name}:{idx}"
        action = event_cfg.action.instantiate(BaseMonitorAction)
        if not self._evaluate_event_condition(job, record, event_cfg.condition, action_id):
            return "continue"

        LOGGER.info(
            "Progress stall on job %s (%s): '%s' stuck at %s for %d polls / %.0fs (mode=%s) -> %s",
            job.job_id,
            runtime.runtime_job_id,
            event_cfg.name,
            record.payload.get("progress_raw", "<never seen>"),
            record.count,
            elapsed_s,
            event_cfg.progress_mode,
            type(action).__name__,
        )

        result = action.execute(self._action_context(record, job))
        self._update_action_state(job, action_id, result)
        # Clear the streak so it must re-accumulate before firing again.
        runtime.events.pop(key, None)
        return self._handle_action_result(job, result)

    def _apply_effect(self, job: JobRecord, effect: str, runtime_id: str | None) -> bool:
        """Advance a job's lifecycle for a terminal/restart action effect.

        Returns True if the job was finished, cancelled, or restarted
        (in which case the caller should stop processing it for this
        poll), False for the "continue" effect.
        """
        if effect == "finished":
            if runtime_id:
                self._get_client(job).remove(runtime_id)
            self._store.mark_finished(job.job_id, "finished")
            return True
        if effect == "cancelled":
            if runtime_id:
                client = self._get_client(job)
                client.cancel(runtime_id)
                client.remove(runtime_id)
            self._store.mark_finished(job.job_id, "cancelled")
            return True
        if effect == "restart":
            self._restart_job(job)
            self._store.upsert(job)
            return True
        return False

    def _status_action(self, job: JobRecord, old_status: str | None, new_status: str | None) -> str:
        """Process state transition events.

        Returns: "continue", "finished", "cancelled", or "restart". A terminal
        or restart effect lets a configured ``state_events`` action (e.g. treat
        a SLURM ``TIMEOUT`` as finished) drive the job lifecycle, the same way
        log events do via :meth:`_process_log_events`.
        """
        runtime = job.runtime
        definition = job.definition

        # Skip if no state change
        if old_status == new_status:
            return "continue"

        # Process each state event configuration (already parsed)
        state_events = getattr(definition, "state_events", None) or []
        for idx, event_cfg in enumerate(state_events):
            if event_cfg.action is None:
                continue

            # Create StateEvent instance
            state_event = StateEvent(event_cfg)

            # Check if event triggers
            if not state_event.check_trigger(old_status, new_status):
                continue

            # Build event metadata
            metadata = state_event.build_metadata(old_status, new_status)
            event_id = build_event_id(job.job_id, event_cfg.name, metadata)
            action_id = f"state:{event_cfg.name}:{idx}"
            action_state = runtime.action_state.get(action_id, {})

            event = EventRecord(
                event_id=event_id,
                name=event_cfg.name,
                source="state",
                payload=metadata,
                metadata={
                    "job_id": job.job_id,
                    "job_name": definition.name,
                    "last_action_ts": float(action_state.get("last_action_ts", 0.0)),
                },
            )

            # Instantiate and execute action
            action = event_cfg.action.instantiate(BaseMonitorAction)
            if self._evaluate_event_condition(job, event, event_cfg.condition, action_id):
                result = action.execute(self._action_context(event, job))
                self._update_action_state(job, action_id, result)
                effect = self._handle_action_result(job, result)
                if effect in ("finished", "cancelled", "restart"):
                    return effect

        return "continue"

    def _action_context(self, event: EventRecord, job: JobRecord):
        from oellm_autoexp.monitor.actions import ActionContext

        return ActionContext(
            event=event,
            job_metadata=self._build_job_metadata(job),
            attempts=job.runtime.attempts,
        )

    def _evaluate_event_condition(
        self,
        job: JobRecord,
        event: EventRecord,
        condition_cfg: MonitorConditionInterface.cfgtype | None,
        action_id: str,
    ) -> bool:
        """Evaluate a single event condition."""
        if condition_cfg is None:
            return True

        action_state = job.runtime.action_state.setdefault(action_id, {})
        condition_state = action_state.setdefault("condition", {})

        if "started_ts" not in condition_state:
            condition_state["started_ts"] = time.time()

        condition = condition_cfg.instantiate(MonitorConditionInterface)
        ctx = ConditionContext(
            event=event,
            job_metadata=self._build_job_metadata(job),
            attempts=job.runtime.attempts,
            # PER-EVENT budget input, distinct from `attempts` above.
            # `attempts` is the JOB-WIDE restart counter, so a MaxAttemptsCondition
            # on one event is really a ceiling on restarts from EVERY cause — on a
            # chained run dominated by healthy wall-clock rollovers that silently
            # disables the stricter events a few segments in. `action_fires`
            # counts only THIS event's own action executions (see
            # _update_action_state) and survives restarts because _restart_job
            # preserves action_state. Consumed by MaxActionFiresCondition.
            # Threaded via `extra` rather than the event metadata because
            # Composite/And/Or/Not all forward `extra` to their children already.
            extra={"action_fires": int(action_state.get("fire_count", 0))},
            state=condition_state,
            started_ts=condition_state.get("started_ts"),
        )
        result = condition.check(ctx)
        result = _apply_persistence(condition_cfg, condition_state, result)
        return result.passed

    def _evaluate_condition(
        self,
        job: JobRecord,
        condition_cfg: MonitorConditionInterface.cfgtype,
        *,
        label: str,
    ) -> ConditionResult:
        state = job.runtime.condition_state.setdefault(label, {})
        if "started_ts" not in state:
            state["started_ts"] = time.time()
        condition = condition_cfg.instantiate(MonitorConditionInterface)
        ctx = ConditionContext(
            job_metadata=self._build_job_metadata(job),
            attempts=job.runtime.attempts,
            state=state,
            started_ts=state.get("started_ts"),
        )
        result = condition.check(ctx)
        return _apply_persistence(condition_cfg, state, result)

    def _build_job_metadata(self, job: JobRecord) -> dict[str, Any]:
        definition = job.definition
        metadata = dict(definition.metadata)
        metadata.setdefault("job_id", job.job_id)
        metadata.setdefault("job_name", definition.name)
        job_class = definition.class_name
        metadata.setdefault("job_class", job_class)
        return metadata

    def _resolve_log_path(self, job: JobRecord) -> Path:
        """Resolve the job's OWN per-job log file (e.g. slurm-<jobid>.log).

        Deliberately does NOT use ``log_path_current``: in a dependency chain all
        jobs share one base_output_dir, so a shared 'current' symlink would make
        every job read whichever job is currently running (cross-contamination).
        The current.* symlinks are a tailing convenience maintained separately by
        :meth:`_update_current_symlinks` on the RUNNING transition.
        """
        definition = job.definition
        runtime = job.runtime
        timestamp = int(runtime.start_ts or time.time())
        return resolve_log_path(
            definition.log_path,
            job_id=runtime.runtime_job_id or job.job_id,
            timestamp=timestamp,
        )

    def _update_current_symlinks(self, job: JobRecord) -> None:
        """Point the job's current.* symlinks at its own log/config.

        Called when a job enters RUNNING so that, in a dependency chain
        (all jobs share one base_output_dir), current.log / current.yaml
        track the job that is actually running rather than the last-
        submitted one. Convenience only; the monitor reads each job's
        own per-job log (see _resolve_log_path).
        """
        definition = job.definition
        runtime_id = job.runtime.runtime_job_id
        if not runtime_id:
            return
        array_suffix = runtime_id.split("_")[-1] if "_" in runtime_id else "0"
        log_current = getattr(definition, "log_path_current", None)
        if log_current and getattr(definition, "log_path", None):
            update_log_symlink(
                expand_log_path(definition.log_path, runtime_id),
                Path(log_current.replace("%a", array_suffix)),
            )
        config_current = getattr(definition, "config_path_current", None)
        if config_current and getattr(definition, "config_path", None):
            update_log_symlink(
                expand_log_path(definition.config_path, runtime_id),
                Path(config_current.replace("%a", array_suffix)),
            )

    def _update_action_state(self, job: JobRecord, action_id: str, result) -> None:
        runtime = job.runtime
        state = runtime.action_state.setdefault(action_id, {})
        state["last_action_ts"] = time.time()
        state["last_status"] = result.status
        # Per-event budget counter read back by MaxActionFiresCondition. Counted
        # here rather than inside the condition because the condition is only a
        # GATE — a child of an And/Or may be evaluated on a poll where the action
        # never runs, so counting at check time would over-count. This runs
        # exactly once per actual execution. Preserved across restarts by
        # _restart_job, which deliberately keeps action_state.
        state["fire_count"] = int(state.get("fire_count", 0)) + 1
        LOGGER.info(
            "Action executed for job '%s' [%s]: special=%s status=%s fires=%d%s",
            job.job_id,
            action_id,
            result.special,
            result.status,
            state["fire_count"],
            f" reason='{result.message}'" if result.message else "",
        )

    def _handle_action_result(self, job: JobRecord, result: ActionResult) -> str:
        """Handle the result of an action execution.

        Returns: "continue", "finished", "cancelled", or "restart"
        """
        # Handle special actions
        if result.special == "restart":
            return "restart"

        if result.special == "cancel":
            return "cancelled"

        if result.special == "finish":
            return "finished"

        # Handle new job submissions using typed config
        if result.action_config is not None:
            if isinstance(result.action_config, NewJobActionConfig) and isinstance(
                result.action_config.job_config, LocalJobConfig
            ):
                self._submit_local_job(result.action_config.job_config)
            elif isinstance(result.action_config, NewJobActionConfig) and isinstance(
                result.action_config.job_config, SlurmJobConfig
            ):
                self._submit_slurm_job(result.action_config.job_config)
            elif isinstance(result.action_config, UpdateChainExcludesActionConfig):
                self._update_chain_excludes(result, job)

        return "continue"

    def _update_chain_excludes(self, result: ActionResult, current_job: JobRecord) -> None:
        """Set the current exclusion list on every pending sibling chain job.

        Reads the exclusion file (resolved by the action and carried in
        ``result.metadata``), then runs ``scontrol update ... ExcNodeList=`` on
        each *pending* Slurm job in the store other than ``current_job``. Pending
        is the only state SLURM lets us edit live; the failing/running job that
        triggered this is skipped (restart it separately to pick up the node).
        """
        if self._slurm_client is None:
            return
        exclude_file = (result.metadata or {}).get("exclude_file") or getattr(
            result.action_config, "exclude_file", ""
        )
        nodelist = oc_exclude_nodes(exclude_file)
        if not nodelist:
            LOGGER.info(
                "UpdateChainExcludes: exclusion list empty (%s), nothing to do", exclude_file
            )
            return

        statuses = self._slurm_client.squeue()
        updated: list[str] = []
        for sibling in self._store.load_all():
            if sibling.job_id == current_job.job_id:
                continue
            if not isinstance(sibling.definition, SlurmJobConfig):
                continue
            runtime_id = sibling.runtime.runtime_job_id
            if not runtime_id or not sibling.runtime.submitted:
                continue
            # Only pending jobs can be live-edited; treat "not yet visible in
            # squeue" (None) as pending since chain jobs are queued up front.
            if statuses.get(runtime_id) not in (None, "PENDING"):
                continue
            try:
                self._slurm_client.update_excludes(runtime_id, nodelist)
                updated.append(runtime_id)
            except Exception as exc:
                LOGGER.warning("UpdateChainExcludes: failed to update job %s: %s", runtime_id, exc)
        LOGGER.info(
            "UpdateChainExcludes: set ExcNodeList=%s on %d pending chain job(s): %s",
            nodelist,
            len(updated),
            updated,
        )

    def _restart_job(self, job: JobRecord) -> None:
        """Restart job preserving condition_state, action_state, and
        attempts."""
        runtime = job.runtime
        client = self._get_client(job)

        # Cancel and remove existing job if it exists
        if runtime.runtime_job_id:
            client.cancel(runtime.runtime_job_id)
            client.remove(runtime.runtime_job_id)

        # Reset runtime fields but preserve state and attempts
        runtime.submitted = False
        runtime.runtime_job_id = None
        runtime.log_cursor = 0
        runtime.start_ts = None
        # Streak records refer to the old run's log; clear so inactivity has to
        # re-accumulate against the fresh log.
        #
        # EXCEPT the HIGH-WATER MARK of a progress_mode: max streak, which is a
        # plain number and not log state. Without it, a restart LOOP is invisible
        # to the progress events: every cycle wipes the streak, so it never
        # reaches its 20/30 min window and the only bound left is the blunt
        # per-event restart budget. Carrying the mark means a job that keeps
        # relaunching and never gets back past the iteration it already reached
        # is finally detectable as "no NET progress", which is exactly what
        # progress_mode: max is for.
        #
        # THE COUNT AND THE ACCUMULATED TIME ARE CARRIED TOO; only the QUEUE GAP
        # is dropped. Zeroing them here would defeat the whole point: a loop
        # whose cycles each die in ~7 min would reset the streak every cycle and
        # could never reach a 30-poll window, so the fast loop -- the shape that
        # actually happens (six jobs in one hour, 2026-08-23) -- would stay
        # invisible. What must NOT be carried is wall-clock time spent PENDING:
        # _process_log_events returns early while the new job's log does not
        # exist, so no poll runs during the wait, and folding it into elapsed_s
        # would make the AND in progress_qualifies degrade to its poll-count
        # half. The record is flagged, and the first poll that actually sees a
        # log re-anchors it so elapsed_s continues from the ACTIVE time already
        # banked (see _process_progress_event).
        #
        # `count` is naturally queue-immune: it only increments on a poll that
        # read a log, so it already measures polls spent RUNNING.
        #
        # progress_last is NOT carried: progress_mode: increase treats the
        # backwards jump of a resume as movement on purpose, so it must start
        # fresh.
        carried: dict[str, Any] = {}
        for key, stored in runtime.events.items():
            payload = (stored or {}).get("payload") or {}
            if "progress_max" not in payload:
                continue
            record = dict(stored)
            record["payload"] = {
                k: v for k, v in payload.items() if k not in ("progress_last", "progress_raw")
            }
            record["payload"]["progress_reanchor"] = True
            carried[key] = record
        runtime.events.clear()
        runtime.events.update(carried)
        # Note: condition_state, action_state, and attempts are preserved

        # Restart the job
        self._start_job(job)
        if job.runtime.runtime_job_id is None:
            LOGGER.warning(f"Failed to re-start job: {job}")
            self._store.mark_finished(job.job_id, "cancelled")

    def _submit_local_job(self, job_config: ConfigInterface) -> None:
        """Submit a new local job (fire-and-forget)."""
        if not self._local_client:
            LOGGER.error("Cannot submit local job: no local client available")
            return

        try:
            job_instance = job_config.instantiate(JobInterface)
            job_id = self._local_client.submit(job_instance)
            LOGGER.info(f"Submitted local job {job_id}")
        except Exception as e:
            LOGGER.error(f"Failed to submit local job: {e}")

    def _submit_slurm_job(self, job_config: ConfigInterface) -> None:
        """Submit a new Slurm job (fire-and-forget)."""
        if not self._slurm_client:
            LOGGER.error("Cannot submit Slurm job: no Slurm client available")
            return

        try:
            job_instance = job_config.instantiate(JobInterface)
            job_id = self._slurm_client.submit(job_instance)
            LOGGER.info(f"Submitted Slurm job {job_id}")
        except Exception as e:
            LOGGER.error(f"Failed to submit Slurm job: {e}")


def _progress_advanced(
    event_cfg: LogEventConfig,
    record: EventRecord,
    last_value: float | None,
    max_value: float | None,
) -> bool:
    """Did the progress counter move this poll? Updates ``record.payload``.

    ``increase`` compares against the PREVIOUS observation and accepts any
    change, including the backwards jump of a resume from checkpoint — a
    healthy ft_launcher restart must not look like a stall.
    ``max`` compares against the running maximum, so re-running iterations the
    job has already done does NOT count as progress.

    A counter seen for the first time counts as movement, which starts the clock
    at the first iteration rather than at job submission. A job that never emits
    a match never moves, so its streak accumulates from the first poll — that is
    what catches a startup hang.
    """
    if event_cfg.progress_mode == "max":
        if max_value is None:
            return False
        previous = record.payload.get("progress_max")
        record.payload["progress_max"] = (
            max_value if previous is None else max(float(previous), max_value)
        )
        return previous is None or max_value > float(previous)

    if last_value is None:
        return False
    previous = record.payload.get("progress_last")
    record.payload["progress_last"] = last_value
    return previous is None or last_value != float(previous)


def _normalize_job_definition(job: JobRecord) -> None:
    """Parse and normalize all nested event configs in the job definition."""
    if job.definition is None:
        return

    # Parse log events
    if hasattr(job.definition, "log_events") and job.definition.log_events:
        parsed_log_events = []
        for item in job.definition.log_events:
            if isinstance(item, LogEventConfig):
                parsed_log_events.append(item)
            elif isinstance(item, dict):
                parsed_log_events.append(parse_config(LogEventConfig, item))
        job.definition.log_events = parsed_log_events

    # Parse state events
    if hasattr(job.definition, "state_events") and job.definition.state_events:
        parsed_state_events = []
        for item in job.definition.state_events:
            if isinstance(item, StateEventConfig):
                parsed_state_events.append(item)
            elif isinstance(item, dict):
                parsed_state_events.append(parse_config(StateEventConfig, item))
        job.definition.state_events = parsed_state_events


def _apply_persistence(
    condition_cfg: MonitorConditionInterface.cfgtype,
    condition_state: dict[str, Any],
    result: ConditionResult,
) -> ConditionResult:
    if condition_state.get("latched_pass"):
        return ConditionResult(passed=True, message=result.message, metadata=result.metadata)
    if condition_state.get("latched_fail"):
        return ConditionResult(passed=False, message=result.message, metadata=result.metadata)
    persistent_pass = bool(getattr(condition_cfg, "persistent_pass", False))
    persistent_fail = bool(getattr(condition_cfg, "persistent_fail", False))
    if result.passed and persistent_pass:
        condition_state["latched_pass"] = True
    if (not result.passed) and persistent_fail:
        condition_state["latched_fail"] = True
    return result


def _import_registry() -> None:
    import oellm_autoexp.monitor.actions  # noqa: F401
    import oellm_autoexp.monitor.conditions  # noqa: F401
    import oellm_autoexp.monitor.submission  # noqa: F401
    import oellm_autoexp.config.conditions  # noqa: F401

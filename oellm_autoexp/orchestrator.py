"""Orchestration utilities tying together configuration, sweeps, and SLURM."""

from __future__ import annotations

import asyncio
import logging
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from collections.abc import Mapping, Sequence, Iterable


from compoconf import asdict

from oellm_autoexp.backends.base import BackendJobSpec, LaunchCommand
from oellm_autoexp.config.evaluator import RuntimeConfig, evaluate
from oellm_autoexp.config.loader import load_config
from oellm_autoexp.config.schema import RootConfig, SlurmConfig
from oellm_autoexp.monitor.controller import JobRegistration, MonitorController
from oellm_autoexp.persistence import MonitorStateStore, StoredJob
from oellm_autoexp.slurm.client import BaseSlurmClient
from oellm_autoexp.slurm.template_renderer import render_template_file
from oellm_autoexp.slurm.validator import validate_job_script
from oellm_autoexp.sweep.expander import SweepPoint, expand_sweep
from oellm_autoexp.sweep.planner import JobPlan, build_job_plans
from oellm_autoexp.utils.start_condition import (
    resolve_start_condition_interval,
    wait_for_start_condition,
)


LOGGER = logging.getLogger(__name__)


@dataclass
class ExecutionPlan:
    config: RootConfig
    runtime: RuntimeConfig
    sweep_points: list[SweepPoint]
    jobs: list[JobPlan]


@dataclass
class RenderedArtifacts:
    job_scripts: list[Path]
    sweep_json: Path | None = None
    array_script: Path | None = None
    array_job_name: str | None = None
    sweep_entries: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SubmissionResult:
    """Return value capturing submission side-effects."""

    controller: MonitorController
    state_store: MonitorStateStore
    submitted_job_ids: list[str]

    @property
    def session_id(self) -> str:
        return self.state_store.session_id

    @property
    def submitted_jobs(self) -> list[str]:
        return list(self.submitted_job_ids)


def build_execution_plan(config: Path | RootConfig) -> ExecutionPlan:
    if isinstance(config, (str, Path)):
        root = load_config(config)
    else:
        root = config
    runtime = evaluate(root)
    points = expand_sweep(root.sweep)
    jobs = build_job_plans(root, points)
    return ExecutionPlan(config=root, runtime=runtime, sweep_points=points, jobs=jobs)


def render_scripts(plan: ExecutionPlan) -> RenderedArtifacts:
    template_path = Path(plan.config.slurm.template_path)

    preferred_script_dir = Path(plan.config.slurm.script_dir)
    try:
        preferred_script_dir.mkdir(parents=True, exist_ok=True)
        script_dir = preferred_script_dir
    except OSError as exc:
        fallback_script_dir = Path.cwd() / ".oellm_autoexp" / "scripts"
        fallback_script_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.warning(
            "Unable to create script directory %s (%s); using fallback %s",
            preferred_script_dir,
            exc,
            fallback_script_dir,
        )
        script_dir = fallback_script_dir

    job_scripts: list[Path] = []
    sweep_entries: list[dict[str, Any]] = []
    write_fallback_dir: Path | None = None

    for job_it, job in enumerate(plan.jobs):
        spec = BackendJobSpec(parameters=dict(job.parameters))
        plan.runtime.backend.validate(spec)
        launch_cmd = plan.runtime.backend.build_launch_command(spec)

        replacements = _build_replacements(plan.runtime, job, launch_cmd, escape_str=True)
        script_path = script_dir / f"{job.name}_{job_it}.sbatch"
        try:
            rendered = render_template_file(template_path, script_path, replacements)
        except OSError as exc:
            if write_fallback_dir is None:
                write_fallback_dir = Path.cwd() / ".oellm_autoexp" / "scripts"
                write_fallback_dir.mkdir(parents=True, exist_ok=True)
            fallback_path = write_fallback_dir / script_path.name
            LOGGER.warning(
                "Unable to write script to %s (%s); using fallback %s",
                script_path,
                exc,
                fallback_path,
            )
            rendered = render_template_file(template_path, fallback_path, replacements)
            script_path = fallback_path
        validate_job_script(rendered, job.name)
        job_scripts.append(script_path)

        sweep_entries.append(_build_sweep_entry(job, launch_cmd))

    sweep_path: Path | None = None
    if plan.config.sweep.store_sweep_json or plan.config.slurm.array:
        sweep_path = _write_sweep_json(plan, sweep_entries)

    array_script: Path | None = None
    array_job_name: str | None = None
    if plan.config.slurm.array and plan.jobs:
        array_job_name = f"{plan.config.project.name}-array"
        array_script = _render_array_script(
            plan,
            script_dir,
            template_path,
            sweep_path,
            array_job_name,
        )

    return RenderedArtifacts(
        job_scripts=job_scripts,
        sweep_json=sweep_path,
        array_script=array_script,
        array_job_name=array_job_name,
        sweep_entries=sweep_entries,
    )


def _ensure_state_store(plan: ExecutionPlan, *, session_id: str | None = None) -> MonitorStateStore:
    monitoring_state_dir = plan.config.project.monitoring_state_dir or plan.runtime.state_dir
    state_store = MonitorStateStore(monitoring_state_dir, session_id=session_id)
    if not state_store.session_path.exists():
        config_dict = asdict(plan.config)
        state_store.save_session(config_dict, plan.config.project.name)
        LOGGER.info("Created monitoring session: %s", state_store.session_path)
    return state_store


def _initialize_monitor_controller(
    plan: ExecutionPlan,
    client: BaseSlurmClient,
    state_store: MonitorStateStore,
) -> tuple[MonitorController, set[str]]:
    saved_jobs = state_store.load()
    controller = MonitorController(
        plan.runtime.monitor,
        client,
        plan.runtime.restart_policies,
        state_store=state_store,
    )
    restored_names = _restore_saved_jobs(controller, client, saved_jobs.values())
    return controller, restored_names


def submit_jobs(
    plan: ExecutionPlan,
    artifacts: RenderedArtifacts,
    slurm_client: BaseSlurmClient | None = None,
) -> SubmissionResult:
    client = slurm_client or plan.runtime.slurm_client
    state_store = _ensure_state_store(plan)
    controller, restored_names = _initialize_monitor_controller(plan, client, state_store)
    submitted_job_ids: list[str] = []

    monitor_config = plan.runtime.monitor.config

    job_script_map = {job.name: script for job, script in zip(plan.jobs, artifacts.job_scripts)}

    pending_jobs = [job for job in plan.jobs if job.name not in restored_names]

    use_array = (
        plan.config.slurm.array
        and artifacts.array_script is not None
        and getattr(client, "supports_array", False)
        and pending_jobs
    )

    if use_array:
        if not pending_jobs:
            return SubmissionResult(
                controller=controller,
                state_store=state_store,
                submitted_job_ids=submitted_job_ids,
            )

        for job in pending_jobs:
            if job.start_condition_cmd:
                interval = resolve_start_condition_interval(
                    job.start_condition_interval_seconds,
                    monitor_config,
                )
                wait_for_start_condition(
                    job.start_condition_cmd,
                    interval_seconds=interval,
                    logger=LOGGER,
                )

        submit_array = getattr(client, "submit_array")
        job_ids: list[str] = submit_array(  # type: ignore[misc]
            artifacts.array_job_name or plan.config.project.name,
            artifacts.array_script,
            [job.log_path for job in pending_jobs],
            [job.name for job in pending_jobs],
        )

        if len(job_ids) != len(pending_jobs):
            raise RuntimeError("SLURM client returned mismatched job ids for array submission")

        for job_id, job in zip(job_ids, pending_jobs):
            job_metadata: dict[str, Any] = {"parameters": dict(job.parameters)}
            if job.inactivity_threshold_seconds is not None:
                job_metadata["inactivity_threshold_seconds"] = job.inactivity_threshold_seconds

            registration = JobRegistration(
                name=job.name,
                script_path=artifacts.array_script,
                log_path=job.log_path,
                metadata=job_metadata,
                output_paths=job.output_paths,
                start_condition_cmd=job.start_condition_cmd,
                start_condition_interval_seconds=job.start_condition_interval_seconds,
                termination_string=job.termination_string,
                termination_command=job.termination_command,
                inactivity_threshold_seconds=job.inactivity_threshold_seconds,
            )
            controller.register_job(job_id, registration)
            submitted_job_ids.append(job_id)

        return SubmissionResult(
            controller=controller,
            state_store=state_store,
            submitted_job_ids=submitted_job_ids,
        )

    for job in pending_jobs:
        script = job_script_map[job.name]
        if job.start_condition_cmd:
            interval = resolve_start_condition_interval(
                job.start_condition_interval_seconds,
                monitor_config,
            )
            wait_for_start_condition(
                job.start_condition_cmd,
                interval_seconds=interval,
                logger=LOGGER,
            )
        job_metadata: dict[str, Any] = {"parameters": dict(job.parameters)}
        if job.inactivity_threshold_seconds is not None:
            job_metadata["inactivity_threshold_seconds"] = job.inactivity_threshold_seconds

        job_id = client.submit(job.name, script, job.log_path)
        registration = JobRegistration(
            name=job.name,
            script_path=script,
            log_path=job.log_path,
            metadata=job_metadata,
            output_paths=job.output_paths,
            start_condition_cmd=job.start_condition_cmd,
            start_condition_interval_seconds=job.start_condition_interval_seconds,
            termination_string=job.termination_string,
            termination_command=job.termination_command,
            inactivity_threshold_seconds=job.inactivity_threshold_seconds,
        )
        controller.register_job(job_id, registration)
        submitted_job_ids.append(job_id)

    return SubmissionResult(
        controller=controller, state_store=state_store, submitted_job_ids=submitted_job_ids
    )


def load_monitor_controller(
    plan: ExecutionPlan,
    slurm_client: BaseSlurmClient | None = None,
    *,
    session_id: str | None = None,
) -> SubmissionResult:
    """Load monitor controller without submitting new jobs.

    Returns a ``SubmissionResult`` with an empty ``submitted_job_ids`` list so callers
    can reuse the same structure as ``submit_jobs`` when wiring the monitor.
    """

    client = slurm_client or plan.runtime.slurm_client
    state_store = _ensure_state_store(plan, session_id=session_id)
    controller, restored_names = _initialize_monitor_controller(plan, client, state_store)

    # Warn if the execution plan contains jobs that were never submitted.
    pending_jobs = [job.name for job in plan.jobs if job.name not in restored_names]
    if pending_jobs:
        LOGGER.warning(
            "Monitoring session %s missing submissions for jobs: %s",
            state_store.session_id,
            ", ".join(pending_jobs),
        )

    return SubmissionResult(controller=controller, state_store=state_store, submitted_job_ids=[])


async def execute_plan(
    plan: ExecutionPlan,
    controller: MonitorController,
) -> None:
    if controller is None:
        raise ValueError("execute_plan requires an initialized MonitorController")

    if plan.runtime.monitor.__class__.__name__ == "NullMonitor":
        controller.clear_state()
        return

    monitor_interval = getattr(
        plan.runtime.monitor.config,
        "poll_interval_seconds",
        getattr(plan.runtime.monitor.config, "check_interval_seconds", 60),
    )

    while controller.jobs():
        await asyncio.sleep(max(1, int(monitor_interval)))
        await controller.observe_once()

    controller.clear_state()


def execute_plan_sync(plan: ExecutionPlan, controller: MonitorController) -> None:
    asyncio.run(execute_plan(plan, controller))


def _build_replacements(
    runtime: RuntimeConfig,
    job: JobPlan,
    launch_cmd: LaunchCommand,
    escape_str: bool = True,
    sbatch_overrides: dict[str, str] = {},
) -> dict[str, str]:
    sbatch_directives = _format_sbatch_directives(
        runtime.root.slurm, job, sbatch_overrides=sbatch_overrides
    )
    backend_cmd = _format_command(launch_cmd.argv, escape_str=escape_str)
    launcher_raw = runtime.root.slurm.launcher_cmd.strip()
    launcher = f"{launcher_raw} " if launcher_raw else ""
    launcher_env_flags = ""
    launcher_env_exports = ""
    if runtime.root.slurm.env or launch_cmd.env:
        environ = dict(**runtime.root.slurm.env)
        environ.update(**launch_cmd.env)
        env_flags = " ".join(f"--env {key}=${key}" for key in environ.keys())
        env_exports = "; ".join(f"export {key}='\"${key}\"'" for key in environ.keys())
        if env_flags:
            launcher_env_flags = f"{env_flags} "
        if env_exports:
            launcher_env_exports = f"{env_exports};"

    def escape_for_double_quotes(s: str) -> str:
        s = s.replace("\\", "\\\\")  # Backslash first!
        s = s.replace("'", "'\"'\"'")
        return s

    launcher = escape_for_double_quotes(launcher)
    # Only escape backend_cmd if escape_str is True
    # When False (e.g., for array jobs), preserve shell variable expansion
    if escape_str:
        backend_cmd = escape_for_double_quotes(backend_cmd)

    if "{{env_flags}}" in launcher:
        launcher = launcher.replace("{{env_flags}}", launcher_env_flags)
    if "{{env_exports}}" in launcher:
        launcher = launcher.replace("{{env_exports}}", launcher_env_exports)

    launcher_cmd = f"{launcher}{backend_cmd}"
    srun_extras = asdict(runtime.root.slurm.srun)
    srun_opts = _format_srun_options(runtime.root.slurm.srun_opts, srun_extras)
    if srun_opts:
        srun_opts = f"{srun_opts} "

    repl = {
        "name": job.name,
        "output_dir": str(job.output_dir),
        "log_path": str(job.log_path),
        "launcher_cmd": launcher_cmd,
        "launcher": launcher,
        "launcher_env_flags": launcher_env_flags,
        "backend_cmd": backend_cmd,
        "env_exports": _format_env(runtime.root.slurm.env),
        "sbatch_directives": sbatch_directives,
        "srun_opts": srun_opts,
    }
    if job.start_condition_cmd:
        repl["start_condition_cmd"] = job.start_condition_cmd
    if job.termination_string:
        repl["termination_string"] = job.termination_string
    if job.termination_command:
        repl["termination_command"] = job.termination_command
    if job.inactivity_threshold_seconds is not None:
        repl["inactivity_threshold_seconds"] = str(job.inactivity_threshold_seconds)
    repl.update(job.parameters)
    repl.update({"project": runtime.root.project.name})
    return repl


def _build_sweep_entry(job: JobPlan, launch_cmd: LaunchCommand) -> dict[str, Any]:
    return {
        "name": job.name,
        "output_dir": str(job.output_dir),
        "log_path": str(job.log_path),
        "parameters": dict(job.parameters),
        "launch": {
            "argv": list(launch_cmd.argv),
            "env": dict(launch_cmd.env),
        },
    }


def _write_sweep_json(plan: ExecutionPlan, entries: list[dict[str, Any]]) -> Path:
    base_output = Path(plan.config.project.base_output_dir)
    try:
        base_output.mkdir(parents=True, exist_ok=True)
        sweep_dir = base_output
    except OSError as exc:
        fallback_output = Path.cwd() / ".oellm_autoexp" / "output"
        fallback_output.mkdir(parents=True, exist_ok=True)
        LOGGER.warning(
            "Unable to create base output dir %s (%s); using fallback %s",
            base_output,
            exc,
            fallback_output,
        )
        sweep_dir = fallback_output
    sweep_path = sweep_dir / "sweep.json"
    payload = {
        "project": plan.config.project.name,
        "jobs": entries,
    }
    try:
        sweep_path.write_text(json.dumps(payload, indent=2))
    except OSError as exc:
        fallback_output = Path.cwd() / ".oellm_autoexp" / "output"
        fallback_output.mkdir(parents=True, exist_ok=True)
        sweep_path = fallback_output / "sweep.json"
        LOGGER.warning(
            "Unable to write sweep metadata to %s (%s); using fallback %s",
            sweep_dir / "sweep.json",
            exc,
            sweep_path,
        )
        sweep_path.write_text(json.dumps(payload, indent=2))
    return sweep_path


def _restore_saved_jobs(
    controller: MonitorController,
    slurm_client: BaseSlurmClient,
    saved_jobs: Iterable[StoredJob],
) -> set[str]:
    restored: set[str] = set()
    for saved in saved_jobs:
        registration = JobRegistration(
            name=saved.name,
            script_path=Path(saved.script_path),
            log_path=Path(saved.log_path),
            metadata=dict(saved.metadata),
            output_paths=[Path(p) for p in saved.output_paths],
            termination_string=saved.termination_string,
            termination_command=saved.termination_command,
            inactivity_threshold_seconds=saved.inactivity_threshold_seconds,
            start_condition_cmd=saved.start_condition_cmd,
            start_condition_interval_seconds=saved.start_condition_interval_seconds,
        )
        if hasattr(slurm_client, "register_job"):
            try:
                slurm_client.register_job(  # type: ignore[attr-defined]
                    saved.job_id,
                    saved.name,
                    Path(saved.script_path),
                    Path(saved.log_path),
                )
            except TypeError:
                pass
        controller.register_job(saved.job_id, registration, attempts=saved.attempts)
        restored.add(saved.name)
    return restored


def _format_command(argv: Sequence[str], escape_str: bool = True) -> str:
    # do ? escape ENV VARIABLES
    # return " ".join(shlex.quote(arg) if escape_str else arg for arg in argv)
    def _escape(s: str) -> str:
        s = s.replace("\\", "\\\\")
        if " " in s and not (s.startswith("'") and s.endswith("'")):
            s = "'" + s + "'"
        return s

    return " ".join(_escape(arg) if escape_str else arg for arg in argv)


def _format_env(base_env: Mapping[str, object]) -> str:
    return "\n".join(f"export {key}={value}" for key, value in base_env.items())


def _format_sbatch_directives(
    slurm_conf: SlurmConfig, job: JobPlan, sbatch_overrides: dict[str, str] = {}
) -> str:
    sbatch_kwargs = asdict(slurm_conf.sbatch)
    sbatch_kwargs.update(sbatch_overrides)
    sbatch_values = {
        k.replace("_", "-"): v
        for k, v in sbatch_kwargs.items()
        if v is not None and not k.startswith("_")
    }
    override_values = {
        k.replace("_", "-"): v
        for k, v in slurm_conf.sbatch_overrides.items()
        if v is not None and not k.startswith("_")
    }
    sbatch_values.update(override_values)
    sbatch_values.setdefault("job-name", job.name)
    sbatch_values.setdefault("output", str(job.log_path))
    lines = [f"#SBATCH --{key}={value}" for key, value in sbatch_values.items()]
    lines.extend(slurm_conf.sbatch_extra_directives)
    return "\n".join(lines)


def _format_srun_options(srun_opts: str, extras: Mapping[str, Any]) -> str:
    options: list[str] = []
    if extras:
        for key, value in extras.items():
            if key.startswith("_"):
                continue
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    options.append(flag)
            else:
                options.append(f"{flag}={value}")
    srun_opts = srun_opts.strip()
    if srun_opts:
        options.append(srun_opts)
    return " ".join(options).strip()


def _render_array_script(
    plan: ExecutionPlan,
    script_dir: Path,
    template_path: Path,
    sweep_path: Path | None,
    array_job_name: str,
) -> Path:
    if sweep_path is None:
        raise RuntimeError("sweep.json must be available for array submission")

    array_log_dir = Path(plan.config.slurm.log_dir)
    try:
        array_log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        fallback_log_dir = Path.cwd() / ".oellm_autoexp" / "logs"
        fallback_log_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.warning(
            "Unable to create log directory %s (%s); using fallback %s",
            array_log_dir,
            exc,
            fallback_log_dir,
        )
        array_log_dir = fallback_log_dir
    array_log_path = array_log_dir / f"{array_job_name}_%a.log"
    array_output_base = Path(plan.config.project.base_output_dir)
    try:
        array_output_base.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        fallback_output = Path.cwd() / ".oellm_autoexp" / "output"
        fallback_output.mkdir(parents=True, exist_ok=True)
        LOGGER.warning(
            "Unable to create array output dir %s (%s); using fallback %s",
            array_output_base,
            exc,
            fallback_output,
        )
        array_output_base = fallback_output
    array_output_dir = array_output_base / f"{array_job_name}_%a"

    runner_script = Path("scripts/run_sweep_entry.py")
    launcher_cmd = LaunchCommand(
        argv=[
            str(runner_script),
            "--sweep",
            str(sweep_path),
            "--index",
            "$SLURM_ARRAY_TASK_ID",
        ],
        env={"SLURM_ARRAY_TASK_ID": "$SLURM_ARRAY_TASK_ID"},
    )

    synthetic_job = JobPlan(
        name=array_job_name,
        parameters={},
        output_dir=array_output_dir,
        log_path=array_log_path,
        output_paths=[],
    )

    replacements = _build_replacements(
        plan.runtime,
        synthetic_job,
        launcher_cmd,
        escape_str=False,
        sbatch_overrides={"array": f"0-{len(plan.sweep_points) - 1}"},
    )
    script_path = script_dir / f"{array_job_name}.sbatch"
    try:
        rendered = render_template_file(template_path, script_path, replacements)
    except OSError as exc:
        fallback_script_dir = Path.cwd() / ".oellm_autoexp" / "scripts"
        fallback_script_dir.mkdir(parents=True, exist_ok=True)
        fallback_script_path = fallback_script_dir / f"{array_job_name}.sbatch"
        LOGGER.warning(
            "Unable to write array script to %s (%s); using fallback %s",
            script_path,
            exc,
            fallback_script_path,
        )
        rendered = render_template_file(template_path, fallback_script_path, replacements)
        script_path = fallback_script_path
    validate_job_script(rendered, array_job_name)
    return script_path


__all__ = [
    "ExecutionPlan",
    "RenderedArtifacts",
    "SubmissionResult",
    "build_execution_plan",
    "render_scripts",
    "submit_jobs",
    "load_monitor_controller",
    "execute_plan",
    "execute_plan_sync",
]

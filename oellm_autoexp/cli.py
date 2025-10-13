"""Command line interface for oellm_autoexp."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import click

from oellm_autoexp.config import schema
from oellm_autoexp.config.evaluator import evaluate
from oellm_autoexp.config.loader import (
    ensure_registrations,
    load_config_reference,
    load_monitoring_reference,
)
from oellm_autoexp.monitor.controller import JobRegistration, MonitorController
from oellm_autoexp.orchestrator import (
    build_execution_plan,
    execute_plan_sync,
    render_scripts,
    submit_jobs,
)
from oellm_autoexp.slurm.client import FakeSlurmClient, FakeSlurmClientConfig
from oellm_autoexp.utils.run import run_with_tee


@click.group()
def cli() -> None:
    """Unified entry point for OELLM experiment automation."""


@cli.command()
@click.option("--config-ref", default="autoexp")
@click.option("--config-dir", type=click.Path(path_type=Path), default=Path("config"))
@click.option("--override", multiple=True, help="Hydra-style overrides")
def plan(config_ref: str, config_dir: Path, override: tuple[str, ...]) -> None:
    """Validate configuration and emit a summary."""

    root = load_config_reference(config_ref, config_dir, override)
    execution_plan = build_execution_plan(root)
    artifacts = render_scripts(execution_plan)
    summary = {
        "project": execution_plan.config.project.name,
        "output_dir": str(execution_plan.config.project.base_output_dir),
        "backend_config": execution_plan.runtime.backend.config.class_name,
        "jobs": len(execution_plan.jobs),
        "array": bool(artifacts.array_script is not None),
    }
    click.echo(json.dumps(summary, indent=2))


@cli.command()
@click.option("--config-ref", default="autoexp")
@click.option("--config-dir", type=click.Path(path_type=Path), default=Path("config"))
@click.option("--fake", is_flag=True, help="Use the in-memory SLURM simulator")
@click.option("--override", multiple=True, help="Hydra-style overrides")
def submit(config_ref: str, config_dir: Path, override: tuple[str, ...], fake: bool) -> None:
    """Render scripts and submit jobs."""
    root = load_config_reference(config_ref, config_dir, override)
    execution_plan = build_execution_plan(root)
    artifacts = render_scripts(execution_plan)

    slurm_client = execution_plan.runtime.slurm_client
    if fake:
        slurm_client = FakeSlurmClient(FakeSlurmClientConfig())
        slurm_client.configure(execution_plan.config.slurm)

    controller = submit_jobs(execution_plan, artifacts, slurm_client)
    for state in controller.jobs():
        click.echo(f"submitted {state.name} -> job {state.job_id} -> log: {state.registration.log_path}")

    execute_plan_sync(
        execution_plan,
        slurm_client=slurm_client,
        artifacts=artifacts,
        controller=controller,
    )


@cli.command()
@click.option("--config-ref", default="autoexp")
@click.option("--monitor-config", type=click.Path(path_type=Path), help="Path to monitoring YAML")
@click.option("--config-dir", type=click.Path(path_type=Path), default=Path("config"))
@click.option("--log", "logs", multiple=True, help="JOB=PATH to the SLURM log to follow")
@click.option("--job", "jobs", multiple=True, help="JOB=JOB_ID to reuse an existing SLURM id")
@click.option("--slurm-state", "slurm_states", multiple=True, help="JOB=STATE to seed fake SLURM status")
@click.option("--action", "actions", multiple=True, help="ACTION=command template executed on signal")
@click.option("--loop", is_flag=True, help="Continuously poll instead of a single iteration")
@click.option("--interval", type=int, help="Polling interval override (seconds)")
@click.option("--dry-run", is_flag=True, help="Render commands without executing them")
@click.option("--json-output", is_flag=True, help="Emit JSON lines for each detected action")
@click.option("--override", multiple=True, help="Hydra-style overrides")
def monitor(
    config_ref: str | None,
    monitor_config: Path | None,
    config_dir: Path,
    logs: tuple[str, ...],
    jobs: tuple[str, ...],
    slurm_states: tuple[str, ...],
    actions: tuple[str, ...],
    loop: bool,
    interval: int | None,
    dry_run: bool,
    json_output: bool,
    override: tuple[str, ...],
) -> None:
    """Run monitoring standalone using monitoring config and log signals."""

    if not logs:
        raise click.ClickException("At least one --log JOB=PATH argument is required")

    monitor_obj, slurm_client, policies = _load_monitor_components(
        config_ref,
        monitor_config,
        config_dir,
        override,
    )
    controller = MonitorController(monitor_obj, slurm_client, policies)

    job_id_map: dict[str, int] = {}
    requested_job_ids: dict[str, int] = {}
    for entry in jobs:
        key, value = _parse_assignment(entry, "--job")
        requested_job_ids[key] = int(value)

    slurm_state_map: dict[str, str] = {}
    for entry in slurm_states:
        key, value = _parse_assignment(entry, "--slurm-state")
        slurm_state_map[key] = value

    action_map: dict[str, str] = {}
    for entry in actions:
        key, value = _parse_assignment(entry, "--action")
        action_map[key] = value

    for log_entry in logs:
        name, path_value = _parse_assignment(log_entry, "--log")
        log_path = Path(path_value).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.touch(exist_ok=True)
        script_path = log_path.with_suffix(".sbatch")
        script_path.parent.mkdir(parents=True, exist_ok=True)

        requested_id = requested_job_ids.get(name)
        if isinstance(slurm_client, FakeSlurmClient):
            if requested_id is not None:
                job_id = slurm_client.register_job(requested_id, name, script_path, log_path)
            else:
                job_id = slurm_client.submit(name, script_path, log_path)
        else:
            job_id = requested_id if requested_id is not None else slurm_client.submit(name, script_path, log_path)

        controller.register_job(
            job_id,
            JobRegistration(
                name=name,
                script_path=script_path,
                log_path=log_path,
                metadata={"log_path": str(log_path)},
            ),
        )
        job_id_map[name] = job_id

        state_override = slurm_state_map.get(name)
        if state_override and hasattr(slurm_client, "set_state"):
            slurm_client.set_state(job_id, state_override)

    poll_interval = interval or _monitor_interval_seconds(monitor_obj)

    exit_code = 0
    while True:
        result = controller.observe_once_sync()
        for action in result.actions:
            payload = {
                "job_id": action.job_id,
                "job_name": action.job_name,
                **{k: str(v) for k, v in action.metadata.items()},
            }
            _emit_action(action.action, payload, action_map, dry_run, json_output)
            if not dry_run:
                exit_code = max(exit_code, _execute_mapped_action(action.action, payload, action_map))
        controller.drain_actions()

        if not loop:
            break
        time.sleep(max(1, int(poll_interval)))

    if exit_code:
        raise SystemExit(exit_code)


def _load_monitor_components(
    config_ref: str | None,
    monitor_config: Path | None,
    config_dir: Path,
    overrides: tuple[str, ...],
):
    if monitor_config is not None:
        ensure_registrations()
        monitor_cfg = load_monitoring_reference(monitor_config, config_dir, overrides)
        monitor = monitor_cfg.instantiate(schema.MonitorInterface)
        slurm_client = FakeSlurmClient(FakeSlurmClientConfig())
        return monitor, slurm_client, {}

    ref = config_ref or "autoexp"
    root = load_config_reference(ref, config_dir, overrides)
    runtime = evaluate(root)
    return runtime.monitor, runtime.slurm_client, runtime.restart_policies


def _monitor_interval_seconds(monitor) -> int:
    return int(
        getattr(
            monitor.config,
            "check_interval_seconds",
            getattr(monitor.config, "poll_interval_seconds", 60),
        )
    )


def _parse_assignment(value: str, option: str) -> tuple[str, str]:
    if "=" not in value:
        raise click.BadParameter(f"Expected KEY=VALUE format for {option}")
    key, raw = value.split("=", 1)
    key = key.strip()
    raw = raw.strip()
    if not key or not raw:
        raise click.BadParameter(f"Invalid assignment for {option}: {value}")
    return key, raw


def _emit_action(
    action: str, payload: dict[str, str], action_map: dict[str, str], dry_run: bool, json_output: bool
) -> None:
    if json_output:
        click.echo(json.dumps({"action": action, "payload": payload}))
    else:
        meta_items = [f"{k}={v}" for k, v in sorted(payload.items()) if k not in {"job_name", "job_id"}]
        suffix = f" ({', '.join(meta_items)})" if meta_items else ""
        click.echo(f"[{action}] job={payload['job_name']}#{payload['job_id']}{suffix}")
    if dry_run and action in action_map:
        preview = _render_action_command(action_map[action], payload)
        click.echo(f"DRY-RUN command: {preview}")


def _execute_mapped_action(action: str, payload: dict[str, str], action_map: dict[str, str]) -> int:
    template = action_map.get(action)
    if not template:
        return 0
    command = _render_action_command(template, payload)
    completed = run_with_tee(command, shell=True, check=False)
    if completed.returncode != 0:
        click.echo(
            f"Command '{command}' for action '{action}' failed with return code {completed.returncode}",
            err=True,
        )
    return completed.returncode


def _render_action_command(template: str, payload: dict[str, str]) -> str:
    try:
        return template.format(**payload)
    except KeyError as exc:  # pragma: no cover - validation path
        missing = exc.args[0]
        raise click.ClickException(f"Missing field '{missing}' for action template '{template}'") from exc


def main() -> None:  # pragma: no cover - exercised via CLI
    cli()


if __name__ == "__main__":  # pragma: no cover
    main()

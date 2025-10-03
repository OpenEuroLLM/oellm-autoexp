"""Command line interface for oellm_autoexp."""

from __future__ import annotations

import json
from pathlib import Path

import click

from oellm_autoexp.config.loader import load_config_reference
from oellm_autoexp.orchestrator import build_execution_plan, render_scripts
from oellm_autoexp.slurm.fake_sbatch import FakeSlurm


@click.group()
def cli() -> None:
    """Unified entry point for OELLM experiment automation."""


@cli.command()
@click.argument("config_ref", default="autoexp")
@click.option("--config-dir", type=click.Path(path_type=Path), default=Path("config"))
@click.option("-o", "--override", multiple=True, help="Hydra-style overrides")
def plan(config_ref: str, config_dir: Path, override: tuple[str, ...]) -> None:
    """Validate configuration and emit a summary."""

    root = load_config_reference(config_ref, config_dir, override)
    execution_plan = build_execution_plan(root)
    render_scripts(execution_plan)
    summary = {
        "project": execution_plan.config.project.name,
        "output_dir": str(execution_plan.config.project.base_output_dir),
        "backend_config": execution_plan.runtime.backend.config.class_name,
        "jobs": len(execution_plan.jobs),
    }
    click.echo(json.dumps(summary, indent=2))


@cli.command()
@click.argument("config_ref", default="autoexp")
@click.option("--config-dir", type=click.Path(path_type=Path), default=Path("config"))
@click.option("-o", "--override", multiple=True, help="Hydra-style overrides")
@click.option("--fake", is_flag=True, help="Use the in-memory SLURM simulator")
def submit(config_ref: str, config_dir: Path, override: tuple[str, ...], fake: bool) -> None:
    """Render scripts and submit jobs."""

    root = load_config_reference(config_ref, config_dir, override)
    execution_plan = build_execution_plan(root)
    scripts = render_scripts(execution_plan)

    if not fake:
        raise click.ClickException("Real SLURM submission not implemented yet")

    slurm = FakeSlurm()
    for job, script in zip(execution_plan.jobs, scripts):
        job_id = slurm.submit(job.name, script, job.log_path)
        click.echo(f"submitted {job.name} -> job {job_id}")



def main() -> None:  # pragma: no cover - exercised via CLI
    cli()


if __name__ == "__main__":  # pragma: no cover
    main()

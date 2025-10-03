"""Orchestration utilities tying together configuration, sweeps, and SLURM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import shlex

from oellm_autoexp.config.evaluator import RuntimeConfig, evaluate
from oellm_autoexp.config.loader import load_config
from oellm_autoexp.config.schema import RootConfig, SlurmConfig
from oellm_autoexp.slurm.template_renderer import render_template_file
from oellm_autoexp.slurm.validator import validate_job_script
from oellm_autoexp.sweep.expander import SweepPoint, expand_sweep
from oellm_autoexp.sweep.planner import JobPlan, build_job_plans
from oellm_autoexp.backends.base import BackendJobSpec, LaunchCommand


@dataclass
class ExecutionPlan:
    config: RootConfig
    runtime: RuntimeConfig
    sweep_points: List[SweepPoint]
    jobs: List[JobPlan]


def build_execution_plan(config: Path | RootConfig) -> ExecutionPlan:
    if isinstance(config, (str, Path)):
        root = load_config(config)
    else:
        root = config
    runtime = evaluate(root)
    points = expand_sweep(root.sweep)
    jobs = build_job_plans(root, points)
    return ExecutionPlan(config=root, runtime=runtime, sweep_points=points, jobs=jobs)


def render_scripts(plan: ExecutionPlan) -> List[Path]:
    rendered_paths: List[Path] = []
    template_path = Path(plan.config.slurm.template_path)
    for job in plan.jobs:
        spec = BackendJobSpec(parameters=dict(job.parameters))
        plan.runtime.backend.validate(spec)
        launch_cmd = plan.runtime.backend.build_launch_command(spec)

        replacements = _build_replacements(plan.runtime, job, launch_cmd)
        script_path = Path(plan.config.slurm.script_dir) / f"{job.name}.sbatch"
        rendered = render_template_file(template_path, script_path, replacements)
        validate_job_script(rendered, job.name)
        rendered_paths.append(script_path)
    return rendered_paths


def _build_replacements(runtime: RuntimeConfig, job: JobPlan, launch_cmd: LaunchCommand) -> Dict[str, str]:
    sbatch_directives = _format_sbatch_directives(runtime.root.slurm, job)
    launcher_cmd = _compose_launcher_command(runtime.root.slurm, launch_cmd)
    srun_opts = runtime.root.slurm.srun_opts.strip()
    if srun_opts:
        srun_opts = f"{srun_opts} "

    repl = {
        "name": job.name,
        "output_dir": str(job.output_dir),
        "log_path": str(job.log_path),
        "launcher_cmd": launcher_cmd,
        "env_exports": _format_env(runtime.root.project.environment, launch_cmd.environment),
        "sbatch_directives": sbatch_directives,
        "srun_opts": srun_opts,
    }
    repl.update(job.parameters)
    repl.update({"project": runtime.root.project.name})
    return repl


def _format_command(argv: Sequence[str]) -> str:
    return " ".join(shlex.quote(arg) for arg in argv)


def _format_env(project_env: Mapping[str, object], command_env: Mapping[str, object]) -> str:
    merged: Dict[str, str] = {}
    merged.update({k: str(v) for k, v in project_env.items()})
    merged.update({k: str(v) for k, v in command_env.items()})
    if not merged:
        return ""
    return "\n".join(f"export {key}={value}" for key, value in merged.items())


def _format_sbatch_directives(slurm_conf: SlurmConfig, job: JobPlan) -> str:
    overrides = dict(slurm_conf.sbatch_overrides)
    overrides.setdefault("job-name", job.name)
    overrides.setdefault("output", str(job.log_path))
    lines = [f"#SBATCH --{k.replace('_', '-')}={v}" for k, v in overrides.items()]
    lines.extend(slurm_conf.sbatch_extra_directives)
    return "\n".join(lines)


def _compose_launcher_command(slurm_conf: SlurmConfig, launch_cmd: LaunchCommand) -> str:
    launcher_prefix = slurm_conf.launcher_cmd.strip()
    backend_cmd = _format_command(launch_cmd.argv)
    if launcher_prefix:
        return f"{launcher_prefix} {backend_cmd}".strip()
    return backend_cmd


__all__ = ["ExecutionPlan", "build_execution_plan", "render_scripts"]

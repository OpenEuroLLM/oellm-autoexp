"""Build execution plans from sweep outputs."""

from __future__ import annotations

from dataclasses import dataclass, field, MISSING
from pathlib import Path
import re

from oellm_autoexp.config.schema import RootConfig
from .expander import SweepPoint


@dataclass(kw_only=True)
class JobPlan:
    """Normalized job description used by downstream modules."""

    name: str = field(default_factory=MISSING)
    parameters: dict[str, str] = field(default_factory=MISSING)
    output_dir: str = field(default_factory=MISSING)
    log_path: str = field(default_factory=MISSING)
    output_paths: list[str] = field(default_factory=list)
    start_condition_cmd: str | None = None
    start_condition_interval_seconds: int | None = None
    termination_string: str | None = None
    termination_command: str | None = None
    inactivity_threshold_seconds: int | None = None


def build_job_plans(config: RootConfig, points: list[SweepPoint]) -> list[JobPlan]:
    base_output = Path(config.project.base_output_dir)
    project_name = config.project.name

    plans: list[JobPlan] = []
    for point in points:
        context: dict[str, str] = {
            "project": project_name,
            "index": str(point.index),
        }
        # TODO: Enable sweeping based on global naming scheme, not just within backend.
        context.update(
            {key.replace(".", "___"): str(value) for key, value in point.parameters.items()}
        )
        job_name = config.sweep.name_template.format(**context)
        t = config.sweep.name_template
        m = re.search(r"\{([^\{]*)\.([^\}]*)\}", t)
        while m:
            t = t[: m.start()] + "{" + m.group(1) + "___" + m.group(2) + "}"
            m = re.search(r"\{([^\{]*)\.([^\}]*)\}", t)

        output_dir = str(base_output / job_name)
        log_template = config.monitoring.log_path_template
        format_context = {**context, "output_dir": str(output_dir), "name": job_name}
        log_path = log_template.format(**format_context)
        monitoring_config = config.monitoring
        output_templates = getattr(monitoring_config, "output_paths", [])
        resolved_outputs: list[str] = []
        for template in output_templates:
            try:
                resolved_outputs.append(template.format(**format_context))
            except KeyError:
                resolved_outputs.append(template)

        start_condition_cmd = getattr(monitoring_config, "start_condition_cmd", None)
        start_condition_interval = getattr(
            monitoring_config,
            "start_condition_interval_seconds",
            None,
        )
        termination_string = getattr(monitoring_config, "termination_string", None)
        termination_command = getattr(monitoring_config, "termination_command", None)
        inactivity_threshold = getattr(
            monitoring_config,
            "inactivity_threshold_seconds",
            None,
        )

        filtered_params: dict[str, str] = {}
        for key, value in point.parameters.items():
            if value is None:
                continue
            normalized = key.lower()
            if normalized in {
                "job.start_condition",
                "job.start_condition_cmd",
                "start_condition",
                "start_condition_cmd",
                "monitoring.start_condition_cmd",
            }:
                start_condition_cmd = str(value)
                continue
            if normalized in {
                "job.start_condition_interval_seconds",
                "start_condition_interval_seconds",
                "monitoring.start_condition_interval_seconds",
            }:
                try:
                    start_condition_interval = int(value)
                except (TypeError, ValueError):
                    start_condition_interval = None
                continue
            if normalized in {
                "job.termination_string",
                "termination_string",
                "monitoring.termination_string",
            }:
                termination_string = str(value)
                continue
            if normalized in {
                "job.termination_command",
                "termination_command",
                "monitoring.termination_command",
            }:
                termination_command = str(value)
                continue
            if normalized in {
                "job.inactivity_threshold_seconds",
                "inactivity_threshold_seconds",
                "monitoring.inactivity_threshold_seconds",
            }:
                try:
                    inactivity_threshold = int(value)
                except (TypeError, ValueError):
                    inactivity_threshold = None
                continue
            filtered_params[key] = str(value)

        plans.append(
            JobPlan(
                name=job_name,
                parameters=filtered_params,
                output_dir=output_dir,
                log_path=log_path,
                output_paths=resolved_outputs,
                start_condition_cmd=start_condition_cmd,
                start_condition_interval_seconds=start_condition_interval,
                termination_string=termination_string,
                termination_command=termination_command,
                inactivity_threshold_seconds=inactivity_threshold,
            )
        )
    return plans


__all__ = ["JobPlan", "build_job_plans"]

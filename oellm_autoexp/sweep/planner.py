"""Build execution plans from sweep outputs."""

from __future__ import annotations

from dataclasses import dataclass, field, MISSING
from pathlib import Path

from compoconf import asdict
from typing import Any
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


def flatten_config(config: RootConfig | dict[str, Any], connector: str = "."):
    """
    >>> flatten({"key": {"subkey": "val"}})
    {"key.subkey": "val"}
    """
    if not isinstance(config, dict):
        cfg_dict = asdict(config)
    else:
        cfg_dict = config

    def _flat(d: tuple | list | dict | Any, prefix: str = ""):
        res = {}
        if isinstance(d, dict):
            for key, val in d.items():
                res.update(_flat(val, prefix=prefix + connector + key if prefix else key))
        elif isinstance(d, (list, tuple)):
            for idx, val in enumerate(cfg_dict):
                res.update(_flat(val, prefix=prefix + connector + str(idx) if prefix else key))
        else:
            res = {prefix: d}
        return res

    return _flat(cfg_dict, prefix="")


def simple_format(template_str: str, args: dict):
    for key, val in args.items():
        template_str = template_str.replace("{" + key + "}", str(val))
    return template_str


def build_job_plans(config: RootConfig, points: list[SweepPoint]) -> list[JobPlan]:
    base_output = Path(config.project.base_output_dir)
    project_name = config.project.name

    plans: list[JobPlan] = []
    for point in points:
        context: dict[str, str] = {
            "project": project_name,
            "index": str(point.index),
        }
        context.update(flatten_config(config))
        context.update(point.parameters)
        job_name = simple_format(config.sweep.name_template, context)

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


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)

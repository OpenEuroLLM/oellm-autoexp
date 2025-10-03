"""Build execution plans from sweep outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from oellm_autoexp.config.schema import RootConfig
from .expander import SweepPoint


@dataclass
class JobPlan:
    """Normalized job description used by downstream modules."""

    name: str
    parameters: Dict[str, str]
    output_dir: Path
    log_path: Path


def build_job_plans(config: RootConfig, points: list[SweepPoint]) -> list[JobPlan]:
    base_output = Path(config.project.base_output_dir)
    project_name = config.project.name

    plans: list[JobPlan] = []
    for point in points:
        context: Dict[str, str] = {
            "project": project_name,
            "index": str(point.index),
        }
        context.update({key: str(value) for key, value in point.parameters.items()})

        job_name = config.sweep.name_template.format(**context)
        output_dir = base_output / job_name
        log_template = config.monitoring.log_path_template
        log_path = Path(log_template.format(output_dir=output_dir, name=job_name))

        plans.append(
            JobPlan(
                name=job_name,
                parameters={key: str(value) for key, value in point.parameters.items()},
                output_dir=output_dir,
                log_path=log_path,
            )
        )
    return plans


__all__ = ["JobPlan", "build_job_plans"]


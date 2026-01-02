"""Sibling reference resolution for multi-stage training."""

from __future__ import annotations

import logging
import re
from typing import Any
from pathlib import Path

from .planner import JobPlan

LOGGER = logging.getLogger(__name__)


# Pattern to match {sibling.PATTERN.ACCESSOR}
SIBLING_TEMPLATE_PATTERN = re.compile(r"\{sibling\.([^.}]+)\.([^}]+)\}")


def resolve_sibling_references(jobs: list[JobPlan]) -> list[JobPlan]:
    """Resolve sibling references in job parameters.

    Processes {sibling.PATTERN.ACCESSOR} templates where:
    - PATTERN: pattern to match sibling jobs (typically a stage name like "stable")
    - ACCESSOR: field to access from sibling (e.g., "output_dir", "name")

    Args:
        jobs: List of job plans with potential sibling references

    Returns:
        List of job plans with resolved sibling references

    Raises:
        ValueError: If sibling reference cannot be resolved
    """
    LOGGER.info(f"Resolving sibling references for {len(jobs)} jobs")
    # First pass: extract sibling patterns and stage names
    jobs = _extract_metadata(jobs)

    # Second pass: resolve sibling references
    resolved_jobs = []
    for job in jobs:
        resolved_job = _resolve_job_siblings(job, jobs)
        resolved_jobs.append(resolved_job)

    return resolved_jobs


def _extract_metadata(jobs: list[JobPlan]) -> list[JobPlan]:
    """Extract sibling pattern and stage name from job parameters."""
    updated_jobs = []

    for job in jobs:
        # Check if job has stage parameter
        stage_name = job.parameters.get("stage")

        # Extract sibling pattern by checking for sibling references
        sibling_pattern = None
        for key, value in job.parameters.items():
            if isinstance(value, str) and "{sibling." in value:
                # Extract the base pattern (everything before stage in job name)
                # Assume job name format is: base_pattern_{stage_name}
                if stage_name and f"_{stage_name}" in job.name:
                    sibling_pattern = job.name.replace(f"_{stage_name}", "")
                break

        # If no sibling references but has stage, create pattern anyway
        if sibling_pattern is None and stage_name:
            if f"_{stage_name}" in job.name:
                sibling_pattern = job.name.replace(f"_{stage_name}", "")
            else:
                sibling_pattern = job.name

        # Create new JobPlan with updated metadata
        updated_job = JobPlan(
            name=job.name,
            parameters=job.parameters,
            output_dir=job.output_dir,
            log_path=job.log_path,
            log_path_current=_compute_log_path_current(job),
            output_paths=job.output_paths,
            start_condition_cmd=job.start_condition_cmd,
            start_condition_interval_seconds=job.start_condition_interval_seconds,
            start_conditions=job.start_conditions,  # ADDED: Preserve start_conditions
            termination_string=job.termination_string,
            termination_command=job.termination_command,
            inactivity_threshold_seconds=job.inactivity_threshold_seconds,
            cancel_conditions=job.cancel_conditions,
            sibling_pattern=sibling_pattern,
            stage_name=stage_name,
        )
        updated_jobs.append(updated_job)

    return updated_jobs


def _compute_log_path_current(job: JobPlan) -> str:
    """Compute the log_path_current (symlink) from log_path."""
    log_path = Path(job.log_path)
    log_dir = log_path.parent
    # Replace slurm-%j.out with current.log
    return str(log_dir / "current.log")


def _resolve_job_siblings(job: JobPlan, all_jobs: list[JobPlan]) -> JobPlan:
    """Resolve sibling references for a single job."""
    resolved_params = {}
    resolved_start_cmd = job.start_condition_cmd
    resolved_term_cmd = job.termination_command
    resolved_start_conditions = job.start_conditions
    resolved_cancel_conditions = job.cancel_conditions

    # Resolve parameters
    for key, value in job.parameters.items():
        if isinstance(value, str):
            resolved_value = _resolve_template(value, job, all_jobs)
            resolved_params[key] = resolved_value
        else:
            resolved_params[key] = value

    # Resolve start_condition_cmd (old approach)
    if resolved_start_cmd:
        resolved_start_cmd = _resolve_template(resolved_start_cmd, job, all_jobs)

    # Resolve termination_command
    if resolved_term_cmd:
        resolved_term_cmd = _resolve_template(resolved_term_cmd, job, all_jobs)

    # Resolve start_conditions (new approach)
    if resolved_start_conditions:
        resolved_start_conditions = _resolve_conditions_list(
            resolved_start_conditions, job, all_jobs
        )

    # Resolve cancel_conditions
    if resolved_cancel_conditions:
        resolved_cancel_conditions = _resolve_conditions_list(
            resolved_cancel_conditions, job, all_jobs
        )

    return JobPlan(
        name=job.name,
        parameters=resolved_params,
        output_dir=job.output_dir,
        log_path=job.log_path,
        log_path_current=job.log_path_current,
        output_paths=job.output_paths,
        start_condition_cmd=resolved_start_cmd,
        start_condition_interval_seconds=job.start_condition_interval_seconds,
        start_conditions=resolved_start_conditions,
        termination_string=job.termination_string,
        termination_command=resolved_term_cmd,
        inactivity_threshold_seconds=job.inactivity_threshold_seconds,
        cancel_conditions=resolved_cancel_conditions,
        sibling_pattern=job.sibling_pattern,
        stage_name=job.stage_name,
    )


def _resolve_template(template: str, job: JobPlan, all_jobs: list[JobPlan]) -> str:
    """Resolve sibling references in a template string."""

    def replace_match(match: re.Match) -> str:
        pattern = match.group(1)
        accessor = match.group(2)

        # Find sibling job matching the pattern
        sibling = _find_sibling(job, pattern, all_jobs)
        if not sibling:
            raise ValueError(
                f"Cannot resolve sibling reference {{sibling.{pattern}.{accessor}}} "
                f"for job {job.name}: no sibling found matching pattern '{pattern}'"
            )

        # Access the field
        value = _access_field(sibling, accessor)
        if value is None:
            raise ValueError(
                f"Cannot resolve sibling reference {{sibling.{pattern}.{accessor}}} "
                f"for job {job.name}: sibling {sibling.name} has no field '{accessor}'"
            )

        return str(value)

    return SIBLING_TEMPLATE_PATTERN.sub(replace_match, template)


def _find_sibling(job: JobPlan, pattern: str, all_jobs: list[JobPlan]) -> JobPlan | None:
    """Find sibling job matching the pattern.

    The pattern can be:
    - A stage name (e.g., "stable") - finds job with same sibling_pattern but different stage
    - A full job name pattern
    """
    # First try: pattern is a stage name
    if job.sibling_pattern:
        for candidate in all_jobs:
            if candidate.stage_name == pattern and candidate.sibling_pattern == job.sibling_pattern:
                return candidate

    # Second try: pattern matches part of job name
    for candidate in all_jobs:
        if pattern in candidate.name:
            return candidate

    return None


def _access_field(job: JobPlan, accessor: str) -> Any:
    """Access a field from a job plan."""
    # Direct attribute access
    if hasattr(job, accessor):
        return getattr(job, accessor)

    # Parameter access
    if accessor in job.parameters:
        return job.parameters[accessor]

    return None


def _resolve_conditions_list(
    conditions: list[dict[str, Any]],
    job: JobPlan,
    all_jobs: list[JobPlan],
) -> list[dict[str, Any]]:
    """Resolve sibling references in a list of conditions (start_conditions or
    cancel_conditions)."""
    resolved = []

    for condition in conditions:
        resolved_condition = {}
        for key, value in condition.items():
            if isinstance(value, str):
                resolved_value = _resolve_template(value, job, all_jobs)
                resolved_condition[key] = resolved_value
            else:
                resolved_condition[key] = value
        resolved.append(resolved_condition)

    return resolved


__all__ = [
    "resolve_sibling_references",
    "SIBLING_TEMPLATE_PATTERN",
]

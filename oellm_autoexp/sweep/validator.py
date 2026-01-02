"""Validation for execution plans."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .planner import JobPlan

LOGGER = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of execution plan validation."""

    is_valid: bool
    errors: list[str]
    warnings: list[str]

    def __str__(self) -> str:
        lines = []
        if self.errors:
            lines.append("Errors:")
            for error in self.errors:
                lines.append(f"  - {error}")
        if self.warnings:
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")
        if self.is_valid:
            lines.append("Validation passed.")
        return "\n".join(lines)


def validate_execution_plan(jobs: list[JobPlan]) -> ValidationResult:
    """Validate an execution plan before submission.

    Checks for:
    - Unresolved sibling references
    - Circular dependencies
    - Duplicate job names
    - Invalid cancel_conditions

    Args:
        jobs: List of job plans to validate

    Returns:
        ValidationResult with errors and warnings
    """
    LOGGER.info(f"Validating execution plan with {len(jobs)} jobs")
    errors: list[str] = []
    warnings: list[str] = []

    # Check for duplicate job names
    job_names = [job.name for job in jobs]
    duplicates = [name for name in job_names if job_names.count(name) > 1]
    if duplicates:
        errors.append(f"Duplicate job names found: {set(duplicates)}")

    # Check each job
    for job in jobs:
        # Check for unresolved sibling references in parameters
        for key, value in job.parameters.items():
            if isinstance(value, str) and "{sibling." in value:
                errors.append(
                    f"Unresolved sibling reference in job '{job.name}', parameter '{key}': {value}"
                )

        # Check for unresolved sibling references in start_condition_cmd
        if job.start_condition_cmd and "{sibling." in job.start_condition_cmd:
            errors.append(
                f"Unresolved sibling reference in job '{job.name}', start_condition_cmd: {job.start_condition_cmd}"
            )

        # Check for unresolved sibling references in termination_command
        if job.termination_command and "{sibling." in job.termination_command:
            errors.append(
                f"Unresolved sibling reference in job '{job.name}', termination_command: {job.termination_command}"
            )

        # Check for unresolved sibling references in cancel_conditions
        for condition in job.cancel_conditions:
            for key, value in condition.items():
                if isinstance(value, str) and "{sibling." in value:
                    errors.append(
                        f"Unresolved sibling reference in job '{job.name}', cancel_condition '{key}': {value}"
                    )

        # Validate cancel_conditions structure
        for idx, condition in enumerate(job.cancel_conditions):
            if not isinstance(condition, dict):
                errors.append(
                    f"Invalid cancel_condition in job '{job.name}' at index {idx}: expected dict, got {type(condition)}"
                )
            elif "class_name" not in condition:
                warnings.append(
                    f"Cancel condition in job '{job.name}' at index {idx} missing 'class_name' field"
                )

    # Check for circular dependencies (simplified check)
    # A job depends on another if it references it in start_condition_cmd or cancel_conditions
    dependency_graph = _build_dependency_graph(jobs)
    cycles = _find_cycles(dependency_graph)
    if cycles:
        errors.append(f"Circular dependencies detected: {cycles}")

    is_valid = len(errors) == 0

    return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)


def _build_dependency_graph(jobs: list[JobPlan]) -> dict[str, set[str]]:
    """Build dependency graph from jobs."""
    graph: dict[str, set[str]] = {}

    for job in jobs:
        deps: set[str] = set()

        # Check start_condition_cmd for job name references
        if job.start_condition_cmd:
            for other_job in jobs:
                if other_job.name != job.name and other_job.name in job.start_condition_cmd:
                    deps.add(other_job.name)

        # Check cancel_conditions for job name references
        for condition in job.cancel_conditions:
            for value in condition.values():
                if isinstance(value, str):
                    for other_job in jobs:
                        if other_job.name != job.name and other_job.name in value:
                            deps.add(other_job.name)

        graph[job.name] = deps

    return graph


def _find_cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    """Find cycles in dependency graph using DFS."""
    visited: set[str] = set()
    rec_stack: set[str] = set()
    cycles: list[list[str]] = []

    def dfs(node: str, path: list[str]) -> None:
        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        for neighbor in graph.get(node, set()):
            if neighbor not in visited:
                dfs(neighbor, path.copy())
            elif neighbor in rec_stack:
                # Found a cycle
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:] + [neighbor]
                cycles.append(cycle)

        rec_stack.remove(node)

    for node in graph:
        if node not in visited:
            dfs(node, [])

    return cycles


__all__ = [
    "ValidationResult",
    "validate_execution_plan",
]

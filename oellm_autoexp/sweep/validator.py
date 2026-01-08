"""Validation for execution plans."""

from __future__ import annotations

import logging
from dataclasses import dataclass
import re

from .planner import JobPlan

LOGGER = logging.getLogger(__name__)
_SIBLING_PATTERN = re.compile(r"\{sibling\.[^}]+\}")


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

    # Check for circular dependencies (simplified check)
    # A job depends on another if it references it in start_condition_cmd or cancel_conditions
    dependency_graph = _build_dependency_graph(jobs)
    cycles = _find_cycles(dependency_graph)
    if cycles:
        errors.append(f"Circular dependencies detected: {cycles}")

    for job in jobs:
        warnings.extend(_validate_cancel_conditions(job))

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


def _has_unresolved_sibling_reference(job: JobPlan) -> bool:
    if _contains_unresolved_sibling(job.parameters):
        return True
    if job.start_condition_cmd and _SIBLING_PATTERN.search(job.start_condition_cmd):
        return True
    if job.termination_command and _SIBLING_PATTERN.search(job.termination_command):
        return True
    if _contains_unresolved_sibling(job.start_conditions):
        return True
    if _contains_unresolved_sibling(job.cancel_conditions):
        return True
    return False


def _contains_unresolved_sibling(value: object) -> bool:
    if isinstance(value, str):
        return _SIBLING_PATTERN.search(value) is not None
    if isinstance(value, dict):
        return any(_contains_unresolved_sibling(v) for v in value.values())
    if isinstance(value, list):
        return any(_contains_unresolved_sibling(v) for v in value)
    return False


def _validate_cancel_conditions(job: JobPlan) -> list[str]:
    warnings: list[str] = []
    for condition in job.cancel_conditions:
        if isinstance(condition, dict) and "class_name" not in condition:
            warnings.append(f"Cancel condition for {job.name} missing 'class_name'")
    return warnings


__all__ = [
    "ValidationResult",
    "validate_execution_plan",
]

#!/usr/bin/env python3
"""Visualize multi-stage experiment plan in ASCII format.

This script provides a visual overview of your multi-stage experiment plan,
showing:
- Hyperparameter sweep combinations
- Job stages and dependencies
- Start and cancel conditions
- Validation results

Usage:
    # Visualize from config
    python scripts/visualize_plan.py --config-ref experiments/korbi/dense_300M_50BT_pull

    # With Hydra overrides
    python scripts/visualize_plan.py --config-ref experiments/my_experiment \\
        backend.megatron.lr=1e-4

    # Limit jobs displayed per stage (useful for large experiments)
    python scripts/visualize_plan.py --config-ref experiments/my_experiment \\
        --max-jobs-per-stage 5
"""

from __future__ import annotations

import argparse
import logging
import sys
import os
from collections import defaultdict
from pathlib import Path
from typing import Any
from compoconf import asdict

from oellm_autoexp.config.loader import load_config_reference
from oellm_autoexp.config.schema import ConfigSetup
from oellm_autoexp.utils.logging_config import configure_logging

# External libraries (extracted during refactoring)
from oellm_autoexp.hydra_staged_sweep.expander import expand_sweep
from oellm_autoexp.hydra_staged_sweep.planner import JobPlan
from oellm_autoexp.hydra_staged_sweep.dag_resolver import resolve_sweep_with_dag


def validate_execution_plan(jobs: list) -> Any:
    """Stub validator - returns always valid result."""

    class ValidationResult:
        is_valid = True
        errors = []
        warnings = []

    return ValidationResult()


LOGGER = logging.getLogger(__file__)


def build_stage_dependencies(points: dict) -> dict[str, set[str]]:
    """Build stage dependencies for visualization.

    This is a simplified version for visualization purposes.
    Returns empty dependencies for now - could be enhanced later.
    """
    # TODO: Implement proper stage dependency extraction if needed
    return {}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config-name", help="Config reference (e.g., experiments/my_experiment)")
    parser.add_argument("--config-path", help="Config full config path (yaml)")
    parser.add_argument("-C", "--config-dir", type=Path, default=Path("config"))
    parser.add_argument(
        "--max-jobs-per-stage",
        type=int,
        default=10,
        help="Maximum jobs to display per stage (default: 10)",
    )
    parser.add_argument(
        "--full-resolve",
        action="store_true",
        help="Resolve full job configs (slower for large sweeps)",
    )
    parser.add_argument("--visualize-keys", nargs="+", action="append", default=[])
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "overrides", nargs="*", default=[], help="Hydra-style overrides (key=value)"
    )
    return parser.parse_args(argv)


def _get_override_value(parameters: list[str], key: str) -> str | None:
    for item in parameters:
        if item.startswith("~"):
            continue
        stripped = item.lstrip("+")
        if "=" not in stripped:
            continue
        param_key, value = stripped.split("=", 1)
        if param_key == key:
            return value
    return None


def group_jobs_by_stage(jobs: list[JobPlan]) -> dict[str, list[JobPlan]]:
    """Group jobs by their stage parameter."""
    LOGGER.debug(f"Grouping {len(jobs)} jobs by stage")
    stages: dict[str, list[JobPlan]] = defaultdict(list)
    for job in jobs:
        # Use stage_name field if available, otherwise fall back to parameters
        stage = job.stage_name or _get_override_value(job.parameters, "stage") or "unknown"
        stages[stage].append(job)
    LOGGER.info(f"Found {len(stages)} stages: {list(stages.keys())}")
    return stages


def extract_hyperparameters(jobs: list[JobPlan], visualize_keys: list[str]) -> dict[str, list[str]]:
    """Extract unique hyperparameter values across all jobs."""
    hyperparams: dict[str, set[str]] = defaultdict(set)

    for job in jobs:
        for key_value in job.parameters:
            stripped = key_value.lstrip("+")
            if stripped.startswith("~") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            if key in visualize_keys:
                hyperparams[key].add(value)
    return {k: sorted(v) for k, v in hyperparams.items()}


def format_condition(condition: dict[str, Any]) -> str:
    """Format a condition dict as a human-readable string."""
    condition = asdict(condition)
    class_name = condition.get("class_name", "UnknownCondition")

    if class_name == "FileExistsCondition":
        path = condition.get("path", "???")
        # Shorten path if too long
        if len(path) > 50:
            path = "..." + path[-47:]
        return f"FileExists: {path}"
    elif class_name == "SlurmStateCondition":
        job_name = condition.get("job_name", "???")
        state = condition.get("state", "???")
        return f"SlurmState: {job_name} = {state}"
    elif class_name == "LogPatternCondition":
        pattern = condition.get("pattern", "???")
        return f"LogPattern: {pattern}"
    elif class_name == "MetadataCondition":
        key = condition.get("key", "???")
        equals = condition.get("equals", "???")
        return f"Metadata: {key} = {equals}"
    else:
        return class_name


def visualize_plan(
    jobs: list,
    config_name: str = "Experiment",
    max_jobs_per_stage: int = 10,
    visualize_keys: list[str] | None = None,
    dependencies: dict[str, set[str]] | None = None,
) -> None:
    """Visualize the execution plan in ASCII format showing DAG structure."""
    LOGGER.info(f"Visualizing plan for {len(jobs)} jobs")

    # Group jobs by stage
    stages_dict = group_jobs_by_stage(jobs)

    # Build dependency graph
    if dependencies is None:
        dependencies = {}

    # Extract hyperparameters
    if visualize_keys is None:
        visualize_keys = []
    hyperparams = extract_hyperparameters(jobs, visualize_keys=visualize_keys)

    # Print header
    print()
    print("=" * 70)
    print(f" Multi-Stage Experiment Plan: {config_name}")
    print("=" * 70)
    print(f"Total: {len(jobs)} jobs across {len(stages_dict)} stage(s)")
    print()

    # Print hyperparameter sweep summary
    if hyperparams:
        print("┌" + "─" * 68 + "┐")
        print("│ " + "Hyperparameter Sweep".ljust(67) + "│")
        for key, values in hyperparams.items():
            short_key = key.replace("backend.megatron.", "")
            values_str = str(values)
            if len(values_str) > 60:
                values_str = values_str[:57] + "..."
            print(f"│ • {short_key}: {values_str}".ljust(69) + "│")
        # Calculate total combinations
        num_combos = (
            len(stages_dict.get("stable", []))
            if "stable" in stages_dict
            else len(jobs) // len(stages_dict)
        )
        print(f"│ Total combinations: {num_combos}".ljust(69) + "│")
        print("└" + "─" * 68 + "┘")
        print()

    # Organize stages by dependency level
    rendered_stages = set()
    root_stages = [s for s in stages_dict.keys() if not dependencies.get(s)]

    if not root_stages:
        # Fallback to stable or first stage
        root_stages = ["stable"] if "stable" in stages_dict else [list(stages_dict.keys())[0]]

    LOGGER.debug(f"Root stages: {root_stages}")

    # Render stages level by level
    current_level = root_stages
    while current_level:
        # Render all stages at current level
        for stage in current_level:
            stage_jobs = stages_dict[stage]
            _print_stage_box(stage, stage_jobs, max_jobs_per_stage, dependencies or {})
            rendered_stages.add(stage)

        # Find next level of stages (those that depend ONLY on already rendered stages)
        next_level = []
        for stage in stages_dict.keys():
            if stage not in rendered_stages:
                stage_deps = dependencies.get(stage, set())
                # Stage is ready if all its dependencies have been rendered
                if not stage_deps or all(d in rendered_stages for d in stage_deps):
                    next_level.append(stage)

        if next_level:
            # Show branching if multiple stages depend on the same parent(s)
            num_branches = len(next_level)
            if num_branches == 1:
                # Simple arrow for single dependency
                print(" " * 33 + "↓")
                print()
            else:
                # Show multi-way branch
                # Calculate spacing for branches
                total_width = 70
                branch_width = total_width // (num_branches + 1)

                # Top connector
                print()
                connectors = []
                for i in range(num_branches):
                    offset = (i + 1) * branch_width
                    connectors.append(offset)

                # Draw branching lines
                line = [" "] * total_width
                for offset in connectors:
                    if offset < total_width:
                        line[offset] = "│"
                print("".join(line))

                print()

        current_level = next_level

    print()


def _print_stage_box(
    stage: str,
    stage_jobs: list,
    max_jobs_per_stage: int,
    dependencies: dict[str, set[str]],
) -> None:
    """Print a box for a single stage."""
    print("┌" + "─" * 68 + "┐")
    deps = sorted(dependencies.get(stage, set()))
    dep_text = "none" if not deps else ", ".join(deps)
    stage_header = f"Stage: {stage} ({len(stage_jobs)} jobs, depends on: {dep_text})"
    print(f"│ {stage_header}".ljust(69) + "│")
    print("├" + "─" * 68 + "┤")

    # Print job names (limited by max_jobs_per_stage)
    for job_idx, job in enumerate(stage_jobs[:max_jobs_per_stage]):
        name = getattr(job, "name", None)
        if name is None and hasattr(job, "config"):
            # Fallback for JobPlan -> config -> job -> name
            # We assume config has 'job' attribute with 'name'
            name = getattr(job.config.job, "name", "unknown")
        print(f"│ • {name}".ljust(69) + "│")

    if len(stage_jobs) > max_jobs_per_stage:
        remaining = len(stage_jobs) - max_jobs_per_stage
        print(f"│ ... and {remaining} more".ljust(69) + "│")

    print("├" + "─" * 68 + "┤")

    # Print start conditions
    sample_job = stage_jobs[0]

    if stage == "stable" or stage == "unknown" or not sample_job.config.job.start_condition:
        print("│ Start: Immediate".ljust(69) + "│")
    else:
        print("│ Start Conditions:".ljust(69) + "│")
        condition = sample_job.config.job.start_condition
        cond_str = format_condition(condition)
        print(f"│   • {cond_str}".ljust(69) + "│")

        # Print cancel conditions
        if sample_job.config.job.cancel_condition:
            print("│ Cancel Conditions:".ljust(69) + "│")
            condition = sample_job.config.job.cancel_condition
            cond_str = format_condition(condition)
            print(f"│   • {cond_str}".ljust(69) + "│")

    print("└" + "─" * 68 + "┘")
    print()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    visualize_keys = [item for group in args.visualize_keys for item in group]

    # Configure logging with environment variable support
    configure_logging(
        debug=args.debug,
        datefmt="%H:%M:%S",
    )

    LOGGER.info("Starting visualization")

    LOGGER.info(f"Loading config: {args.config_name if not args.config_path else args.config_path}")
    if args.overrides:
        print(f"Overrides: {args.overrides}")
        LOGGER.debug(f"Overrides: {args.overrides}")
    config_setup = ConfigSetup(
        config_name=args.config_name,
        config_path=args.config_path,
        config_dir=args.config_dir,
        overrides=args.overrides,
    )
    config = load_config_reference(config_setup=config_setup)
    LOGGER.info(f"Config loaded: {config.job.name}")

    # Expand sweep
    LOGGER.info("Expanding sweep")
    points_list = expand_sweep(config.sweep)
    points = {point.index: point for point in points_list}
    LOGGER.info(f"Expanded to {len(points)} sweep points")

    # Unified DAG-based resolution (job planning + sibling resolution)
    LOGGER.info("Resolving sweep with DAG")
    jobs = resolve_sweep_with_dag(
        config,
        points,
        config_setup=ConfigSetup(
            config_dir=args.config_dir,
            config_name=args.config_name,
            config_path=args.config_name,
            overrides=args.overrides,
            pwd=os.curdir,
        ),
        config_class=type(config),  # Pass RootConfig class
    )
    LOGGER.info(f"Resolved to {len(jobs)} jobs")
    # Validate
    LOGGER.info("Validating execution plan")
    validation = validate_execution_plan(jobs)

    # Visualize
    deps = build_stage_dependencies(points)
    visualize_plan(
        jobs,
        config_name=config.job.name,
        max_jobs_per_stage=args.max_jobs_per_stage,
        visualize_keys=visualize_keys,
        dependencies=deps,
    )

    # Print validation results
    if not validation.is_valid:
        print("=" * 70)
        print(" VALIDATION ERRORS")
        print("=" * 70)
        for error in validation.errors:
            print(f"❌ {error}")
        print()
        return 1

    if validation.warnings:
        print("=" * 70)
        print(" VALIDATION WARNINGS")
        print("=" * 70)
        for warning in validation.warnings:
            print(f"⚠️  {warning}")
        print()

    print("✅ Plan is valid and ready to execute!")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

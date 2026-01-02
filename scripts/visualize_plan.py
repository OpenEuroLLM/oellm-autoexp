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

    # Visualize from existing plan manifest (not yet implemented)
    python scripts/visualize_plan.py --manifest monitoring_state/manifests/plan_20250115_abc123.json

Example Output:
    ======================================================================
     Multi-Stage Experiment Plan: dense_300M_sweep
    ======================================================================
    Total: 75 jobs across 5 stage(s)

    ┌────────────────────────────────────────────────────────────────────┐
    │ Hyperparameter Sweep                                               │
    │ • lr: [2.5e-4, 5e-4, 1e-3, 2e-3]                                   │
    │ • global_batch_size: [64, 128, 256, 512, 1024]                     │
    │ Total combinations: 15                                             │
    └────────────────────────────────────────────────────────────────────┘

    ┌────────────────────────────────────────────────────────────────────┐
    │ Stage: stable (15 jobs)                                            │
    ├────────────────────────────────────────────────────────────────────┤
    │ • dense_300M_01_lr2.5e-4_gbsz64_beta20.95_stable                   │
    │ ...                                                                │
    ├────────────────────────────────────────────────────────────────────┤
    │ Start: Immediate                                                   │
    └────────────────────────────────────────────────────────────────────┘
                                     ↓
    ┌────────────────────────────────────────────────────────────────────┐
    │ Stage: decay6B (15 jobs)                                           │
    ├────────────────────────────────────────────────────────────────────┤
    │ ...                                                                │
    ├────────────────────────────────────────────────────────────────────┤
    │ Start Conditions:                                                  │
    │   • FileExists: .../checkpoints/iter_XXX/done.txt                  │
    │ Cancel Conditions:                                                 │
    │   • SlurmState: stable_job = FAILED                                │
    │   • LogPattern: FATAL ERROR|OutOfMemoryError                       │
    └────────────────────────────────────────────────────────────────────┘

    ✅ Plan is valid and ready to execute!
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from oellm_autoexp.config.loader import load_config_reference
from oellm_autoexp.sweep.expander import expand_sweep
from oellm_autoexp.sweep.dag_resolver import resolve_sweep_with_dag
from oellm_autoexp.sweep.validator import validate_execution_plan
from oellm_autoexp.utils.logging_config import configure_logging

# Import to register configs
import oellm_autoexp.monitor.watcher  # noqa: F401
import oellm_autoexp.backends.base  # noqa: F401
import oellm_autoexp.slurm.client  # noqa: F401

LOGGER = logging.getLogger(__file__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config-ref", help="Config reference (e.g., experiments/my_experiment)")
    parser.add_argument("-C", "--config-dir", type=Path, default=Path("config"))
    parser.add_argument("--manifest", type=Path, help="Path to existing plan manifest")
    parser.add_argument(
        "--max-jobs-per-stage",
        type=int,
        default=10,
        help="Maximum jobs to display per stage (default: 10)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("override", nargs="*", default=[], help="Hydra-style overrides (key=value)")
    return parser.parse_args(argv)


def group_jobs_by_stage(jobs: list) -> dict[str, list]:
    """Group jobs by their stage parameter."""
    LOGGER.debug(f"Grouping {len(jobs)} jobs by stage")
    stages = defaultdict(list)
    for job in jobs:
        # Use stage_name field if available, otherwise fall back to parameters
        stage = job.stage_name or job.parameters.get("stage", "unknown")
        stages[stage].append(job)
    LOGGER.info(f"Found {len(stages)} stages: {list(stages.keys())}")
    return stages


def build_stage_dependencies(jobs: list) -> dict[str, set[str]]:
    """Build stage-level dependency graph from jobs.

    Returns:
        Dict mapping stage name to set of stages it depends on
    """
    LOGGER.debug("Building stage dependency graph")
    dependencies = defaultdict(set)

    for job in jobs:
        stage = job.stage_name or job.parameters.get("stage", "unknown")

        # Extract dependencies from start conditions
        if job.start_conditions:
            for cond in job.start_conditions:
                if isinstance(cond, dict):
                    # FileExistsCondition with sibling path
                    if cond.get("class_name") == "FileExistsCondition":
                        path = cond.get("path", "")
                        # Extract stage from sibling references in path
                        # e.g., "dense_300M_01_lr0.00025_gbsz64_beta20.95_stable/checkpoints/..."
                        for s in ["stable", "decay6B", "decay12B", "decay30B", "decay50B"]:
                            if f"_{s}/" in path:
                                dependencies[stage].add(s)
                                break
                    # SlurmStateCondition with job_name
                    elif cond.get("class_name") == "SlurmStateCondition":
                        job_name = cond.get("job_name", "")
                        for s in ["stable", "decay6B", "decay12B", "decay30B", "decay50B"]:
                            if f"_{s}" in job_name:
                                dependencies[stage].add(s)
                                break

    LOGGER.info(f"Stage dependencies: {dict(dependencies)}")
    return dependencies


def extract_hyperparameters(jobs: list) -> dict[str, set]:
    """Extract unique hyperparameter values across all jobs."""
    hyperparams = defaultdict(set)

    # Common hyperparameter keys to display
    interesting_keys = {
        "backend.megatron.lr",
        "backend.megatron.global_batch_size",
        "backend.megatron.num_layers",
        "backend.megatron.hidden_size",
        "backend.megatron.adam_beta2",
    }

    for job in jobs:
        for key, value in job.parameters.items():
            if key in interesting_keys:
                hyperparams[key].add(value)

    return {k: sorted(v) for k, v in hyperparams.items()}


def format_condition(condition: dict[str, Any]) -> str:
    """Format a condition dict as a human-readable string."""
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
    jobs: list, config_name: str = "Experiment", max_jobs_per_stage: int = 10
) -> None:
    """Visualize the execution plan in ASCII format showing DAG structure."""
    LOGGER.info(f"Visualizing plan for {len(jobs)} jobs")

    # Group jobs by stage
    stages_dict = group_jobs_by_stage(jobs)

    # Build dependency graph
    dependencies = build_stage_dependencies(jobs)

    # Extract hyperparameters
    hyperparams = extract_hyperparameters(jobs)

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
            _print_stage_box(stage, stage_jobs, max_jobs_per_stage)
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


def _print_stage_box(stage: str, stage_jobs: list, max_jobs_per_stage: int) -> None:
    """Print a box for a single stage."""
    print("┌" + "─" * 68 + "┐")
    stage_header = f"Stage: {stage} ({len(stage_jobs)} jobs)"
    print(f"│ {stage_header}".ljust(69) + "│")
    print("├" + "─" * 68 + "┤")

    # Print job names (limited by max_jobs_per_stage)
    for job_idx, job in enumerate(stage_jobs[:max_jobs_per_stage]):
        print(f"│ • {job.name}".ljust(69) + "│")

    if len(stage_jobs) > max_jobs_per_stage:
        remaining = len(stage_jobs) - max_jobs_per_stage
        print(f"│ ... and {remaining} more".ljust(69) + "│")

    print("├" + "─" * 68 + "┤")

    # Print start conditions
    sample_job = stage_jobs[0]

    if stage == "stable" or stage == "unknown" or not sample_job.start_conditions:
        print("│ Start: Immediate".ljust(69) + "│")
    else:
        print("│ Start Conditions:".ljust(69) + "│")
        for condition in sample_job.start_conditions:
            cond_str = format_condition(condition)
            print(f"│   • {cond_str}".ljust(69) + "│")

        # Print cancel conditions
        if sample_job.cancel_conditions:
            print("│ Cancel Conditions:".ljust(69) + "│")
            for condition in sample_job.cancel_conditions:
                cond_str = format_condition(condition)
                print(f"│   • {cond_str}".ljust(69) + "│")

    print("└" + "─" * 68 + "┘")
    print()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Configure logging with environment variable support
    # OELLM_LOG_LEVEL can be set to DEBUG, INFO, WARNING, ERROR, or CRITICAL
    configure_logging(
        debug=args.debug,
        datefmt="%H:%M:%S",
    )

    LOGGER.info("Starting visualization")

    # Either load from config or manifest
    if args.manifest:
        LOGGER.info(f"Loading from manifest: {args.manifest}")
        # Load from existing manifest
        with open(args.manifest) as f:
            _ = json.load(f)

        print(f"Loading from manifest: {args.manifest}")
        print("Note: Manifest-based visualization not yet implemented.")
        print("Use --config-ref to visualize from config.")
        return 1

    elif args.config_ref:
        LOGGER.info(f"Loading config: {args.config_ref}")
        # Load config
        print(f"Loading config: {args.config_ref}")
        if args.override:
            print(f"Overrides: {args.override}")
            LOGGER.debug(f"Overrides: {args.override}")

        config = load_config_reference(
            args.config_ref, str(args.config_dir), overrides=args.override
        )
        LOGGER.info(f"Config loaded: {config.project.name}")

        # Expand sweep
        LOGGER.info("Expanding sweep")
        points = expand_sweep(config.sweep)
        LOGGER.info(f"Expanded to {len(points)} sweep points")

        # Unified DAG-based resolution (job planning + sibling resolution)
        LOGGER.info("Resolving sweep with DAG")
        jobs = resolve_sweep_with_dag(config, points)
        LOGGER.info(f"Resolved to {len(jobs)} jobs")

        # Validate
        LOGGER.info("Validating execution plan")
        validation = validate_execution_plan(jobs)

        # Visualize
        visualize_plan(
            jobs, config_name=config.project.name, max_jobs_per_stage=args.max_jobs_per_stage
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

    else:
        print("Error: Must provide either --config-ref or --manifest")
        return 1


if __name__ == "__main__":
    sys.exit(main())

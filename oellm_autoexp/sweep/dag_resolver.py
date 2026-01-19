"""DAG resolution and sibling replacement - imports from hydra_staged_sweep with minimal wrapper."""

from typing import Any

from hydra_staged_sweep.dag_resolver import (
    resolve_sweep_with_dag as _hs_resolve_sweep_with_dag,
    build_dependency_dag_from_points,
    param_to_cmdlines,
    config_to_cmdline,
    extract_sibling_patterns,
    find_sibling_by_group_path,
)
from hydra_staged_sweep.expander import SweepPoint
from hydra_staged_sweep.planner import JobPlan as BaseJobPlan
from hydra_staged_sweep.config.schema import ConfigSetup as HSConfigSetup

__all__ = [
    "resolve_sweep_with_dag",
    "build_dependency_dag_from_points",
    "param_to_cmdlines",
    "config_to_cmdline",
    "extract_sibling_patterns",
    "find_sibling_by_group_path",
    "build_stage_dependencies",
]


def _convert_to_extended_jobplan(base_job: BaseJobPlan) -> Any:
    """Convert hydra_staged_sweep's minimal JobPlan to oellm's extended
    JobPlan.

    Extracts oellm-specific fields from the resolved config.
    """
    from oellm_autoexp.sweep.planner import JobPlan

    cfg = base_job.config

    # Extract fields from resolved config
    project_name = getattr(cfg.project, "name", "")
    output_dir = getattr(cfg, "output_dir", "")

    # Handle log paths
    log_path = getattr(cfg.project, "log_path", "")
    log_path_current = getattr(cfg.project, "log_path_current", "")

    # Extract start/cancel conditions from job config
    job_config = getattr(cfg, "job", None)
    start_conditions = []
    cancel_conditions = []
    start_condition_cmd = None
    start_condition_interval_seconds = None
    termination_string = None
    termination_command = None
    inactivity_threshold_seconds = None
    output_paths = []

    if job_config:
        start_conditions = getattr(job_config, "start_conditions", [])
        cancel_conditions = getattr(job_config, "cancel_conditions", [])
        start_condition_cmd = getattr(job_config, "start_condition_cmd", None)
        start_condition_interval_seconds = getattr(
            job_config, "start_condition_interval_seconds", None
        )
        termination_string = getattr(job_config, "termination_string", None)
        termination_command = getattr(job_config, "termination_command", None)
        inactivity_threshold_seconds = getattr(job_config, "inactivity_threshold_seconds", None)
        output_paths = getattr(job_config, "output_paths", [])

    # Create extended JobPlan
    return JobPlan(
        config=cfg,
        parameters=base_job.parameters,
        sibling_pattern=base_job.sibling_pattern,
        stage_name=base_job.stage_name,
        name=project_name,
        output_dir=output_dir,
        log_path=log_path,
        log_path_current=log_path_current,
        output_paths=list(output_paths) if output_paths else [],
        start_condition_cmd=start_condition_cmd,
        start_condition_interval_seconds=start_condition_interval_seconds,
        start_conditions=list(start_conditions) if start_conditions else [],
        cancel_conditions=list(cancel_conditions) if cancel_conditions else [],
        termination_string=termination_string,
        termination_command=termination_command,
        inactivity_threshold_seconds=inactivity_threshold_seconds,
    )


def resolve_sweep_with_dag(
    config: Any,
    points: list[SweepPoint] | dict[int, SweepPoint],
    config_setup: Any,
    config_class: type | None = None,
) -> list[Any]:
    """Wrapper around hydra_staged_sweep resolve_sweep_with_dag that uses
    oellm's config loader.

    This ensures compatibility with oellm's RootConfig schema while
    using the well-tested DAG resolution from hydra_staged_sweep.
    Converts the minimal JobPlan to oellm's extended JobPlan with all
    required fields.
    """
    # Import here to avoid circular dependency
    from oellm_autoexp.config.loader import load_config_reference
    from oellm_autoexp.config.schema import RootConfig

    # Use RootConfig if not specified
    if config_class is None:
        config_class = RootConfig

    # Convert oellm ConfigSetup to hydra_staged_sweep ConfigSetup if needed
    if hasattr(config_setup, "config_ref"):
        hs_config_setup = HSConfigSetup(
            pwd=config_setup.pwd,
            config_name=config_setup.config_name,
            config_path=config_setup.config_path,
            config_dir=config_setup.config_dir,
            override=config_setup.override,
        )
    else:
        hs_config_setup = config_setup

    # Monkeypatch the load_config_reference in hydra_staged_sweep temporarily
    import hydra_staged_sweep.dag_resolver

    orig_loader = hydra_staged_sweep.dag_resolver.load_config_reference

    def patched_loader(
        config_name=None, config_path=None, config_dir=None, overrides=None, config_class=RootConfig
    ):
        """Patched loader that uses oellm's loader."""
        if config_path:
            return load_config_reference(config_path, config_dir, overrides)
        else:
            return load_config_reference(config_name, config_dir, overrides)

    try:
        hydra_staged_sweep.dag_resolver.load_config_reference = patched_loader
        base_jobs = _hs_resolve_sweep_with_dag(config, points, hs_config_setup, config_class)

        # Convert minimal JobPlan to extended JobPlan
        extended_jobs = [_convert_to_extended_jobplan(job) for job in base_jobs]
        return extended_jobs
    finally:
        hydra_staged_sweep.dag_resolver.load_config_reference = orig_loader


def build_stage_dependencies(points: dict[int, SweepPoint]) -> dict[str, set[str]]:
    """Build stage-level dependency map from sweep points.

    Thin wrapper around build_dependency_dag_from_points that groups by
    stage.
    """
    dag = build_dependency_dag_from_points(points)
    stage_by_index = {
        idx: (point.parameters.get("stage") or "unknown") for idx, point in points.items()
    }
    dependencies: dict[str, set[str]] = {}
    for idx, stage in stage_by_index.items():
        dependencies.setdefault(stage, set())
        for parent in dag.predecessors(idx):
            parent_stage = stage_by_index.get(parent, "unknown")
            if parent_stage != stage:
                dependencies[stage].add(parent_stage)
    return dependencies

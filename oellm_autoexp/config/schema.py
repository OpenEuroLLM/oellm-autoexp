"""Configuration dataclasses and registries for oellm_autoexp.

These types are designed for use with compoconf so that new
implementations can be registered declaratively from configuration
files.
"""

from __future__ import annotations

from dataclasses import dataclass, field, MISSING
from typing import Any

from compoconf import (
    ConfigInterface,
    RegistrableConfigInterface,
    register_interface,
    NonStrictDataclass,
)

# ---------------------------------------------------------------------------
# Core interfaces
# ---------------------------------------------------------------------------


@register_interface
class BackendInterface(RegistrableConfigInterface):
    """Abstract training backend.

    Implementations translate job parameters into launch commands and perform
    backend-specific validation. They receive a config dataclass defined via
    ``BackendInterface.cfgtype``.
    """


@register_interface
class MonitorInterface(RegistrableConfigInterface):
    """Monitoring implementation responsible for observing job execution."""


@register_interface
class SlurmClientInterface(RegistrableConfigInterface):
    """Abstraction layer over SLURM command interactions."""


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class ProjectConfig(ConfigInterface):
    """Project-level metadata and defaults.

    Attributes:
        name: Project name used in job naming
        base_output_dir: Per-run output directory (may include timestamps for unique runs)
        log_path: (str) path template as used in slurm (%a, %A etc.)
        log_path_current: (str) path for a symlink to the latest log output - fixed
        monitoring_state_dir: Stable directory for monitoring sessions (NO timestamps).
            This directory persists across runs and enables --monitor-all to find sessions.
            Defaults to a stable location (strips timestamps from base_output_dir).
        resume: Whether to resume monitoring from persisted state
    """

    name: str = ""
    base_output_dir: str = field(default=MISSING)
    log_path: str = field(default=MISSING)  # template as used by SLURM
    log_path_current: str = field(
        default=MISSING
    )  # fixed path for a symlink to the latest log file, known before submission
    monitoring_state_dir: str | None = None  # Stable, cross-run monitoring state
    resume: bool = True


@dataclass(kw_only=True)
class SweepConfig(ConfigInterface):
    """Sweep expansion settings.

    Supports both legacy grid format and new composable groups format:

    Legacy format (backward compatible):
        grids: list[dict[str, list[Any]]]

    New composable format:
        type: "product" | "list"
        groups: list[dict with 'type' and 'params'/'configs']

    ``base_values`` contain default substitutions applied before the
    sweep is generated.
    """

    class_name: str = "Sweep"
    # Legacy format
    grids: list[dict[str, list[Any]]] | None = None
    # New composable format
    type: str | None = None  # "product" or "list"
    groups: list[dict[str, Any]] | None = None
    base_values: dict[str, Any] = field(default_factory=dict)
    store_sweep_json: bool = True
    filter: str | None = None


@dataclass(init=False)
class SrunConfig(NonStrictDataclass):
    pass


@dataclass(init=False)
class SbatchConfig(NonStrictDataclass):
    account: str | None = None
    nodes: int | None = None
    partition: str | None = None
    qos: str | None = None
    time: str = "0-01:00:00"


@dataclass(kw_only=True)
class ContainerConfig(ConfigInterface):
    """Container runtime configuration for reproducible execution."""

    class_name: str = "Container"
    image: str | None = None
    runtime: str = "singularity"
    bind: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    pwd: str | None = None
    python: str = "python"


@dataclass(kw_only=True)
class SlurmConfig(ConfigInterface):
    """Parameters for SBATCH rendering and submission."""

    class_name: str = "Slurm"
    template_path: str = field(default_factory=MISSING)
    script_dir: str = field(default_factory=MISSING)
    log_dir: str = field(default_factory=MISSING)
    array: bool = True
    submit_cmd: str = "sbatch"
    squeue_cmd: str = "squeue"
    cancel_cmd: str = "scancel"
    sacct_cmd: str = "sacct"
    launcher_cmd: str = ""
    srun_opts: str = ""
    launcher_env_passthrough: bool = False
    env: dict[str, Any] = field(default_factory=dict)
    srun: SrunConfig = field(default_factory=SrunConfig)
    sbatch: SbatchConfig = field(default_factory=SbatchConfig)
    sbatch_overrides: dict[str, Any] = field(default_factory=dict)
    sbatch_extra_directives: list[str] = field(default_factory=list)
    test_only: bool = False
    client: SlurmClientInterface.cfgtype | None = None


@dataclass(kw_only=True)
class SchedulerConfig(ConfigInterface):
    """Optional scheduler-level throttling and limits."""

    class_name: str = "Scheduler"
    max_jobs: int | None = None
    submit_rate_limit_seconds: float | None = None


@dataclass(kw_only=True)
class JobConfig(ConfigInterface):
    """Per-job lifecycle configuration."""

    start_condition_cmd: str | None = None
    start_condition_interval_seconds: int | None = None
    start_conditions: list[dict[str, Any]] = field(default_factory=list)
    cancel_conditions: list[dict[str, Any]] = field(default_factory=list)
    inactivity_threshold_seconds: int | None = None


@dataclass(kw_only=True)
class RootConfig(ConfigInterface):
    """Top-level configuration schema."""

    class_name: str = "Root"
    project: ProjectConfig = field(default_factory=MISSING)
    sweep: SweepConfig = field(default_factory=MISSING)
    slurm: SlurmConfig = field(default_factory=MISSING)
    monitoring: MonitorInterface.cfgtype = field(default_factory=MISSING)
    job: JobConfig = field(default_factory=JobConfig)
    backend: BackendInterface.cfgtype = field(default_factory=MISSING)
    container: ContainerConfig | None = None
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    metadata: dict[str, Any] = field(default_factory=dict)
    stage: str = ""
    # these are usually set internally
    sibling: dict[str, Any] = field(default_factory=dict)
    index: int | tuple[int] = 0
    project_name: str = ""
    job_name: str = ""
    output_dir: str = ""


@dataclass(kw_only=True)
class ConfigSetup:
    pwd: str
    config_ref: str
    config_dir: str
    override: list[str] = field(default=list)

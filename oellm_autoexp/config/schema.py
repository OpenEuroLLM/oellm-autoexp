"""Configuration dataclasses and registries for oellm_autoexp.

These types are designed for use with compoconf so that new
implementations can be registered declaratively from configuration
files.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field, MISSING
from typing import Any

from compoconf import (
    ConfigInterface,
    RegistrableConfigInterface,
    register_interface,
    NonStrictDataclass,
)

# Import base classes from hydra_staged_sweep
from hydra_staged_sweep.config.schema import StagedSweepRoot

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


# SweepConfig imported from hydra_staged_sweep above


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
class RootConfig(StagedSweepRoot):
    """Top-level configuration schema - extends hydra_staged_sweep with oellm-specific fields.

    Base fields from StagedSweepRoot:
        sweep: SweepConfig
        stage: str
        index: int | tuple[int]
        sibling: dict[str, Any]

    Additional oellm-specific fields below:
    """

    class_name: str = "Root"
    # oellm-specific configuration sections
    project: ProjectConfig = field(default_factory=MISSING)
    slurm: SlurmConfig = field(default_factory=MISSING)
    monitoring: MonitorInterface.cfgtype = field(default_factory=MISSING)
    job: JobConfig = field(default_factory=JobConfig)
    backend: BackendInterface.cfgtype = field(default_factory=MISSING)
    container: ContainerConfig | None = None
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    metadata: dict[str, Any] = field(default_factory=dict)
    # Runtime fields set during orchestration
    project_name: str = ""
    job_name: str = ""
    output_dir: str = ""


@dataclass
class ConfigSetup:
    """Config setup - compatible with hydra_staged_sweep.

    Uses config_name/config_path like hydra_staged_sweep but maintains
    config_ref as alias for backward compatibility.
    """

    pwd: str
    config_name: str | None = None
    config_path: str | None = None
    config_dir: str | None = None
    override: list[str] = field(default_factory=list)
    # Backward compatibility field
    config_ref: str | None = field(default=None, init=True)

    def __post_init__(self):
        # Backward compatibility: convert config_ref to config_name/config_path
        if self.config_ref is not None:
            ref_path = Path(self.config_ref)
            if ref_path.exists():
                self.config_path = str(ref_path)
                self.config_name = None
            else:
                self.config_name = self.config_ref
                self.config_path = None

        # Ensure at least one way to specify config is provided
        if self.config_path is None and self.config_name is None:
            raise ValueError("Either config_path, config_name, or config_ref must be specified")

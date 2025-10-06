"""Configuration dataclasses and registries for oellm_autoexp.

These types are designed for use with compoconf so that new implementations can
be registered declaratively from configuration files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from compoconf import ConfigInterface, RegistrableConfigInterface, register_interface, NonStrictDataclass

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
class RestartPolicyInterface(RegistrableConfigInterface):
    """Decision logic for handling job errors/stalls."""


@register_interface
class MonitorInterface(RegistrableConfigInterface):
    """Monitoring implementation responsible for observing job execution."""


@register_interface
class SlurmClientInterface(RegistrableConfigInterface):
    """Abstraction layer over SLURM command interactions."""


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ProjectConfig(ConfigInterface):
    """Project-level metadata and defaults."""

    name: str
    base_output_dir: Path
    state_dir: Optional[Path] = None
    resume: bool = True


@dataclass
class SweepConfig(ConfigInterface):
    """Sweep expansion settings.

    ``axes`` holds the raw sweep definition (potentially nested Hydra-style
    mappings). ``base_values`` contain default substitutions applied before the
    sweep is generated.
    """

    axes: Any
    base_values: Dict[str, Any] = field(default_factory=dict)
    name_template: str = "{project}_{index}"
    store_sweep_json: bool = True


@dataclass(init=False)
class SrunConfig(NonStrictDataclass):
    pass


@dataclass(init=False)
class SbatchConfig(NonStrictDataclass):
    account: Optional[str] = None
    nodes: Optional[int] = None
    partition: Optional[str] = None
    qos: Optional[str] = None
    time: str = "0-01:00:00"


@dataclass
class SlurmConfig(ConfigInterface):
    """Parameters for SBATCH rendering and submission."""

    template_path: Path
    script_dir: Path
    log_dir: Path
    array: bool = True
    submit_cmd: str = "sbatch"
    squeue_cmd: str = "squeue"
    cancel_cmd: str = "scancel"
    launcher_cmd: str = ""
    srun_opts: str = ""
    launcher_env_passthrough: bool = False
    environment: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    srun: SrunConfig = field(default_factory=SrunConfig)
    sbatch: SbatchConfig = field(default_factory=SbatchConfig)
    sbatch_overrides: Dict[str, Any] = field(default_factory=dict)
    sbatch_extra_directives: List[str] = field(default_factory=list)
    test_only: bool = False
    client: SlurmClientInterface.cfgtype | None = None


@dataclass
class RestartPolicyConfig(ConfigInterface):
    """Entry describing how to react to specific error modes."""

    mode: Literal["stall", "crash", "timeout", "success"]
    implementation: RestartPolicyInterface.cfgtype
    max_retries: Optional[int] = None


@dataclass
class SchedulerConfig(ConfigInterface):
    """Optional scheduler-level throttling and limits."""

    max_jobs: Optional[int] = None
    submit_rate_limit_seconds: Optional[float] = None


@dataclass
class RootConfig(ConfigInterface):
    """Top-level configuration schema."""

    project: ProjectConfig
    sweep: SweepConfig
    slurm: SlurmConfig
    monitoring: MonitorInterface.cfgtype
    backend: BackendInterface.cfgtype
    restart_policies: List[RestartPolicyConfig] = field(default_factory=list)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    metadata: Dict[str, Any] = field(default_factory=dict)

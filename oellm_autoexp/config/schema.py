"""Configuration dataclasses and registries for oellm_autoexp.

These types are designed for use with compoconf so that new
implementations can be registered declaratively from configuration
files.
"""

from __future__ import annotations

# Setup library paths first
import oellm_autoexp._libs  # noqa: F401

from dataclasses import dataclass, field, MISSING
from typing import Any

from compoconf import (
    ConfigInterface,
    RegistrableConfigInterface,
    register_interface,
)

# Import base classes from hydra_staged_sweep
from hydra_staged_sweep.config.schema import StagedSweepRoot, SweepConfig
from monitor.submission import SlurmJobConfig, SlurmConfig
from monitor.local_client import LocalCommandClientConfig
from monitor.slurm_client import SlurmClientConfig

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
class RootConfig(StagedSweepRoot):
    """Top-level configuration schema - extends hydra_staged_sweep with oellm-specific fields.

    Base fields from StagedSweepRoot:
        sweep: SweepConfig
        stage: str
        index: int | tuple[int]
        sibling: dict[str, Any]

    Additional oellm-specific fields below:
    """

    # oellm-specific configuration sections
    project: ProjectConfig = field(
        default_factory=MISSING
    )  # defines project core parts (directories, name)
    slurm: SlurmConfig = field(default_factory=MISSING)  # defines slurm setup
    job: SlurmJobConfig = field(
        default_factory=MISSING
    )  # defines job interactions (when to start, cancel, finish)
    backend: BackendInterface.cfgtype = field(
        default_factory=MISSING
    )  # defines what is actually running
    container: ContainerConfig | None = None  # defines container setup
    sweep: SweepConfig | None = (
        None  # defines a surrounding sweep (already inherited from StagedSweepRoot)
    )

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class RunEnvConfig:
    slurm_client: SlurmClientConfig
    local_client: LocalCommandClientConfig


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

    def __post_init__(self):
        # Ensure at least one way to specify config is provided
        if self.config_path is None and self.config_name is None:
            raise ValueError("Either config_path or config_name must be specified")

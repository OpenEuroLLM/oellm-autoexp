"""Configuration dataclasses and registries for oellm_autoexp.

These types are designed for use with compoconf so that new
implementations can be registered declaratively from configuration
files.
"""

from __future__ import annotations

# Setup library paths first
import oellm_autoexp._libs  # noqa: F401

from dataclasses import dataclass, field, MISSING
from typing import Any, TypedDict

from compoconf import ConfigInterface, RegistrableConfigInterface, register_interface

from oellm_autoexp.postprocess import PostProcessStepInterface

# Import base classes from oellm_autoexp.hydra_staged_sweep
from oellm_autoexp.hydra_staged_sweep.config.schema import (
    StagedSweepRoot,
    SweepConfig,
    ConfigSetup as BaseConfigSetup,
)
from oellm_autoexp.monitor.submission import SlurmJobConfig as SlurmJobConfigBase, SlurmConfig
from oellm_autoexp.monitor.local_client import LocalCommandClientConfig
from oellm_autoexp.monitor.slurm_client import SlurmClientConfig

# ---------------------------------------------------------------------------
# Core interfaces
# ---------------------------------------------------------------------------


class EmptyDict(TypedDict):
    pass


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
class CondaConfig(ConfigInterface):
    """Conda runtime configuration for local/host execution."""

    class_name: str = "Conda"
    env_name: str = "base"
    conda_prefix: str | None = None
    activate_script: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    python: str = "python"


@dataclass(kw_only=True)
class VenvConfig(ConfigInterface):
    """Virtualenv runtime configuration for local/host execution."""

    class_name: str = "Venv"
    venv_path: str = ".venv"
    activate_script: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    python: str = "python"


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
class SlurmJobConfig(SlurmJobConfigBase):
    base_output_dir: str = field(default_factory=MISSING)
    chain_repeat: int = 1

    # --- submit-gate cost estimate ------------------------------------------
    # All optional and estimate-only: nothing here reaches SLURM or the trainer,
    # it only sharpens the "~N GPU-h. Proceed?" gate in run_autoexp.py.
    #
    # WHY IT EXISTS. The gate used to price every job at its FULL --time, which
    # is right for a run that trains to its wall clock and badly wrong for a
    # measurement run that exits after a fixed number of iterations. The
    # 1024-node PP/VPP campaign was quoted 35,499 GPU-h and actually cost 9,404,
    # because `exit_interval: 50` ends each arm in 8-10 min of a 40 min wall.
    # A gate that overstates by 4x gets clicked through, which defeats it.
    #
    # Set est_step_time_s from a MEASURED step time and the estimate becomes
    #     per segment = min(--time, exit_duration_in_mins, startup + steps*step)
    # Steps are taken from est_steps, else derived from the backend
    # (exit_interval -> train_iters -> train_samples/global_batch_size), so for
    # the common measurement case only est_step_time_s has to be set.
    est_step_time_s: float | None = None
    # Steps this ONE job segment runs. Leave None to derive from the backend.
    est_steps: int | None = None
    # Container start + rendezvous + init before iteration 1. NOT negligible at
    # scale: >=256-node jobs on JUPITER are silent for 8+ min before the first
    # step, which dominates the cost of any short measurement run.
    est_startup_min: float = 0.0


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
    slurm: SlurmConfig = field(default_factory=MISSING)  # defines slurm setup
    job: SlurmJobConfig = field(
        default_factory=MISSING
    )  # defines job interactions (when to start, cancel, finish)
    backend: BackendInterface.cfgtype = field(
        default_factory=MISSING
    )  # defines what is actually running
    container: EmptyDict | ContainerConfig | CondaConfig | VenvConfig = field(
        default_factory=EmptyDict
    )  # defines container setup
    sweep: EmptyDict | SweepConfig = field(
        default_factory=EmptyDict
    )  # defines a surrounding sweep (already inherited from StagedSweepRoot)

    postprocess: dict[str, PostProcessStepInterface.cfgtype] = field(
        default_factory=dict
    )  # optional post-processing steps (e.g. ckpt conversion, eval)

    aux: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.container and self.slurm:
            assert self.slurm.env["MACHINE_NAME"] == self.container.env["MACHINE_NAME"]


@dataclass(kw_only=True)
class RunEnvConfig:
    slurm_client: SlurmClientConfig
    local_client: LocalCommandClientConfig


@dataclass(kw_only=True)
class ConfigSetup(BaseConfigSetup):
    """Config setup - compatible with hydra_staged_sweep.

    Uses config_name/config_path like hydra_staged_sweep but maintains
    """

    pwd: str | None = None
    config_name: str | None = None
    config_path: str | None = None
    config_dir: str | None = None
    overrides: list[str] = field(default_factory=list)
    monitor_state_dir: str = "./monitor_state"

    def __post_init__(self):
        # Ensure at least one way to specify config is provided
        if self.config_path is None and self.config_name is None:
            raise ValueError("Either config_path or config_name must be specified")

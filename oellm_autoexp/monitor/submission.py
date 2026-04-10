"""Job registration and state management for submission."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, MISSING
from typing import Any, TYPE_CHECKING

from compoconf import (
    ConfigInterface,
    register,
)
from oellm_autoexp.monitor.conditions import MonitorConditionInterface
from oellm_autoexp.monitor.actions import JobInterface, LogEventConfig, StateEventConfig

# Import slurm_gen optionally
try:
    from oellm_autoexp.slurm_gen import SlurmConfig
except ImportError:
    SlurmConfig = Any

if TYPE_CHECKING:
    from oellm_autoexp.slurm_gen import SlurmConfig

LOGGER = logging.getLogger(__name__)


@dataclass(kw_only=True)
class BaseJob:
    # can be a slurm job name template with %i, %A, %a
    log_path: str = field(default=MISSING)
    # config_path
    # should be a fixed name known at config time, should contain %a for array index
    log_path_current: str | None = None
    config_path: str | None = None
    config_path_current: str | None = None
    name: str = ""
    log_events: list[LogEventConfig] = field(default_factory=list)
    state_events: list[StateEventConfig] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    start_condition: MonitorConditionInterface.cfgtype | None = None
    cancel_condition: MonitorConditionInterface.cfgtype | None = None
    finish_condition: MonitorConditionInterface.cfgtype | None = None
    array_len: int = 1
    base_config: Any = field(default_factory=dict)


@dataclass(kw_only=True)
class SlurmJobConfig(BaseJob, ConfigInterface):
    """Configuration for a job to be monitored."""

    slurm: SlurmConfig = field(default_factory=MISSING)


@dataclass(kw_only=True)
class LocalJobConfig(BaseJob, ConfigInterface):
    command: list[str] = field(default_factory=list)
    extra_args: list[str] = field(default_factory=list)
    array_args: list[list[str]] | None = (
        None  # potentially appended to the main command for array jobs
    )
    log_to_file: bool = True
    name: str = ""

    def __post_init__(self):
        if self.array_args:
            self.array_len = max(1, len(self.array_args))


@register
class SlurmJob(JobInterface):
    config: SlurmJobConfig


@register
class LocalJob(JobInterface):
    config: LocalJobConfig

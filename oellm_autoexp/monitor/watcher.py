from dataclasses import dataclass, field
from typing import Any
from compoconf import ConfigInterface, register
from oellm_autoexp.monitor.states import MonitorStateInterface
from oellm_autoexp.monitor.event_bindings import EventActionConfig
from oellm_autoexp.config.schema import MonitorInterface


class BaseMonitor(MonitorInterface):
    """Base class for monitors."""

    pass


@dataclass
class LogEventConfig(ConfigInterface):
    class_name: str = "LogEvent"
    name: str = ""
    pattern: str = ""
    pattern_type: str = "regex"
    state: MonitorStateInterface.cfgtype | None = None
    actions: list[EventActionConfig] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    extract_groups: dict[str, str] = field(default_factory=dict)


@dataclass
class StateEventConfig(ConfigInterface):
    class_name: str = "StateEvent"
    name: str = ""
    state: MonitorStateInterface.cfgtype | None = None
    actions: list[EventActionConfig] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SlurmLogMonitorConfig(ConfigInterface):
    class_name: str = "SlurmLogMonitor"
    log_path: str = ""
    log_events: list[LogEventConfig] = field(default_factory=list)
    state_events: list[StateEventConfig] = field(default_factory=list)
    check_interval_seconds: float = 60.0
    inactivity_threshold_seconds: float = 0.0
    output_paths: list[str] = field(default_factory=list)


@register
class SlurmLogMonitor(BaseMonitor):
    config: SlurmLogMonitorConfig

    def __init__(self, config: SlurmLogMonitorConfig):
        self.config = config


@dataclass
class NullMonitorConfig(ConfigInterface):
    class_name: str = "NullMonitor"


@register
class NullMonitor(BaseMonitor):
    config: NullMonitorConfig


@dataclass
class MonitoredJob:
    job_id: str
    name: str
    log_path: Any
    check_interval_seconds: float
    state: str | None = None
    termination_string: str | None = None
    termination_command: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    output_paths: list[str] = field(default_factory=list)


__all__ = [
    "MonitorInterface",
    "BaseMonitor",
    "LogEventConfig",
    "StateEventConfig",
    "SlurmLogMonitorConfig",
    "SlurmLogMonitor",
    "NullMonitorConfig",
    "NullMonitor",
    "MonitoredJob",
]

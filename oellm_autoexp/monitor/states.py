"""Monitor state definitions used by controller and monitors."""

from __future__ import annotations

from dataclasses import dataclass

from compoconf import ConfigInterface, RegistrableConfigInterface, register, register_interface


@register_interface
class MonitorStateInterface(RegistrableConfigInterface):
    """Interface representing a monitor-produced state change."""


class BaseMonitorState(MonitorStateInterface):
    config: ConfigInterface

    def __init__(self, config: ConfigInterface) -> None:
        self.config = config

    @property
    def key(self) -> str:  # pragma: no cover - overridden by subclasses
        raise NotImplementedError


@dataclass
class SuccessStateConfig(ConfigInterface):
    class_name: str = "SuccessState"
    key: str = "success"


@register
class SuccessState(BaseMonitorState):
    config: SuccessStateConfig

    @property
    def key(self) -> str:
        return self.config.key


@dataclass
class CrashStateConfig(ConfigInterface):
    class_name: str = "CrashState"
    key: str = "crash"


@register
class CrashState(BaseMonitorState):
    config: CrashStateConfig

    @property
    def key(self) -> str:
        return self.config.key


@dataclass
class StalledStateConfig(ConfigInterface):
    class_name: str = "StalledState"
    key: str = "stall"


@register
class StalledState(BaseMonitorState):
    config: StalledStateConfig

    @property
    def key(self) -> str:
        return self.config.key


@dataclass
class TimeoutStateConfig(ConfigInterface):
    class_name: str = "TimeoutState"
    key: str = "timeout"


@register
class TimeoutState(BaseMonitorState):
    config: TimeoutStateConfig

    @property
    def key(self) -> str:
        return self.config.key


@dataclass
class StartedStateConfig(ConfigInterface):
    class_name: str = "StartedState"
    key: str = "running"


@register
class StartedState(BaseMonitorState):
    config: StartedStateConfig

    @property
    def key(self) -> str:
        return self.config.key


@dataclass
class PendingStateConfig(ConfigInterface):
    class_name: str = "PendingState"
    key: str = "pending"


@register
class PendingState(BaseMonitorState):
    config: PendingStateConfig

    @property
    def key(self) -> str:
        return self.config.key

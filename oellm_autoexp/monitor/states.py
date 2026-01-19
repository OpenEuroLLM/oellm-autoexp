from dataclasses import dataclass
from compoconf import ConfigInterface, register_interface, register, RegistrableConfigInterface


@register_interface
class MonitorStateInterface(RegistrableConfigInterface):
    pass


class BaseMonitorState(MonitorStateInterface):
    def __init__(self, config: ConfigInterface):
        self.config = config


@dataclass
class PendingStateConfig(ConfigInterface):
    class_name: str = "PendingState"


@register
class PendingState(BaseMonitorState):
    config: PendingStateConfig


@dataclass
class StartedStateConfig(ConfigInterface):
    class_name: str = "StartedState"


@register
class StartedState(BaseMonitorState):
    config: StartedStateConfig


@dataclass
class StalledStateConfig(ConfigInterface):
    class_name: str = "StalledState"


@register
class StalledState(BaseMonitorState):
    config: StalledStateConfig


@dataclass
class CrashStateConfig(ConfigInterface):
    class_name: str = "CrashState"


@register
class CrashState(BaseMonitorState):
    config: CrashStateConfig


@dataclass
class TimeoutStateConfig(ConfigInterface):
    class_name: str = "TimeoutState"


@register
class TimeoutState(BaseMonitorState):
    config: TimeoutStateConfig


@dataclass
class SuccessStateConfig(ConfigInterface):
    class_name: str = "SuccessState"


@register
class SuccessState(BaseMonitorState):
    config: SuccessStateConfig


__all__ = [
    "MonitorStateInterface",
    "BaseMonitorState",
    "PendingState",
    "PendingStateConfig",
    "StartedState",
    "StartedStateConfig",
    "StalledState",
    "StalledStateConfig",
    "CrashState",
    "CrashStateConfig",
    "TimeoutState",
    "TimeoutStateConfig",
    "SuccessState",
    "SuccessStateConfig",
]

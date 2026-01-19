from dataclasses import dataclass, field
from typing import Literal, Any
from compoconf import ConfigInterface
from oellm_autoexp.monitor.actions import BaseMonitorAction
from oellm_autoexp.monitor.conditions import MonitorConditionInterface


@dataclass(kw_only=True)
class EventActionConfig(ConfigInterface):
    class_name: str = "EventAction"
    mode: Literal["inline", "queue"] = "inline"
    action: BaseMonitorAction.cfgtype | None = None
    conditions: list[MonitorConditionInterface.cfgtype] = field(default_factory=list)
    aux: dict[str, Any] = field(default_factory=dict)


__all__ = ["EventActionConfig"]

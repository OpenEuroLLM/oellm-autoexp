"""Event action binding helpers for monitor events."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from compoconf import ConfigInterface

from oellm_autoexp.monitor.actions import BaseMonitorAction, MonitorActionInterface
from oellm_autoexp.monitor.conditions import MonitorConditionInterface


@dataclass(kw_only=True)
class EventActionConfig(ConfigInterface):
    """Declarative binding between an event and a monitor action."""

    class_name: str = "EventAction"
    mode: Literal["inline", "queue"] = "inline"
    action: MonitorActionInterface.cfgtype
    conditions: list[MonitorConditionInterface.cfgtype] = field(default_factory=list)


@dataclass
class EventActionBinding:
    """Runtime binding used by the controller."""

    action: BaseMonitorAction
    mode: Literal["inline", "queue"]
    conditions: list[MonitorConditionInterface]


def instantiate_bindings(configs: list[EventActionConfig]) -> list[EventActionBinding]:
    """Instantiate action bindings from configuration."""

    bindings: list[EventActionBinding] = []
    for cfg in configs:
        action = cfg.action.instantiate(MonitorActionInterface)
        conditions = [
            condition.instantiate(MonitorConditionInterface) for condition in cfg.conditions
        ]
        bindings.append(EventActionBinding(action=action, mode=cfg.mode, conditions=conditions))
    return bindings


__all__ = ["EventActionConfig", "EventActionBinding", "instantiate_bindings"]

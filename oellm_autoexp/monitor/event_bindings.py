"""Event action binding helpers for monitor events."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Any

from compoconf import ConfigInterface, parse_config, register

from oellm_autoexp.monitor.actions import BaseMonitorAction, ActionContext, ActionResult
from oellm_autoexp.monitor.conditions import MonitorConditionInterface


@dataclass(kw_only=True)
class EventActionConfig(ConfigInterface):
    """Declarative binding between an event and a monitor action."""

    class_name: str = "EventAction"
    mode: Literal["inline", "queue"] = "inline"
    action: BaseMonitorAction.cfgtype | None = None
    conditions: list[MonitorConditionInterface.cfgtype] = field(default_factory=list)
    aux: dict[str, Any] = field(default_factory=dict)


@register
class EventAction(BaseMonitorAction):
    config: EventActionConfig

    def __init__(self, config: EventActionConfig):
        self.config = config
        self._action = self.config.action.instantiate(BaseMonitorAction)

    def execute(self, context: ActionContext) -> ActionResult:  # pragma: no cover
        return self._action.execute(context)


@dataclass
class EventActionBinding:
    """Runtime binding used by the controller."""

    action: BaseMonitorAction
    mode: Literal["inline", "queue"]
    conditions: list[MonitorConditionInterface]


def _coerce_action_config(payload: Any) -> BaseMonitorAction.cfgtype:
    if hasattr(payload, "instantiate"):
        return payload
    if isinstance(payload, dict):
        return parse_config(BaseMonitorAction.cfgtype, payload)
    raise TypeError(f"Unsupported action config payload: {payload!r}")


def instantiate_bindings(configs: list[Any]) -> list[EventActionBinding]:
    """Instantiate action bindings from configuration."""

    bindings: list[EventActionBinding] = []
    for cfg in configs:
        if isinstance(cfg, EventActionConfig):
            if cfg.action is None:
                raise ValueError("EventActionConfig requires 'action'")
            action_cfg = _coerce_action_config(cfg.action)
            condition_cfgs = cfg.conditions
            mode = cfg.mode
        else:
            action_cfg = _coerce_action_config(cfg)
            condition_cfgs = []
            mode = "inline"
        action = action_cfg.instantiate(BaseMonitorAction)
        conditions = [
            condition.instantiate(MonitorConditionInterface) for condition in condition_cfgs
        ]
        bindings.append(EventActionBinding(action=action, mode=mode, conditions=conditions))
    return bindings


__all__ = ["EventActionConfig", "EventActionBinding", "instantiate_bindings"]

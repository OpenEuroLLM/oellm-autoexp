"""Restart policy implementations registered with compoconf."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

from compoconf import ConfigInterface, register

from oellm_autoexp.config.schema import RestartPolicyInterface

ActionLiteral = Literal["restart", "stop", "adjust"]


@dataclass
class RestartDecision:
    """Result returned by a restart policy."""

    action: ActionLiteral
    reason: str
    adjustments: Dict[str, Any] | None = None


@dataclass
class RestartEvent:
    """Information about a job failure/stall."""

    mode: Literal["stall", "crash", "timeout", "success"]
    attempt: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseRestartPolicy(RestartPolicyInterface):
    """Convenience base class for restart policies."""

    config: ConfigInterface

    def __init__(self, config: ConfigInterface) -> None:
        self.config = config

    def decide(self, event: RestartEvent) -> RestartDecision:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class NoRestartPolicyConfig(ConfigInterface):
    class_name: str = "NoRestartPolicy"
    message: str = "No restart configured"


@register
class NoRestartPolicy(BaseRestartPolicy):
    config: NoRestartPolicyConfig

    def decide(self, event: RestartEvent) -> RestartDecision:
        return RestartDecision(action="stop", reason=self.config.message)


@dataclass
class AlwaysRestartPolicyConfig(ConfigInterface):
    class_name: str = "AlwaysRestartPolicy"
    reason: str = "Retrying job"
    max_retries: Optional[int] = None


@register
class AlwaysRestartPolicy(BaseRestartPolicy):
    config: AlwaysRestartPolicyConfig

    def decide(self, event: RestartEvent) -> RestartDecision:
        if self.config.max_retries is not None and event.attempt >= self.config.max_retries:
            return RestartDecision(action="stop", reason="retry budget exhausted")
        return RestartDecision(action="restart", reason=self.config.reason)


__all__ = [
    "RestartDecision",
    "RestartEvent",
    "BaseRestartPolicy",
    "NoRestartPolicyConfig",
    "AlwaysRestartPolicyConfig",
]

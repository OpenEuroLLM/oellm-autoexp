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


@dataclass
class SelectiveRestartPolicyConfig(ConfigInterface):
    """Restart policy that filters based on signal metadata."""

    class_name: str = "SelectiveRestartPolicy"
    max_retries: Optional[int] = 3
    restart_on_error_types: list[str] = field(default_factory=list)
    restart_on_subsystems: list[str] = field(default_factory=list)
    restart_on_signals: list[str] = field(default_factory=list)
    exclude_error_types: list[str] = field(default_factory=list)
    default_action: Literal["restart", "stop"] = "stop"
    restart_reason: str = "Transient failure detected, restarting"
    stop_reason: str = "Non-recoverable error, stopping"


@register
class SelectiveRestartPolicy(BaseRestartPolicy):
    """Restart policy that selectively restarts based on error metadata.

    This policy inspects the signal metadata to determine if the failure
    is likely recoverable (e.g., NCCL timeouts, CUDA hangs) vs permanent
    (e.g., OOM errors, code bugs).

    Configuration:
    - restart_on_error_types: List of error_type values to restart on
    - restart_on_subsystems: List of subsystem values to restart on
    - restart_on_signals: List of signal names to restart on
    - exclude_error_types: List of error_type values to never restart
    - default_action: What to do if no rules match ("restart" or "stop")
    - max_retries: Maximum number of restart attempts
    """

    config: SelectiveRestartPolicyConfig

    def decide(self, event: RestartEvent) -> RestartDecision:
        # Check retry budget first
        if self.config.max_retries is not None and event.attempt >= self.config.max_retries:
            return RestartDecision(
                action="stop",
                reason=f"retry budget exhausted (attempt {event.attempt}/{self.config.max_retries})",
            )

        metadata = event.metadata
        error_type = metadata.get("error_type")
        subsystem = metadata.get("subsystem")
        signal_name = metadata.get("signal_name")

        # Check exclusions first (highest priority)
        if error_type and error_type in self.config.exclude_error_types:
            return RestartDecision(
                action="stop",
                reason=f"{self.config.stop_reason} (excluded error_type: {error_type})",
            )

        # Check inclusion rules
        should_restart = False
        reason_parts = []

        if signal_name and signal_name in self.config.restart_on_signals:
            should_restart = True
            reason_parts.append(f"signal={signal_name}")

        if error_type and error_type in self.config.restart_on_error_types:
            should_restart = True
            reason_parts.append(f"error_type={error_type}")

        if subsystem and subsystem in self.config.restart_on_subsystems:
            should_restart = True
            reason_parts.append(f"subsystem={subsystem}")

        if should_restart:
            reason = f"{self.config.restart_reason} ({', '.join(reason_parts)})"
            return RestartDecision(action="restart", reason=reason)

        # Fall back to default action
        if self.config.default_action == "restart":
            return RestartDecision(action="restart", reason=self.config.restart_reason)
        else:
            return RestartDecision(action="stop", reason=self.config.stop_reason)


__all__ = [
    "RestartDecision",
    "RestartEvent",
    "BaseRestartPolicy",
    "NoRestartPolicyConfig",
    "NoRestartPolicy",
    "AlwaysRestartPolicyConfig",
    "AlwaysRestartPolicy",
    "SelectiveRestartPolicyConfig",
    "SelectiveRestartPolicy",
]

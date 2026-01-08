"""Backend abstractions used by oellm_autoexp."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Mapping, Sequence

from compoconf import ConfigInterface, register

from oellm_autoexp.config.schema import BackendInterface

LOGGER = logging.getLogger(__name__)


@dataclass(kw_only=True)
class LaunchCommand:
    """Concrete command to be executed on the cluster."""

    argv: Sequence[str] = field(default_factory=list)
    env: Mapping[str, str] = field(default_factory=dict)
    cwd: str | None = None


@dataclass(kw_only=True)
class BackendJobSpec:
    """Information needed to construct a launch command."""

    parameters: dict[str, Any] = field(default_factory=dict)


class BaseBackend(BackendInterface):
    """Base class for backend adapters."""

    config: ConfigInterface

    def __init__(self, config: ConfigInterface) -> None:
        self.config = config

    def validate(self, spec: BackendJobSpec) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def build_launch_command(self, spec: BackendJobSpec) -> LaunchCommand:  # pragma: no cover
        raise NotImplementedError


@dataclass(kw_only=True)
class NullBackendConfig(ConfigInterface):
    """Backend that echoes sweep parameters for testing."""

    base_command: Sequence[str] = field(
        default_factory=lambda: [
            "echo",
        ]
    )
    extra_cli_args: Sequence[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    dummy: int = 0


@register
class NullBackend(BaseBackend):
    config: NullBackendConfig

    def validate(self, spec: BackendJobSpec) -> None:  # pragma: no cover - trivial
        _ = spec  # deliberate no-op to mirror interface

    def build_launch_command(self, spec: BackendJobSpec) -> LaunchCommand:
        argv: list[str] = [str(arg) for arg in self.config.base_command]
        for arg in sorted(spec.parameters):
            argv.append(str(arg))
        argv.extend(str(arg) for arg in self.config.extra_cli_args)
        env = dict(self.config.env)
        return LaunchCommand(argv=argv, env=env)


__all__ = ["BaseBackend", "BackendJobSpec", "LaunchCommand", "NullBackendConfig"]

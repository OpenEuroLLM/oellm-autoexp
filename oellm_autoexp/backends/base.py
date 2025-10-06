"""Backend abstractions used by oellm_autoexp."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence

from compoconf import ConfigInterface, register

from oellm_autoexp.config.schema import BackendInterface


@dataclass
class LaunchCommand:
    """Concrete command to be executed on the cluster."""

    argv: Sequence[str]
    env: Mapping[str, str] = field(default_factory=dict)
    cwd: str | None = None


@dataclass
class BackendJobSpec:
    """Information needed to construct a launch command."""

    parameters: Dict[str, Any]


class BaseBackend(BackendInterface):
    """Base class for backend adapters."""

    config: ConfigInterface

    def __init__(self, config: ConfigInterface) -> None:
        self.config = config

    def validate(self, spec: BackendJobSpec) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def build_launch_command(self, spec: BackendJobSpec) -> LaunchCommand:  # pragma: no cover
        raise NotImplementedError


@dataclass
class NullBackendConfig(ConfigInterface):
    """Backend that echoes sweep parameters for testing."""

    class_name: str = "NullBackend"
    base_command: Sequence[str] = field(default_factory=lambda: ("echo",))
    extra_cli_args: Sequence[str] = field(default_factory=tuple)
    env: Dict[str, str] = field(default_factory=dict)


@register
class NullBackend(BaseBackend):
    config: NullBackendConfig

    def validate(self, spec: BackendJobSpec) -> None:  # pragma: no cover - trivial
        _ = spec  # deliberate no-op to mirror interface

    def build_launch_command(self, spec: BackendJobSpec) -> LaunchCommand:
        argv: list[str] = [str(arg) for arg in self.config.base_command]
        for key, value in sorted(spec.parameters.items()):
            argv.extend([str(key), str(value)])
        argv.extend(str(arg) for arg in self.config.extra_cli_args)
        env = dict(self.config.env)
        return LaunchCommand(argv=argv, env=env)


__all__ = ["BaseBackend", "BackendJobSpec", "LaunchCommand", "NullBackendConfig"]

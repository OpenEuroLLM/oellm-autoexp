"""Backend abstractions used by oellm_autoexp."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from collections.abc import Sequence

from compoconf import ConfigInterface, register

from oellm_autoexp.config.schema import BackendInterface

LOGGER = logging.getLogger(__name__)


class BaseBackendConfig(ConfigInterface):
    env: dict[str, str]


class BaseBackend(BackendInterface):
    """Base class for backend adapters."""

    config: BaseBackendConfig

    def __init__(self, config: BaseBackendConfig) -> None:
        self.config = config

    def validate(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def build_launch_command(self) -> str:  # pragma: no cover
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

    def validate(self) -> None:  # pragma: no cover - trivial
        pass

    def build_launch_command(self) -> str:
        argv: list[str] = [str(arg) for arg in self.config.base_command]
        argv.extend(str(arg) for arg in self.config.extra_cli_args)
        return " ".join(argv)


__all__ = ["BaseBackend", "NullBackendConfig"]

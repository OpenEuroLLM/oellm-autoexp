"""Megatron backend adapter that mirrors megatron-train behaviour."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from compoconf import ConfigInterface, register

from oellm_autoexp.backends.base import BaseBackend, BackendJobSpec, LaunchCommand
from oellm_autoexp.backends.megatron_args import (
    MegatronArgMetadata,
    build_cmdline_args,
    get_arg_metadata,
    get_megatron_parser,
)


@dataclass
class MegatronBackendConfig(ConfigInterface):
    """Configuration for the Megatron backend."""

    class_name: str = "MegatronBackend"
    launcher_script: Path = Path("scripts/run_megatron.sh")
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    extra_cli_args: list[str] = field(default_factory=list)


@register
class MegatronBackend(BaseBackend):
    config: MegatronBackendConfig

    def __init__(self, config: MegatronBackendConfig) -> None:
        super().__init__(config)
        self.config = config
        self._parser = get_megatron_parser()
        self._arg_metadata: Dict[str, MegatronArgMetadata] = get_arg_metadata(self._parser)

    def validate(self, spec: BackendJobSpec) -> None:
        merged_args = self._merge_args(spec)
        cli_args = build_cmdline_args(merged_args, self._parser, self._arg_metadata)
        try:
            self._parser.parse_args(cli_args)
        except SystemExit as exc:  # pragma: no cover - argparse exits
            raise ValueError("Invalid Megatron arguments") from exc

    def build_launch_command(self, spec: BackendJobSpec) -> LaunchCommand:
        merged_args = self._merge_args(spec)
        cli_args = build_cmdline_args(merged_args, self._parser, self._arg_metadata)
        argv = [str(self.config.launcher_script), *cli_args, *self.config.extra_cli_args]
        environment = {**self.config.environment}
        return LaunchCommand(argv=argv, environment=environment)

    def _merge_args(self, spec: BackendJobSpec) -> Dict[str, Any]:
        args: Dict[str, Any] = {}
        args.update(self._filter_megatron_args(self.config.config_overrides))
        args.update(self._filter_megatron_args(spec.parameters))
        return args

    def _filter_megatron_args(self, params: Dict[str, Any]) -> Dict[str, Any]:
        filtered: Dict[str, Any] = {}
        for key, value in params.items():
            dest = self._normalize_key(key)
            if dest in self._arg_metadata:
                filtered[dest] = value
        return filtered

    @staticmethod
    def _normalize_key(key: str) -> str:
        key = key.replace("-", "_")
        if key.startswith("megatron."):
            key = key.split(".", 1)[1]
        return key


__all__ = ["MegatronBackendConfig", "MegatronBackend"]

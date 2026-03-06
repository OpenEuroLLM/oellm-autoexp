"""TorchTitan/Titan-OELLM backend adapter."""

from __future__ import annotations

import logging
import shlex
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal
import importlib

from compoconf import NonStrictDataclass, asdict, register

from oellm_autoexp.backends.base import BaseBackend, BaseBackendConfig
from oellm_autoexp.backends.titan.config_schema import JobConfig as TitanJobConfigStrict

LOGGER = logging.getLogger(__name__)


@dataclass(init=False)
class TitanJobConfigGeneric(NonStrictDataclass):
    """Generic (non-strict) Titan job config for advanced/custom setups."""

    class_name: str = "TitanJobConfigGeneric"


@dataclass(init=False)
class TitanBackendConfig(NonStrictDataclass, BaseBackendConfig):
    """Configuration for the Titan/TorchTitan backend."""

    class_name: str = "TitanBackend"
    env: dict[str, str] = field(default_factory=dict)

    # Job configuration (strict schema by default)
    titan: TitanJobConfigStrict | TitanJobConfigGeneric = field(default_factory=TitanJobConfigStrict)

    # Optional extension hooks
    custom_config_module: str = ""
    require_custom_module: bool = False

    # TorchTitan launch
    launcher_module: str = "torchtitan.train"
    torchrun_args: dict[str, Any] = field(default_factory=dict)
    extra_cli_args: list[str] = field(default_factory=list)

    # Cluster arg resolution (Titan-OELLM helper)
    cluster_args_source: Literal["none", "titan_oellm"] = (
        "none"  # none | titan_oellm - titan_oellm uses the titan_oellm internal path resolution
    )
    validate_cluster_paths: bool = False
    dataset: str = "slimpajama_627b"
    tokenizer: str = "neox"
    cluster: str | None = None
    config_file: str = "base_plus.toml"
    project_root: str = "."
    cluster_args: str = ""
    cluster_args_overrides: dict[str, Any] = field(default_factory=dict)

    # TOML artifact
    toml_output_path: str = ""
    write_toml_provenance: bool = False

    # Validation
    full_schema_validation: bool = False


@register
class TitanBackend(BaseBackend):
    config: TitanBackendConfig

    def __init__(self, config: TitanBackendConfig) -> None:
        super().__init__(config)
        self.config = config

    def validate(self) -> None:
        if not self.config.full_schema_validation:
            return

        # Optional: validate via TorchTitan ConfigManager if available.
        try:
            from torchtitan.config import ConfigManager, JobConfig as TorchTitanJobConfig
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "TorchTitan not available for full validation. Install torchtitan or disable full_schema_validation."
            ) from exc

        args = [f"--job.config_file={self._ensure_toml()}", *self.config.extra_cli_args]
        ConfigManager(config_cls=TorchTitanJobConfig).parse_args(args)

    def build_launch_command(self) -> str:
        toml_path = self._ensure_toml()

        torchrun = self._build_torchrun_cmd()
        cmd = f"{torchrun} -m {self.config.launcher_module} --job.config_file={toml_path}"

        if self.config.extra_cli_args:
            cmd = f"{cmd} {' '.join(self.config.extra_cli_args)}"
        return cmd

    def _build_torchrun_cmd(self) -> str:
        parts = ["torchrun"]
        for key, value in self.config.torchrun_args.items():
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    parts.append(flag)
                continue
            parts.append(f"{flag}={value}")
        return " ".join(parts)

    def _ensure_toml(self) -> str:
        output_path = self.config.toml_output_path or "config.titan.toml"
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self._build_toml_dict()
        toml_text = _dump_toml(data)
        path.write_text(toml_text, encoding="utf-8")
        if self.config.write_toml_provenance:
            self._write_provenance(path, data)
        return str(path)

    def _build_toml_dict(self) -> dict[str, Any]:
        data = asdict(self.config.titan)

        _prune_internal_keys(data)

        merged_cls = self._resolve_job_config_class()
        if merged_cls is not None:
            data = _apply_schema_filter(data, merged_cls)

        if self.config.custom_config_module:
            if self.config.require_custom_module:
                try:
                    importlib.import_module(self.config.custom_config_module)
                except Exception as exc:
                    raise RuntimeError(
                        f"custom_config_module not importable: {self.config.custom_config_module}"
                    ) from exc
            data.setdefault("job", {})["custom_config_module"] = self.config.custom_config_module

        # Apply cluster args if requested
        if self.config.cluster_args_source == "titan_oellm":
            args = self._resolve_cluster_args()
            overrides = _parse_cli_overrides(args)
            _apply_overrides(data, overrides)

        if self.config.cluster_args:
            overrides = _parse_cli_overrides(self.config.cluster_args)
            _apply_overrides(data, overrides)

        if self.config.cluster_args_overrides:
            _apply_overrides(data, self.config.cluster_args_overrides)

        return data

    def _resolve_cluster_args(self) -> str:
        try:
            from titan_oellm.cluster_config import get_cli_args
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Titan-OELLM not available for cluster arg resolution.") from exc

        return get_cli_args(
            dataset=self.config.dataset,
            tokenizer=self.config.tokenizer,
            cluster=self.config.cluster,
            config_file=self.config.config_file,
            project_root=self.config.project_root,
            validate=self.config.validate_cluster_paths,
        )

    def _write_provenance(self, toml_path: Path, data: dict[str, Any]) -> None:
        provenance_path = toml_path.with_suffix(toml_path.suffix + ".provenance.json")
        payload = {
            "toml_path": str(toml_path),
            "custom_config_module": self.config.custom_config_module,
            "cluster_args_source": self.config.cluster_args_source,
            "validate_cluster_paths": self.config.validate_cluster_paths,
            "dataset": self.config.dataset,
            "tokenizer": self.config.tokenizer,
            "cluster": self.config.cluster,
            "config_file": self.config.config_file,
            "project_root": self.config.project_root,
            "cluster_args": self.config.cluster_args,
            "cluster_args_overrides": dict(self.config.cluster_args_overrides),
            "launcher_module": self.config.launcher_module,
            "torchrun_args": dict(self.config.torchrun_args),
            "extra_cli_args": list(self.config.extra_cli_args),
            "job_config": data,
        }
        try:
            provenance_path.write_text(
                __import__("json").dumps(payload, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:  # pragma: no cover - best effort
            LOGGER.warning("Failed to write Titan provenance: %s", exc)

    def _resolve_job_config_class(self) -> type | None:
        if not self.config.custom_config_module:
            return None
        try:
            module = importlib.import_module(self.config.custom_config_module)
        except Exception as exc:
            if self.config.require_custom_module:
                raise RuntimeError(f"custom_config_module not importable: {self.config.custom_config_module}") from exc
            LOGGER.warning(
                "custom_config_module not importable (%s); skipping schema merge.",
                self.config.custom_config_module,
            )
            return None

        custom_cls = getattr(module, "JobConfig", None)
        if custom_cls is None:
            if self.config.require_custom_module:
                raise RuntimeError(f"custom_config_module missing JobConfig: {self.config.custom_config_module}")
            LOGGER.warning("custom_config_module missing JobConfig; skipping schema merge.")
            return None

        return _merge_configs(TitanJobConfigStrict, custom_cls)


def _merge_configs(base, custom) -> type:
    from dataclasses import field, fields, is_dataclass, make_dataclass

    result = []
    b_map = {f.name: f for f in fields(base)}
    c_map = {f.name: f for f in fields(custom)}

    for name, f in b_map.items():
        if name in c_map and is_dataclass(f.type) and is_dataclass(c_map[name].type):
            m_type = _merge_configs(f.type, c_map[name].type)
            result.append((name, m_type, field(default_factory=m_type)))
        elif name in c_map:
            result.append((name, c_map[name].type, c_map[name]))
        else:
            result.append((name, f.type, f))

    for name, f in c_map.items():
        if name not in b_map:
            result.append((name, f.type, f))

    return make_dataclass(f"Merged{base.__name__}", result, bases=(base,))


def _apply_schema_filter(data: dict[str, Any], schema: type) -> dict[str, Any]:
    if not is_dataclass(schema):
        return data
    allowed = {f.name: f for f in fields(schema)}
    filtered: dict[str, Any] = {}
    for key, value in data.items():
        if key not in allowed:
            continue
        field_def = allowed[key]
        if isinstance(value, dict) and is_dataclass(field_def.type):
            filtered[key] = _apply_schema_filter(value, field_def.type)
        else:
            filtered[key] = value
    return filtered


def _prune_internal_keys(data: dict[str, Any]) -> None:
    data.pop("class_name", None)
    data.pop("_non_strict", None)
    for key, value in list(data.items()):
        if isinstance(value, dict):
            _prune_internal_keys(value)


def _parse_cli_overrides(arg_str: str) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for token in shlex.split(arg_str):
        if not token.startswith("--"):
            continue
        key_val = token[2:]
        if "=" not in key_val:
            continue
        key, value = key_val.split("=", 1)
        if key in {"job.config_file", "job.config-file"}:
            continue
        _set_nested(overrides, key.split("."), _coerce_scalar(value))
    return overrides


def _coerce_scalar(value: str) -> Any:
    low = value.lower()
    if low in {"true", "false"}:
        return low == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _set_nested(root: dict[str, Any], keys: list[str], value: Any) -> None:
    curr = root
    for key in keys[:-1]:
        curr = curr.setdefault(key, {})
    curr[keys[-1]] = value


def _apply_overrides(data: dict[str, Any], overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            _apply_overrides(data[key], value)
        else:
            data[key] = value


def _dump_toml(data: dict[str, Any]) -> str:
    lines: list[str] = []

    def format_value(val: Any) -> str | None:
        if val is None:
            return None
        if isinstance(val, bool):
            return "true" if val else "false"
        if isinstance(val, (int, float)):
            return str(val)
        if isinstance(val, str):
            return repr(val)
        if isinstance(val, list):
            items = []
            for item in val:
                item_str = format_value(item)
                if item_str is None:
                    item_str = '""'
                items.append(item_str)
            return f"[{', '.join(items)}]"
        return repr(val)

    def write_table(prefix: str, obj: dict[str, Any]) -> None:
        simple: dict[str, Any] = {}
        nested: dict[str, dict[str, Any]] = {}
        for key, val in obj.items():
            if isinstance(val, dict):
                nested[key] = val
            else:
                simple[key] = val

        if prefix:
            lines.append(f"[{prefix}]")
        for key, val in simple.items():
            rendered = format_value(val)
            if rendered is None:
                continue
            lines.append(f"{key} = {rendered}")
        if simple and nested:
            lines.append("")
        for key, val in nested.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            write_table(new_prefix, val)
            lines.append("")

    write_table("", data)
    return "\n".join(line for line in lines if line.strip() != "" or lines)


__all__ = [
    "TitanBackendConfig",
    "TitanBackend",
    "TitanJobConfigGeneric",
]

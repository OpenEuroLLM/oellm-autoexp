"""Build execution plans - extends hydra_staged_sweep with oellm-specific fields."""

from __future__ import annotations

from dataclasses import dataclass, field, MISSING
from typing import Any

from compoconf import asdict

# Import base JobPlan from hydra_staged_sweep
from hydra_staged_sweep.planner import JobPlan as BaseJobPlan


@dataclass(kw_only=True)
class JobPlan(BaseJobPlan):
    """Extended job plan with oellm-specific fields.

    Extends the minimal hydra_staged_sweep JobPlan with fields for:
    - Job naming and output paths
    - Start/cancel conditions
    - Termination criteria
    - Monitoring configuration
    """

    # Job identification
    name: str = field(default_factory=MISSING)
    output_dir: str = field(default_factory=MISSING)
    log_path: str = field(default_factory=MISSING)
    log_path_current: str = field(default_factory=MISSING)
    output_paths: list[str] = field(default_factory=list)

    # Start conditions (old blocking approach - DEPRECATED)
    start_condition_cmd: str | None = None
    start_condition_interval_seconds: int | None = None

    # Start conditions (new async approach with monitor-driven submission)
    start_conditions: list[dict[str, Any]] = field(default_factory=list)

    # Termination and monitoring
    termination_string: str | None = None
    termination_command: str | None = None
    inactivity_threshold_seconds: int | None = None

    # Cancel conditions (proactive failure detection)
    cancel_conditions: list[dict[str, Any]] = field(default_factory=list)


def flatten_config(config: Any, connector: str = ".") -> dict[str, Any]:
    """Flatten nested config to dotted keys.

    >>> flatten_config({"key": {"subkey": "val"}})
    {"key.subkey": "val"}
    """
    if not isinstance(config, dict):
        cfg_dict = asdict(config)
    else:
        cfg_dict = config

    def _flat(d: tuple | list | dict | Any, prefix: str = "") -> dict[str, Any]:
        res = {}
        if isinstance(d, dict):
            for key, val in d.items():
                res.update(_flat(val, prefix=prefix + connector + key if prefix else key))
        elif isinstance(d, (list, tuple)):
            for idx, val in enumerate(d):
                res.update(_flat(val, prefix=prefix + connector + str(idx) if prefix else str(idx)))
        else:
            res = {prefix: d}
        return res

    return _flat(cfg_dict, prefix="")


def simple_format(template_str: str, args: dict) -> str:
    """Format template string with dotted key support.

    Args:
        template_str: Template with placeholders like {key} or {nested.key}
        args: Flat dict with dotted keys (e.g., {'backend.megatron.lr': '1e-4'})

    Returns:
        Formatted string with all placeholders replaced
    """
    for key, val in args.items():
        template_str = template_str.replace("{" + key + "}", str(val))
    return template_str


__all__ = ["JobPlan", "flatten_config", "simple_format"]

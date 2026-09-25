"""Build the generic in-container profiling command."""

from __future__ import annotations

import base64
import json
import shlex

from .config import ProfilingConfig


def _encoded_options(options: dict) -> str:
    payload = json.dumps(options, separators=(",", ":"), sort_keys=True).encode()
    return base64.urlsafe_b64encode(payload).decode()


def wrap_launch_command(config: ProfilingConfig, launch_command: str) -> str:
    if not config.enabled:
        return launch_command
    domains = [
        name
        for name in ("kernel", "runtime", "communication", "memory", "markers")
        if getattr(config.capture, name)
    ]
    prefix = [
        "python",
        "scripts/profiling/run_profile.py",
        "--provider",
        config.provider,
        "--mode",
        config.mode,
        "--ranks",
        ",".join(str(rank) for rank in config.ranks),
        "--output-dir",
        config.output_dir,
        "--capture",
        ",".join(domains),
        "--provider-options",
        _encoded_options(config.provider_options),
        "--analysis-options",
        _encoded_options(
            {
                "enabled": config.analysis.enabled,
                "auto_run": config.analysis.auto_run,
                "execution": config.analysis.execution,
                "steady_start_iteration": config.analysis.steady_start_iteration,
                "steady_end_iteration": config.analysis.steady_end_iteration,
                "canvas": config.analysis.canvas,
                "partition": config.analysis.partition,
                "account": config.analysis.account,
                "cpus": config.analysis.cpus,
                "memory": config.analysis.memory,
                "time": config.analysis.time,
            }
        ),
        "--",
    ]
    return " ".join(shlex.quote(part) for part in prefix) + " " + launch_command


def analysis_submission_command(config: ProfilingConfig) -> str:
    """Return an outer-shell command that submits the generated analysis script."""

    if (
        not config.enabled
        or not config.analysis.enabled
        or not config.analysis.auto_run
        or config.analysis.execution != "dependent_slurm"
    ):
        return ""
    analysis_dir = shlex.quote(f"{config.output_dir}/analysis")
    return (
        f'ANALYSIS_SCRIPT=$(ls -t {analysis_dir}/analyze-*.sbatch 2>/dev/null | head -n 1)\n'
        'if [ -n "$ANALYSIS_SCRIPT" ]; then\n'
        '  sbatch --dependency=afterok:${SLURM_JOB_ID} "$ANALYSIS_SCRIPT"\n'
        "else\n"
        f'  echo "warning: profiling analysis script not found under {analysis_dir}" >&2\n'
        "fi"
    )


__all__ = ["analysis_submission_command", "wrap_launch_command"]

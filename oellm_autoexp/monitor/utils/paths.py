"""Path and string utilities for SLURM log handling."""

from __future__ import annotations

from pathlib import Path


def expand_log_path(log_path: str | Path, job_id: str) -> Path:
    """Expand SLURM-style tokens in a log path.

    Supported tokens:
        %j: Full job ID
        %A: Master job ID (for arrays)
        %a: Array task index
    """
    log_str = str(log_path)
    if "_" in job_id:
        base_id, array_idx = job_id.split("_", 1)
        log_str = log_str.replace("%A", base_id).replace("%a", array_idx)
    else:
        # If not an array job, %A and %a typically resolve to %j and nothing or %j
        # but for simplicity and consistency with our current usage:
        log_str = log_str.replace("%A", job_id).replace("%a", "0")

    log_str = log_str.replace("%j", job_id)
    return Path(log_str)


def update_log_symlink(target: Path, symlink_path: Path) -> None:
    symlink_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if symlink_path.is_symlink() or symlink_path.exists():
            symlink_path.unlink()
    except OSError:  # pragma: no cover
        pass
    try:
        target = target.absolute()
        symlink_path = symlink_path.absolute()
        symlink_path.symlink_to(target)
    except OSError:  # pragma: no cover
        pass


def resolve_log_path(
    log_path: str | Path,
    *,
    job_id: str | None = None,
    timestamp: int | None = None,
) -> Path:
    resolved = Path(log_path)
    if job_id:
        resolved = expand_log_path(resolved, job_id)
    if timestamp is not None:
        resolved = Path(str(resolved).replace("%t", str(timestamp)))
    return resolved

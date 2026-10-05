"""Cross-platform capture and analysis for accelerator timelines."""

from .analysis import analyze, parse_iteration_log, select_steady_window
from .models import KernelEvent, ProfileArtifacts, ProfilingError

__all__ = [
    "KernelEvent",
    "ProfileArtifacts",
    "ProfilingError",
    "analyze",
    "parse_iteration_log",
    "select_steady_window",
]

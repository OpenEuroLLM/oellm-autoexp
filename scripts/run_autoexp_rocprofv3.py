#!/usr/bin/env python3
"""Deprecated compatibility launcher for ``profiling=rocprofv3``.

New commands should invoke ``scripts/run_autoexp.py`` directly with the
``profiling=rocprofv3`` Hydra choice.

Pass ``--all-ranks`` to profile every SLURM task. Do not also pass
``backend.megatron.profile`` / ``use_pytorch_profiler``.

After the job, look in ``${job.base_output_dir}/profiling/`` and analyze with::

    uv run scripts/profiling/analyze_profile.py <job>/profiling
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_run_autoexp():
    path = Path(__file__).resolve().parent / "run_autoexp.py"
    spec = importlib.util.spec_from_file_location("run_autoexp", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inject(argv: list[str]) -> list[str]:
    all_ranks = False
    forwarded: list[str] = []
    for arg in argv:
        if arg == "--all-ranks":
            all_ranks = True
            continue
        forwarded.append(arg)

    extras: list[str] = []
    if not any(arg.lstrip("+").startswith("profiling=") for arg in forwarded):
        extras.append("profiling=rocprofv3")
    if all_ranks and not any("profiling.ranks=" in arg for arg in forwarded):
        extras.append("profiling.ranks=[]")
    return [*forwarded, *extras]


def main(argv: list[str] | None = None) -> None:
    forwarded = _inject(list(argv if argv is not None else sys.argv[1:]))
    _load_run_autoexp().main(forwarded)


if __name__ == "__main__":
    main()

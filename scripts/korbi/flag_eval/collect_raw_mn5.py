"""Shim: the collector is oellm-eval's ``oellm.reports.collect`` (the JUPITER collect_raw.py, for
oellm-eval run dirs with --runs and worker results roots with --root DIR=CHECKPOINT).

    python3 collect_raw_mn5.py --checkpoints <c> --runs <dir> ... [--chrf-patched] [--datasets <csv>] -o out.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "submodules" / "oellm-eval"))
from oellm.reports.collect import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

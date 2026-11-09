"""Shim: the missing-row check is oellm-eval's ``oellm.reports.check``.

    python3 flag_missing_rows.py --csv <export>.tasks.csv (--suite <yaml> | --groups ... | --plan plan.json) \\
        [-o <export>.missing.csv] [--strict]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "submodules" / "oellm-eval"))
from oellm.reports.check import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())

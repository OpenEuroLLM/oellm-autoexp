"""Puts this repo's oellm-eval checkout (submodules/oellm-eval) on sys.path for the shims below:
the FLAG runners live in oellm-eval (oellm/runner, oellm/reports); these scripts keep the old entry
points of older configs and launchers working."""
import sys
from pathlib import Path

OELLM = Path(__file__).resolve().parents[3] / "submodules" / "oellm-eval"
if str(OELLM) not in sys.path:
    sys.path.insert(0, str(OELLM))

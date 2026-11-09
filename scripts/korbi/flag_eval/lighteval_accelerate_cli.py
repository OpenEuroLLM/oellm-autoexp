"""Shim: oellm-eval's ``oellm.runner.lighteval_cli`` (lighteval's accelerate command without Typer).

    python lighteval_accelerate_cli.py accelerate <model_args> <tasks> [--load-tasks-multilingual] [--output-dir DIR] \\
        [--save-details] [--max-samples N]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "submodules" / "oellm-eval"))
from oellm.runner.lighteval_cli import main  # noqa: E402

if __name__ == "__main__":
    args = sys.argv[1:]
    raise SystemExit(main(args if args[:1] == ["accelerate"] else ["accelerate", *args]))

"""The old FLAG entry points delegate to oellm-eval (scripts/korbi/flag_eval shims)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.korbi.flag_eval import flag_lmeval  # noqa: E402


def test_flag_lmeval_old_arguments_become_the_session_runners():
    got = flag_lmeval.translate(["--export", "/x/exp_1", "--out", "/o", "--names", "vllm/a/0", "--limit", "20",
                                 "--suite", "s.yaml", "--tp", "1"])
    assert got == ["--model", "/x/exp_1", "--root", "/o/exp_1_limit20", "--names", "vllm/a/0", "--limit", "20",
                   "--tp", "1"]
    assert flag_lmeval.translate(["--model", "/m", "--root", "/r"]) == ["--model", "/m", "--root", "/r"]


def test_flag_lmeval_skips_done_rows_through_the_shim(tmp_path):
    root = tmp_path / "out" / "exp_1"
    (root / "done").mkdir(parents=True)
    (root / "done" / "a_0.ok").write_text("{}")
    r = subprocess.run([sys.executable, str(REPO / "scripts/korbi/flag_eval/flag_lmeval.py"), "--export",
                        str(tmp_path / "exp_1"), "--out", str(tmp_path / "out"), "--names", "vllm/a/0"],
                       capture_output=True, text=True, cwd=REPO, timeout=120)
    assert r.returncode == 0 and "1 rows, 0 to run" in r.stdout, r.stdout + r.stderr


def test_lighteval_shims_name_their_backend():
    from oellm.runner.lighteval_cli import parse

    assert parse(["vllm", "m", "t", "--save-details"])[3] == {"save_details": True}
    for b in ("vllm", "accelerate"):
        r = subprocess.run([sys.executable, str(REPO / f"scripts/korbi/flag_eval/lighteval_{b}_cli.py")],
                           capture_output=True, text=True, timeout=60)
        assert "usage" in (r.stdout + r.stderr)

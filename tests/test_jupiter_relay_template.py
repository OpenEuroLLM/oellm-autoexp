"""templates/jupiter_relay.sbatch: the worker-only SIGTERM relay renders to valid bash with the trap, the
background main step and the wait loop, and its pkill pattern hits torchrun's workers but not the agent."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from oellm_autoexp.slurm_gen.template_renderer import render_template

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "templates" / "jupiter_relay.sbatch"
REPL = dict(sbatch_directives="#SBATCH --nodes=4\n#SBATCH --signal=B:TERM@240", env_exports="export X=1",
            job_name="t", srun_opts="--exclusive ", launcher_cmd="apptainer exec img.sif", command="python -m torch.distributed.run pretrain_gpt.py --a 1")


def test_renders_to_valid_bash(tmp_path):
    text = render_template(TEMPLATE.read_text(), REPL)
    f = tmp_path / "job.sbatch"
    f.write_text(text)
    assert subprocess.run(["bash", "-n", str(f)], capture_output=True).returncode == 0
    assert "trap relay_workers USR1 TERM" in text
    assert re.search(r"^srun --exclusive bash -c '.*' &$", text, re.M)
    assert 'wait "$main_pid"' in text and 'exit "$rc"' in text
    assert "srun --overlap" in text


def test_relay_targets_direct_children_of_the_agents():
    text = TEMPLATE.read_text()
    m = re.search(r'pgrep -f "([^"]+torch.distributed.run)"', text)
    assert m, "agent pattern"
    pat = m.group(1)
    assert re.search(pat, "/opt/venv/bin/python  -u -m torch.distributed.run --nnodes 4 pretrain_gpt.py")
    assert re.search(pat, "python3 -m torch.distributed.run pretrain_gpt.py")
    assert re.search(pat, "bash -c export X=1; apptainer exec img.sif python -u -m torch.distributed.run pretrain_gpt.py") is None
    assert re.search(pat, "apptainer exec img.sif python -u -m torch.distributed.run pretrain_gpt.py") is None
    assert re.search(pat, "/opt/venv/bin/python -u pretrain_gpt.py --num-layers 64") is None
    assert 'pgrep -P "$a"' in text
    assert "kill -TERM" in text and "pkill" not in text
    assert "B:USR1" in text

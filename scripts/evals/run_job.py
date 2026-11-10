#!/usr/bin/env python3
"""Run the export job of a typed eval launch (scripts/evals/launch.py) from its JobSpec JSON, inside
the MN5 image (scripts/evals/job.sh plain): export_main_run_v2.sh (EXPORT_ONLY), the identity
chat-template view, the export's current model code, <export>/.downstream_ready. The eval units run
oellm-eval's worker (job.sh worker; submodules/oellm-eval/oellm/runner). ``--print`` shows the
commands and their environment changes without running them.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.evals.spec import JobSpec, SpecError, load  # noqa: E402


@dataclass
class Step:
    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)  # set on top of the job's environment
    cwd: str | None = None
    check: bool = False  # a failure ends the job (else: noted, the next step runs)


def export_steps(job: JobSpec, py: str) -> list[Step]:
    e = job.export
    env = {"EXPORT_ONLY": "1", "EXPERT_DTYPE": e.expert_dtype, "HF_EXPORTER": e.exporter,
           "OELLM_EXPORT_MAX_POSITIONS": str(e.max_positions)}
    root = str(Path(job.export_dir).parent)
    if e.run_dir:  # any training run (export_run.sh: export only, into the export dir itself)
        first = Step(["bash", "scripts/korbi/hf_export/export_run.sh", e.run_dir, str(e.step), job.export_dir], env,
                     check=True)
    else:
        first = Step(["bash", "scripts/korbi/hf_export/export_main_run_v2.sh", e.model, str(e.step), root], env, check=True)
    steps = [first,
             Step([py, "-c", "import sys, pathlib; sys.exit(0 if pathlib.Path(sys.argv[1]).is_dir() else 1)",
                   job.export_dir], check=True)]
    if not Path(job.view_dir).is_dir():
        steps.append(Step([py, "submodules/oellm-eval/containers/prepare_base_model_view.py", "--source",
                           job.export_dir, "--out", job.view_dir, "--template", "identity"], check=True))
    steps.append(Step(["bash", "scripts/korbi/hf_export/refresh_code.sh", job.export_dir, job.view_dir]))
    steps.append(Step(["touch", f"{job.export_dir}/.downstream_ready"], check=True))
    return steps


def run(job: JobSpec) -> int:
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    rc = 0
    for s in export_steps(job, sys.executable):
        print(f"[run_job] {' '.join(s.argv[:4])} ... (cwd {s.cwd or REPO})", flush=True)
        r = subprocess.call(s.argv, env={**os.environ, **s.env}, cwd=s.cwd or REPO)
        if r != 0:
            print(f"[run_job] exit {r}: {s.argv[:4]}", flush=True)
            if s.check:
                return r
            rc = rc or r
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", type=Path, help="the export job's JobSpec JSON")
    ap.add_argument("--print", action="store_true", help="print the commands, run nothing")
    a = ap.parse_args(argv)
    try:
        job = load(JobSpec, json.loads(a.spec.read_text()))
    except SpecError as e:
        print(f"[run_job] bad job spec {a.spec}: {e}", file=sys.stderr)
        return 2
    print(f"[run_job] {job.name}: export of {job.export_name}", flush=True)
    if a.print:
        for s in export_steps(job, sys.executable):
            print(json.dumps({"argv": s.argv, "env": s.env, "cwd": s.cwd}))
        return 0
    return run(job)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run one job of a typed eval launch (scripts/evals/launch.py) from its JobSpec JSON, inside the MN5
image (scripts/evals/job.sh sets up the runtime first):

    export          export_main_run_v2.sh (EXPORT_ONLY), the identity chat-template view, the
                    export's current model code, <export>/.downstream_ready
    lm_eval_worker  flag_lmeval.py --queue: one engine, queue items until the queue is empty
    harness_group   lighteval (one call for all rows) or evalchemy (one call per task), then a done
                    marker per row whose result is there (oellm_rows_done.py)

Every harness command is an argument list built here from typed fields -- the same commands as
mn5_oellm_run_one.sh / run_flag_fast_mn5.sh, without a shell in between. ``--print`` shows the
commands and their environment changes without running them.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
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


def key_of(task: str) -> str:
    return task.replace("/", "_").replace(":", "_").replace("|", "_").replace(" ", "_")


def _flag(x: bool) -> str:
    return "True" if x else "False"


def model_dir(job: JobSpec) -> str:
    """The identity chat-template view where there is one (evalchemy / lighteval need it)."""
    return job.view_dir if Path(job.view_dir).is_dir() else job.export_dir


def results_root(job: JobSpec) -> Path:
    return Path(job.out) / (job.export_name + (f"_limit{job.limit}" if job.limit else ""))


def group_key(job: JobSpec) -> str:
    """The output dir key of a harness call: the row's own, or one per group of rows (by its rows,
    as mn5_oellm_run_one.sh: a rerun of the same remaining rows writes to the same dir)."""
    if len(job.rows) == 1:
        return f"{key_of(job.rows[0].task)}_{job.rows[0].n_shot}"
    tasks = ",".join(r.task for r in job.rows)
    shots = ",".join(str(r.n_shot) for r in job.rows)
    return "group_" + hashlib.sha1(f"{tasks}\n{shots}\n".encode()).hexdigest()[:10]


def pending_rows(job: JobSpec) -> list:
    done = results_root(job) / job.harness / "done"
    return [r for r in job.rows if not (done / f"{key_of(r.task)}_{r.n_shot}.ok").exists()]


def export_steps(job: JobSpec, py: str) -> list[Step]:
    e = job.export
    env = {"EXPORT_ONLY": "1", "EXPERT_DTYPE": e.expert_dtype, "HF_EXPORTER": e.exporter,
           "OELLM_EXPORT_MAX_POSITIONS": str(e.max_positions)}
    root = str(Path(job.export_dir).parent)
    steps = [Step(["bash", "scripts/korbi/hf_export/export_main_run_v2.sh", e.model, str(e.step), root], env, check=True),
             Step([py, "-c", "import sys, pathlib; sys.exit(0 if pathlib.Path(sys.argv[1]).is_dir() else 1)",
                   job.export_dir], check=True)]
    if not Path(job.view_dir).is_dir():
        steps.append(Step([py, "submodules/oellm-eval/containers/prepare_base_model_view.py", "--source",
                           job.export_dir, "--out", job.view_dir, "--template", "identity"], check=True))
    steps.append(Step(["bash", "scripts/korbi/hf_export/refresh_code.sh", job.export_dir, job.view_dir]))
    steps.append(Step(["touch", f"{job.export_dir}/.downstream_ready"], check=True))
    return steps


def lm_eval_steps(job: JobSpec, py: str) -> list[Step]:
    g = job.engine
    argv = [py, "scripts/korbi/flag_eval/flag_lmeval.py", "--export", job.export_dir, "--out", job.out,
            "--shard", str(job.shard), "--num-shards", "1", "--suite", job.suite_file,
            "--tp", str(g.tp), "--dp", str(g.dp), "--ep", "1" if g.expert_parallel else "0",
            "--gpu-mem", str(g.gpu_memory_utilization), "--max-num-seqs", str(g.max_num_seqs),
            "--max-num-batched-tokens", str(g.max_num_batched_tokens),
            "--request-batch-size", str(g.request_batch_size) if g.request_batch_size else "auto"]
    if g.response_cache:
        argv.append("--response-cache")
    if job.limit:
        argv += ["--limit", str(job.limit)]
    argv += ["--queue", job.queue]
    return [Step(argv, job.model_code.env(), check=True)]


def harness_env(job: JobSpec) -> dict[str, str]:
    p = job.paths
    env = dict(job.model_code.env())
    env["STANZA_RESOURCES_DIR"] = p.stanza
    env["OELLM_STAGE"] = p.oellm_stage
    env["OELLM_INCLUDE"] = str(REPO / "submodules/oellm-eval/oellm/resources/custom_lm_eval_tasks")
    front = [p.pylibs_oellm, p.oellm_stage]
    if job.harness == "lighteval" and Path(p.pylibs_lighteval).is_dir():
        # lighteval pins its own metric deps (latex2sympy2_extended, nltk, sacrebleu, ...)
        front.insert(0, p.pylibs_lighteval)
    env["PYTHONPATH"] = ":".join(front + [os.environ.get("PYTHONPATH", "")]).rstrip(":")
    return env


def lighteval_steps(job: JobSpec, py: str, rows: list) -> list[Step]:
    g = job.engine
    model_args = (f"model_name={model_dir(job)},trust_remote_code=True,dtype=bfloat16,"
                  f"gpu_memory_utilization={g.gpu_memory_utilization},max_num_seqs={g.max_num_seqs},"
                  f"max_num_batched_tokens={g.max_num_batched_tokens},")
    if g.max_model_length is not None:
        model_args += f"max_model_length={g.max_model_length},"
    model_args += f"tensor_parallel_size={g.tp}"
    tasks = ",".join(f"{r.task}|{r.n_shot}" for r in rows)
    out = results_root(job) / "lighteval" / f"lighteval_{group_key(job)}"
    argv = [py, str(REPO / "scripts/korbi/flag_eval/lighteval_vllm_cli.py"), "vllm", model_args, tasks,
            "--load-tasks-multilingual", "--output-dir", str(out)]
    if job.limit:
        argv += ["--max-samples", str(job.limit)]
    return [Step(argv, harness_env(job))]


def evalchemy_steps(job: JobSpec, py: str, rows: list) -> list[Step]:
    g = job.engine
    model_args = (f"pretrained={model_dir(job)},trust_remote_code=True,dtype=bfloat16,"
                  f"gpu_memory_utilization={g.gpu_memory_utilization},max_num_seqs={g.max_num_seqs},"
                  f"tensor_parallel_size={g.tp},enforce_eager={_flag(job.model_code.enforce_eager)}")
    if g.dp > 1:
        model_args += f",data_parallel_size={g.dp},data_parallel_backend=mp"
    if g.expert_parallel:
        model_args += ",enable_expert_parallel=True"
    if g.max_model_length is not None:
        model_args += f",max_model_len={g.max_model_length}"
    steps = []
    # one call per task: evalchemy writes all results at the end, and one task's crash lost a whole
    # 8-task group (11-06); the load is minutes since the parallel prefetch / GPU dequant
    for r in rows:
        out = results_root(job) / "evalchemy" / f"evalchemy_{key_of(r.task)}_{r.n_shot}"
        argv = [py, "-m", "eval.eval", "--model", "vllm", "--tasks", r.task, "--model_args", model_args,
                "--batch_size", "auto", "--output_path", str(out)]
        if job.limit:
            argv += ["--limit", str(job.limit)]
        steps.append(Step(argv, harness_env(job), cwd=f"{job.paths.oellm_stage}/evalchemy"))
    return steps


def done_step(job: JobSpec, py: str, rows: list) -> Step:
    root = results_root(job)
    results = root / job.harness if job.harness == "evalchemy" else root / job.harness / f"{job.harness}_{group_key(job)}"
    return Step([py, "scripts/korbi/flag_eval/oellm_rows_done.py", "--suite", job.harness, "--results", str(results),
                 "--lm-done", str(root / "done"), "--out-done", str(root / job.harness / "done"),
                 "--limit", str(job.limit), "--rows", "\n".join(f"{r.task}|{r.n_shot}" for r in rows)],
                check=True)


def steps_of(job: JobSpec, py: str) -> list[Step]:
    if job.kind == "export":
        return export_steps(job, py)
    if job.kind == "lm_eval_worker":
        return lm_eval_steps(job, py)
    rows = pending_rows(job)
    if not rows:
        return []
    run = lighteval_steps if job.harness == "lighteval" else evalchemy_steps
    return run(job, py, rows) + [done_step(job, py, rows)]


def _uva_cleanup(job: JobSpec) -> None:
    if job.model_code is not None and job.model_code.engram_tables == "uva":
        for f in glob.glob(f"{os.environ.get('OELLM_ENGRAM_UVA_DIR', '/dev/shm')}/oellm_engram_*"):
            try:
                os.unlink(f)
            except OSError:
                pass


def _drop_old_typer() -> None:
    # the vLLM wheel pins an older typer; lighteval's CLI metadata needs the image's
    dest = os.environ.get("DEST")
    if not dest:
        return
    for d in [f"{dest}/typer", *glob.glob(f"{dest}/typer-*.dist-info"), *glob.glob(f"{dest}/typer_slim-*.dist-info")]:
        shutil.rmtree(d, ignore_errors=True)


def run(job: JobSpec) -> int:
    py = sys.executable
    steps = steps_of(job, py)
    if not steps:
        print(f"[run_job] {job.name}: every row is done")
        return 0
    if job.kind == "harness_group" and job.harness == "lighteval":
        _drop_old_typer()
    # Engram tables staged in /dev/shm (uva) outlive the processes: remove a killed job's now and
    # this job's at the end (pool steps run inside a long host job: no epilog cleans them)
    _uva_cleanup(job)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    rc = 0
    try:
        for s in steps:
            env = {**os.environ, **s.env}
            print(f"[run_job] {' '.join(s.argv[:4])} ... (cwd {s.cwd or REPO})", flush=True)
            r = subprocess.call(s.argv, env=env, cwd=s.cwd or REPO)
            if r != 0:
                print(f"[run_job] exit {r}: {s.argv[:4]}", flush=True)
                if s.check:
                    return r
                rc = rc or r
    finally:
        _uva_cleanup(job)
    if rc and job.kind == "harness_group":
        print(f"[run_job] a harness call exited {rc}, but every row has its result")
        rc = 0
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", type=Path, help="the job's JobSpec JSON")
    ap.add_argument("--print", action="store_true", help="print the commands, run nothing")
    a = ap.parse_args(argv)
    try:
        job = load(JobSpec, json.loads(a.spec.read_text()))
    except SpecError as e:
        print(f"[run_job] bad job spec {a.spec}: {e}", file=sys.stderr)
        return 2
    print(f"[run_job] {job.name}: {job.kind} {job.harness or ''} on {job.export_name}, {len(job.rows)} row(s)",
          flush=True)
    if a.print:
        for s in steps_of(job, sys.executable):
            print(json.dumps({"argv": s.argv, "env": s.env, "cwd": s.cwd}))
        return 0
    return run(job)


if __name__ == "__main__":
    raise SystemExit(main())

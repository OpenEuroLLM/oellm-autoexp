#!/usr/bin/env python3
"""Launch the FLAG evals of one checkpoint from a typed spec (scripts/evals/spec.py): the export
(unless it is ready), lm-eval through queue workers, lighteval / evalchemy in balanced one-load
groups -- all as pool jobs, the eval jobs gated on <export>/.downstream_ready. Run on the MN5
login node from a synced snapshot, with the autoexp environment:

    $R/autoexp_env/bin/python scripts/evals/launch.py config/evals/flag_full_mn5.yaml --step 13000
    ... config/evals/flag_short_mn5.yaml --step 13000 --set lm_eval.workers=12 --dry-run

Each job's settings go into a JSON spec (<out>/<export>/launches/<id>/jobs/<job>.json, a JobSpec);
the job runs ``bash scripts/evals/job.sh <runtime> <that file>`` -- no task name, list or setting
passes through a shell or Hydra interpolation. The pool sweep for run_autoexp is rendered next to it
(launch.yaml, composed against this repo's config/ via hydra.searchpath).

Rows with a done marker are left out; lm-eval rows already queued, running or failed are not
queued again, so a relaunch adds only what is missing. --dry-run prints the plan and writes nothing.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.evals.spec import (  # noqa: E402
    EvalSpec, GroupedSpec, JobSpec, Row, SpecError, eval_spec, to_dict,
)
from scripts.korbi.flag_eval import flag_queue  # noqa: E402

ROW_TIMES = REPO / "config/eval_tasks/flag_evals_row_times.json"     # job seconds incl. a load
ENGINE_TIMES = REPO / "config/eval_tasks/flag_evals_engine_times.json"  # engine seconds, lm-eval
JOB_CONFIG = "experiments/korbi/production/evals/eval_job_mn5"
LOAD_S = 600.0  # the model load inside a reference row time
EXPORT_EST_S = 900.0
SAFE = re.compile(r"^[A-Za-z0-9_./@=-]+$")


def key_of(task: str) -> str:
    """The harness runners' file key of a task name (mn5_oellm_run_one.sh key_of)."""
    return re.sub(r"[/:| ]", "_", task)


# ---------------------------------------------------------------- rows


def select_rows(spec: EvalSpec) -> list[Row]:
    suite = yaml.safe_load((REPO / spec.rows.suite_file).read_text())["tasks"]
    by_name = {r["name"]: r for r in suite}
    if spec.rows.names is not None:
        missing = [n for n in spec.rows.names if n not in by_name]
        if missing:
            raise SpecError(f"rows.names not in {spec.rows.suite_file}: {missing}")
        chosen = [by_name[n] for n in dict.fromkeys(spec.rows.names)]
    else:
        chosen = suite
    exclude = set(spec.rows.exclude)
    harness_excl = {"lighteval": set(spec.lighteval.exclude_tasks), "evalchemy": set(spec.evalchemy.exclude_tasks)}
    out = []
    for r in chosen:
        if r["suite"] not in spec.rows.harnesses or r["name"] in exclude:
            continue
        if r["task"] in harness_excl.get(r["suite"], ()):
            continue
        out.append(Row(name=r["name"], task=str(r["task"]), n_shot=int(r["n_shot"]), suite=r["suite"]))
    return out


def results_root(spec: EvalSpec) -> Path:
    """<out>/<export>[_limit<l>]: where the runners put results and done markers."""
    name = spec.export.export_name
    lim = f"_limit{spec.rows.limit}" if spec.rows.limit else ""
    return Path(spec.paths.out) / f"{name}{lim}"


def row_done(spec: EvalSpec, row: Row) -> bool:
    root = results_root(spec)
    if row.suite == "lm-eval-harness":
        return (root / "done" / f"{row.task}_{row.n_shot}.ok").exists()
    return (root / row.suite / "done" / f"{key_of(row.task)}_{row.n_shot}.ok").exists()


# ---------------------------------------------------------------- planning


def _times(path: Path) -> dict[str, float]:
    return json.loads(path.read_text()) if path.exists() else {}


def expected_seconds(rows: list[Row], times: dict[str, float]) -> list[float]:
    """Per row: its reference job time minus the load, else its harness's median."""
    key = lambda r: f"{r.suite}|{r.task}|{r.n_shot}"  # noqa: E731
    known: dict[str, list[float]] = {}
    for k, t in times.items():
        known.setdefault(k.split("|", 1)[0], []).append(max(60.0, t - LOAD_S))
    med = {s: sorted(v)[len(v) // 2] for s, v in known.items()}
    return [max(60.0, times[key(r)] - LOAD_S) if key(r) in times else med.get(r.suite, 600.0) for r in rows]


def balance(secs: list[float], groups: int) -> list[list[int]]:
    """Longest first onto the least loaded group; each group's members in input order."""
    import heapq

    heap = [(0.0, g) for g in range(groups)]
    out: list[list[int]] = [[] for _ in range(groups)]
    for s, i in sorted(((s, i) for i, s in enumerate(secs)), reverse=True):
        load, g = heapq.heappop(heap)
        out[g].append(i)
        heapq.heappush(heap, (load + s, g))
    return [sorted(g) for g in out if g]


def group_jobs(spec: EvalSpec, harness: str, rows: list[Row], times: dict[str, float]) -> list[tuple[list[Row], float]]:
    cfg: GroupedSpec = getattr(spec, harness)
    if not rows:
        return []
    secs = expected_seconds(rows, times)
    n = max(cfg.groups, math.ceil(len(rows) / cfg.max_rows) if cfg.max_rows else 1)
    n = min(n, len(rows))
    out = []
    for members in balance(secs, n):
        loads = len(members) if harness == "evalchemy" else 1  # evalchemy: one call per task
        out.append(([rows[i] for i in members], loads * LOAD_S + sum(secs[i] for i in members)))
    return out


def wall_hours(limit_h: int | None, est_s: float) -> int:
    return limit_h if limit_h is not None else min(48, max(2, math.ceil(3 * est_s / 3600)))


def plan(spec: EvalSpec) -> tuple[list[tuple[JobSpec, dict]], list[tuple[str, float]], list[Row]]:
    """The launch's jobs (each with its pool settings), the lm-eval queue items, and the rows
    left out as done."""
    p = spec.paths.resolved()
    export_name = spec.export.export_name
    export_dir = f"{p.exports}/{export_name}"
    view_dir = f"{p.views}/{export_name}"
    ready = Path(export_dir, ".downstream_ready").exists()
    rows = select_rows(spec)
    done = [r for r in rows if row_done(spec, r)]
    todo = [r for r in rows if not row_done(spec, r)]
    common = dict(export_name=export_name, export_dir=export_dir, view_dir=view_dir, out=p.out,
                  paths=p, suite_file=str(REPO / spec.rows.suite_file), limit=spec.rows.limit)
    tag = f"_{spec.tag}" if spec.tag else ""
    jobs: list[tuple[JobSpec, dict]] = []
    gate = None if ready else f"{export_dir}/.downstream_ready"
    if not ready:
        if spec.export.step is None:
            raise SpecError(f"{export_dir} is not ready (no .downstream_ready) and export.step is not set")
        jobs.append((JobSpec(kind="export", name=f"ev_{export_name}_export{tag}", export=spec.export,
                             est_s=EXPORT_EST_S, **common),
                     dict(priority=50, time_h=1, use_excluded_nodes=True, gate=None, runtime="plain")))

    items: list[tuple[str, float]] = []
    lm_rows = [r for r in todo if r.suite == "lm-eval-harness"]
    if spec.lm_eval.enabled and lm_rows:
        # a test run (rows.limit) evaluates the first documents of each row: no parts
        split_s = 0 if spec.rows.limit else spec.lm_eval.split_s
        items = flag_queue.plan([r.name for r in lm_rows], _times(ENGINE_TIMES), split_s, spec.lm_eval.max_parts)
        queue = str(results_root(spec) / "queue_lm")
        total = sum(s for _, s in items)
        workers = min(spec.lm_eval.workers, len(items))
        lim = spec.lm_eval.job
        for w in range(workers):
            jobs.append((JobSpec(kind="lm_eval_worker", name=f"ev_{export_name}_lm_w{w}{tag}", harness="lm-eval-harness",
                                 engine=spec.lm_eval.engine, model_code=spec.lm_eval.model_code, queue=queue,
                                 shard=w, est_s=total / workers, **common),
                         dict(priority=lim.priority, time_h=lim.time_h if lim.time_h is not None else 12,
                              use_excluded_nodes=lim.use_excluded_nodes, gate=gate, runtime="vllm")))

    row_times = _times(ROW_TIMES)
    for harness, short in (("lighteval", "le"), ("evalchemy", "ec")):
        cfg: GroupedSpec = getattr(spec, harness)
        h_rows = [r for r in todo if r.suite == harness]
        if not cfg.enabled:
            continue
        for g, (members, est) in enumerate(group_jobs(spec, harness, h_rows, row_times), start=1):
            jobs.append((JobSpec(kind="harness_group", name=f"ev_{export_name}_{short}_g{g}{tag}", harness=harness,
                                 rows=members, engine=cfg.engine, model_code=cfg.model_code, est_s=est, **common),
                         dict(priority=cfg.job.priority, time_h=wall_hours(cfg.job.time_h, est),
                              use_excluded_nodes=cfg.job.use_excluded_nodes, gate=gate, runtime="vllm")))
    for job, _ in jobs:
        if not SAFE.match(job.name):
            raise SpecError(f"job name {job.name!r}: unsafe characters")
    return jobs, items, done


# ---------------------------------------------------------------- rendering


class _QuotingDumper(yaml.SafeDumper):
    """Quotes every string with a colon: "01:00:00" unquoted is a YAML 1.1 sexagesimal number."""


_QuotingDumper.add_representer(
    str, lambda d, v: d.represent_scalar("tag:yaml.org,2002:str", v, style='"' if ":" in v else None))


def render_launch(spec: EvalSpec, jobs: list[tuple[JobSpec, dict]], launch_dir: Path) -> str:
    """The primary run_autoexp config of the launch: one list-sweep point per job, all values
    plain literals (the job's settings are in its JSON spec)."""
    p = spec.paths.resolved()
    points = []
    for job, pool in jobs:
        spec_path = str(launch_dir / "jobs" / f"{job.name}.json")
        for v in (spec_path, p.logs):
            if not SAFE.match(v):
                raise SpecError(f"path {v!r}: unsafe characters for the job command")
        pt = {
            "aux.job": job.name,
            "aux.runtime": pool["runtime"],
            "aux.spec": spec_path,
            "job.est_step_time_s": int(max(60, job.est_s)),
            "slurm.sbatch.time": f"{pool['time_h']:02d}:00:00",
            "job.metadata.pool_priority": pool["priority"],
            "job.metadata.pool_use_excluded_nodes": pool["use_excluded_nodes"],
        }
        if job.kind == "export":
            pt["backend.env.PYTHONPATH"] = "${aux.export_pythonpath}"
        if pool["gate"]:
            pt["job.start_condition"] = {"class_name": "FileExistsCondition", "path": pool["gate"],
                                         "persistent_pass": True}
        points.append(pt)
    cfg = {
        "hydra": {"searchpath": [f"file://{REPO / 'config'}"]},
        "defaults": [f"/{JOB_CONFIG}", "_self_"],
        "aux": {"log_root": p.logs},
        "slurm": {"sbatch": {"account": spec.slurm.account, "qos": spec.slurm.qos}},
        "sweep": {"type": "list", "store_sweep_json": True,
                  "groups": [{"type": "list", "configs": points}]},
    }
    head = ("# @package _global_\n# GENERATED by scripts/evals/launch.py -- one pool job per sweep point; "
            "each job's settings are in its jobs/<name>.json\n")
    return head + yaml.dump(cfg, Dumper=_QuotingDumper, sort_keys=False, width=200)


def launch_id(spec: EvalSpec) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}{'_' + spec.tag if spec.tag else ''}"


def write_launch(spec: EvalSpec, jobs, launch_dir: Path) -> Path:
    (launch_dir / "jobs").mkdir(parents=True, exist_ok=True)
    for job, _ in jobs:
        (launch_dir / "jobs" / f"{job.name}.json").write_text(json.dumps(to_dict(job), indent=1) + "\n")
    (launch_dir / "spec.json").write_text(json.dumps(to_dict(spec), indent=1) + "\n")
    path = launch_dir / "launch.yaml"
    path.write_text(render_launch(spec, jobs, launch_dir))
    return path


def print_plan(spec: EvalSpec, jobs, items, done) -> None:
    print(f"[launch] export {spec.export.export_name}: {len(done)} row(s) done already, {len(jobs)} job(s)")
    if items:
        print(f"[launch] lm-eval queue: {len(items)} item(s), {sum(s for _, s in items) / 3600:.1f} engine-h")
    for job, pool in jobs:
        what = (f"{len(job.rows)} rows" if job.rows else job.kind)
        gate = "gated" if pool["gate"] else "now"
        print(f"  {job.name:58s} {what:16s} est {job.est_s / 3600:5.1f} h  limit {pool['time_h']:2d} h  "
              f"prio {pool['priority']}  {gate}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", type=Path, help="an EvalSpec YAML file (config/evals/*.yaml)")
    ap.add_argument("--step", type=int, help="export.step")
    ap.add_argument("--model", help="export.model")
    ap.add_argument("--export", help="export.name: an existing export directory name")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a spec field, the value parsed as YAML (repeatable)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; write and submit nothing")
    ap.add_argument("--render-only", type=Path, metavar="DIR",
                    help="write the launch files below DIR and stop (no queue, no submission)")
    ap.add_argument("--autoexp", default="scripts/korbi/mn5_autoexp.sh",
                    help="the run_autoexp wrapper of this cluster")
    a = ap.parse_args(argv)
    sets = list(a.set)
    for flag, key in ((a.step, "export.step"), (a.model, "export.model"), (a.export, "export.name")):
        if flag is not None:
            sets.append(f"{key}={flag}")
    try:
        spec = eval_spec(yaml.safe_load(a.spec.read_text()) or {}, sets)
        lid = launch_id(spec)
        root = a.render_only if a.render_only else results_root(spec) / "launches"
        launch_dir = root / lid
        jobs, items, done = plan(spec)
    except SpecError as e:
        print(f"[launch] spec error: {e}", file=sys.stderr)
        return 2
    print_plan(spec, jobs, items, done)
    if not jobs:
        print("[launch] nothing to do")
        return 0
    if a.dry_run:
        return 0
    cfg = write_launch(spec, jobs, launch_dir)
    print(f"[launch] {cfg}")
    if a.render_only:
        return 0
    if items:
        q = flag_queue.Queue(results_root(spec) / "queue_lm")
        added = q.init(items, results_root(spec) / "done")
        print(f"[launch] queued {added} of {len(items)} lm-eval item(s); {q.status()}")
    p = spec.paths.resolved()
    cmd = ["bash", a.autoexp, "scripts/run_autoexp.py", "--no-monitor",
           "--monitor-state-dir", f"{p.hybrid_env}/pool_state", "--pool", p.pool,
           "-C", str(launch_dir), "--config-name", "launch", "--yes"]
    print("[launch] " + " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=REPO)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Launch the FLAG evals of one checkpoint from a typed spec (scripts/evals/spec.py) as pool jobs:
the export (unless it is ready), then one job per unit of an oellm-eval plan (submodules/oellm-eval,
oellm/runner: lm-eval sessions over a shared row queue, lighteval and evalchemy groups), the eval
jobs gated on <export>/.downstream_ready. Run on a login node from a synced snapshot, with the
autoexp environment:

    $R/autoexp_env/bin/python scripts/evals/launch.py config/evals/flag_full_mn5.yaml --step 13000
    ... config/evals/flag_short_mn5.yaml --step 13000 --set lm_eval.workers=12 --dry-run

A launch directory (<out>/<export>/launches/<id>/) holds plan.json (oellm-eval's plan: rows, units,
every setting), runtime.json (how the harnesses run on this cluster's nodes), jobs/export.json (the
export job's JobSpec), launch.yaml (the pool sweep for run_autoexp, composed against this repo's
config/ via hydra.searchpath; one point per job, plain literals) and manifest.json (what the launch
planned: plan / runtime hashes, the code snapshot's SYNC_STAMP, the stage files' hashes). A unit's
job runs ``bash scripts/evals/job.sh worker <plan.json> <unit id>`` -- no task name, list or setting
passes through a shell or Hydra interpolation; every unit run adds a provenance record.

Rows with a done marker are left out; lm-eval items already queued, running or failed are not
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
OELLM = REPO / "submodules" / "oellm-eval"
for _p in (REPO, OELLM):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from oellm.reports.manifest import write_manifest  # noqa: E402
from oellm.runner import plan as oplan  # noqa: E402
from oellm.runner.layout import Layout  # noqa: E402
from oellm.runner.layout import Row as ORow  # noqa: E402
from oellm.runner.layout import results_root as oellm_results_root  # noqa: E402
from oellm.runner.runtime import Runtime  # noqa: E402
from scripts.evals.spec import (  # noqa: E402
    GPUS_PER_NODE, EvalSpec, GroupedSpec, JobSpec, SpecError, eval_spec, to_dict,
)

ROW_TIMES = REPO / "config/eval_tasks/flag_evals_row_times.json"     # job seconds incl. a load
ENGINE_TIMES = REPO / "config/eval_tasks/flag_evals_engine_times.json"  # engine seconds, lm-eval
EXPORT_EST_S = 900.0
SAFE = re.compile(r"^[A-Za-z0-9_./@=-]+$")
# files of the oellm-eval stage that decide how rows are run and scored (patched or bug-prone)
STAGE_FILES = (
    "evalchemy/eval/task.py",
    "evalchemy/eval/chat_benchmarks/HumanEval/human_eval/execution.py",
    "evalchemy/eval/chat_benchmarks/HumanEval/eval_instruct.py",
    "evalchemy/eval/chat_benchmarks/LiveCodeBench/eval_instruct.py",
    "lighteval/metrics/metrics_corpus.py",
    "lighteval/models/vllm/vllm_model.py",
    "lighteval/main_vllm.py",
)


# ---------------------------------------------------------------- rows


def select_rows(spec: EvalSpec) -> list[ORow]:
    if spec.rows.suite_file:
        suite = yaml.safe_load((REPO / spec.rows.suite_file).read_text())["tasks"]
        rows = [ORow(name=t["name"], task=str(t["task"]), n_shot=int(t["n_shot"]), harness=t["suite"]) for t in suite]
    else:
        rows = oplan.rows_from_task_groups(spec.rows.groups)
    by_name = {r.name: r for r in rows}
    if spec.rows.names is not None:
        missing = [n for n in spec.rows.names if n not in by_name]
        if missing:
            raise SpecError(f"rows.names not in the rows of {spec.rows.suite_file or spec.rows.groups}: {missing}")
        rows = [by_name[n] for n in dict.fromkeys(spec.rows.names)]
    exclude = set(spec.rows.exclude)
    harness_excl = {"lighteval": set(spec.lighteval.exclude_tasks), "evalchemy": set(spec.evalchemy.exclude_tasks)}
    return [r for r in rows if r.harness in spec.rows.harnesses and r.name not in exclude
            and r.task not in harness_excl.get(r.harness, ())]


def results_root(spec: EvalSpec) -> Path:
    """<out>/<export>[_limit<l>]: where the runners put results and done markers (oellm layout)."""
    return oellm_results_root(spec.paths.out, spec.export.export_name, spec.rows.limit)


# ---------------------------------------------------------------- planning


def _times(path: Path) -> dict[str, float]:
    return json.loads(path.read_text()) if path.exists() else {}


def _engine(e) -> oplan.Engine:
    return oplan.Engine(tp=e.tp, dp=e.dp, expert_parallel=e.expert_parallel, max_num_seqs=e.max_num_seqs,
                        max_num_batched_tokens=e.max_num_batched_tokens, max_model_length=e.max_model_length,
                        gpu_memory_utilization=e.gpu_memory_utilization, request_batch_size=e.request_batch_size,
                        response_cache=e.response_cache, replicas=e.replicas)


def _group(cfg: GroupedSpec) -> oplan.GroupSettings | None:
    if not cfg.enabled:
        return None
    return oplan.GroupSettings(groups=cfg.groups, max_rows=cfg.max_rows, exclude_tasks=list(cfg.exclude_tasks),
                               engine=_engine(cfg.engine), env=cfg.model_code.env(), backend=cfg.backend)


def wall_hours(limit_h: int | None, est_s: float) -> int:
    return limit_h if limit_h is not None else min(48, max(2, math.ceil(3 * est_s / 3600)))


def _git_rev(path: Path) -> str:
    try:
        rev = subprocess.run(["git", "-C", str(path), "rev-parse", "--short=7", "HEAD"], capture_output=True,
                             text=True, timeout=30).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=30).stdout.strip()
        if rev:
            return rev + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        pass
    stamp = REPO / "SYNC_STAMP"  # a synced snapshot has no .git: its stamp names the commit
    if stamp.is_file():
        for line in stamp.read_text().splitlines():
            parts = line.split()
            if len(parts) > 2 and parts[0] == "submodules/oellm-eval":
                return parts[2] + ("" if parts[-1] == "dirty=da39a3ee5e6b" else "-dirty")
    return "unknown"


def build(spec: EvalSpec) -> tuple[JobSpec | None, oplan.Plan, dict[str, dict]]:
    """The export job (None when the export is ready), the oellm-eval plan, and the pool settings
    per unit id."""
    p = spec.paths.resolved()
    name = spec.export.export_name
    export_dir, view_dir = f"{p.exports}/{name}", f"{p.views}/{name}"
    ready = Path(export_dir, ".downstream_ready").exists()
    sfx = f"_{spec.tag}" if spec.tag else ""
    export_job = None
    if not ready:
        if spec.export.step is None:
            raise SpecError(f"{export_dir} is not ready (no .downstream_ready) and export.step is not set")
        export_job = JobSpec(kind="export", name=f"ev_{name}_export{sfx}", export_name=name, export_dir=export_dir,
                             view_dir=view_dir, paths=p, export=spec.export, est_s=EXPORT_EST_S)
    lm = spec.lm_eval
    session = (oplan.SessionSettings(workers=lm.workers, split_s=lm.split_s, max_parts=lm.max_parts,
                                     engine=_engine(lm.engine), env=lm.model_code.env()) if lm.enabled else None)
    try:
        # the identity view: made by the export job, else there or not (then the plain model)
        view = view_dir if (not ready or Path(view_dir).is_dir()) else export_dir
        plan = oplan.build_plan(model_name=name, model_dir=export_dir, view_dir=view, out=p.out,
                                rows=select_rows(spec), lm_eval=session, lighteval=_group(spec.lighteval),
                                evalchemy=_group(spec.evalchemy), limit=spec.rows.limit,
                                task_groups=[] if spec.rows.suite_file else list(spec.rows.groups.values()),
                                oellm_rev=_git_rev(OELLM), engine_times=_times(ENGINE_TIMES),
                                row_times=_times(ROW_TIMES), tag=spec.tag)
    except ValueError as e:
        raise SpecError(str(e)) from None
    gate = None if ready else f"{export_dir}/.downstream_ready"
    pools = {}
    for u in plan.units:
        if u.kind == "lm_eval_session":
            lim, time_h = lm.job, lm.job.time_h if lm.job.time_h is not None else 12
        else:
            lim = getattr(spec, u.kind).job
            time_h = wall_hours(lim.time_h, u.est_s)
        pools[u.id] = dict(priority=lim.priority, time_h=time_h, use_excluded_nodes=lim.use_excluded_nodes,
                           gate=gate, job=f"ev_{name}_{u.id}")
    for job in [export_job.name] if export_job else []:
        if not SAFE.match(job):
            raise SpecError(f"job name {job!r}: unsafe characters")
    for pool in pools.values():
        if not SAFE.match(pool["job"]):
            raise SpecError(f"job name {pool['job']!r}: unsafe characters")
    return export_job, plan, pools


def runtime(spec: EvalSpec, plan: oplan.Plan) -> Runtime:
    """The cluster's runtime.json: MN5's node-local vLLM install (job.sh sets $INFER_PY) or images."""
    p = spec.paths.resolved()
    rs = spec.runtime
    if rs.kind == "apptainer":
        return Runtime(kind="apptainer", python="python", images=dict(rs.images), apptainer_args=list(rs.apptainer_args),
                       binds=list(rs.binds), env=dict(rs.env), dirs={"evalchemy": rs.evalchemy_dir}, gpus=GPUS_PER_NODE,
                       hang_minutes=rs.hang_minutes, name="apptainer")
    stage = p.oellm_stage
    harness_env = {"STANZA_RESOURCES_DIR": p.stanza, "OELLM_STAGE": stage,
                   "OELLM_INCLUDE": str(OELLM / "oellm/resources/custom_lm_eval_tasks")}
    front = [p.pylibs_oellm, stage]
    lighteval_front = ([p.pylibs_lighteval] if Path(p.pylibs_lighteval).is_dir() else []) + front
    uva = any(u.env.get("OELLM_ENGRAM_TABLES") == "uva" for u in plan.units)
    return Runtime(
        kind="local", python="${INFER_PY}", env=dict(rs.env),
        # the stage and the harness libraries for lighteval / evalchemy only (the lm-eval sessions
        # run on the vLLM install's own lm-eval); lighteval's metric pins first
        harness_env={"lighteval": harness_env, "evalchemy": harness_env},
        harness_pythonpath={"lighteval": lighteval_front, "evalchemy": front},
        dirs={"evalchemy": f"{stage}/evalchemy"}, gpus=GPUS_PER_NODE,
        # Engram tables staged in /dev/shm (uva) outlive the processes; pool steps run inside a
        # long host job, so no epilog removes them
        cleanup_globs=["/dev/shm/oellm_engram_*"] if uva else [],
        # the vLLM wheel pins an older typer; lighteval's CLI metadata needs the image's
        prune={"lighteval": ["$DEST/typer", "$DEST/typer-*.dist-info", "$DEST/typer_slim-*.dist-info"]},
        hash_files=[f"{stage}/{f}" for f in STAGE_FILES], hang_minutes=rs.hang_minutes, name="mn5")


# ---------------------------------------------------------------- rendering


class _QuotingDumper(yaml.SafeDumper):
    """Quotes every string with a colon: "01:00:00" unquoted is a YAML 1.1 sexagesimal number."""


_QuotingDumper.add_representer(
    str, lambda d, v: d.represent_scalar("tag:yaml.org,2002:str", v, style='"' if ":" in v else None))


def render_launch(spec: EvalSpec, export_job: JobSpec | None, plan: oplan.Plan, pools: dict[str, dict],
                  launch_dir: Path) -> str:
    """The primary run_autoexp config of the launch: one list-sweep point per job, all values
    plain literals (the settings are in plan.json / jobs/export.json)."""
    p = spec.paths.resolved()
    plan_path = str(launch_dir / "plan.json")
    points = []
    if export_job is not None:
        points.append({"aux.job": export_job.name, "aux.runtime": "plain",
                       "aux.spec": str(launch_dir / "jobs" / "export.json"), "aux.unit": "-",
                       "job.est_step_time_s": int(export_job.est_s), "slurm.sbatch.time": "01:00:00",
                       "job.metadata.pool_priority": 50, "job.metadata.pool_use_excluded_nodes": True,
                       "backend.env.PYTHONPATH": "${aux.export_pythonpath}"})
    for u in plan.units:
        pool = pools[u.id]
        pt = {"aux.job": pool["job"], "aux.runtime": "worker", "aux.spec": plan_path, "aux.unit": u.id,
              "job.est_step_time_s": int(max(60, u.est_s)), "slurm.sbatch.time": f"{pool['time_h']:02d}:00:00",
              "job.metadata.pool_priority": pool["priority"],
              "job.metadata.pool_use_excluded_nodes": pool["use_excluded_nodes"]}
        if pool["gate"]:
            pt["job.start_condition"] = {"class_name": "FileExistsCondition", "path": pool["gate"],
                                         "persistent_pass": True}
        points.append(pt)
    for pt in points:
        for v in (pt["aux.spec"], pt["aux.unit"], p.logs):
            if v != "-" and not SAFE.match(v):
                raise SpecError(f"{v!r}: unsafe characters for the job command")
    cfg = {
        "hydra": {"searchpath": [f"file://{REPO / 'config'}"]},
        "defaults": [f"/{spec.slurm.job_config}", "_self_"],
        "aux": {"log_root": p.logs},
        "slurm": {"sbatch": {"account": spec.slurm.account, "qos": spec.slurm.qos}},
        "sweep": {"type": "list", "store_sweep_json": True, "groups": [{"type": "list", "configs": points}]},
    }
    head = ("# @package _global_\n# GENERATED by scripts/evals/launch.py -- one pool job per sweep point: the "
            "export (jobs/export.json) and one per unit of plan.json\n")
    return head + yaml.dump(cfg, Dumper=_QuotingDumper, sort_keys=False, width=200)


def launch_id(spec: EvalSpec) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}{'_' + spec.tag if spec.tag else ''}"


def write_launch(spec: EvalSpec, export_job: JobSpec | None, plan: oplan.Plan, pools: dict[str, dict],
                 launch_dir: Path) -> Path:
    launch_dir.mkdir(parents=True, exist_ok=True)
    if export_job is not None:
        (launch_dir / "jobs").mkdir(exist_ok=True)
        (launch_dir / "jobs" / "export.json").write_text(json.dumps(to_dict(export_job), indent=1) + "\n")
    plan.write(launch_dir / "plan.json")
    runtime(spec, plan).write(launch_dir / "runtime.json")
    (launch_dir / "spec.json").write_text(json.dumps(to_dict(spec), indent=1) + "\n")
    path = launch_dir / "launch.yaml"
    path.write_text(render_launch(spec, export_job, plan, pools, launch_dir))
    stamp = REPO / "SYNC_STAMP"
    caller = {"autoexp": str(REPO), "sync_stamp": stamp.read_text().splitlines() if stamp.is_file() else [],
              "spec": str(launch_dir / "spec.json")}
    write_manifest(launch_dir, launch_dir / "plan.json", launch_dir / "runtime.json", caller=caller)
    return path


def print_plan(spec: EvalSpec, export_job: JobSpec | None, plan: oplan.Plan, pools: dict[str, dict]) -> None:
    layout = Layout(plan.root)
    done = sum(layout.is_done(r) for r in plan.rows)
    print(f"[launch] export {spec.export.export_name}: {done} row(s) done already, "
          f"{len(plan.units) + (export_job is not None)} job(s)")
    if plan.lm_items:
        print(f"[launch] lm-eval queue: {len(plan.lm_items)} item(s), "
              f"{sum(s for _, s in plan.lm_items) / 3600:.1f} engine-h")
    if export_job is not None:
        print(f"  {export_job.name:58s} {'export':16s} est {export_job.est_s / 3600:5.1f} h  limit  1 h  prio 50  now")
    for u in plan.units:
        pool = pools[u.id]
        what = f"{len(u.rows)} rows" if u.rows else u.kind
        print(f"  {pool['job']:58s} {what:16s} est {u.est_s / 3600:5.1f} h  limit {pool['time_h']:2d} h  "
              f"prio {pool['priority']}  {'gated' if pool['gate'] else 'now'}")


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
        export_job, plan, pools = build(spec)
    except SpecError as e:
        print(f"[launch] spec error: {e}", file=sys.stderr)
        return 2
    print_plan(spec, export_job, plan, pools)
    if not plan.units and export_job is None:
        print("[launch] nothing to do")
        return 0
    if a.dry_run:
        return 0
    root = a.render_only if a.render_only else results_root(spec) / "launches"
    launch_dir = root / launch_id(spec)
    cfg = write_launch(spec, export_job, plan, pools, launch_dir)
    print(f"[launch] {cfg}\n[launch] {launch_dir / 'manifest.json'}")
    if a.render_only:
        return 0
    if plan.lm_items:
        print(f"[launch] lm-eval queue: {oplan.init_queue(plan)}")
    p = spec.paths.resolved()
    cmd = ["bash", a.autoexp, "scripts/run_autoexp.py", "--no-monitor",
           "--monitor-state-dir", f"{p.hybrid_env}/pool_state", "--pool", p.pool,
           "-C", str(launch_dir), "--config-name", "launch", "--yes"]
    print("[launch] " + " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=REPO)


if __name__ == "__main__":
    raise SystemExit(main())

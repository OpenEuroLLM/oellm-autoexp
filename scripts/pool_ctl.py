#!/usr/bin/env python3
"""Inspect and steer a sub-SLURM-job pool from the login node.

Jobs go INTO a pool with ``run_autoexp.py --pool <pool_dir> ...``; this is
everything else. Commands that change monitor state (cancel, close, open, drain)
are requests the pool's monitor executes on its next poll; the others act on
the pool's files directly and are picked up by the supervisors.

  pool_ctl.py <pool_dir> status                  hosts, runs, queue
  pool_ctl.py <pool_dir> cancel <job|run|name>   cancel a job (stops its run)
  pool_ctl.py <pool_dir> preempt <run>           graceful stop + requeue (resumes from checkpoint)
  pool_ctl.py <pool_dir> exit-host <host>        preempt the host's run, release the allocation
  pool_ctl.py <pool_dir> drain [<host> ...]      finish current runs, then release (all hosts if none given)
  pool_ctl.py <pool_dir> close | open            stop / resume submitting hosts
  pool_ctl.py <pool_dir> set key=value ...       change pool settings (live), e.g. keep_pending=1
  pool_ctl.py <pool_dir> exclude <node> ...      never launch a run on these nodes again
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from oellm_autoexp.monitor.pool import (  # noqa: E402
    PoolHostConfig,
    PoolLayout,
    read_env_file,
    read_run_state,
    submit_request,
)


def _age(path: Path) -> str:
    try:
        return f"{time.time() - path.stat().st_mtime:.0f}s"
    except OSError:
        return "-"


def _fmt_ts(value: str | float | None) -> str:
    try:
        return time.strftime("%m-%d %H:%M:%S", time.localtime(float(value)))
    except (TypeError, ValueError):
        return "-"


def _job_records(session_dir: Path) -> dict[str, dict]:
    """Run id -> record summary, from the monitor session's job files."""
    out: dict[str, dict] = {}
    for path in sorted(session_dir.glob("*.job.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        runtime = payload.get("runtime") or {}
        rid = runtime.get("runtime_job_id")
        if rid:
            out[rid] = {
                "job_id": payload.get("job_id"),
                "final": runtime.get("final_state"),
                "status": runtime.get("last_status"),
                "attempts": runtime.get("attempts"),
            }
    return out


def cmd_status(layout: PoolLayout, _args) -> int:
    meta = layout.load_meta()
    cfg = layout.load_config()
    state = layout.load_state()
    session_dir = Path(meta.get("session_dir", ""))
    records = _job_records(session_dir) if session_dir.is_dir() else {}
    print(f"pool {cfg.name}  {layout.root}")
    print(
        f"  session {session_dir}  monitor tick {_age(layout.monitor_alive)} ago"
        + ("  [CLOSED]" if state.get("closed") else "")
    )
    print(
        f"  max_hosts_running={cfg.max_hosts_running} keep_pending={cfg.keep_pending} "
        f"min_hosts={cfg.min_hosts} handover={cfg.handover} idle_exit_s={cfg.idle_exit_s:.0f}"
    )
    print("\nhosts:")
    for hdir in sorted(layout.hosts_dir.glob("host*")):
        info = read_env_file(hdir / "info.env")
        st = (hdir / "state").read_text().split() if (hdir / "state").exists() else []
        controls = ",".join(p.name for p in sorted((hdir / "control").glob("*"))) or "-"
        exited = (hdir / "exited").read_text().strip() if (hdir / "exited").exists() else ""
        host_record = (
            next(
                (
                    r
                    for r in (json.loads(p.read_text()) for p in session_dir.glob("*.job.json"))
                    if (r.get("definition") or {}).get("metadata", {}).get("pool_host_id")
                    == hdir.name
                ),
                None,
            )
            if session_dir.is_dir()
            else None
        )
        slurm_id = (host_record or {}).get("runtime", {}).get("runtime_job_id") or info.get(
            "SLURM_JOB_ID", "-"
        )
        runtime = (host_record or {}).get("runtime", {})
        # a closed record keeps the LAST POLLED state (e.g. RUNNING): show how it ended instead
        slurm_state = (
            (f"ended:{runtime['final_state']}" if runtime.get("final_state") else None)
            or runtime.get("last_status")
            or "-"
        )
        print(
            f"  {hdir.name}  job {slurm_id:<10} {slurm_state:<10} nodes {info.get('NUM_NODES', '-'):<4} "
            f"started {_fmt_ts(info.get('START_TS'))}  heartbeat {_age(hdir / 'alive'):<6} "
            f"state {' '.join(st[:2]) or '-':<22} control {controls}"
            + (f"  exited: {exited}" if exited else "")
        )
    print("\nruns:")
    for rdir in sorted(layout.runs_dir.iterdir() if layout.runs_dir.exists() else []):
        if not (rdir / "run.json").exists():
            continue
        run = json.loads((rdir / "run.json").read_text())
        rs = read_run_state(layout, rdir.name)
        rec = records.get(rdir.name, {})
        try:
            launch = json.loads((rdir / "launch_provenance.json").read_text())
            changed = launch.get("changed_since_enqueue")
        except (OSError, json.JSONDecodeError):
            changed = None
        print(
            f"  {rdir.name:<22} {rs.status:<10} {run.get('name', ''):<40} host {rs.host or '-':<8} "
            f"started {_fmt_ts(rs.started.get('ts'))}"
            + (f"  rc {rs.code} {rs.reason}" if rs.reason else "")
            + (f"  [job {rec['job_id']} {rec.get('final') or 'active'}]" if rec else "")
            + (f"  CODE CHANGED SINCE ENQUEUE ({len(changed)} change(s))" if changed else "")
        )
    pending = (
        sorted(p.name for p in layout.requests_dir.glob("*.json"))
        if layout.requests_dir.exists()
        else []
    )
    if pending:
        print(f"\nrequests not yet processed by the monitor: {pending}")
    return 0


# A monitor tick older than this means nobody will execute a request soon.
MONITOR_STALE_S = 300.0


def _pool_jobs(layout: PoolLayout) -> list[dict]:
    """The pool's job records (raw JSON) from its monitor session."""
    session_dir = Path(layout.load_meta().get("session_dir", ""))
    out = []
    for path in sorted(session_dir.glob("*.job.json")) if session_dir.is_dir() else []:
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        pool_dir = (record.get("definition") or {}).get("pool_dir")
        if pool_dir and Path(pool_dir).absolute() == layout.root:
            out.append(record)
    return out


def resolve_target(layout: PoolLayout, target: str) -> tuple[list[str], list[str]]:
    """Job ids a cancel ``target`` means: (active ids, explanations of misses).

    ``target`` may be a job id, a job name, or ANY run id of the job (also an
    earlier attempt's: run.json names its job). Nothing is written here, so a
    target that only matches finished jobs is refused up front instead of
    becoming a request the monitor silently answers with "no match".
    """
    run_job = run_name = None
    run_json = layout.run_dir(target) / "run.json"
    if run_json.exists():
        try:
            run = json.loads(run_json.read_text())
            run_job, run_name = run.get("job_id") or None, run.get("name") or None
        except (OSError, json.JSONDecodeError):
            pass
    active, finished = [], []
    for record in _pool_jobs(layout):
        job_id = record.get("job_id")
        runtime = record.get("runtime") or {}
        names = {
            job_id,
            (record.get("definition") or {}).get("name"),
            runtime.get("runtime_job_id"),
        }
        name = (record.get("definition") or {}).get("name")
        # a run id: the job it names (by id; by name for runs written without one)
        of_run = job_id == run_job if run_job else (run_name is not None and name == run_name)
        if target not in names and not of_run:
            continue
        if runtime.get("final_state"):
            when = runtime.get("end_ts")
            stamp = time.strftime("%H:%M:%S", time.localtime(when)) if when else "?"
            finished.append(f"{job_id} already ended ({runtime['final_state']} at {stamp})")
        else:
            active.append(job_id)
    misses = finished or ([] if active else [f"nothing in this pool matches {target!r}"])
    return active, misses


def _await_result(layout: PoolLayout, request: Path, timeout_s: float) -> int:
    """Wait for the monitor to execute a request; print its result."""
    done = layout.requests_dir / "done" / request.name
    try:
        tick_age = time.time() - layout.monitor_alive.stat().st_mtime
    except OSError:
        tick_age = float("inf")
    if tick_age > MONITOR_STALE_S:
        age = "never" if tick_age == float("inf") else f"{tick_age:.0f}s ago"
        print(
            f"  no live monitor (last tick: {age}): the request waits until a monitor "
            "attaches (python scripts/monitor_autoexp.py --session-dir <session from pool.json>)"
        )
        return 0
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if done.exists():
            try:
                result = str(json.loads(done.read_text()).get("result", "?"))
            except (OSError, json.JSONDecodeError):
                time.sleep(0.5)
                continue
            print(f"  monitor: {result}")
            failed = result.startswith(("no active job", "unknown op", "error"))
            return 1 if failed else 0
        time.sleep(1.0)
    print(
        f"  not executed within {timeout_s:.0f}s; it stays queued (pool_ctl status shows pending requests)"
    )
    return 0


def cmd_request(op: str):
    def run(layout: PoolLayout, args) -> int:
        payload = {}
        if op == "cancel":
            active, misses = resolve_target(layout, args.target)
            if not active:
                for line in misses:
                    print(f"nothing to cancel: {line}", file=sys.stderr)
                return 1
            if len(active) > 1 and not args.all:
                print(
                    f"{args.target!r} matches {len(active)} active jobs: {', '.join(active)}\n"
                    "  name one of them, or pass --all to cancel every one",
                    file=sys.stderr,
                )
                return 1
            # exact ids, so the monitor cancels precisely what was resolved here
            payload["targets"] = active
            print(f"cancelling {', '.join(active)}")
        if op == "drain" and args.hosts:
            payload["hosts"] = args.hosts
        path = submit_request(layout.root, op, **payload)
        print(f"request {op} written: {path.name}")
        if getattr(args, "no_wait", False):
            return 0
        return _await_result(layout, path, args.wait)

    return run


_HOOK_MARK = "# Pool adoption hook (oellm_autoexp/monitor/pool.py, `pool_ctl adopt`)"


def _hook_ids(path: Path) -> set[str] | None:
    """Job ids of an installed adoption hook; None if the file is not ours."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith(_HOOK_MARK):
        return None
    import ast

    for line in text.splitlines():
        if line.startswith("_ADOPTED = "):
            value = line.split("=", 1)[1].split("#", 1)[0].strip()
            return set(ast.literal_eval(value) or ()) if value != "set()" else set()
    return None


def render_hook(ids: set[str], adopt_dir: Path, hold_max_s: int) -> str:
    template = (Path(__file__).resolve().parent / "pool" / "adopt_sitecustomize.py").read_text()
    text = (
        template.replace("{__ADOPTED_JOB_IDS__}", repr(set(sorted(ids))) if ids else "set()")
        .replace("__ADOPT_DIR__", repr(str(adopt_dir)))
        .replace("__HOLD_MAX_S__", str(int(hold_max_s)))
    )
    compile(text, "sitecustomize.py", "exec")  # never install a hook that does not parse
    return text


def cmd_attach(layout: PoolLayout, args) -> int:
    """Attach to a RUNNING allocation, as `adopt` does to a queued one but without a hook: the
    allocation must run nothing that uses its nodes (an idle host of another pool, kept alive with a
    long idle_exit_s). The monitor starts this pool's supervisor into it as an --overlap step; it
    stops its runs adopt_stop_margin_s before the allocation's end and never cancels it."""
    import subprocess

    jid = str(args.job_id)
    row = subprocess.run(
        ["squeue", "-h", "-j", jid, "-o", "%T|%D|%a|%l"], capture_output=True, text=True
    ).stdout.strip()
    if not row:
        print(f"job {jid} is not in the queue", file=sys.stderr)
        return 1
    state, nodes, account, limit = row.split("|")
    if state != "RUNNING":
        print(
            f"job {jid} is {state}: attach takes a running allocation (adopt a queued one)",
            file=sys.stderr,
        )
        return 1
    adopt_dir = layout.root / "adopt"
    adopt_dir.mkdir(parents=True, exist_ok=True)
    # what the adoption hook writes when it holds the job's step: the monitor's go-ahead
    (adopt_dir / f"{jid}.started").write_text(
        f"{time.time():.0f} attached (running allocation, no hook)\n"
    )
    path = submit_request(
        layout.root, "adopt", job_id=jid, nodes=int(nodes), account=account, time=limit
    )
    print(f"request adopt (attach) written: {path.name}")
    if args.no_wait:
        return 0
    return _await_result(layout, path, args.wait)


def cmd_adopt(layout: PoolLayout, args) -> int:
    """Adopt a QUEUED SLURM job as a host of this pool: install the fail-closed hook into its
    working directory (its own training step will never run) and register it with the monitor,
    which starts the supervisor into the allocation once it runs."""
    import subprocess

    jid = str(args.job_id)
    row = subprocess.run(
        ["squeue", "-h", "-j", jid, "-o", "%T|%D|%a|%l|%Z"], capture_output=True, text=True
    ).stdout.strip()
    if not row:
        print(f"job {jid} is not in the queue", file=sys.stderr)
        return 1
    state, nodes, account, limit, workdir = row.split("|")
    hook = Path(args.workdir or workdir) / "sitecustomize.py"
    ids = _hook_ids(hook) if hook.exists() else set()
    if ids is None:
        print(f"{hook} exists and is not an adoption hook: refusing", file=sys.stderr)
        return 1
    if args.remove:
        ids.discard(jid)
        if ids:
            hook.write_text(
                render_hook(ids, layout.root / "adopt", args.hold_max_s), encoding="utf-8"
            )
        else:
            hook.unlink(missing_ok=True)
        print(
            f"job {jid} removed from {hook} ({'still holds ' + ', '.join(sorted(ids)) if ids else 'hook removed'})"
        )
        return 0
    if state != "PENDING":
        print(
            f"job {jid} is {state}: only a queued job can be adopted (its step must not have started)",
            file=sys.stderr,
        )
        return 1
    (layout.root / "adopt").mkdir(parents=True, exist_ok=True)
    text = render_hook(ids | {jid}, layout.root / "adopt", args.hold_max_s)
    tmp = hook.with_name(f".sitecustomize.{jid}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(hook)  # replace, never edit in place: a starting job may be reading it
    assert jid in (_hook_ids(hook) or set()), "hook does not list the job after install"
    print(f"hook installed: {hook} (holds {', '.join(sorted(ids | {jid}))})")
    path = submit_request(
        layout.root,
        "adopt",
        job_id=jid,
        nodes=int(nodes),
        account=account,
        time=limit,
        workdir=str(hook.parent),
    )
    print(f"request adopt written: {path.name}")
    if args.no_wait:
        return 0
    return _await_result(layout, path, args.wait)


def cmd_preempt(layout: PoolLayout, args) -> int:
    rd = layout.run_dir(args.run)
    if not (rd / "run.json").exists():
        print(f"no such run: {args.run}", file=sys.stderr)
        return 1
    state = read_run_state(layout, args.run)
    if state.status != "RUNNING":
        detail = f" (rc {state.code} {state.reason})" if state.reason else ""
        print(
            f"run {args.run} is {state.status}{detail}: only a RUNNING run can be preempted",
            file=sys.stderr,
        )
        return 1
    (rd / "preempt").write_text(f"pool_ctl {time.time()}\n")
    print(f"run {args.run}: preempt requested (graceful stop, then requeued by the monitor)")
    return 0


def cmd_exit_host(layout: PoolLayout, args) -> int:
    path = layout.host_dir(args.host) / "control" / "exit"
    if not path.parent.parent.exists():
        print(f"no such host: {args.host}", file=sys.stderr)
        return 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("pool_ctl exit-host\n")
    print(f"host {args.host}: exit requested")
    return 0


def cmd_set(layout: PoolLayout, args) -> int:
    meta = layout.load_meta()
    config = dict(meta.get("config") or {})
    types = {f.name: f.type for f in dataclasses.fields(PoolHostConfig)}
    for item in args.assignments:
        if "=" not in item:
            print(f"expected key=value, got {item!r}", file=sys.stderr)
            return 1
        key, value = item.split("=", 1)
        if key not in types:
            print(f"unknown pool setting {key!r}; known: {sorted(types)}", file=sys.stderr)
            return 1
        kind = str(types[key])
        if kind in ("bool", "<class 'bool'>"):
            parsed = value.lower() in ("1", "true", "yes", "on")
        elif kind in ("int", "<class 'int'>"):
            parsed = int(value)
        elif kind in ("float", "<class 'float'>"):
            parsed = float(value)
        else:
            parsed = value
        config[key] = parsed
        print(f"  {key} = {parsed!r}")
    meta["config"] = config
    meta["updated"] = time.time()
    layout.write_meta(meta)
    return 0


def cmd_exclude(layout: PoolLayout, args) -> int:
    existing = (
        set(layout.exclude_nodes.read_text().split()) if layout.exclude_nodes.exists() else set()
    )
    with layout.exclude_nodes.open("a") as fh:
        for node in args.nodes:
            if node not in existing:
                fh.write(node + "\n")
    print(f"excluded for future launches: {' '.join(args.nodes)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("pool_dir", type=Path)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status").set_defaults(func=cmd_status)

    def waiting(p):
        p.add_argument("--wait", type=float, default=120.0, help="seconds to wait for the monitor")
        p.add_argument("--no-wait", action="store_true", help="write the request and return")
        return p

    p = waiting(sub.add_parser("cancel"))
    p.add_argument("target", help="job id, job name, or any run id of the job")
    p.add_argument("--all", action="store_true", help="cancel every active job the name matches")
    p.set_defaults(func=cmd_request("cancel"))
    p = waiting(sub.add_parser("adopt", help="adopt a queued SLURM job as a host of this pool"))
    p.add_argument("job_id")
    p.add_argument("--workdir", help="where the job's step runs Python (default: its WorkDir)")
    p.add_argument(
        "--hold-max-s",
        type=int,
        default=1800,
        help="the held step leaves (rc 0, no training) if no supervisor attaches by then",
    )
    p.add_argument("--remove", action="store_true", help="take the job out of the hook again")
    p.set_defaults(func=cmd_adopt)
    p = waiting(
        sub.add_parser(
            "attach",
            help="take over a RUNNING allocation that runs no training of its own (e.g. an idle"
            " host of another pool) as a host of this pool",
        )
    )
    p.add_argument("job_id")
    p.set_defaults(func=cmd_attach)
    p = sub.add_parser("preempt")
    p.add_argument("run")
    p.set_defaults(func=cmd_preempt)
    p = sub.add_parser("exit-host")
    p.add_argument("host")
    p.set_defaults(func=cmd_exit_host)
    p = waiting(sub.add_parser("drain"))
    p.add_argument("hosts", nargs="*")
    p.set_defaults(func=cmd_request("drain"))
    waiting(sub.add_parser("close")).set_defaults(func=cmd_request("close"))
    waiting(sub.add_parser("open")).set_defaults(func=cmd_request("open"))
    p = sub.add_parser("set")
    p.add_argument("assignments", nargs="+")
    p.set_defaults(func=cmd_set)
    p = sub.add_parser("exclude")
    p.add_argument("nodes", nargs="+")
    p.set_defaults(func=cmd_exclude)
    args = parser.parse_args(argv)
    layout = PoolLayout(args.pool_dir)
    if not layout.exists():
        print(f"{layout.root} is not a pool (no pool.json)", file=sys.stderr)
        return 2
    return args.func(layout, args)


if __name__ == "__main__":
    sys.exit(main())

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
  pool_ctl.py <pool_dir> pin <run> <host> | unpin <run>  a queued run waits for that host only
  pool_ctl.py <pool_dir> freeze | unfreeze <host> no new run on the host, never released (resize prep)
  pool_ctl.py <pool_dir> shrink <host> --keep N  plan a resize of a frozen host to N nodes; --apply does it
  pool_ctl.py <pool_dir> park [<host> ...]       plan parking pending hosts (Nice: priority 1); --apply
  pool_ctl.py <pool_dir> promote [<host> ...]    plan Nice=0 again (full priority back); --apply
  pool_ctl.py <pool_dir> queue                   all my pending jobs per account / QOS, in SLURM's order

Resizing a host (2026-11-10): `freeze` it, wait until no run is active on it (or preempt them), then
`shrink --keep N --apply`, then `unfreeze`. A step running during `scontrol update NodeList=` stays
RESIZING and keeps its GPUs after it ends ("Insufficient GRES" for every later step there: p134c host006
11-06, p264 host003 11-10), also on nodes the resize keeps. shrink keeps the batch node, every node a
run holds, and free non-excluded nodes up to N; it writes the removed nodes into <host>/exclude_nodes and
<host>/inuse/zz_shrunk_removed_nodes (old supervisors keep their start-time node list) before the update.
A resize changes a SLURM job: only with the user's explicit go (AGENTS.md).

Parking hosts (2026-11-18, docs/mn5_renice.md): `park` sets Nice=250000 on pending hosts (priority 1,
behind every job in the partition), `promote` sets Nice=0 (the exact priority back; age kept growing).
It is no gate: a parked job still starts by backfill or when nothing else wants the nodes, and the
first of a fully parked group still takes the next free QOS slot. Without host ids both act on every
pending host of the pool and set its submit_nice (the hosts the monitor queues later). Plan by
default; --apply re-nices existing SLURM jobs: only with the user's explicit go (AGENTS.md).
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


def _loads_or_none(path: Path) -> dict | None:
    """Parse a job record, tolerating an empty or corrupt file.

    pool_state/pool_mn5-pool-p134c held 11 records of 100 % NUL bytes (2026-11-24), and an
    unguarded json.loads in `status` turned that into a JSONDecodeError for the whole pool.
    """
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _job_records(session_dir: Path) -> dict[str, dict]:
    """Run id -> record summary, from the monitor session's job files."""
    out: dict[str, dict] = {}
    for path in sorted(session_dir.glob("*.job.json")):
        payload = _loads_or_none(path)
        if payload is None:
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
    queued = _slurm_jobs(list(_host_jobs(layout).values()))  # nice / priority, {} without squeue
    for hdir in sorted(layout.hosts_dir.glob("host*")):
        info = read_env_file(hdir / "info.env")
        st = (hdir / "state").read_text().split() if (hdir / "state").exists() else []
        controls = ",".join(p.name for p in sorted((hdir / "control").glob("*"))) or "-"
        exited = (hdir / "exited").read_text().strip() if (hdir / "exited").exists() else ""
        host_record = (
            next(
                (
                    r
                    for r in (
                        rec
                        for rec in (_loads_or_none(p) for p in session_dir.glob("*.job.json"))
                        if rec is not None
                    )
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
            + (
                f"  nice {queued[slurm_id]['nice']} priority {queued[slurm_id]['priority']}"
                if queued.get(slurm_id, {}).get("state") == "PENDING"
                else ""
            )
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


def cmd_hold(layout: PoolLayout, args) -> int:
    """hold / unhold a host: a held host is never released by the monitor (not idle, not for
    queued work it cannot run, not after a handover); its wall clock or exit-host still end it."""
    hdir = layout.host_dir(args.host)
    if not hdir.exists():
        print(f"no such host: {args.host}", file=sys.stderr)
        return 1
    path = hdir / "control" / "hold"
    if args.cmd == "hold":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("pool_ctl hold\n")
        print(f"host {args.host}: held (the monitor never releases it)")
    else:
        path.unlink(missing_ok=True)
        print(f"host {args.host}: released from hold")
    return 0


def cmd_pin(layout: PoolLayout, args) -> int:
    """pin / unpin a queued run to one host: it then waits for that host instead of taking the
    newest free one.

    The pin is ``runs/<rid>/pin_host``, which the monitor reads on every pass (a record field
    would be overwritten by the monitor's own writes). A pinned run that its host cannot fit
    waits there; no other host takes it.
    """
    active, misses = resolve_target(layout, args.run)
    if not active:
        for line in misses:
            print(f"nothing to pin: {line}", file=sys.stderr)
        return 1
    if len(active) > 1:
        print(f"{args.run} names {len(active)} active jobs: {active}", file=sys.stderr)
        return 1
    job_id = active[0]
    for record in _pool_jobs(layout):
        if record.get("job_id") != job_id:
            continue
        rid = (record.get("runtime") or {}).get("runtime_job_id")
        if not rid:
            break
        path = layout.run_dir(rid) / "pin_host"
        if args.cmd == "pin":
            if not layout.host_dir(args.host).exists():
                print(f"no such host: {args.host}", file=sys.stderr)
                return 1
            path.write_text(f"{args.host}\n")
            print(f"{rid}: pinned to {args.host}")
        else:
            path.unlink(missing_ok=True)
            print(f"{rid}: pin cleared")
        return 0
    print(f"no active run for {job_id}", file=sys.stderr)
    return 1


def cmd_freeze(layout: PoolLayout, args) -> int:
    """freeze / unfreeze a host: a frozen host gets no new run and is never released by the
    monitor (pool.PoolManager._frozen); its running runs continue."""
    hdir = layout.host_dir(args.host)
    if not hdir.exists():
        print(f"no such host: {args.host}", file=sys.stderr)
        return 1
    path = hdir / "control" / "freeze"
    if args.cmd == "freeze":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"pool_ctl freeze {time.time():.0f}\n")
        print(f"host {args.host}: frozen (no new run, never released; running runs continue)")
    else:
        path.unlink(missing_ok=True)
        print(f"host {args.host}: unfrozen")
    return 0


def _slurm(argv: list[str]) -> str:
    import subprocess

    return subprocess.run(argv, capture_output=True, text=True).stdout


def _hostnames(nodelist: str, slurm=_slurm) -> list[str]:
    nodelist = nodelist.strip()
    return [] if not nodelist else slurm(["scontrol", "show", "hostnames", nodelist]).split()


def shrink_plan(layout: PoolLayout, hid: str, keep: int, slurm=_slurm) -> dict:
    """What `shrink` would do: the host's live allocation, the nodes runs hold, the batch node, the
    active run steps, and the kept / removed node sets."""
    hdir = layout.host_dir(hid)
    info = read_env_file(hdir / "info.env")
    jid = str(info.get("SLURM_JOB_ID") or "").strip()
    if not jid:
        raise SystemExit(f"host {hid}: no SLURM_JOB_ID in info.env")
    alloc = sorted(set(_hostnames(slurm(["squeue", "-h", "-j", jid, "-o", "%N"]), slurm)))
    if not alloc:
        raise SystemExit(f"job {jid} has no allocation (not running?)")
    job = slurm(["scontrol", "show", "job", jid])
    batch = next((w.split("=", 1)[1] for w in job.split() if w.startswith("BatchHost=")), "")
    inuse_dir = hdir / "inuse"
    held, runs = set(), {}
    for f in sorted(inuse_dir.glob("*")) if inuse_dir.exists() else []:
        nodes = {n for n in f.read_text().replace("\n", ",").split(",") if n}
        if f.name.startswith("zz_"):
            continue  # nodes a previous shrink removed: not in the allocation
        runs[f.name] = sorted(nodes & set(alloc))
        held |= nodes & set(alloc)
    steps = [
        line.split()[0]
        for line in slurm(["squeue", "-s", "-h", "-j", jid, "-o", "%i %j"]).splitlines()
        if line.split() and not line.split()[0].endswith((".batch", ".extern"))
    ]
    excl_file = hdir / "exclude_nodes"
    excluded = set(excl_file.read_text().split()) if excl_file.exists() else set()
    must = sorted(held | ({batch} if batch else set()))
    if len(must) > keep:
        raise SystemExit(f"{len(must)} nodes must stay (batch node + runs' nodes) > keep {keep}")
    free = [n for n in alloc if n not in held and n != batch and n not in excluded]
    kept = sorted(set(must) | set(free[: keep - len(must)]))
    if len(kept) < keep:
        raise SystemExit(f"only {len(kept)} usable nodes for keep {keep}")
    return {"job": jid, "alloc": alloc, "batch": batch, "runs": runs, "steps": steps,
            "keep": kept, "remove": sorted(set(alloc) - set(kept))}


def cmd_shrink(layout: PoolLayout, args, slurm=_slurm) -> int:
    hid = args.host
    if not (layout.host_dir(hid) / "control" / "freeze").exists():
        print(f"host {hid} is not frozen: pool_ctl freeze {hid} first (no run may start during a resize)",
              file=sys.stderr)
        return 1
    plan = shrink_plan(layout, hid, args.keep, slurm)
    print(f"host {hid} job {plan['job']}: {len(plan['alloc'])} -> {len(plan['keep'])} nodes "
          f"(batch {plan['batch']}; {sum(len(v) for v in plan['runs'].values())} held by "
          f"{len(plan['runs'])} run(s); remove {len(plan['remove'])})")
    for rid, nodes in plan["runs"].items():
        print(f"  run {rid}: {len(nodes)} node(s) kept")
    if plan["steps"]:
        msg = (f"run steps active: {' '.join(plan['steps'])} -- a resize now leaves their GPUs reserved "
               "after they end (their nodes are then lost for GPU work)")
        if not args.accept_gpu_leak:
            print(f"refused: {msg}; wait / preempt, or --accept-gpu-leak", file=sys.stderr)
            return 1
        print(f"warning: {msg}")
    if not args.apply:
        print("plan only (--apply resizes the SLURM job: only with the user's go)")
        return 0
    hdir = layout.host_dir(hid)
    excl = hdir / "exclude_nodes"
    have = set(excl.read_text().split()) if excl.exists() else set()
    with excl.open("a") as fh:
        for n in plan["remove"]:
            if n not in have:
                fh.write(n + "\n")
    zz = hdir / "inuse" / "zz_shrunk_removed_nodes"
    old = {n for n in (zz.read_text().split(",") if zz.exists() else []) if n.strip()}
    zz.parent.mkdir(parents=True, exist_ok=True)
    tmp = zz.with_suffix(".tmp")
    tmp.write_text(",".join(sorted(old | set(plan["remove"]))))
    tmp.replace(zz)
    keep_list = slurm(["scontrol", "show", "hostlist", ",".join(plan["keep"])]).strip()
    slurm(["scontrol", "update", f"JobId={plan['job']}", f"NodeList={keep_list}"])
    now = len(_hostnames(slurm(["squeue", "-h", "-j", plan["job"], "-o", "%N"]), slurm))
    if now != len(plan["keep"]):
        print(f"host {hid}: the job has {now} node(s) after the update, {len(plan['keep'])} expected",
              file=sys.stderr)
        return 1
    print(f"host {hid}: resized to {now} node(s); unfreeze with pool_ctl unfreeze {hid}")
    return 0


# ---------------------------------------------------------------- park / promote (2026-11-18)
# Nice subtracts 1:1 from the SLURM priority, floored at 1; the highest priority MareNostrum 5
# computes is 210500 (age 100000 + fairshare 100000 + job size 10000 + QOS 500), so this parks any
# job at priority 1. Nothing else changes (submit / accrue time, age keeps growing) and Nice=0
# gives the exact priority back; users cannot go below 0 (docs/mn5_renice.md).
PARK_NICE = 250_000


def _host_jobs(layout: PoolLayout) -> dict[str, str]:
    """host id -> SLURM job id of the pool's hosts whose record is still open."""
    session_dir = Path(layout.load_meta().get("session_dir", ""))
    out: dict[str, str] = {}
    for path in sorted(session_dir.glob("*.job.json")) if session_dir.is_dir() else []:
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        md = (record.get("definition") or {}).get("metadata") or {}
        runtime = record.get("runtime") or {}
        if (
            md.get("pool_role") == "host"
            and Path(str(md.get("pool_dir", ""))).absolute() == layout.root
            and runtime.get("runtime_job_id")
            and not runtime.get("final_state")
        ):
            out[md["pool_host_id"]] = str(runtime["runtime_job_id"])
    return out


def _slurm_jobs(job_ids, slurm=_slurm) -> dict[str, dict]:
    """job id -> {state, nice, priority} for the jobs SLURM still knows ({} on no answer)."""
    if not job_ids:
        return {}
    try:
        text = slurm(["squeue", "-h", "-j", ",".join(job_ids), "-o", "%i|%T|%y|%Q"])
    except OSError:
        return {}
    out = {}
    for line in text.splitlines():
        parts = line.strip().split("|")
        if len(parts) == 4 and parts[0]:
            try:
                out[parts[0]] = {"state": parts[1], "nice": int(parts[2]), "priority": int(parts[3])}
            except ValueError:
                continue
    return out


def _ledger(layout: PoolLayout, hid: str, jid: str, text: str) -> None:
    with open(layout.ledger, "a") as fh:
        fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{hid}\t{jid}\t-\t{text}\n")


def _set_submit_nice(layout: PoolLayout, nice: int) -> None:
    meta = layout.load_meta()
    meta["config"] = {**(meta.get("config") or {}), "submit_nice": nice}
    meta["updated"] = time.time()
    layout.write_meta(meta)


def cmd_nice(layout: PoolLayout, args, slurm=_slurm) -> int:
    """park / promote: the PENDING hosts of the pool (or the named ones) to Nice=--nice / Nice=0.

    Plan by default; --apply runs `scontrol update JobId=.. Nice=..` (an edit of an existing SLURM
    job: only with the user's go, AGENTS.md), reads the value back, writes <host>/control/parked and
    a ledger line. Without named hosts it also sets the pool's submit_nice, so the hosts the
    monitor queues later follow. Running jobs are left alone (nice no longer matters there).
    """
    want = int(args.nice) if args.cmd == "park" else 0
    if want < 0:
        print("negative nice needs root on SLURM", file=sys.stderr)
        return 1
    hosts = _host_jobs(layout)
    named = list(args.hosts)
    missing = [h for h in named if h not in hosts]
    if missing:
        print(f"no open SLURM job for host(s): {' '.join(missing)}", file=sys.stderr)
        return 1
    targets = {h: hosts[h] for h in (named or sorted(hosts))}
    jobs = _slurm_jobs(list(targets.values()), slurm)
    if targets and not jobs:
        print("no answer from squeue", file=sys.stderr)
        return 1
    change = []
    for hid, jid in targets.items():
        job = jobs.get(jid)
        if job is None:
            print(f"  {hid}  job {jid}: not in the queue any more, left alone")
            continue
        if job["state"] != "PENDING":
            print(f"  {hid}  job {jid}: {job['state']}, left alone (nice only orders the queue)")
            continue
        same = job["nice"] == want
        if same:
            expect = "unchanged"
        elif job["priority"] <= 1 and want < job["nice"]:
            expect = "its natural priority (hidden while floored at 1)"
        else:
            expect = f"~{max(1, job['priority'] + job['nice'] - want)}"
        print(f"  {hid}  job {jid}: nice {job['nice']} -> {want}  priority {job['priority']} -> "
              f"{expect}")
        if not same:
            change.append((hid, jid, job))
    pool_wide = not named
    cfg = layout.load_config()
    if pool_wide and cfg.submit_nice != want:
        print(f"  submit_nice {cfg.submit_nice} -> {want} (hosts the monitor submits from now on)")
    if not args.apply:
        print("plan only (--apply re-nices the SLURM jobs: only with the user's go)")
        return 0
    if pool_wide:
        _set_submit_nice(layout, want)
    rc = 0
    for hid, jid, job in change:
        slurm(["scontrol", "update", f"JobId={jid}", f"Nice={want}"])
        now = _slurm_jobs([jid], slurm).get(jid)
        if now is None or now["nice"] != want:
            print(f"  {hid}  job {jid}: nice is {now['nice'] if now else 'unknown'} after the "
                  f"update, {want} expected", file=sys.stderr)
            rc = 1
            continue
        marker = layout.host_dir(hid) / "control" / "parked"
        if want:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(f"nice={want} ts={time.time():.0f}\n")
        else:
            marker.unlink(missing_ok=True)
        _ledger(layout, hid, jid, f"{args.cmd} nice {job['nice']} -> {want}, priority "
                f"{job['priority']} -> {now['priority']}")
        print(f"  {hid}  job {jid}: nice {want}, priority {now['priority']}")
    return rc


def cmd_queue(layout: PoolLayout, _args, slurm=_slurm) -> int:
    """All of the user's pending jobs, per account / QOS, in SLURM's order (priority, then job id),
    with the running count; '*' = this pool's hosts, 'P' = parked (nice > 0). Read-only."""
    try:
        pending = slurm(["squeue", "-h", "--me", "-t", "PD", "-o", "%i|%j|%y|%Q|%a|%q|%D|%r"])
        running = slurm(["squeue", "-h", "--me", "-t", "R", "-o", "%a|%q"])
    except OSError as exc:
        print(f"squeue failed: {exc}", file=sys.stderr)
        return 1
    mine = set(_host_jobs(layout).values())
    groups: dict[tuple[str, str], list[list[str]]] = {}
    for line in pending.splitlines():
        parts = line.strip().split("|")
        if len(parts) == 8:
            groups.setdefault((parts[4], parts[5]), []).append(parts)
    n_running: dict[tuple[str, str], int] = {}
    for line in running.splitlines():
        parts = line.strip().split("|")
        if len(parts) == 2:
            n_running[(parts[0], parts[1])] = n_running.get((parts[0], parts[1]), 0) + 1
    for (account, qos), jobs in sorted(groups.items()):
        jobs.sort(key=lambda p: (-int(p[3] or 0), int(p[0].split("_")[0] or 0)))
        parked = sum(1 for p in jobs if int(p[2] or 0) > 0)
        print(f"{account} / {qos}: {len(jobs)} pending ({parked} parked), "
              f"{n_running.get((account, qos), 0)} running")
        for p in jobs:
            flags = ("*" if p[0] in mine else " ") + ("P" if int(p[2] or 0) > 0 else " ")
            print(f"  {flags} {p[0]:<10} {p[1]:<28} nice {p[2]:>7} priority {p[3]:>7} "
                  f"nodes {p[6]:>4}  {p[7]}")
        if jobs and parked == len(jobs):
            print("  (all parked: the first of them still takes the next free slot)")
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
    for name in ("hold", "unhold"):
        p = sub.add_parser(name, help="never release / release again a host (see cmd_hold)")
        p.add_argument("host")
        p.set_defaults(func=cmd_hold)
    for name in ("pin", "unpin"):
        p = sub.add_parser(
            name, help="pin a queued run to one host / clear the pin (runs/<rid>/pin_host)"
        )
        p.add_argument("run", help="job id, job name, or any run id of the job")
        if name == "pin":
            p.add_argument("host")
        p.set_defaults(func=cmd_pin)
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
    for name in ("freeze", "unfreeze"):
        p = sub.add_parser(name, help="no new run on / never release a host while a resize is prepared")
        p.add_argument("host")
        p.set_defaults(func=cmd_freeze)
    p = sub.add_parser("shrink", help="resize a frozen host to --keep nodes (plan; --apply does it)")
    p.add_argument("host")
    p.add_argument("--keep", type=int, required=True)
    p.add_argument("--apply", action="store_true", help="write the markers and scontrol update (user's go)")
    p.add_argument("--accept-gpu-leak", action="store_true",
                   help="resize although run steps are active (their nodes' GPUs stay reserved)")
    p.set_defaults(func=cmd_shrink)
    for name in ("park", "promote"):
        p = sub.add_parser(
            name,
            help=("lower the SLURM priority of pending hosts (Nice; plan, --apply does it)"
                  if name == "park" else "Nice=0 again: the full priority back (plan; --apply)"),
        )
        p.add_argument("hosts", nargs="*", help="host ids; none = every pending host + submit_nice")
        if name == "park":
            p.add_argument("--nice", type=int, default=PARK_NICE)
        p.add_argument("--apply", action="store_true", help="scontrol update Nice (user's go)")
        p.set_defaults(func=cmd_nice)
    sub.add_parser("queue", help="the user's pending jobs per account / QOS in SLURM's order"
                   ).set_defaults(func=cmd_queue)
    args = parser.parse_args(argv)
    layout = PoolLayout(args.pool_dir)
    if not layout.exists():
        print(f"{layout.root} is not a pool (no pool.json)", file=sys.stderr)
        return 2
    return args.func(layout, args)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""A tiny local SLURM for tests: sbatch, squeue, sacct, scancel, srun, scontrol.

Put ``tests/fakeslurm/bin`` first on PATH and set ``FAKESLURM_DIR`` to a scratch
directory; the real ``SlurmClient`` and the pool supervisor then run unchanged
against local processes. Stdlib only.

What it models (enough to test sub-SLURM-job pools honestly):

* ``sbatch`` parses ``#SBATCH`` lines (nodes, output, time, job-name, signal),
  copies the script and starts a detached runner. A job stays PENDING while
  ``$FAKESLURM_DIR/jobs/<id>/hold`` exists (created at submission when
  ``FAKESLURM_HOLD=1``), so a test decides when an allocation "starts".
  Nodes are named ``n<id>-<k>``.
* the runner enforces the time limit: ``--signal=[B:]SIG@margin`` (B: = batch
  shell only), SIGTERM at the limit then SIGKILL, state TIMEOUT. At the end of a
  job every step's processes are killed, like proctrack/cgroup.
* ``srun`` (inside a job) creates a step, validates node counts and nodelists
  against the allocation, WAITS while a non-overlap step holds any of its nodes
  (what real SLURM does), runs one process per task locally with
  SLURM_NODEID/SLURM_PROCID/SLURM_STEP_* set, forwards SIGTERM/SIGINT/SIGUSR1/2
  to its tasks and exits with the highest task exit code.
* ``scancel`` on a job (TERM, then KILL after 2 s; CANCELLED) or a step
  (``<id>.<step>``, with ``--signal``/``-s``).
* ``squeue`` (``--jobs``/``-j``, ``-h``, ``-o``/``--format`` with %i %T %j %L,
  ``-s`` for steps) and ``sacct`` (``--jobs``, ``--parsable2``, JobID,State).
* ``scontrol show hostnames <list>`` with ``a,b`` and ``pre[1-3,5]`` forms.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("FAKESLURM_DIR", "/nonexistent-fakeslurm"))
JOBS = ROOT / "jobs"
FINAL = ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT")


# ----------------------------------------------------------------- helpers
def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _job(jid: str) -> dict:
    return _read_json(JOBS / jid / "job.json")


def _parse_time(text: str | None) -> float | None:
    if not text:
        return None
    t = text.strip()
    days = 0
    if "-" in t:
        d, t = t.split("-", 1)
        days = int(d)
        parts = [int(p) for p in t.split(":")] + [0, 0]
        h, m, s = parts[:3]
    else:
        parts = [int(p) for p in t.split(":")]
        if len(parts) == 1:
            h, m, s = 0, parts[0], 0
        elif len(parts) == 2:
            h, m, s = 0, parts[0], parts[1]
        else:
            h, m, s = parts[:3]
    return float(days * 86400 + h * 3600 + m * 60 + s)


def expand_hostlist(text: str) -> list[str]:
    out: list[str] = []
    for token in re.findall(r"[^,\[]+(?:\[[^\]]*\])?[^,]*", text or ""):
        token = token.strip()
        if not token:
            continue
        m = re.fullmatch(r"(.*)\[([^\]]*)\](.*)", token)
        if not m:
            out.append(token)
            continue
        pre, body, post = m.groups()
        for part in body.split(","):
            if "-" in part:
                a, b = part.split("-", 1)
                width = len(a)
                for i in range(int(a), int(b) + 1):
                    out.append(f"{pre}{i:0{width}d}{post}")
            else:
                out.append(f"{pre}{part}{post}")
    return out


def _signal_group(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _sig(name: str) -> int:
    name = name.upper()
    if name.isdigit():
        return int(name)
    return getattr(signal, name if name.startswith("SIG") else "SIG" + name)


def _steps(jid: str) -> list[tuple[Path, dict]]:
    d = JOBS / jid / "steps"
    return [(p, _read_json(p)) for p in sorted(d.glob("*.json"))] if d.exists() else []


def _step_running(step: dict) -> bool:
    """RUNNING and at least one task process group still alive (a SIGKILLed
    srun never records its end)."""
    return step.get("state") == "RUNNING" and any(_alive(p) for p in step.get("pgids", []))


def _kill_steps(jid: str, sig: int) -> None:
    for _, step in _steps(jid):
        if _step_running(step):
            for pgid in step.get("pgids", []):
                _signal_group(pgid, sig)


# ----------------------------------------------------------------- sbatch
def cmd_sbatch(argv: list[str]) -> int:
    parsable = False
    cli: dict[str, str] = {}
    script = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--parsable":
            parsable = True
        elif a.startswith("--") and "=" in a:
            k, v = a[2:].split("=", 1)
            cli[k] = v
        elif a.startswith("-") and script is None:
            pass
        else:
            script = a
            break
        i += 1
    if script is None:
        print("sbatch: error: no script", file=sys.stderr)
        return 1
    text = Path(script).read_text()
    opts: dict[str, str] = {}
    for line in text.splitlines():
        m = re.match(r"#SBATCH\s+--([\w-]+)(?:[= ](.*))?$", line.strip())
        if m:
            opts[m.group(1)] = (m.group(2) or "true").strip()
    opts.update(cli)
    JOBS.mkdir(parents=True, exist_ok=True)
    with open(ROOT / "lock", "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        counter = ROOT / "next_id"
        jid = int(counter.read_text()) if counter.exists() else 1000
        counter.write_text(str(jid + 1))
    jid = str(jid)
    d = JOBS / jid
    (d / "steps").mkdir(parents=True)
    (d / "script.sh").write_text(text)
    nodes = int(opts.get("nodes", "1"))
    nodelist = [f"n{jid}-{k}" for k in range(nodes)]
    output = (
        opts.get("output", f"slurm-{jid}.out")
        .replace("%j", jid)
        .replace("%x", opts.get("job-name", "job"))
    )
    if not os.path.isabs(output):
        output = os.path.join(os.getcwd(), output)
    job = {
        "id": jid,
        "name": opts.get("job-name", Path(script).name),
        "account": opts.get("account"),
        "nodes": nodes,
        "nodelist": nodelist,
        "time_s": _parse_time(opts.get("time")),
        "signal": opts.get("signal"),
        "output": output,
        "cwd": os.getcwd(),
        "state": "PENDING",
        "submit_ts": time.time(),
    }
    _write_json(d / "job.json", job)
    if os.environ.get("FAKESLURM_HOLD") == "1":
        (d / "hold").write_text("")
    subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), "_run", jid],
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=open(d / "runner.log", "w"),
        stderr=subprocess.STDOUT,
        env={**os.environ, "FAKESLURM_DIR": str(ROOT)},
    )
    print(jid if parsable else f"Submitted batch job {jid}")
    return 0


def _update(jid: str, **fields) -> dict:
    path = JOBS / jid / "job.json"
    job = _read_json(path)
    job.update(fields)
    _write_json(path, job)
    return job


def cmd_run(argv: list[str]) -> int:
    """Detached per-job runner: pending -> running -> final state."""
    jid = argv[0]
    d = JOBS / jid
    while (d / "hold").exists():
        if (d / "cancel").exists():
            _update(jid, state="CANCELLED", end_ts=time.time())
            return 0
        time.sleep(0.1)
    if (d / "cancel").exists():
        _update(jid, state="CANCELLED", end_ts=time.time())
        return 0
    job = _job(jid)
    env = {
        **os.environ,
        "SLURM_JOB_ID": jid,
        "SLURM_JOBID": jid,
        "SLURM_JOB_NAME": job["name"],
        "SLURM_JOB_NODELIST": ",".join(job["nodelist"]),
        "SLURM_NODELIST": ",".join(job["nodelist"]),
        "SLURM_JOB_NUM_NODES": str(job["nodes"]),
        "SLURM_NNODES": str(job["nodes"]),
        "SLURM_SUBMIT_DIR": job["cwd"],
        "FAKESLURM_DIR": str(ROOT),
    }
    Path(job["output"]).parent.mkdir(parents=True, exist_ok=True)
    out = open(job["output"], "a")
    proc = subprocess.Popen(
        ["bash", str(d / "script.sh")],
        cwd=job["cwd"],
        env=env,
        stdout=out,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    start = time.time()
    _update(jid, state="RUNNING", start_ts=start, batch_pid=proc.pid)
    limit = job.get("time_s")
    sig_spec = job.get("signal")
    sig_batch_only, sig_num, sig_margin, sig_sent = False, None, 0.0, False
    if sig_spec:
        spec = sig_spec
        if spec.startswith("B:"):
            sig_batch_only, spec = True, spec[2:]
        name, _, margin = spec.partition("@")
        sig_num, sig_margin = _sig(name), float(margin or 60)
    final = None
    term_ts = None
    while True:
        rc = proc.poll()
        if rc is not None:
            final = final or ("COMPLETED" if rc == 0 else "FAILED")
            break
        now = time.time()
        if term_ts is None and (d / "cancel").exists():
            final, term_ts = "CANCELLED", now
            out.write(
                f"slurmstepd: error: *** JOB {jid} ON {job['nodelist'][0]} CANCELLED AT {time.strftime('%Y-%m-%dT%H:%M:%S')} ***\n"
            )
            out.flush()
            _signal_group(proc.pid, signal.SIGTERM)
            _kill_steps(jid, signal.SIGTERM)
        if limit is not None:
            if sig_num is not None and not sig_sent and now - start >= limit - sig_margin:
                sig_sent = True
                if sig_batch_only:
                    os.kill(proc.pid, sig_num)
                else:
                    _signal_group(proc.pid, sig_num)
                    _kill_steps(jid, sig_num)
            if term_ts is None and now - start >= limit:
                final, term_ts = "TIMEOUT", now
                out.write(
                    f"slurmstepd: error: *** JOB {jid} ON {job['nodelist'][0]} CANCELLED AT {time.strftime('%Y-%m-%dT%H:%M:%S')} DUE TO TIME LIMIT ***\n"
                )
                out.flush()
                _signal_group(proc.pid, signal.SIGTERM)
                _kill_steps(jid, signal.SIGTERM)
        if term_ts is not None and now - term_ts >= 2.0:
            _signal_group(proc.pid, signal.SIGKILL)
            _kill_steps(jid, signal.SIGKILL)
        time.sleep(0.1)
    # end of the allocation: nothing of it survives (proctrack/cgroup)
    _signal_group(proc.pid, signal.SIGKILL)
    _kill_steps(jid, signal.SIGKILL)
    _update(jid, state=final, rc=proc.returncode, end_ts=time.time())
    return 0


# ----------------------------------------------------------------- srun
_SRUN_VALUE_SHORT = {
    "-N": "nodes",
    "-n": "ntasks",
    "-w": "nodelist",
    "-J": "job-name",
    "-c": "cpus-per-task",
}


def cmd_srun(argv: list[str]) -> int:
    opts: dict[str, str] = {}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--"):
            if "=" in a:
                k, v = a[2:].split("=", 1)
                opts[k] = v
            else:
                opts[a[2:]] = "true"
        elif a in _SRUN_VALUE_SHORT:
            opts[_SRUN_VALUE_SHORT[a]] = argv[i + 1]
            i += 1
        elif a.startswith("-") and len(a) > 2 and a[:2] in _SRUN_VALUE_SHORT:
            opts[_SRUN_VALUE_SHORT[a[:2]]] = a[2:]
        else:
            break
        i += 1
    command = argv[i:]
    jid = opts.get("jobid") or os.environ.get("SLURM_JOB_ID")
    if not jid or not (JOBS / jid).exists():
        print("srun: error: not inside a fakeslurm allocation", file=sys.stderr)
        return 1
    job = _job(jid)
    alloc = job["nodelist"]
    if "nodelist" in opts:
        nodes = expand_hostlist(opts["nodelist"])
        bad = [n for n in nodes if n not in alloc]
        if bad:
            print(f"srun: error: nodes {bad} are not part of job {jid} ({alloc})", file=sys.stderr)
            return 1
    else:
        nodes = alloc[: int(opts.get("nodes", len(alloc)))]
    nnodes = int(opts.get("nodes", len(nodes)))
    if nnodes != len(nodes):
        print(
            f"srun: error: --nodes={nnodes} but {len(nodes)} node(s) in the nodelist",
            file=sys.stderr,
        )
        return 1
    if nnodes > len(alloc):
        print(
            f"srun: error: --nodes={nnodes} exceeds the allocation ({len(alloc)})", file=sys.stderr
        )
        return 1
    tpn = int(opts.get("ntasks-per-node", "1"))
    ntasks = int(opts.get("ntasks", nnodes * tpn))
    if ntasks != nnodes * tpn:
        print(
            f"srun: error: --ntasks={ntasks} != nodes*ntasks-per-node={nnodes * tpn}",
            file=sys.stderr,
        )
        return 1
    overlap = opts.get("overlap") == "true"
    name = opts.get("job-name", Path(command[0]).name if command else "step")

    steps_dir = JOBS / jid / "steps"
    with open(ROOT / "lock", "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        step_id = len(list(steps_dir.glob("*.json")))
        step_path = steps_dir / f"{step_id:05d}.json"
        _write_json(
            step_path,
            {
                "id": f"{jid}.{step_id}",
                "name": name,
                "nodes": nodes,
                "state": "WAITING",
                "overlap": overlap,
                "pgids": [],
            },
        )
    # a step waits for its nodes, like SLURM ("Requested nodes are busy")
    warned = False
    while not overlap:
        busy = [
            s
            for p, s in _steps(jid)
            if p != step_path
            and _step_running(s)
            and not s.get("overlap")
            and set(s.get("nodes", [])) & set(nodes)
        ]
        if not busy:
            break
        if not warned:
            print(
                f"srun: Job {jid} step creation temporarily disabled, retrying (Requested nodes are busy)",
                file=sys.stderr,
                flush=True,
            )
            warned = True
        time.sleep(0.2)
        if _job(jid).get("state") in FINAL:
            return 1

    procs = []
    for node_idx, node in enumerate(nodes):
        for local in range(tpn):
            env = {
                **os.environ,
                "SLURM_JOB_ID": jid,
                "SLURM_STEP_ID": str(step_id),
                "SLURM_STEP_NODELIST": ",".join(nodes),
                "SLURM_STEP_NUM_NODES": str(nnodes),
                "SLURM_NNODES": str(nnodes),
                "SLURM_NODEID": str(node_idx),
                "SLURM_PROCID": str(node_idx * tpn + local),
                "SLURM_LOCALID": str(local),
                "SLURM_NTASKS": str(ntasks),
                "SLURMD_NODENAME": node,
                "FAKESLURM_NODE": node,
            }
            procs.append(subprocess.Popen(command, env=env, start_new_session=True))
    _write_json(
        step_path,
        {
            "id": f"{jid}.{step_id}",
            "name": name,
            "nodes": nodes,
            "state": "RUNNING",
            "overlap": overlap,
            "pgids": [p.pid for p in procs],
        },
    )

    def forward(signum, _frame):
        for p in procs:
            _signal_group(p.pid, signum)

    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1, signal.SIGUSR2):
        signal.signal(s, forward)
    rcs = []
    for p in procs:
        while True:
            try:
                rcs.append(p.wait())
                break
            except InterruptedError:  # pragma: no cover
                continue
    step = _read_json(step_path)
    step["state"] = "COMPLETED"
    _write_json(step_path, step)
    rc = max((r if r >= 0 else 128 - r for r in rcs), default=0)
    return rc


# ----------------------------------------------------------------- queries
def _all_jobs() -> list[dict]:
    return [_read_json(p / "job.json") for p in sorted(JOBS.glob("*"))] if JOBS.exists() else []


def _fmt(fmt: str, row: dict) -> str:
    return re.sub(r"%([a-zA-Z])", lambda m: str(row.get(m.group(1), "")), fmt)


def cmd_squeue(argv: list[str]) -> int:
    jobs_filter = None
    fmt = "%i %T"
    header = True
    steps = False
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--jobs", "-j"):
            jobs_filter = set(argv[i + 1].split(","))
            i += 1
        elif a.startswith("--jobs="):
            jobs_filter = set(a.split("=", 1)[1].split(","))
        elif a in ("--format", "-o"):
            fmt = argv[i + 1]
            i += 1
        elif a.startswith("--format="):
            fmt = a.split("=", 1)[1]
        elif a in ("-h", "--noheader"):
            header = False
        elif a in ("-s", "--steps"):
            steps = True
        i += 1
    rows = []
    for job in _all_jobs():
        if not job or (jobs_filter and job["id"] not in jobs_filter):
            continue
        if job.get("state") not in ("PENDING", "RUNNING"):
            continue
        if steps:
            for _, step in _steps(job["id"]):
                if _step_running(step):
                    rows.append({"i": step["id"], "j": step["name"], "T": "RUNNING"})
            continue
        left = ""
        if job.get("time_s") and job.get("start_ts"):
            rem = max(0, int(job["start_ts"] + job["time_s"] - time.time()))
            left = f"{rem // 3600}:{rem % 3600 // 60:02d}:{rem % 60:02d}"
        rows.append({"i": job["id"], "T": job["state"], "j": job["name"], "L": left})
    if header:
        print(_fmt(fmt, {"i": "JOBID", "T": "STATE", "j": "NAME", "L": "TIME_LEFT"}))
    for row in rows:
        print(_fmt(fmt, row))
    return 0


def cmd_sacct(argv: list[str]) -> int:
    ids = None
    parsable = False
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--jobs", "-j"):
            ids = set(argv[i + 1].split(","))
            i += 1
        elif a in ("--parsable2", "-P"):
            parsable = True
        i += 1
    sep = "|" if parsable else " "
    for job in _all_jobs():
        if job and (ids is None or job["id"] in ids):
            print(f"{job['id']}{sep}{job.get('state')}")
    return 0


def cmd_scancel(argv: list[str]) -> int:
    sig = None
    targets = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--signal="):
            sig = _sig(a.split("=", 1)[1])
        elif a in ("-s", "--signal"):
            sig = _sig(argv[i + 1])
            i += 1
        elif not a.startswith("-"):
            targets.append(a)
        i += 1
    for target in targets:
        if "." in target:
            jid, sid = target.split(".", 1)
            path = JOBS / jid / "steps" / f"{int(sid):05d}.json"
            step = _read_json(path)
            if not _step_running(step):
                continue
            for pgid in step.get("pgids", []):
                _signal_group(pgid, sig if sig is not None else signal.SIGKILL)
            continue
        d = JOBS / target
        if not d.exists():
            print(f"scancel: error: Invalid job id {target}", file=sys.stderr)
            continue
        job = _job(target)
        if sig is not None and job.get("state") == "RUNNING":
            _signal_group(job["batch_pid"], sig)
            _kill_steps(target, sig)
        elif sig is None:
            (d / "cancel").write_text("")
    return 0


def cmd_scontrol(argv: list[str]) -> int:
    if argv[:2] == ["show", "hostnames"]:
        text = argv[2] if len(argv) > 2 else os.environ.get("SLURM_JOB_NODELIST", "")
        for h in expand_hostlist(text):
            print(h)
        return 0
    if argv[:1] == ["update"]:
        return 0
    print(f"scontrol: unsupported {shlex.join(argv)}", file=sys.stderr)
    return 1


def cmd_sinfo(argv: list[str]) -> int:
    """``sinfo -h -N -n <nodes> -o "%N %t"``: node states from
    ``$FAKESLURM_DIR/node_states`` (``<node> <state>`` lines), ``alloc``
    otherwise."""
    nodes: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] in ("-n", "--nodes"):
            nodes = expand_hostlist(argv[i + 1])
            i += 1
        i += 1
    states = {}
    path = ROOT / "node_states"
    if path.exists():
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                states[parts[0]] = parts[1]
    for node in nodes:
        print(f"{node} {states.get(node, 'alloc')}")
    return 0


def cmd_killall(_argv: list[str]) -> int:
    """Test teardown: end every job (batch shells, steps) right now."""
    for job in _all_jobs():
        if not job:
            continue
        (JOBS / job["id"] / "hold").unlink(missing_ok=True)
        (JOBS / job["id"] / "cancel").write_text("")
        if job.get("batch_pid"):
            _signal_group(job["batch_pid"], signal.SIGKILL)
        _kill_steps(job["id"], signal.SIGKILL)
    return 0


def cmd_crash(argv: list[str]) -> int:
    """Test hook: the job's nodes die (SIGKILL everything, no signal, no
    cleanup)."""
    job = _job(argv[0])
    if job.get("batch_pid"):
        _signal_group(job["batch_pid"], signal.SIGKILL)
    _kill_steps(argv[0], signal.SIGKILL)
    return 0


COMMANDS = {
    "_killall": cmd_killall,
    "_crash": cmd_crash,
    "sinfo": cmd_sinfo,
    "sbatch": cmd_sbatch,
    "_run": cmd_run,
    "srun": cmd_srun,
    "squeue": cmd_squeue,
    "sacct": cmd_sacct,
    "scancel": cmd_scancel,
    "scontrol": cmd_scontrol,
}


def main() -> int:
    if not os.environ.get("FAKESLURM_DIR"):
        print("fakeslurm: FAKESLURM_DIR is not set", file=sys.stderr)
        return 2
    return COMMANDS[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    sys.exit(main())

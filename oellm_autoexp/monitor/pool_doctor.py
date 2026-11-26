"""pool_ctl doctor: a consistency check of a pool, and the per-node probe (2026-11-25).

docs/pool_allocation_improvement.md §3.3 / §5.4. ``doctor_report`` only reads (files, and with
``slurm`` squeue / scontrol); the monitor runs it without SLURM as a periodic health check
(PoolHostConfig.health_check_s, <pool>/health.txt). ``probe_report`` starts one ``srun --overlap``
step per node of a host's allocation (pool_ctl doctor --probe): GPU count and InfiniBand ports.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from oellm_autoexp.monitor.pool import (
    PoolLayout,
    _atomic_write,
    env_key_lines,
    held_jobs,
    read_env_file,
)

# A monitor tick older than this means nobody executes requests (pool_ctl uses the same value).
MONITOR_STALE_S = 300.0


def run_slurm(argv: list[str]) -> str:
    return subprocess.run(argv, capture_output=True, text=True).stdout


def _hostnames(nodelist: str, slurm) -> list[str]:
    nodelist = nodelist.strip()
    return [] if not nodelist else slurm(["scontrol", "show", "hostnames", nodelist]).split()


def _loads_or_none(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


# A request older than this was not executed by a monitor.
REQUEST_STALE_S = 300.0
# A live host that lost this many runs (rc host_lost: its supervisor or allocation went away while
# they ran) is a problem: lost hosts are not counted against the jobs (P3, 2026-11-26).
LOST_RUNS_PROBLEM = 3
# A run whose own EXCLUDE covers more than this share of its host's allocation is suspicious
# (11-24: 262 of 264 nodes, "needs 256 node(s), only 2 usable").
EXCLUDE_SHARE_WARN = 0.5


def _node_set(tokens: list[str], slurm) -> set[str]:
    """Node names from exclusion tokens; hostlist ranges expanded (through scontrol) when slurm
    is given, else kept as written."""
    out: set[str] = set()
    for tok in tokens:
        if slurm is not None and "[" in tok:
            out.update(_hostnames(tok, slurm) or [tok])
        elif tok:
            out.add(tok)
    return out


def _tokens(text: str | None) -> list[str]:
    return [t for t in (text or "").replace(",", "\n").split() if t and not t.startswith("#")]


def doctor_report(layout: PoolLayout, slurm=run_slurm, now: float | None = None) -> list[tuple[str, str]]:
    """(level, line) pairs of a read-only consistency check; level is "ok", "info" or "PROBLEM".

    Every check maps to a failure of 2026-11-24 (docs/pool_allocation_improvement.md §3.3):
    stale monitor, unreadable or duplicate live records, info.env vs the live allocation,
    exclusions vs slack, a run's EXCLUDE written twice or covering its host, unexecuted requests.
    """
    now = time.time() if now is None else now
    out: list[tuple[str, str]] = []
    try:
        tick = now - layout.monitor_alive.stat().st_mtime
    except OSError:
        tick = None
    if tick is None or tick > MONITOR_STALE_S:
        out.append(("PROBLEM", f"monitor: last tick {'never' if tick is None else f'{tick:.0f}s ago'}"))
    else:
        out.append(("ok", f"monitor: last tick {tick:.0f}s ago"))

    session_dir = Path(layout.load_meta().get("session_dir", ""))
    corrupt, live_by_job = [], {}
    for path in sorted(session_dir.glob("*.job.json")) if session_dir.is_dir() else []:
        record = _loads_or_none(path)
        if record is None:
            corrupt.append(path.name)
            continue
        definition = record.get("definition") or {}
        if (definition.get("metadata") or {}).get("pool_role") == "host":
            continue
        if (record.get("runtime") or {}).get("final_state"):
            continue
        pool_dir = definition.get("pool_dir")
        if not pool_dir or Path(pool_dir).absolute() != layout.root:
            continue
        key = (definition.get("metadata") or {}).get("pool_singleton") or definition.get("name")
        live_by_job.setdefault(str(key), []).append(record.get("job_id"))
    if corrupt:
        out.append(("PROBLEM", f"records: {len(corrupt)} unreadable: {', '.join(corrupt[:5])}"))
    for key, ids in sorted(live_by_job.items()):
        if len(ids) > 1:
            out.append(("PROBLEM", f"job {key}: {len(ids)} live records {sorted(ids)} (1 expected)"))
    out.append(("info", f"records: {sum(len(v) for v in live_by_job.values())} live run record(s)"))

    pool_excl = _tokens(_read_text(layout.exclude_nodes))
    hosts: dict[str, dict] = {}
    for hdir in sorted(layout.hosts_dir.glob("host*")):
        info = read_env_file(hdir / "info.env")
        if not info or (hdir / "exited").exists():
            continue
        listed = [n for n in (info.get("NODELIST") or "").split(",") if n]
        jid = (info.get("SLURM_JOB_ID") or "").strip()
        live = None
        if slurm is not None and jid:
            text = slurm(["squeue", "-h", "-j", jid, "-o", "%N"]).strip().splitlines()
            live = _hostnames(text[0], slurm) if text and text[0] != "(null)" else None
        alloc = set(live if live else listed)
        excl = _node_set(pool_excl + _tokens(_read_text(hdir / "exclude_nodes")), slurm) & alloc
        hosts[hdir.name] = {"alloc": alloc, "excl": excl, "available": len(alloc) - len(excl)}
        lost = [ln for ln in (_read_text(hdir / "lost_runs") or "").splitlines() if ln.strip()]
        if len(lost) >= LOST_RUNS_PROBLEM:
            out.append(("PROBLEM", f"host {hdir.name}: lost {len(lost)} run(s) (host_lost; "
                        "pool_ctl doctor --probe, or exit-host)"))
        line = f"host {hdir.name} job {jid or '-'}: info.env {len(listed)} node(s)"
        if live is None:
            level = "info"
            line += ", live allocation unknown"
        elif set(listed) - set(live):
            level = "PROBLEM"
            gone = sorted(set(listed) - set(live))
            line += f", live {len(live)}: STALE, gone {','.join(gone[:5])}"
        else:
            level = "ok"
            line += f", live {len(live)}"
        out.append((level, line + f"; excluded {len(excl)}, available {len(alloc) - len(excl)}"))

    for rdir in sorted(layout.runs_dir.iterdir() if layout.runs_dir.exists() else []):
        if (rdir / "rc").exists() or not (rdir / "run.json").exists():
            continue
        text = _read_text(rdir / "meta.env") or ""
        meta = read_env_file(rdir / "meta.env")
        try:
            need = int(meta.get("NODES") or 1)
        except ValueError:
            need = 1
        n_lines = env_key_lines(text, "EXCLUDE")
        if n_lines > 1:
            out.append(("PROBLEM", f"run {rdir.name}: meta.env has {n_lines} EXCLUDE= lines (1 expected)"))
        host = (_read_text(rdir / "assigned") or "").strip()
        if host in hosts:
            h = hosts[host]
            own = _node_set(_tokens(meta.get("EXCLUDE")), slurm) & h["alloc"]
            available = len(h["alloc"]) - len(h["excl"] | own)
            share = len(own) / max(1, len(h["alloc"]))
            level = "PROBLEM" if available < need or share > EXCLUDE_SHARE_WARN else "ok"
            out.append((level, f"run {rdir.name} on {host}: need {need}, own EXCLUDE {len(own)} in the "
                        f"allocation, available {available}, slack {available - need:+d}"))
        elif not host:
            best = max((h["available"] for h in hosts.values()), default=None)
            if best is not None and best < need:
                out.append(("PROBLEM", f"run {rdir.name} (queued): need {need}, the largest host has "
                            f"{best} available"))

    held_warn_s = layout.load_config().held_warn_s
    for payload in held_jobs(layout) if held_warn_s > 0 else []:
        md = ((payload.get("definition") or {}).get("metadata") or {})
        by = md.get("pool_submitted_by") or {}
        since = float(by.get("ts") or now)
        if now - since > held_warn_s:
            out.append(("PROBLEM", f"held >{held_warn_s / 3600:g}h: {payload.get('job_id')} by "
                        f"{by.get('user', '?')} since {time.strftime('%m-%d %H:%M', time.localtime(since))} "
                        f"(pool_ctl permit {payload.get('job_id')} | deny)"))
    stale = [
        p.name
        for p in (layout.requests_dir.glob("*.json") if layout.requests_dir.exists() else [])
        if now - p.stat().st_mtime > REQUEST_STALE_S
    ]
    if stale:
        out.append(("PROBLEM", f"requests: {len(stale)} not executed for >{REQUEST_STALE_S:.0f}s: {stale[:3]}"))
    return out


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text()
    except OSError:
        return None


# ---------------------------------------------------------------- periodic health check
def health_check(layout: PoolLayout, previous: set[str] | None = None) -> tuple[set[str], list[str]]:
    """The monitor's health check: doctor_report without SLURM calls (the monitor tracks the live
    allocations itself, alloc_check_s). Writes <pool>/health.txt (time + every line) and returns
    (the PROBLEM lines, the lines that are new since ``previous``)."""
    report = [(level, line) for level, line in doctor_report(layout, slurm=None)
              if not line.startswith("monitor:")]  # the monitor asking about itself
    problems = {line for level, line in report if level == "PROBLEM"}
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    text = f"# health check {stamp}: {len(problems)} problem(s) (pool_ctl doctor for details)\n"
    text += "".join(f"{level:<8} {line}\n" for level, line in report)
    try:
        _atomic_write(layout.root / "health.txt", text)
    except OSError:
        pass
    return problems, sorted(problems - (previous or set()))


def read_health(layout: PoolLayout) -> tuple[str, int, float | None]:
    """(first line, problem count, age_s) of the last health check; ("", 0, None) if none."""
    path = layout.root / "health.txt"
    try:
        text = path.read_text()
        age = time.time() - path.stat().st_mtime
    except OSError:
        return "", 0, None
    lines = text.splitlines()
    return (lines[0] if lines else ""), sum(1 for ln in lines if ln.startswith("PROBLEM")), age


# ---------------------------------------------------------------- per-node probe (§5.4)
# One step per node: a chunked step fails as a whole and names no node. --overlap --mem=0 is
# the step an allocation shares with running runs (phase-0 probe 2026-10-27); no --gres, so it
# works on nodes whose GPUs a run holds (nvidia-smi only reads).
PROBE_CMD = (
    'echo "GPUS $(nvidia-smi -L 2>/dev/null | grep -c ^GPU)"; n=0; a=0; '
    "for s in /sys/class/infiniband/*/ports/*/state; do [ -e \"$s\" ] || continue; n=$((n+1)); "
    'grep -q ACTIVE "$s" && a=$((a+1)); done; echo "IB $a/$n"'
)


def probe_node(job_id: str, node: str, timeout_s: float = 120.0) -> str:
    try:
        proc = subprocess.run(
            ["srun", "--overlap", "--mem=0", f"--jobid={job_id}", f"--nodelist={node}",
             "--nodes=1", "--ntasks=1", "--cpus-per-task=1", "--job-name=doctor-probe",
             "bash", "-c", PROBE_CMD],
            capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except OSError as exc:
        return f"ERROR {exc}"
    return proc.stdout + proc.stderr


def parse_probe(text: str) -> tuple[int | None, int | None, int | None]:
    """(gpus, active IB ports, IB ports) from a probe's output; None where it did not answer."""
    gpus = active = total = None
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "GPUS" and parts[1].isdigit():
            gpus = int(parts[1])
        elif len(parts) == 2 and parts[0] == "IB" and "/" in parts[1]:
            a, n = parts[1].split("/", 1)
            if a.isdigit() and n.isdigit():
                active, total = int(a), int(n)
    return gpus, active, total


def probe_report(
    layout: PoolLayout,
    *,
    hosts: list[str] | None = None,
    gpus: int = 4,
    parallel: int = 16,
    slurm=run_slurm,
    probe=None,
) -> list[tuple[str, str]]:
    """Probe every node of the pool's live hosts (or ``hosts``), one step per node. A node is a
    PROBLEM with fewer than ``gpus`` GPUs, fewer ACTIVE InfiniBand ports than the host's usual
    count (the fleet norm: mlx5_3 is DOWN on every MN5 node, 5 of 6 ACTIVE is normal), or no
    answer. Writes <host>/doctor_probe.txt; excludes nothing (that stays a decision)."""
    probe = probe or probe_node
    out: list[tuple[str, str]] = []
    for hdir in sorted(layout.hosts_dir.glob("host*")):
        if hosts and hdir.name not in hosts:
            continue
        info = read_env_file(hdir / "info.env")
        jid = (info.get("SLURM_JOB_ID") or "").strip()
        if not info or not jid or (hdir / "exited").exists():
            continue
        if (hdir / "control" / "freeze").exists():
            # a step running during a resize keeps its GPUs reserved afterwards (pool_ctl shrink)
            out.append(("info", f"host {hdir.name}: frozen (resize prepared), not probed"))
            continue
        text = slurm(["squeue", "-h", "-j", jid, "-o", "%N"]).strip().splitlines()
        nodes = _hostnames(text[0], slurm) if text and text[0] != "(null)" else []
        if not nodes:
            out.append(("info", f"host {hdir.name} job {jid}: no live allocation, not probed"))
            continue
        with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
            answers = dict(zip(nodes, pool.map(lambda n: probe(jid, n), nodes)))
        parsed = {n: parse_probe(a) for n, a in answers.items()}
        norm = Counter(p[1] for p in parsed.values() if p[1] is not None).most_common(1)
        ib_norm = norm[0][0] if norm else None
        bad = []
        for node, (g, a, n) in parsed.items():
            why = []
            if g is None:
                why.append(f"no answer ({answers[node].strip().splitlines()[-1][:80] if answers[node].strip() else 'empty'})")
            elif g < gpus:
                why.append(f"{g} GPU(s), expected {gpus}")
            if g is not None and ib_norm is not None and (a is None or a < ib_norm):
                why.append(f"IB {a}/{n} ACTIVE, host norm {ib_norm}")
            if why:
                bad.append((node, "; ".join(why)))
        lines = [f"{n} gpus={parsed[n][0]} ib={parsed[n][1]}/{parsed[n][2]}" for n in nodes]
        try:
            _atomic_write(hdir / "doctor_probe.txt", f"# {time.strftime('%Y-%m-%dT%H:%M:%S')} job "
                          f"{jid}\n" + "\n".join(lines) + "\n")
        except OSError:
            pass
        out.append(("ok" if not bad else "PROBLEM",
                    f"host {hdir.name} job {jid}: probed {len(nodes)} node(s), {len(bad)} bad"
                    + (f", IB norm {ib_norm} ACTIVE" if ib_norm is not None else "")))
        out += [("PROBLEM", f"  {node}: {why}") for node, why in bad]
    return out

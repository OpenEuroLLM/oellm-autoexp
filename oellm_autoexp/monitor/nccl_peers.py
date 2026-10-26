"""Bounded, opt-in post-failure node attribution; no scheduler mutations.

Run as a restart pre-command or via RestartAction.nccl_peer_scan. Four rules exclude a node:

corroborated_down_peer  A warning alone never excludes its reporter. Only a uniquely
                        mapped allocated peer with a fresh DOWN+NOT_RESPONDING snapshot
                        and incident-window LastBusyTime qualifies.
multi_path_peer         The same allocated peer, reached over at least two distinct
                        (reporter, local HCA) paths, failed with IBV_WC_RETRY_EXC_ERR in
                        the attempt window. A broken local HCA fails on that HCA only;
                        several local HCAs failing towards one remote address put the
                        fault on the remote side (job 2454305: jpbo-037-01 mlx5_0 and
                        mlx5_3 -> jpbo-033-32-interconnect-1, peer never went DOWN).
missing_nccl_sidecar    An allocated host that wrote no NCCL_DEBUG_FILE (nccl-<host>.log)
                        during the failed attempt, while nearly all others did, and whose
                        NCCL output instead reached the job log on stderr ("NCCL version"
                        fallback lines, at least one per missing host). Such a node cannot
                        create files on the shared filesystem, and its contribution to the
                        first collective was garbage: job 2251402 and the five restarts
                        2256442..2256692 (2026-10-09/10) all died loading iter_<garbage>
                        with exactly these hosts silent (jpbo-007-44/48 every time).
fatal_rank_signature    A rank logged a node-local hardware/filesystem fault (GPU contained
                        error, uncorrectable ECC, a data shard its node cannot see), at
                        most FATAL_HOST_CAP hosts did, and the host the rank maps to
                        (rank // ranks-per-node over the ordered allocation) is
                        corroborated by Slurm: srun reported that host's task exiting with
                        an error, or the site health check drained it after the attempt.
                        Jobs 2347262/2347802 (shard), 2387082, 2391569, 2400415, 2500294,
                        2500813, 2502095 (GPU) -- one host each, every time.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time

WARNING = re.compile(
    r"^\[(?P<time>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] "
    r"(?P<reporter>[\w.-]+):\d+:\d+ .*NCCL WARN NET/IB: "
    r"Got completion from peer (?P<ip>[0-9a-fA-F:.]+)<\d+> "
    r"with status=IBV_WC_RETRY_EXC_ERR\(12\)(?:.*\bhca (?P<hca>\S+))?"
)
FAILED = {"FAILED", "NODE_FAIL", "CANCELLED", "TIMEOUT", "BOOT_FAIL", "PREEMPTED"}
EXCLUDING_RULES = (
    "corroborated_down_peer",
    "multi_path_peer",
    "missing_nccl_sidecar",
    "fatal_rank_signature",
)
NCCL_FALLBACK = re.compile(r"NCCL version \d")
# missing_nccl_sidecar: NCCL must demonstrably have started across the job.
SIDECAR_MIN_FRESH = 0.95
# IB warnings whose peer is mapped, allocated and in the window, and that only failed the
# DOWN+NOT_RESPONDING check: multi_path_peer may still apply.
PEER_ALIVE = {
    "peer_not_down_and_unresponsive",
    "peer_outage_not_in_attempt_window",
    "missing_peer_outage_time",
}
# Node-local faults: a healthy node does not produce these, whatever its peers do.
FATAL_SIGNATURE = re.compile(
    r"\[rank(?P<rank>\d+)\]:.*?(?P<sig>Invalid access of peer GPU memory over nvlink"
    r"|uncorrectable ECC error|\.idx and \.bin files cannot be found)"
)
SRUN_ERROR = re.compile(
    r"srun: error: (?P<host>[\w.-]+): task (?P<task>\d+): Exited with exit code"
)
WORLD_SIZE = re.compile(r"using world size: (\d+)")
# More hosts than this with a node-local signature is a cascade or a bug, not a node.
FATAL_HOST_CAP = 2
INTERCONNECT_SUFFIX = re.compile(r"-interconnect-\d+$")


def stamp(value):
    # NCCL and Slurm timestamps must use the same host-local timezone.
    return datetime.fromisoformat(value).timestamp()


def command(argv):
    return subprocess.run(
        argv, check=True, capture_output=True, text=True, timeout=10
    ).stdout.strip()


def snapshot(job_id):
    if not re.fullmatch(r"\d+", job_id):
        raise ValueError("expected numeric Slurm allocation ID")
    rows = command(
        ["sacct", "-X", "-n", "-P", "-j", job_id, "-o", "JobIDRaw,State%40,Start,End,NodeList%8000"]
    ).splitlines()
    matches = [r.split("|") for r in rows if r.split("|")[0].strip() == job_id]
    if len(matches) != 1:
        raise ValueError("missing or ambiguous Slurm allocation")
    _, state, start, end, hostlist, *_ = matches[0]
    if state.split()[0] not in FAILED:
        return {"job_id": job_id, "state": state, "hosts": []}, {}
    hosts = command(["scontrol", "show", "hostnames", hostlist]).splitlines()
    if not hosts or len(hosts) != len(set(hosts)):
        raise ValueError("missing or ambiguous allocation hosts")
    job = dict(job_id=job_id, state=state, start=stamp(start), end=stamp(end), hosts=hosts)
    if job["end"] < job["start"]:
        raise ValueError("invalid allocation time window")
    # One scheduler query, rather than hundreds of per-node requests.
    nodes = {}
    for line in command(["scontrol", "show", "nodes", "--oneliner"]).splitlines():
        fields = dict(re.findall(r"(\w+)=([^\s]+)", line))
        if fields.get("NodeName") in hosts:
            nodes[fields["NodeName"]] = fields
    return job, nodes


def reverse_lookup(ip):
    """Host name for an IB address.

    Slurm's NodeAddr is the management address; NCCL reports the
    interconnect one (jpbo-033-32-interconnect-1.jupiter.internal).
    """
    try:
        name = socket.gethostbyaddr(ip)[0]
    except (OSError, UnicodeError):
        return None
    return INTERCONNECT_SUFFIX.sub("", name.split(".")[0])


def peers_for(ip, addresses, hosts):
    peers = addresses.get(ip)
    if peers:
        return peers
    name = reverse_lookup(ip)
    return [name] if name in hosts else []


def scan(log_root, job, nodes, cap=4, job_log=None):
    result = dict(
        job=job, scanned_at=datetime.now().isoformat(), files=0, bytes=0, events=[], exclude=[]
    )
    if job["state"].split()[0] not in FAILED:
        return result
    addresses = {}
    for name, info in nodes.items():
        try:
            address = str(ipaddress.ip_address(info.get("NodeAddr", "")))
        except ValueError:
            continue
        addresses.setdefault(address, []).append(name)
    resolved = {}
    # Only direct sidecars from this allocation. No recursive/glob-wide scan.
    for reporter in job["hosts"]:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", reporter):
            raise ValueError("unsafe node name")
        path = Path(log_root) / f"nccl-{reporter}.log"
        if not path.exists():
            continue
        size = path.stat().st_size
        result["files"] += 1
        result["bytes"] += size
        if size > 16 * 1024**2 or result["bytes"] > 128 * 1024**2:
            raise ValueError("NCCL scan size limit exceeded; no exclusions applied")
        with path.open(errors="replace") as stream:
            for number, line in enumerate(stream, 1):
                m = WARNING.search(line)
                if not m:
                    continue
                reason = "corroborated_down_peer"
                ip = str(ipaddress.ip_address(m["ip"]))
                if ip not in resolved:
                    resolved[ip] = peers_for(ip, addresses, job["hosts"])
                peers = resolved[ip]
                peer = peers[0] if len(peers) == 1 else None
                info = nodes.get(peer, {})
                when = stamp(m["time"])
                if m["reporter"].split(".")[0] != reporter.split(".")[0]:
                    reason = "reporter_filename_mismatch"
                elif not job["start"] <= when <= job["end"] + 120:
                    reason = "outside_attempt_window"
                elif peer is None or peer not in job["hosts"] or peer == reporter:
                    reason = "peer_not_uniquely_mapped_to_other_allocated_node"
                elif not {"DOWN", "NOT_RESPONDING"} <= set(info.get("State", "").split("+")):
                    reason = "peer_not_down_and_unresponsive"
                else:
                    try:
                        busy = stamp(info.get("LastBusyTime", ""))
                        if not job["start"] <= busy <= job["end"] + 120:
                            reason = "peer_outage_not_in_attempt_window"
                    except ValueError:
                        reason = "missing_peer_outage_time"
                event = dict(
                    rule=reason,
                    node=peer,
                    reporter=reporter,
                    peer_ip=m["ip"],
                    hca=m["hca"],
                    log_path=str(path),
                    log_line_no=number,
                    log_line=line.strip(),
                    scheduler_node=info,
                )
                event["event_id"] = hashlib.sha256(
                    f"{job['job_id']}:{path}:{number}:{line}".encode()
                ).hexdigest()[:16]
                result["events"].append(event)
    paths = {}
    for e in result["events"]:
        if e["rule"] in PEER_ALIVE and e["hca"]:
            paths.setdefault(e["node"], set()).add((e["reporter"], e["hca"]))
    for e in result["events"]:
        if e["rule"] in PEER_ALIVE and len(paths.get(e["node"], ())) >= 2:
            e["rule"] = "multi_path_peer"
    if job_log is not None:
        result["events"].extend(missing_sidecars(log_root, job, nodes, job_log))
        result["events"].extend(fatal_signatures(job, nodes, job_log))
    result["exclude"] = sorted(
        {e["node"] for e in result["events"] if e["rule"] in EXCLUDING_RULES}
    )
    if len(result["exclude"]) > cap:
        result["exclude"] = []
        result["refused"] = "too many peers; requires review"
    return result


def missing_sidecars(log_root, job, nodes, job_log):
    """Events for allocated hosts without an NCCL_DEBUG_FILE written in this
    attempt."""
    hosts = job["hosts"]
    window = (job["start"], job["end"] + 120)
    fresh, missing = [], []
    for host in hosts:
        path = Path(log_root) / f"nccl-{host}.log"
        try:
            written = window[0] <= path.stat().st_mtime <= window[1]
        except OSError:
            written = False
        (fresh if written else missing).append(host)
    if not missing or len(fresh) < SIDECAR_MIN_FRESH * len(hosts):
        return []
    fallback = []
    path = Path(job_log)
    if path.is_file():
        with path.open(errors="replace") as stream:
            fallback = [
                (n, line.strip()) for n, line in enumerate(stream, 1) if NCCL_FALLBACK.search(line)
            ]
    corroborated = len(fallback) >= len(missing)
    events = []
    for host in missing:
        event = {
            "rule": "missing_nccl_sidecar" if corroborated else "missing_sidecar_uncorroborated",
            "node": host,
            "reporter": None,
            "peer_ip": None,
            "log_path": str(Path(log_root) / f"nccl-{host}.log"),
            "log_line_no": None,
            "log_line": f"{len(fresh)}/{len(hosts)} hosts wrote their NCCL log; "
            f"{len(fallback)} stderr fallback line(s) in {path}",
            "fallback_lines": fallback[:8],
            "scheduler_node": nodes.get(host, {}),
        }
        event["event_id"] = hashlib.sha256(f"{job['job_id']}:sidecar:{host}".encode()).hexdigest()[
            :16
        ]
        events.append(event)
    return events


def health_drained(info, job):
    """The site health check drained the node at the end of this attempt
    (JUPITER: Reason=pshealthcheck ..., set in the job epilogue)."""
    if "DRAIN" not in info.get("State", "") or not info.get("Reason", "").startswith(
        "pshealthcheck"
    ):
        return False
    try:
        busy = stamp(info.get("LastBusyTime", ""))
    except ValueError:
        return False
    return job["start"] <= busy <= job["end"] + 900


def fatal_signatures(job, nodes, job_log):
    """Events for hosts whose ranks logged a node-local fault."""
    path = Path(job_log)
    if not path.is_file():
        return []
    hosts = job["hosts"]
    world, ranks, srun = None, {}, set()
    with path.open(errors="replace") as stream:
        for number, line in enumerate(stream, 1):
            if world is None and (m := WORLD_SIZE.search(line)):
                world = int(m[1])
            if m := FATAL_SIGNATURE.search(line):
                ranks.setdefault(int(m["rank"]), (number, line.strip(), m["sig"]))
            if m := SRUN_ERROR.search(line):
                task = int(m["task"])
                if task < len(hosts) and hosts[task] == m["host"]:
                    srun.add(m["host"])
    if not ranks:
        return []
    per_node = world // len(hosts) if world and world % len(hosts) == 0 else None
    by_host = {}
    for rank, evidence in sorted(ranks.items()):
        host = hosts[rank // per_node] if per_node and rank < world else None
        by_host.setdefault(host, (rank, *evidence))
    cascade = None in by_host or len(by_host) > FATAL_HOST_CAP
    events = []
    for host, (rank, number, line, sig) in by_host.items():
        info = nodes.get(host, {})
        drained = health_drained(info, job)
        if cascade:
            rule = "fatal_rank_signature_unattributed"
        elif host in srun or drained:
            rule = "fatal_rank_signature"
        else:
            rule = "fatal_rank_signature_uncorroborated"
        event = {
            "rule": rule,
            "node": host,
            "reporter": None,
            "peer_ip": None,
            "rank": rank,
            "signature": sig,
            "srun_error": host in srun,
            "health_drained": drained,
            "log_path": str(path),
            "log_line_no": number,
            "log_line": line[:500],
            "scheduler_node": info,
        }
        event["event_id"] = hashlib.sha256(
            f"{job['job_id']}:fatal:{host}:{rank}".encode()
        ).hexdigest()[:16]
        events.append(event)
    return events


def apply(result, exclusion_file, audit_file):
    """Append with a bounded advisory lock; never rewrite existing
    exclusions."""
    if not result["events"]:
        return []
    if not Path(exclusion_file).is_file():
        raise ValueError("refuse to create a missing shared exclusion list")
    with open(exclusion_file, "a+") as exclusions:
        deadline = time.monotonic() + 3
        while True:
            try:
                fcntl.flock(exclusions, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("exclusion lock busy")
                time.sleep(0.05)
        exclusions.seek(0)
        present = {s.strip() for s in exclusions if s.strip() and not s.startswith("#")}
        additions = sorted(set(result["exclude"]) - present)
        # Evidence is durable before the exclusion changes. Repeated scans may
        # append audit observations but never duplicate exclusion entries.
        with open(audit_file, "a") as audit:
            audit.write(json.dumps(dict(result, newly_excluded=additions)) + "\n")
            audit.flush()
            os.fsync(audit.fileno())
        for node in additions:
            evidence = next(
                e for e in result["events"] if e["node"] == node and e["rule"] in EXCLUDING_RULES
            )
            exclusions.write(
                f"\n# autoexp {evidence['rule']} event {evidence['event_id']} job {result['job']['job_id']}; evidence {audit_file}\n{node}\n"
            )
        exclusions.flush()
        os.fsync(exclusions.fileno())
        return additions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    parser.add_argument("--log-root", required=True, type=Path)
    parser.add_argument("--exclude-file", required=True, type=Path)
    parser.add_argument("--audit-file", type=Path)
    parser.add_argument(
        "--job-log", type=Path, help="the failed attempt's log (missing_nccl_sidecar rule)"
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    job, nodes = snapshot(args.job)
    result = scan(args.log_root, job, nodes, job_log=args.job_log)
    if args.apply:
        result["newly_excluded"] = apply(
            result, args.exclude_file, args.audit_file or args.log_root / "nccl-peer-events.jsonl"
        )
    print(json.dumps(result))


if __name__ == "__main__":
    main()

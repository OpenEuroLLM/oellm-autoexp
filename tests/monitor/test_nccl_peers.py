"""Historical evidence replay and real restart lifecycle with fake scheduler
I/O."""

import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from oellm_autoexp.monitor import nccl_peers as peers
from oellm_autoexp.monitor.actions import (
    ActionContext,
    EventRecord,
    RestartAction,
    RestartActionConfig,
)
from oellm_autoexp.monitor.loop import JobRecord, JobRuntime, MonitorLoop
from oellm_autoexp.slurm_gen.generator import build_sbatch_directives
from oellm_autoexp.slurm_gen.schema import SlurmConfig, SbatchConfig

EVIDENCE = json.loads((Path(__file__).parent / "fixtures/nccl_peer_1725379.json").read_text())


def setup(tmp_path):
    e = EVIDENCE
    job = dict(
        job_id=e["job_id"],
        state=e["state"],
        start=peers.stamp(e["start"]),
        end=peers.stamp(e["end"]),
        hosts=[e["reporter"], e["peer"]],
    )
    nodes = {
        e["peer"]: copy.deepcopy(e["node"]),
        e["reporter"]: dict(NodeName=e["reporter"], NodeAddr="10.0.0.1", State="IDLE"),
    }
    (tmp_path / f"nccl-{e['reporter']}.log").write_text(e["warning"] + "\n")
    return job, nodes


def test_exact_historical_warning_excludes_peer_not_reporter(tmp_path):
    job, nodes = setup(tmp_path)
    result = peers.scan(tmp_path, job, nodes)
    assert result["exclude"] == ["jpbo-071-42"]
    exclusions = tmp_path / "exclude"
    exclusions.write_text("# preserve\njpbo-001-01\n")
    audit = tmp_path / "events.jsonl"
    assert peers.apply(result, exclusions, audit) == ["jpbo-071-42"]
    assert peers.apply(result, exclusions, audit) == []
    assert exclusions.read_text().startswith("# preserve\njpbo-001-01\n")
    assert exclusions.read_text().splitlines().count("jpbo-071-42") == 1
    assert "jpbo-022-09" not in exclusions.read_text()
    assert (
        json.loads(audit.read_text().splitlines()[0])["events"][0]["log_line"]
        == EVIDENCE["warning"]
    )


@pytest.mark.parametrize(
    "case",
    [
        "healthy",
        "down_only",
        "unresponsive_only",
        "old_busy",
        "missing_busy",
        "foreign_peer",
        "ambiguous_ip",
        "old_warning",
        "late_warning",
        "wrong_reporter",
        "completed",
        "running",
        "cuda_cleanup",
    ],
)
def test_insufficient_or_stale_evidence_never_excludes(tmp_path, case):
    job, nodes = setup(tmp_path)
    node = nodes[EVIDENCE["peer"]]
    if case == "healthy":
        node["State"] = "IDLE"
    if case == "down_only":
        node["State"] = "DOWN"
    if case == "unresponsive_only":
        node["State"] = "ALLOCATED+NOT_RESPONDING"
    if case == "old_busy":
        node["LastBusyTime"] = "2026-09-08T10:40:21"
    if case == "missing_busy":
        node.pop("LastBusyTime")
    if case == "foreign_peer":
        job["hosts"].remove(EVIDENCE["peer"])
    if case == "ambiguous_ip":
        nodes[EVIDENCE["reporter"]]["NodeAddr"] = node["NodeAddr"]
    if case == "old_warning":
        job["start"] = job["end"] + 100
    if case == "late_warning":
        job["end"] -= 200
    if case == "completed":
        job["state"] = "COMPLETED"
    if case == "running":
        job["state"] = "RUNNING"
    p = tmp_path / f"nccl-{EVIDENCE['reporter']}.log"
    if case == "wrong_reporter":
        p.write_text(EVIDENCE["warning"].replace("jpbo-022-09", "jpbo-090-25"))
    if case == "cuda_cleanup":
        p.write_text(
            "[2026-09-09 19:17:41] jpbo-090-25:366286:375408 [1] misc/strongstream.cc:403 NCCL WARN Cuda failure 'unspecified launch failure'\n"
        )
    assert peers.scan(tmp_path, job, nodes)["exclude"] == []


def test_mass_failure_requires_review(tmp_path):
    job, nodes = setup(tmp_path)
    result = peers.scan(tmp_path, job, nodes, cap=0)
    assert result["exclude"] == [] and result["refused"]


def test_missing_exclusion_file_or_audit_failure_does_not_create_exclusions(tmp_path):
    job, nodes = setup(tmp_path)
    result = peers.scan(tmp_path, job, nodes)
    f = tmp_path / "missing"
    with pytest.raises(ValueError):
        peers.apply(result, f, tmp_path / "audit")
    f.write_text("jpbo-001-01\n")
    with pytest.raises(OSError):
        peers.apply(result, f, tmp_path / "absent" / "audit")
    assert f.read_text() == "jpbo-001-01\n"


def fake_scheduler(tmp_path, monkeypatch):
    e = EVIDENCE
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    responses = {
        "sacct": f"{e['job_id']}|FAILED|{e['start']}|{e['end']}|{e['reporter']},{e['peer']}|\n",
        "hosts": e["reporter"] + "\n" + e["peer"] + "\n",
        "nodes": " ".join(f"{k}={v}" for k, v in e["node"].items())
        + "\n"
        + f"NodeName={e['reporter']} NodeAddr=10.0.0.1 State=IDLE\n",
    }
    for name in ["sacct", "scontrol"]:
        p = bin_dir / name
        p.write_text(
            f'#!{sys.executable}\nimport sys\ndata={responses!r}\nprint(data["sacct" if {name!r}=="sacct" else "hosts" if "hostnames" in sys.argv else "nodes"],end="")\n'
        )
        p.chmod(0o755)
    import os

    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])


@pytest.mark.parametrize("fault", [None, "timeout", "error"])
def test_cancel_teardown_real_peer_hook_then_render_excludes(tmp_path, monkeypatch, fault):
    setup(tmp_path)
    fake_scheduler(tmp_path, monkeypatch)
    exclude = tmp_path / "exclude"
    exclude.write_text("jpbo-001-01\n")
    logs = tmp_path / "logs"
    logs.mkdir()
    log = logs / f"slurm-{EVIDENCE['job_id']}.log"
    log.touch()
    slurm = SlurmConfig(
        name="fake",
        template_path=str(tmp_path / "unused"),
        script_dir=str(tmp_path),
        log_dir=str(logs),
        exclude_file=str(exclude),
        sbatch=SbatchConfig(),
    )
    definition = SimpleNamespace(slurm=slurm)
    job = JobRecord(
        job_id="fake",
        definition=definition,
        runtime=JobRuntime(
            submitted=True, runtime_job_id=EVIDENCE["job_id"], last_status="RUNNING", attempts=1
        ),
    )
    loop = MonitorLoop.__new__(MonitorLoop)
    loop._pending_restart_hooks = {}
    actions = []
    loop._store = SimpleNamespace(upsert=lambda j: None)
    loop._get_client = lambda j: SimpleNamespace(
        cancel=lambda rid: actions.append("cancel"), remove=lambda rid: None
    )
    loop._build_job_metadata = lambda j: {}
    loop._resolve_log_path = lambda j: log
    loop._restart_job = lambda j: actions.append("\n".join(build_sbatch_directives(slurm)))
    cfg = RestartActionConfig(cancel_first=True, nccl_peer_scan=True, exclude_file=str(exclude))
    res = RestartAction(cfg).execute(
        ActionContext(event=EventRecord(event_id="x", name="x", source="log"), job_metadata={})
    )
    assert res.metadata["nccl_peer_scan"] is True
    loop._pending_restart_hooks[job.job_id] = res.metadata
    assert loop._apply_effect(job, "restart", EVIDENCE["job_id"])
    assert actions == ["cancel"] and not (tmp_path / "nccl-peer-events.jsonl").exists()
    assert loop._resume_deferred_restart(job, EVIDENCE["job_id"])
    assert actions == ["cancel"]
    job.runtime.last_status = "CANCELLED"
    if fault:

        def fail(*args, **kwargs):
            if fault == "timeout":
                raise subprocess.TimeoutExpired("scan", 60)
            raise OSError("scanner unavailable")

        monkeypatch.setattr(subprocess, "run", fail)
    assert loop._resume_deferred_restart(job, EVIDENCE["job_id"])
    assert len(actions) == 2 and actions[0] == "cancel"
    assert ("jpbo-071-42" in actions[1]) == (fault is None)
    assert "jpbo-022-09" not in actions[1]
    assert job.runtime.deferred_restart == {}


def test_size_limit_fails_without_partial_apply(tmp_path):
    job, nodes = setup(tmp_path)
    with (tmp_path / f"nccl-{EVIDENCE['peer']}.log").open("wb") as f:
        f.truncate(17 * 1024**2)
    with pytest.raises(ValueError, match="size limit"):
        peers.scan(tmp_path, job, nodes)


def test_busy_exclusion_lock_times_out_without_writing(tmp_path, monkeypatch):
    import fcntl

    job, nodes = setup(tmp_path)
    result = peers.scan(tmp_path, job, nodes)
    exclusions = tmp_path / "exclude"
    exclusions.write_text("jpbo-001-01\n")
    with exclusions.open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        ticks = iter([0, 4])
        monkeypatch.setattr(peers.time, "monotonic", lambda: next(ticks))
        with pytest.raises(TimeoutError):
            peers.apply(result, exclusions, tmp_path / "audit")
    assert exclusions.read_text() == "jpbo-001-01\n"
    assert not (tmp_path / "audit").exists()


def test_snapshot_refuses_missing_accounting_without_exclusion(monkeypatch):
    monkeypatch.setattr(peers, "command", lambda argv: "")
    with pytest.raises(ValueError, match="allocation"):
        peers.snapshot("1725379")
    with pytest.raises(ValueError, match="numeric"):
        peers.snapshot("1725379; echo bad")


# --- missing_nccl_sidecar: jobs 2251402 / 2256442..2256692 (2026-10-09/10) -----------------
# The faulty nodes wrote no nccl-<host>.log (NCCL fell back to stderr: one "NCCL version"
# line each in the job log) while the other 510 of 512 hosts did.
def sidecar_job(tmp_path, n=40, missing=("jpbo-007-44", "jpbo-007-48"), fallback=2, stale=()):
    hosts = [f"jpbo-0{10 + i // 16:02d}-{i % 16 + 1:02d}" for i in range(n - len(missing))] + list(
        missing
    )
    job = {
        "job_id": "2256692",
        "state": "FAILED",
        "start": 1_000_000.0,
        "end": 1_000_300.0,
        "hosts": hosts,
    }
    import os

    for h in hosts:
        if h in missing and h not in stale:
            continue
        p = tmp_path / f"nccl-{h}.log"
        p.write_text("NCCL version 2.29.3+cuda13.1\n")
        t = job["start"] - 3600 if h in stale else job["start"] + 60
        os.utime(p, (t, t))
    log = tmp_path / "slurm-2256692.log"
    log.write_text(
        "[default0]: loading\n"
        + "[default0]:NCCL version 2.29.3+cuda13.1\n" * fallback
        + "[default2]:WARNING: on rank 86 found iteration 100000 ...\n"
    )
    nodes = {h: {"NodeName": h, "State": "ALLOCATED"} for h in hosts}
    return job, nodes, log


def test_missing_sidecar_with_stderr_fallback_excludes_exactly_those_hosts(tmp_path):
    job, nodes, log = sidecar_job(tmp_path)
    result = peers.scan(tmp_path, job, nodes, job_log=log)
    assert result["exclude"] == ["jpbo-007-44", "jpbo-007-48"]
    assert {e["rule"] for e in result["events"]} == {"missing_nccl_sidecar"}
    exclusions = tmp_path / "exclude"
    exclusions.write_text("# keep\njpbo-001-01\n")
    assert peers.apply(result, exclusions, tmp_path / "audit.jsonl") == [
        "jpbo-007-44",
        "jpbo-007-48",
    ]
    text = exclusions.read_text()
    assert (
        text.startswith("# keep\njpbo-001-01\n") and "# autoexp missing_nccl_sidecar event" in text
    )
    assert peers.apply(result, exclusions, tmp_path / "audit.jsonl") == []


def test_a_sidecar_left_by_an_earlier_attempt_does_not_count(tmp_path):
    job, nodes, log = sidecar_job(tmp_path, stale=("jpbo-007-44",))
    assert peers.scan(tmp_path, job, nodes, job_log=log)["exclude"] == [
        "jpbo-007-44",
        "jpbo-007-48",
    ]


@pytest.mark.parametrize(
    "case", ["no_job_log", "uncorroborated", "nccl_never_started", "all_stale"]
)
def test_missing_sidecar_needs_both_signals(tmp_path, case):
    kw = {"fallback": 1} if case == "uncorroborated" else {}
    if case == "nccl_never_started":
        kw = {
            "missing": tuple(f"jpbo-099-{i:02d}" for i in range(1, 11)),
            "fallback": 10,
        }  # 10 of 40
    if case == "all_stale":
        kw = {"n": 4, "missing": ("jpbo-007-44",), "fallback": 4, "stale": ("jpbo-007-44",)}
    job, nodes, log = sidecar_job(tmp_path, **kw)
    if case == "all_stale":  # every sidecar from an earlier attempt: NCCL not shown to have started
        import os

        for p in tmp_path.glob("nccl-*.log"):
            os.utime(p, (job["start"] - 3600,) * 2)
    result = peers.scan(tmp_path, job, nodes, job_log=None if case == "no_job_log" else log)
    assert result["exclude"] == []
    if case == "uncorroborated":
        assert {e["rule"] for e in result["events"]} == {"missing_sidecar_uncorroborated"}


def test_more_missing_sidecars_than_the_cap_requires_review(tmp_path):
    missing = tuple(f"jpbo-099-{i:02d}" for i in range(1, 6))
    job, nodes, log = sidecar_job(tmp_path, n=200, missing=missing, fallback=5)
    result = peers.scan(tmp_path, job, nodes, job_log=log)
    assert result["exclude"] == [] and result["refused"]

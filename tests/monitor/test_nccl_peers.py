"""Historical evidence replay and real restart lifecycle with fake scheduler
I/O."""

import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import socket

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


# --- fatal_rank_signature: real lines from the 2026-10 continuation run --------------------
# (job, rank, srun host/task, signature line); 4 ranks per node, 512 hosts.
FATAL = {
    "2391569": (
        1629,
        "jpbo-085-47",
        407,
        "[default1]:[rank1629]: torch.AcceleratorError: CUDA error: Invalid access of peer GPU memory over nvlink or a hardware error",
    ),
    "2400415": (
        1151,
        "jpbo-082-33",
        287,
        "[default3]:[rank1151]:[E1019 06:26:39.430918320 ProcessGroupNCCL.cpp:2109] [PG ID 6 PG GUID 4576(TENSOR_MODEL_PARALLEL_GROUP) Rank 3] Process group watchdog thread terminated with exception: CUDA error: Invalid access of peer GPU memory over nvlink or a hardware error",
    ),
    "2347262": (
        1748,
        "jpbo-104-26",
        437,
        "[default0]:[rank1748]: AssertionError: One or both of the .idx and .bin files cannot be found at the path prefix /e/fscratch/e-sta-openeurollm/training_v2/training/collection/flag/nemotron-cc-1.0/megatron-lm/high/actual/shard_0000_text_document",
    ),
}
DRAINED = {
    "State": "IDLE+DRAIN",
    "Reason": "pshealthcheck",
    "LastBusyTime": "2026-10-25T10:20:48",
}


def fatal_job(tmp_path, job_id, srun=True, extra_lines=(), world=True):
    rank, host, task, line = FATAL[job_id]
    hosts = [f"jpbo-{1 + i // 48:03d}-{i % 48 + 1:02d}" for i in range(512)]
    hosts[task] = host
    job = {
        "job_id": job_id,
        "state": "FAILED",
        "start": peers.stamp("2026-10-25T09:56:13"),
        "end": peers.stamp("2026-10-25T10:20:36"),
        "hosts": hosts,
    }
    lines = ["[default0]:using world size: 2048, data-parallel size: 128"] if world else []
    lines += ["[default0]: iteration   250700/  894000 | consumed samples: 1", line, *extra_lines]
    # every other rank dies as a victim, with NCCL errors that are not node-local
    lines += [
        "[default2]:[rank574]:[E1023 11:51:14.8] Collective WorkNCCL(SeqNum=17518) ncclRemoteError: A call failed possibly due to a network error"
    ]
    if srun:
        lines.append(f"srun: error: {host}: task {task}: Exited with exit code 1")
    log = tmp_path / f"slurm-{job_id}.log"
    log.write_text("\n".join(lines) + "\n")
    nodes = {h: {"NodeName": h, "State": "ALLOCATED"} for h in hosts}
    return job, nodes, log, host


@pytest.mark.parametrize("job_id", sorted(FATAL))
def test_fatal_signature_excludes_the_one_corroborated_host(tmp_path, job_id):
    job, nodes, log, host = fatal_job(tmp_path, job_id)
    result = peers.scan(tmp_path, job, nodes, job_log=log)
    assert result["exclude"] == [host]
    (event,) = (e for e in result["events"] if e["rule"] == "fatal_rank_signature")
    assert event["rank"] == FATAL[job_id][0] and event["srun_error"]
    exclusions = tmp_path / "exclude"
    exclusions.write_text("jpbo-001-01\n")
    assert peers.apply(result, exclusions, tmp_path / "audit.jsonl") == [host]
    assert "# autoexp fatal_rank_signature event" in exclusions.read_text()


def test_fatal_signature_without_srun_line_needs_the_health_drain(tmp_path):
    # job 2500813: the monitor cancelled first, so srun never reported the task
    job, nodes, log, host = fatal_job(tmp_path, "2400415", srun=False)
    result = peers.scan(tmp_path, job, nodes, job_log=log)
    assert result["exclude"] == []
    assert result["events"][0]["rule"] == "fatal_rank_signature_uncorroborated"
    nodes[host].update(DRAINED)
    assert peers.scan(tmp_path, job, nodes, job_log=log)["exclude"] == [host]


@pytest.mark.parametrize("case", ["other_reason", "old_drain", "not_drained"])
def test_health_drain_must_be_the_site_check_in_this_attempt(tmp_path, case):
    job, nodes, log, host = fatal_job(tmp_path, "2400415", srun=False)
    nodes[host].update(DRAINED)
    if case == "other_reason":
        nodes[host]["Reason"] = "maintenance"
    if case == "old_drain":
        nodes[host]["LastBusyTime"] = "2026-10-24T10:20:48"
    if case == "not_drained":
        nodes[host]["State"] = "IDLE"
    assert peers.scan(tmp_path, job, nodes, job_log=log)["exclude"] == []


def test_srun_line_for_a_different_host_does_not_corroborate(tmp_path):
    job, nodes, log, host = fatal_job(tmp_path, "2391569", srun=False)
    with log.open("a") as f:
        f.write("srun: error: jpbo-001-01: task 0: Exited with exit code 1\n")
    assert peers.scan(tmp_path, job, nodes, job_log=log)["exclude"] == []


def test_fatal_signature_on_many_hosts_is_a_cascade_not_a_node(tmp_path):
    victims = [
        f"[default0]:[rank{r}]: torch.AcceleratorError: CUDA error: Invalid access of peer GPU memory over nvlink or a hardware error"
        for r in (0, 400, 800)
    ]
    job, nodes, log, host = fatal_job(tmp_path, "2391569", extra_lines=victims)
    result = peers.scan(tmp_path, job, nodes, job_log=log)
    assert result["exclude"] == []
    assert {e["rule"] for e in result["events"] if e["rank"] is not None} == {
        "fatal_rank_signature_unattributed"
    }


def test_fatal_signature_without_world_size_is_not_attributed(tmp_path):
    job, nodes, log, host = fatal_job(tmp_path, "2391569", world=False)
    assert peers.scan(tmp_path, job, nodes, job_log=log)["exclude"] == []


def test_victim_nccl_errors_alone_exclude_nothing(tmp_path):
    job, nodes, log, host = fatal_job(tmp_path, "2391569")
    log.write_text(
        "[default0]:using world size: 2048\n"
        "[default2]:[rank574]:[E1023 11:51:14.8] ncclRemoteError: A call failed possibly due to a network error\n"
        "srun: error: jpbo-085-47: task 407: Exited with exit code 1\n"
    )
    result = peers.scan(tmp_path, job, nodes, job_log=log)
    assert result["exclude"] == [] and result["events"] == []


# --- IB peers: job 2454305 (2026-10-23) ----------------------------------------------------
# jpbo-037-01 saw IBV_WC_RETRY_EXC_ERR from 10.128.26.96 on two of its HCAs. That address is
# jpbo-033-32's interconnect rail, not its Slurm NodeAddr (10.2.2.96), so the old lookup
# never mapped it, and the peer never went DOWN.
IB_LINES = [
    "[2026-10-23 11:51:15] jpbo-037-01:1261731:1264960 [0] ib_plugin.c:2033 NCCL WARN NET/IB: Got completion from peer 10.128.26.96<60030> with status=IBV_WC_RETRY_EXC_ERR(12) opcode=IBV_WC_RECV_RDMA_WITH_IMM(129) reqSize=16 vendor_err=129 req_type=Recv hca mlx5_0",
    "[2026-10-23 11:51:17] jpbo-037-01:1261734:1264944 [3] ib_plugin.c:2033 NCCL WARN NET/IB: Got completion from peer 10.128.26.96<48584> with status=IBV_WC_RETRY_EXC_ERR(12) opcode=IBV_WC_RECV_RDMA_WITH_IMM(129) reqSize=0 vendor_err=129 req_type=Recv hca mlx5_3",
]


def ib_job(
    tmp_path, monkeypatch, lines=IB_LINES, dns="jpbo-033-32-interconnect-1.jupiter.internal"
):
    job = {
        "job_id": "2454305",
        "state": "FAILED",
        "start": peers.stamp("2026-10-23T08:49:37"),
        "end": peers.stamp("2026-10-23T11:51:59"),
        "hosts": ["jpbo-037-01", "jpbo-033-32", "jpbo-050-01"],
    }
    nodes = {
        "jpbo-037-01": {"NodeName": "jpbo-037-01", "NodeAddr": "10.2.4.1", "State": "ALLOCATED"},
        "jpbo-033-32": {"NodeName": "jpbo-033-32", "NodeAddr": "10.2.2.96", "State": "ALLOCATED"},
        "jpbo-050-01": {"NodeName": "jpbo-050-01", "NodeAddr": "10.2.6.1", "State": "ALLOCATED"},
    }
    (tmp_path / "nccl-jpbo-037-01.log").write_text("\n".join(lines) + "\n")

    def lookup(ip):
        if dns is None:
            raise socket.herror("unknown host")
        return (dns, [], [ip])

    monkeypatch.setattr(peers.socket, "gethostbyaddr", lookup)
    return job, nodes


def test_ib_peer_on_two_local_hcas_is_excluded_via_its_interconnect_name(tmp_path, monkeypatch):
    job, nodes = ib_job(tmp_path, monkeypatch)
    result = peers.scan(tmp_path, job, nodes)
    assert result["exclude"] == ["jpbo-033-32"]
    assert {e["rule"] for e in result["events"]} == {"multi_path_peer"}


def test_ib_peer_on_one_local_hca_is_not_enough(tmp_path, monkeypatch):
    # the reporter's own HCA could be the broken end
    job, nodes = ib_job(tmp_path, monkeypatch, lines=IB_LINES[:1] * 3)
    result = peers.scan(tmp_path, job, nodes)
    assert result["exclude"] == []
    assert {e["rule"] for e in result["events"]} == {"peer_not_down_and_unresponsive"}


@pytest.mark.parametrize("dns", [None, "jpbo-099-01-interconnect-1.jupiter.internal"])
def test_ib_peer_that_does_not_resolve_to_the_allocation_is_not_excluded(
    tmp_path, monkeypatch, dns
):
    job, nodes = ib_job(tmp_path, monkeypatch, dns=dns)
    result = peers.scan(tmp_path, job, nodes)
    assert result["exclude"] == []
    assert {e["rule"] for e in result["events"]} == {
        "peer_not_uniquely_mapped_to_other_allocated_node"
    }

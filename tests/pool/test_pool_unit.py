"""Unit tests of the pool's monitor side (oellm_autoexp/monitor/pool.py).

No processes: the monitor loop runs against FakeSlurmClient, and the tests play
the supervisor by writing the files it would write (info.env, alive, state,
started, rc). The end-to-end behaviour against real batch shells is in
test_pool_e2e.py.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path

import pytest

from oellm_autoexp.monitor.loop import JobFileStore, JobRecord, MonitorLoop
from oellm_autoexp.monitor.pool import (
    PoolHostConfig,
    PoolLayout,
    PoolManager,
    check_fits_host,
    create_or_update_pool,
    parse_slurm_time,
    pool_tag,
    read_run_state,
    render_step_script,
    submit_request,
    write_rc,
)
from oellm_autoexp.monitor.slurm_client import SlurmClient, SlurmClientConfig
from oellm_autoexp.monitor.submission import SlurmJobConfig
from oellm_autoexp.slurm_gen.client import FakeSlurmClientConfig
from oellm_autoexp.slurm_gen.schema import SbatchConfig, SlurmConfig

REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- fixtures
def _slurm(
    tmp: Path, name: str, *, nodes: int = 2, template: str = "templates/base.sbatch", **sbatch
) -> SlurmConfig:
    sb = SbatchConfig(
        nodes=nodes, time=sbatch.pop("time", "01:00:00"), account=sbatch.pop("account", "acc")
    )
    for key, value in sbatch.items():
        setattr(sb, key, value)
    return SlurmConfig(
        template_path=str(REPO / template),
        script_dir=str(tmp / name),
        log_dir=str(tmp / name),
        name=name,
        command=["echo hello"],
        sbatch=sb,
    )


def _host_definition(tmp: Path, nodes: int = 2) -> SlurmJobConfig:
    slurm = _slurm(tmp, "hostjob", nodes=nodes, template="templates/pool_host.sbatch")
    slurm = replace(
        slurm, command=['bash scripts/pool/supervisor.sh "$AUTOEXP_POOL_DIR" "$AUTOEXP_POOL_HOST"']
    )
    return SlurmJobConfig(name="hostjob", log_path=str(tmp / "pool" / "slurm-%j.log"), slurm=slurm)


def _run_definition(
    tmp: Path, pool_dir: Path, name: str, *, nodes: int = 1, priority: int = 0, **sbatch
) -> SlurmJobConfig:
    return SlurmJobConfig(
        name=name,
        log_path=str(tmp / name / "slurm-%j.log"),
        slurm=_slurm(tmp, name, nodes=nodes, **sbatch),
        pool_dir=str(pool_dir),
        metadata={"pool_priority": priority, "pool_enqueued_ts": time.time()},
    )


class Env:
    def __init__(self, tmp: Path, **cfg) -> None:
        self.tmp = tmp
        self.session = tmp / "session"
        self.store = JobFileStore(self.session)
        self.layout = create_or_update_pool(
            tmp / "pool",
            host_definition=_host_definition(tmp),
            config=PoolHostConfig(name="unitpool", **{"min_remaining_s": 0, **cfg}),
            session_dir=self.session,
        )
        self.slurm = SlurmClient(SlurmClientConfig(base_client=FakeSlurmClientConfig()))
        self.loop = MonitorLoop(
            self.store, slurm_client=self.slurm, show_poll_state=False, no_error_catching=True
        )
        self.fake = self.slurm._client

    def add_run(self, name: str, **kw) -> str:
        record = JobRecord(
            job_id=name, definition=_run_definition(self.tmp, self.layout.root, name, **kw)
        )
        self.store.upsert(record)
        return name

    def poll(self, n: int = 1) -> None:
        for _ in range(n):
            self.loop.observe_once()

    def rec(self, job_id: str) -> JobRecord:
        return self.store.load(job_id, include_finished=True)

    def hosts(self) -> list[JobRecord]:
        hosts = [r for r in self.store.load_all(include_finished=True) if PoolManager.is_host(r)]
        return sorted(hosts, key=lambda r: r.job_id)

    def host_up(
        self, record: JobRecord, *, start_ts: float | None = None, idle_since: float | None = None
    ) -> str:
        """Play SLURM + supervisor: the host's job is RUNNING and its
        supervisor is up."""
        hid = record.definition.metadata["pool_host_id"]
        self.fake.set_state(record.runtime.runtime_job_id, "RUNNING")
        hdir = self.layout.host_dir(hid)
        (hdir / "assign").mkdir(parents=True, exist_ok=True)
        start = start_ts or time.time()
        (hdir / "info.env").write_text(
            f"SLURM_JOB_ID={record.runtime.runtime_job_id}\nNUM_NODES=2\nNODELIST=a,b\nSTART_TS={int(start)}\n"
        )
        (hdir / "alive").write_text("")
        (hdir / "state").write_text(f"idle {int(idle_since or start)} 2\n")
        return hid

    def start_run(self, rid: str, hid: str) -> None:
        (self.layout.run_dir(rid) / "started").write_text(
            f"host={hid}\njob=1\nstep_index=1\nnodes=a\nts={int(time.time())}\n"
        )

    def assigned(self, hid: str) -> list[str]:
        d = self.layout.host_dir(hid) / "assign"
        return sorted(p.name for p in d.iterdir()) if d.exists() else []


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    return Env(tmp_path)


# ---------------------------------------------------------------- helpers
def test_parse_slurm_time():
    assert parse_slurm_time("12:00:00") == 43200
    assert parse_slurm_time("1-02:00:00") == 93600
    assert parse_slurm_time("30") == 1800
    assert parse_slurm_time("05:30") == 330
    assert parse_slurm_time("2-0") == 172800
    assert parse_slurm_time(None) is None
    assert parse_slurm_time("UNLIMITED") is None


def test_pool_tag_has_no_underscore():
    # expand_log_path splits "A_B" into array id / index
    assert pool_tag("speed_ablation_n64") == "speed-ablation-n64"
    assert "_" not in pool_tag("a_b.c")


def test_run_state_mapping(tmp_path):
    layout = PoolLayout(tmp_path)
    for rid, (rc, status) in {
        "p.0001": ("0 completed", "COMPLETED"),
        "p.0002": ("3 failed", "FAILED"),
        "p.0003": ("0 cancelled", "CANCELLED"),
        "p.0004": ("0 preempted", "PREEMPTED"),
        "p.0005": ("-1 host_lost", "PREEMPTED"),
        "p.0006": ("-1 insufficient_nodes", "PREEMPTED"),
        "p.0007": ("143 timeout", "TIMEOUT"),
        "p.0008": ("1 completed", "FAILED"),
    }.items():
        layout.run_dir(rid).mkdir(parents=True)
        (layout.run_dir(rid) / "rc").write_text(rc)
        assert read_run_state(layout, rid).status == status, rid
    layout.run_dir("p.0009").mkdir()
    assert read_run_state(layout, "p.0009").status == "PENDING"
    (layout.run_dir("p.0009") / "started").write_text("host=host001\nts=1\n")
    state = read_run_state(layout, "p.0009")
    assert (state.status, state.host) == ("RUNNING", "host001")


def test_render_step_script_rewrites_allocation_wide_settings(tmp_path):
    slurm = _slurm(
        tmp_path, "train", nodes=2, template="templates/jupiter.sbatch", cpus_per_task=288
    )
    slurm = replace(
        slurm,
        env={
            "MASTER_ADDR": '$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)',
            "MASTER_PORT": "20074",
            "NUM_NODES": "$SLURM_JOB_NUM_NODES",
            "NUM_GPUS": "$((4*SLURM_NNODES))",
        },
        srun_opts="--kill-on-bad-exit=0",
        launcher_cmd="apptainer exec img",
    )
    text = render_step_script(slurm, "pool.0003", log_path=str(tmp_path / "x.log"))
    assert text.startswith("#!/bin/bash\n# [pool] step script of run pool.0003")
    # the job's own header documents what it asked for, but is not live
    assert "\n#SBATCH" not in text and "#   #SBATCH --nodes=2" in text
    assert "SLURM_JOB_NODELIST" not in text.split("echo")[0]
    assert '"$AUTOEXP_STEP_NODELIST"' in text and "export NUM_NODES=$AUTOEXP_STEP_NUM_NODES" in text
    assert "$((4*AUTOEXP_STEP_NUM_NODES))" in text
    assert "export MASTER_PORT=$(( MASTER_PORT + AUTOEXP_STEP_INDEX % 64 ))" in text
    srun = next(line for line in text.splitlines() if line.startswith("srun "))
    # the job's own options first, the pool's pinning last (srun: last one wins)
    assert srun.index("--kill-on-bad-exit=0") < srun.index("--nodes=$AUTOEXP_STEP_NUM_NODES")
    for flag in (
        '--nodelist="$AUTOEXP_STEP_NODELIST"',
        '--job-name="$AUTOEXP_RUN_ID"',
        "--cpus-per-task=288",
        "--ntasks=$AUTOEXP_STEP_NTASKS",
    ):
        assert flag in srun
    # (hybrid-v0.19 renders `bash -c 'ulimit -c 0; <cmd>'`: no core dumps)
    assert "bash -c '" in srun and "apptainer exec img echo hello'" in srun


def test_check_fits_host(tmp_path):
    host = _host_definition(tmp_path, nodes=4)
    assert check_fits_host(_run_definition(tmp_path, tmp_path, "ok", nodes=4), host) == []
    big = check_fits_host(_run_definition(tmp_path, tmp_path, "big", nodes=8), host)
    assert big and "needs 8 nodes" in big[0]
    other = check_fits_host(_run_definition(tmp_path, tmp_path, "acc", account="other"), host)
    assert other and "account" in other[0]


# ---------------------------------------------------------------- submit / cancel
def test_submit_writes_a_self_contained_run(env):
    pm = PoolManager()
    rid = pm.submit(
        _run_definition(
            env.tmp, env.layout.root, "job_a", nodes=2, time="00:30:00", exclude="x1,x2"
        )
    )
    assert rid == "unitpool.0001"
    rd = env.layout.run_dir(rid)
    meta = (rd / "meta.env").read_text()
    assert "NODES=2" in meta and "MAX_RUNTIME_S=1800" in meta and "EXCLUDE=x1,x2" in meta
    assert f"LOG={env.tmp / 'job_a' / 'slurm-unitpool.0001.log'}" in meta
    assert (rd / "step.sh").read_text().startswith("#!/bin/bash")
    assert json.loads((rd / "run.json").read_text())["name"] == "job_a"
    assert pm.squeue() == {rid: "PENDING"}
    assert pm.submit(_run_definition(env.tmp, env.layout.root, "job_b")) == "unitpool.0002"


def test_cancel_semantics(env):
    pm = PoolManager()
    queued = pm.submit(_run_definition(env.tmp, env.layout.root, "q"))
    assigned = pm.submit(_run_definition(env.tmp, env.layout.root, "a"))
    running = pm.submit(_run_definition(env.tmp, env.layout.root, "r"))
    (env.layout.run_dir(assigned) / "assigned").write_text("host001\n")
    (env.layout.run_dir(running) / "started").write_text("host=host001\nts=1\n")
    for rid in (queued, assigned, running):
        pm.cancel(rid)
    # never handed to a supervisor: closed by the manager right away
    assert read_run_state(env.layout, queued).status == "CANCELLED"
    # a supervisor owns these: they only get the stop file, and the supervisor writes rc
    assert read_run_state(env.layout, assigned).status == "PENDING"
    assert read_run_state(env.layout, running).status == "RUNNING"
    assert (env.layout.run_dir(assigned) / "stop").exists() and (
        env.layout.run_dir(running) / "stop"
    ).exists()


# ---------------------------------------------------------------- scheduling
def test_work_creates_and_submits_a_host_then_assigns(env):
    env.add_run("job_a")
    env.poll()  # run queued, host record created
    assert len(env.hosts()) == 1
    rid = env.rec("job_a").runtime.runtime_job_id
    assert read_run_state(env.layout, rid).status == "PENDING"
    env.poll()  # host submitted
    host = env.hosts()[0]
    assert host.runtime.submitted and host.definition.slurm.env["AUTOEXP_POOL_HOST"] == "host001"
    assert (
        Path(host.definition.slurm.script_path).read_text().count("scripts/pool/supervisor.sh") == 1
    )
    env.poll()
    assert env.assigned("host001") == []  # PENDING host: nothing assigned yet
    hid = env.host_up(env.hosts()[0])
    env.poll()
    assert env.assigned(hid) == [rid]
    assert (env.layout.run_dir(rid) / "assigned").read_text().strip() == hid
    # the supervisor runs and finishes it
    env.start_run(rid, hid)
    env.poll()
    assert env.rec("job_a").runtime.last_status == "RUNNING"
    write_rc(env.layout, rid, 0, "completed")
    env.poll()
    assert env.rec("job_a").runtime.final_state == "finished"


def test_no_work_no_host_and_min_hosts(env, tmp_path):
    env.poll(3)
    assert env.hosts() == []
    meta = env.layout.load_meta()
    meta["config"]["min_hosts"] = 1
    env.layout.write_meta(meta)
    env.poll()
    assert len(env.hosts()) == 1


def test_priority_and_fifo_order(env):
    for name, prio in (("first", 0), ("second", 0), ("urgent", 5)):
        env.add_run(name, priority=prio)
        time.sleep(0.01)
    env.poll(2)
    hid = env.host_up(env.hosts()[0])
    env.poll()
    assert env.assigned(hid) == [env.rec("urgent").runtime.runtime_job_id]


def test_run_too_big_for_degraded_host_is_not_assigned(env):
    env.add_run("needs2", nodes=2)
    env.poll(2)
    hid = env.host_up(env.hosts()[0])
    (env.layout.host_dir(hid) / "state").write_text(
        f"idle {int(time.time())} 1\n"
    )  # one node excluded
    env.poll()
    assert env.assigned(hid) == []


def test_host_without_time_left_gets_nothing(env):
    meta = env.layout.load_meta()
    meta["config"]["min_remaining_s"] = 1800
    env.layout.write_meta(meta)
    env.add_run("job_a")
    env.poll(2)
    # 1 h allocation started 45 min ago: 15 min left < 30 min
    hid = env.host_up(env.hosts()[0], start_ts=time.time() - 2700)
    env.poll()
    assert env.assigned(hid) == []


def test_preempted_run_is_requeued_with_a_new_run(env):
    env.add_run("job_a")
    env.poll(2)
    hid = env.host_up(env.hosts()[0])
    env.poll()
    first = env.rec("job_a").runtime.runtime_job_id
    env.start_run(first, hid)
    env.poll()
    write_rc(env.layout, first, 0, "preempted")
    env.poll()
    rec = env.rec("job_a")
    assert rec.runtime.final_state is None
    assert rec.runtime.runtime_job_id != first and rec.runtime.attempts == 2
    assert read_run_state(env.layout, rec.runtime.runtime_job_id).status == "PENDING"


def test_preemption_budget_counts_only_unhealthy_requeues(env):
    meta = env.layout.load_meta()
    meta["config"]["max_attempts"] = 2
    env.layout.write_meta(meta)
    env.add_run("job_a")
    env.poll(2)
    env.host_up(env.hosts()[0])
    # healthy requeues (handover / wall clock / preempt) never use up the budget
    for _ in range(5):
        env.poll()
        write_rc(env.layout, env.rec("job_a").runtime.runtime_job_id, 0, "preempted")
    env.poll()
    assert env.rec("job_a").runtime.final_state is None
    # unhealthy ones do: two are allowed, the third gives up
    for _ in range(3):
        env.poll()
        write_rc(env.layout, env.rec("job_a").runtime.runtime_job_id, -1, "host_lost")
    env.poll()
    rec = env.rec("job_a")
    assert rec.runtime.final_state == "cancelled"
    assert rec.runtime.action_state["pool:requeues"] == {"preempted": 5, "host_lost": 3}


def test_timeout_is_final(env):
    env.add_run("job_a")
    env.poll(2)
    hid = env.host_up(env.hosts()[0])
    env.poll()
    rid = env.rec("job_a").runtime.runtime_job_id
    env.start_run(rid, hid)
    write_rc(env.layout, rid, 143, "timeout")
    env.poll()
    assert env.rec("job_a").runtime.final_state == "cancelled"


def test_host_loss_requeues_the_running_run(env):
    env.add_run("job_a")
    env.poll(2)
    host = env.hosts()[0]
    hid = env.host_up(host)
    env.poll()
    rid = env.rec("job_a").runtime.runtime_job_id
    env.start_run(rid, hid)
    env.poll()
    env.fake.set_state(host.runtime.runtime_job_id, "NODE_FAIL")
    env.poll()
    assert read_run_state(env.layout, rid).reason == "host_lost"
    rec = env.rec("job_a")
    assert rec.runtime.runtime_job_id != rid and rec.runtime.final_state is None
    # the dead host's record is closed (NODE_FAIL is final for a host)
    assert env.rec(host.job_id).runtime.final_state == "cancelled"


def test_stale_supervisor_heartbeat_cancels_the_host(env):
    meta = env.layout.load_meta()
    meta["config"]["host_stale_s"] = 60
    env.layout.write_meta(meta)
    env.add_run("job_a")
    env.poll(2)
    host = env.hosts()[0]
    hid = env.host_up(host)
    env.poll()
    rid = env.rec("job_a").runtime.runtime_job_id
    env.start_run(rid, hid)
    old = time.time() - 120
    os.utime(env.layout.host_dir(hid) / "alive", (old, old))
    env.poll()
    assert read_run_state(env.layout, rid).reason == "host_lost"
    assert env.fake.get_job(host.runtime.runtime_job_id).state == "CANCELLED"


def test_assigned_but_not_started_run_returns_to_the_queue(env):
    env.add_run("job_a")
    env.poll(2)
    host = env.hosts()[0]
    hid = env.host_up(host)
    env.poll()
    rid = env.rec("job_a").runtime.runtime_job_id
    assert env.assigned(hid) == [rid]
    env.fake.set_state(host.runtime.runtime_job_id, "FAILED")
    env.poll()
    assert not (env.layout.run_dir(rid) / "assigned").exists()
    assert env.rec("job_a").runtime.runtime_job_id == rid  # same run, just queued again


def test_idle_host_is_released(env):
    meta = env.layout.load_meta()
    meta["config"]["idle_exit_s"] = 60
    meta["config"]["min_hosts"] = 1
    env.layout.write_meta(meta)
    env.poll(2)
    hid = env.host_up(env.hosts()[0], start_ts=time.time() - 30)
    env.poll()
    assert not (env.layout.host_dir(hid) / "control" / "exit").exists()
    (env.layout.host_dir(hid) / "state").write_text(f"idle {int(time.time() - 120)} 2\n")
    env.poll()
    assert (env.layout.host_dir(hid) / "control" / "exit").exists()


def test_handover_retires_the_oldest_host_and_keeps_one_pending(env):
    meta = env.layout.load_meta()
    meta["config"].update(handover=True, keep_pending=1, max_hosts_running=1)
    env.layout.write_meta(meta)
    env.add_run("train")
    env.poll(2)
    hosts = env.hosts()
    assert len(hosts) == 2  # one to run on, one waiting in the queue
    old = env.host_up(hosts[0], start_ts=time.time() - 2400)
    env.poll()
    rid = env.rec("train").runtime.runtime_job_id
    assert env.assigned(old) == [rid]
    env.start_run(rid, old)
    env.poll()
    # the successor starts: the old host is told to preempt its run and release
    new = env.host_up(env.hosts()[1])
    env.poll()
    assert (env.layout.host_dir(old) / "control" / "exit").exists()
    assert not (env.layout.host_dir(new) / "control" / "exit").exists()
    # ... and the new idle host is NOT released for idleness while the handover runs
    (env.layout.host_dir(new) / "state").write_text(f"idle {int(time.time() - 10_000)} 2\n")
    env.poll()
    assert not (env.layout.host_dir(new) / "control" / "exit").exists()
    # a third host goes into the queue to keep one pending
    assert len([h for h in env.hosts() if h.runtime.final_state is None]) == 3
    # the old supervisor preempts: the run is requeued and goes to the new host
    write_rc(env.layout, rid, 0, "preempted")
    (env.layout.host_dir(old) / "exited").write_text("exit requested\n")
    env.poll(2)
    second = env.rec("train").runtime.runtime_job_id
    assert second != rid and env.assigned(new) == [second]


def test_requests_close_and_cancel(env):
    env.add_run("keep")
    env.add_run("drop")
    env.poll(2)
    submit_request(env.layout.root, "cancel", target="drop")
    env.poll()
    assert env.rec("drop").runtime.final_state == "cancelled"
    assert read_run_state(env.layout, env.rec("drop").runtime.runtime_job_id).status == "CANCELLED"
    assert list((env.layout.requests_dir / "done").glob("*.json"))
    # close: the queued host is cancelled, no new one is created
    submit_request(env.layout.root, "close")
    env.poll(2)
    assert env.layout.load_state()["closed"] is True
    assert all(h.runtime.final_state == "cancelled" for h in env.hosts())
    env.poll(2)
    assert all(h.runtime.final_state is not None for h in env.hosts())
    submit_request(env.layout.root, "open")
    env.poll(2)
    assert any(h.runtime.final_state is None for h in env.hosts())


def test_monitor_restart_rehydrates_pool_runs(env):
    env.add_run("job_a")
    env.poll(2)
    hid = env.host_up(env.hosts()[0])
    env.poll()
    rid = env.rec("job_a").runtime.runtime_job_id
    env.start_run(rid, hid)
    # a fresh monitor process attaches to the session
    loop = MonitorLoop(
        env.store, slurm_client=env.slurm, show_poll_state=False, no_error_catching=True
    )
    loop.rehydrate()
    assert loop._pool.squeue() == {rid: "RUNNING"}


def test_a_young_host_is_not_handed_over_and_its_successor_waits(env):
    meta = env.layout.load_meta()
    meta["config"].update(handover=True, keep_pending=1, idle_exit_s=60, handover_min_age_s=1800)
    env.layout.write_meta(meta)
    env.add_run("train")
    env.poll(2)
    hosts = env.hosts()
    old = env.host_up(hosts[0], start_ts=time.time() - 300)  # 5 min into its allocation
    env.poll()
    rid = env.rec("train").runtime.runtime_job_id
    env.start_run(rid, old)
    new = env.host_up(env.hosts()[1], idle_since=time.time() - 10_000)
    env.poll(2)
    # no ping-pong: the young host keeps its run, the successor waits and is not released
    assert not (env.layout.host_dir(old) / "control" / "exit").exists()
    assert not (env.layout.host_dir(new) / "control" / "exit").exists()
    # once the old host is old enough, the handover happens
    info = env.layout.host_dir(old) / "info.env"
    info.write_text(
        info.read_text().replace(
            next(line for line in info.read_text().splitlines() if line.startswith("START_TS")),
            f"START_TS={int(time.time() - 2000)}",
        )
    )
    env.poll()
    assert (env.layout.host_dir(old) / "control" / "exit").exists()


def test_successor_delay_holds_back_the_pending_host(env):
    meta = env.layout.load_meta()
    meta["config"].update(handover=True, keep_pending=1, successor_delay_s=1800)
    env.layout.write_meta(meta)
    env.add_run("train")
    env.poll(2)
    assert len(env.hosts()) == 1  # the successor is not queued with the first host
    host = env.hosts()[0]
    env.host_up(host, start_ts=time.time() - 600)
    env.poll(2)
    assert len(env.hosts()) == 1  # running host is only 10 min old
    info = env.layout.host_dir("host001") / "info.env"
    info.write_text(
        info.read_text().replace(
            next(line for line in info.read_text().splitlines() if line.startswith("START_TS")),
            f"START_TS={int(time.time() - 2000)}",
        )
    )
    env.poll(2)
    assert len(env.hosts()) == 2


def test_surplus_queued_host_is_cancelled(env):
    meta = env.layout.load_meta()
    meta["config"].update(keep_pending=1, handover=True)
    env.layout.write_meta(meta)
    env.add_run("train")
    env.poll(3)
    hosts = env.hosts()
    assert len(hosts) == 2 and all(h.runtime.submitted for h in hosts)
    env.host_up(hosts[0])
    meta["config"].update(keep_pending=0)
    env.layout.write_meta(meta)
    env.poll()
    second = env.rec(hosts[1].job_id)
    assert second.runtime.final_state == "cancelled"
    assert env.fake.get_job(second.runtime.runtime_job_id).state == "CANCELLED"
    assert env.rec(hosts[0].job_id).runtime.final_state is None  # the running one is kept


def test_failed_host_starts_back_off(env, monkeypatch):
    def broken_submit(_cfg):
        raise RuntimeError("sbatch: command not found")

    monkeypatch.setattr(env.fake, "submit", broken_submit)
    env.loop.no_error_catching = (
        False  # production behaviour: a failed sbatch is logged, not raised
    )
    env.add_run("job_a")
    for _ in range(10):
        env.poll()
    # without the backoff this was one new host record per poll
    assert len(env.hosts()) <= 2
    state = env.layout.load_state()
    assert state["host_failures"] >= 1
    # sbatch works again and the backoff has passed: the next host gets submitted and counted as up
    monkeypatch.undo()
    state["last_host_created_ts"] = 0
    env.layout.write_state(state)
    env.poll(2)
    host = [h for h in env.hosts() if h.runtime.final_state is None][0]
    assert host.runtime.submitted
    env.host_up(host)
    env.poll()
    assert env.layout.load_state()["host_failures"] == 0


# ---------------------------------------------------------------- bad nodes
def test_sbatch_get_set_reach_the_rendered_header():
    from compoconf import asdict

    from oellm_autoexp.monitor.pool import sbatch_get, sbatch_set

    sb = SbatchConfig(nodes=2)
    # `exclude` is a declared field now: getattr / setattr / the rendered header agree
    sb.exclude = "a,b"
    assert asdict(sb)["exclude"] == "a,b" and sb._extras.get("exclude") is None
    # keys that are still non-strict extras: plain getattr does NOT see them
    sb._extras["signal"] = "old"
    assert getattr(sb, "signal", None) is None and sbatch_get(sb, "signal") == "old"
    setattr(sb, "signal", "shadow")  # a shadowing attribute wins on read ...
    assert sbatch_get(sb, "signal") == "shadow"
    sbatch_set(sb, "signal", "B:USR1@60")  # ... and sbatch_set is what reaches the header
    assert asdict(sb)["signal"] == "B:USR1@60" and sbatch_get(sb, "signal") == "B:USR1@60"
    sbatch_set(sb, "nodes", 4)
    assert sb.nodes == 4


def test_new_hosts_avoid_pool_and_fleet_exclusions(env, tmp_path):
    fleet = tmp_path / "fleet_exclude.txt"
    fleet.write_text("jpbo-001-01\njpbo-[002-003]-07\n")
    host = env.layout.load_host_definition()
    host.slurm = replace(host.slurm, exclude_file=str(fleet))
    from compoconf import asdict

    env.layout.host_definition.write_text(json.dumps(asdict(host)))
    env.layout.exclude_nodes.write_text("jpbo-009-09\njpbo-001-01\n")
    env.add_run("job_a")
    env.poll(2)
    script = Path(env.hosts()[0].definition.slurm.script_path).read_text()
    line = next(x for x in script.splitlines() if x.startswith("#SBATCH --exclude="))
    assert line == "#SBATCH --exclude=jpbo-001-01,jpbo-[002-003]-07,jpbo-009-09"
    # the fleet file is read when the host is CREATED: a node added later reaches the next host
    fleet.write_text("jpbo-001-01\njpbo-[002-003]-07\njpbo-004-04\n")
    env.host_up(env.hosts()[0])  # it did start (a host that never starts triggers the backoff)
    env.poll()
    env.fake.set_state(env.hosts()[0].runtime.runtime_job_id, "NODE_FAIL")
    env.poll(3)
    newest = env.hosts()[-1]
    assert "jpbo-004-04" in Path(newest.definition.slurm.script_path).read_text()


def test_a_host_that_cannot_fit_any_queued_job_is_released_at_once(env):
    env.add_run("needs2", nodes=2)
    env.poll(2)
    hid = env.host_up(env.hosts()[0])
    (env.layout.host_dir(hid) / "state").write_text(f"idle {int(time.time())} 1\n")  # a bad node
    env.poll()
    exit_file = env.layout.host_dir(hid) / "control" / "exit"
    assert exit_file.exists() and "needs 2 node(s), 1 usable" in exit_file.read_text()
    (env.layout.host_dir(hid) / "exited").write_text("exit requested\n")
    env.poll(2)
    assert len([h for h in env.hosts() if h.runtime.final_state is None]) >= 1  # a replacement


def test_no_release_loop_for_a_job_bigger_than_every_host(env):
    env.add_run("huge", nodes=8)  # bypasses the enqueue check, as a changed host definition would
    env.poll(2)
    hid = env.host_up(env.hosts()[0])
    env.poll(3)
    assert not (env.layout.host_dir(hid) / "control" / "exit").exists()
    assert len(env.hosts()) == 1


# ---------------------------------------------------------------- fault scan / queued excludes
def test_restart_hooks_attribute_a_failed_pool_run_to_its_node(env, tmp_path):
    env.add_run("train", nodes=2)
    env.poll(2)
    hid = env.host_up(env.hosts()[0])
    env.poll()
    rid = env.rec("train").runtime.runtime_job_id
    host_job = env.hosts()[0].runtime.runtime_job_id
    (env.layout.run_dir(rid) / "started").write_text(
        f"host={hid}\njob={host_job}\nstep_index=1\nnodes=jpbo-001-01,jpbo-001-02\nts={int(time.time()) - 60}\n"
    )
    log = Path(env.rec("train").definition.log_path.replace("%j", rid))
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        "[default0]: using world size: 8, data-parallel size: 8\n"
        "[rank5]: RuntimeError: CUDA error: uncorrectable ECC error encountered\n"
        "srun: error: jpbo-001-02: task 1: Exited with exit code 1\n"
    )
    write_rc(env.layout, rid, 1, "failed")
    excl = tmp_path / "fleet_exclude.txt"
    excl.write_text("# fleet list\n")
    out = tmp_path / "pre.out"
    env.loop._pending_restart_hooks[env.rec("train").job_id] = {
        "nccl_peer_scan": True,
        "exclude_file": str(excl),
        "pre_command": f"echo {{runtime_job_id}} {{run_id}} {{step_nodes}} > {out}",
    }
    env.loop._run_restart_hooks(env.rec("train"), rid)
    # the node whose rank logged the fault (task 1 -> second node) is excluded, with evidence
    assert "jpbo-001-02" in excl.read_text().split() and "jpbo-001-01" not in excl.read_text()
    assert f"job {host_job}" in excl.read_text()
    audit = (log.parent.parent / "nccl-peer-events.jsonl").read_text()
    assert f'"pool_run": "{rid}"' in audit
    # hooks get the HOST's SLURM job id, not the run id, plus the run's own facts
    assert out.read_text().split() == [host_job, rid, "jpbo-001-01,jpbo-001-02"]


def test_queued_successor_gets_exclusions_found_after_its_submission(env, monkeypatch):
    meta = env.layout.load_meta()
    meta["config"].update(keep_pending=1, handover=True)
    env.layout.write_meta(meta)
    env.add_run("train")
    env.poll(3)
    running, queued = env.hosts()
    env.host_up(running)
    env.poll()
    calls = []
    monkeypatch.setattr(
        env.fake, "update_excludes", lambda job_id, nodes: calls.append((job_id, nodes))
    )
    env.layout.exclude_nodes.write_text("jpbo-007-07\n")
    env.poll()
    assert calls == [(queued.runtime.runtime_job_id, "jpbo-007-07")]  # the queued one only
    assert env.rec(queued.job_id).definition.slurm.sbatch.exclude == "jpbo-007-07"
    env.poll(2)
    assert len(calls) == 1  # nothing new, no repeated scontrol calls
    env.layout.exclude_nodes.write_text("jpbo-007-07\njpbo-008-08\n")
    env.poll()
    assert calls[-1] == (queued.runtime.runtime_job_id, "jpbo-007-07,jpbo-008-08")

"""End-to-end tests of sub-SLURM-job pools against tests/fakeslurm.

Real run_autoexp / monitor_autoexp / SlurmClient / supervisor.sh / step
scripts; only SLURM is local. Runs are scripts/sigterm_dummy_trainer.py
(Megatron's SIGTERM-save-exit and checkpoint-resume semantics) under a
fake torchrun agent that, like the real one, kills its workers when IT
gets the signal -- so a graceful stop only works if the supervisor
signals the ranks.

Each test takes 10-90 s of wall time. Skipped if bash/pgrep are missing.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

import pytest

from tests.pool.harness import PoolEnv

pytestmark = pytest.mark.skipif(
    not (shutil.which("bash") and shutil.which("pgrep")), reason="needs bash and pgrep"
)

FAST = ("job.pool_host.idle_exit_s=5", "job.pool_host.poll_s=1")


@pytest.fixture
def pool(tmp_path):
    env = PoolEnv(tmp_path)
    yield env
    env.teardown()


def _iterations(log: str) -> list[int]:
    return [int(m) for m in re.findall(r"iteration\s+(\d+)/", log)]


def _resumed_from(log: str) -> int | None:
    m = re.search(r"loading checkpoint from \S+ at iteration (\d+)", log)
    return int(m.group(1)) if m else None


def _monitor_done(pool: PoolEnv, timeout: float):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pool.monitor.poll() is not None:
            return pool.monitor.returncode
        time.sleep(0.25)
    raise AssertionError(f"monitor still running after {timeout}s\n{pool.debug()}")


# ------------------------------------------------------------------ scenarios
def test_queue_runs_back_to_back_in_one_allocation(pool):
    pool.create_pool(*FAST)
    pool.enqueue("a")
    pool.enqueue("b")
    pool.start_monitor()
    assert _monitor_done(pool, 120) == 0

    hosts = pool.fake_jobs()
    assert len(hosts) == 1, "both jobs must share ONE allocation"
    assert hosts[0]["state"] == "COMPLETED"
    jid = hosts[0]["id"]
    ra, rb = pool.runs_of("dummy_a")[0], pool.runs_of("dummy_b")[0]
    for rid in (ra, rb):
        state = pool.run_state(rid)
        assert (state.status, state.host, state.started["job"]) == ("COMPLETED", "host001", jid)
    assert int(pool.run_state(rb).started["ts"]) >= int(pool.run_state(ra).started["ts"])
    for name in ("dummy_a", "dummy_b"):
        assert pool.record(name)["runtime"]["final_state"] == "finished"
    # main output + sub-named per-run logs + ledger + per-run provenance
    main_log = (pool.pool_dir / f"slurm-{jid}.log").read_text()
    assert (
        f"run {ra} ended: rc=0 (completed)" in main_log and "releasing the allocation" in main_log
    )
    assert (pool.pool_dir / f"slurm-{jid}.1-dummy_a.log").resolve().read_text().count(
        "[after training is done]"
    ) == 1
    assert (pool.pool_dir / f"slurm-{jid}.2-dummy_b.log").exists()
    ledger = (pool.pool_dir / "ledger.tsv").read_text()
    assert f"{ra}\tstart step=1" in ledger and f"{rb}\tend 0 completed" in ledger
    assert list(pool.layout.manifests_dir.glob("plan_*.json"))
    # provenance: the resolved config still parses and renders the same command, and
    # carries the enqueue-time code state; the run records the code it launched with
    import yaml
    from compoconf import parse_config

    import oellm_autoexp.orchestrator  # noqa: F401  (registers the backends)
    from oellm_autoexp.config.schema import BackendInterface, RootConfig

    dumped = yaml.safe_load((pool.root / "runs" / "dummy_a" / f"config-{ra}.yaml").read_text())
    prov = dumped["config"]["metadata"]["provenance"]
    assert prov["repo"]["commit"] and "dirty" in prov["repo"] and "submodules" in prov
    assert Path(prov["manifest"]).name in {
        p.name for p in pool.layout.manifests_dir.glob("plan_*.json")
    }
    cfg = parse_config(RootConfig, dumped["config"])
    record = pool.record("dummy_a")
    command = cfg.backend.instantiate(BackendInterface).build_launch_command()
    assert command == record["definition"]["slurm"]["command"][0]
    launch = json.loads((pool.layout.run_dir(ra) / "launch_provenance.json").read_text())
    assert launch["repo"]["commit"] == prov["repo"]["commit"]
    assert isinstance(launch["changed_since_enqueue"], list)
    assert f"{ra}\tprovenance: code " in ledger


def test_enqueue_and_cancel_while_the_pool_runs(pool):
    pool.create_pool(*FAST)
    pool.enqueue("long", "aux.iters=300", "aux.tick=0.1")
    pool.start_monitor()
    long_rid = pool.wait_for(lambda: pool.runs_of("dummy_long"), 30, "the long run")[0]
    pool.wait_for(lambda: pool.run_state(long_rid).status == "RUNNING", 30, "the long run to start")
    # the live monitor picks new records up on its next poll; nothing is resubmitted
    pool.enqueue("b")
    pool.enqueue("c")
    pool.wait_for(lambda: pool.runs_of("dummy_c"), 30, "c to be queued")
    pool.ctl("cancel", "dummy_c")
    pool.wait_for(
        lambda: pool.record("dummy_c")["runtime"]["final_state"] == "cancelled", 30, "c cancelled"
    )
    assert pool.run_state(pool.runs_of("dummy_c")[0]).started == {}, (
        "a cancelled queued job never starts"
    )
    pool.ctl("cancel", "dummy_long")
    pool.wait_for(
        lambda: pool.run_state(long_rid).status == "CANCELLED", 30, "the long run to stop"
    )
    b_rid = pool.runs_of("dummy_b")[0]
    pool.wait_for(
        lambda: pool.run_state(b_rid).status == "COMPLETED", 60, "b to run after the cancel"
    )
    assert len(pool.fake_jobs()) == 1
    pool.wait_for(
        lambda: pool.record("dummy_long")["runtime"]["final_state"] == "cancelled",
        30,
        "long closed",
    )


def test_preempt_saves_requeues_and_resumes_from_the_checkpoint(pool):
    pool.create_pool(*FAST)
    pool.enqueue("a", "aux.iters=150", "aux.tick=0.1")
    pool.start_monitor()
    first = pool.wait_for(lambda: pool.runs_of("dummy_a"), 30, "run")[0]
    pool.wait_for(lambda: max(_iterations(pool.run_log(first)) or [0]) >= 25, 60, "25 iterations")
    pool.ctl("preempt", first)
    pool.wait_for(lambda: pool.run_state(first).status == "PREEMPTED", 30, "the preemption")
    log = pool.run_log(first)
    assert "exiting program after receiving SIGTERM." in log, (
        "the RANK must get the signal, not the agent"
    )
    assert "tearing workers down" not in log
    saved = max(_iterations(log))
    second = pool.wait_for(
        lambda: [r for r in pool.runs_of("dummy_a") if r != first], 30, "the requeue"
    )[0]
    pool.wait_for(
        lambda: pool.record("dummy_a")["runtime"]["final_state"] == "finished",
        60,
        "the resumed run to finish",
    )
    assert _resumed_from(pool.run_log(second)) == saved
    # sigterm_rollover (the job's own policy) and the pool's PREEMPTED requeue must not BOTH restart
    assert len(pool.runs_of("dummy_a")) == 2
    rec = pool.record("dummy_a")
    assert rec["runtime"]["attempts"] == 2 and rec["runtime"]["final_state"] == "finished"
    assert len(pool.fake_jobs()) == 1, "the requeue stays in the same allocation"


def test_preempt_uses_the_jobs_own_exit_signal(pool):
    """A job with Megatron --exit-signal SIGUSR2 (our MareNostrum production runs) must get
    SIGUSR2 on a preemption: SIGTERM would kill it without a save."""
    pool.create_pool(*FAST)
    pool.enqueue("u", "aux.iters=150", "aux.tick=0.1", "aux.extra_args=--exit-signal SIGUSR2")
    pool.start_monitor()
    first = pool.wait_for(lambda: pool.runs_of("dummy_u"), 30, "run")[0]
    meta = (pool.pool_dir / "runs" / first / "meta.env").read_text()
    assert "STOP_SIGNAL=USR2" in meta, meta
    pool.wait_for(lambda: max(_iterations(pool.run_log(first)) or [0]) >= 25, 60, "25 iterations")
    pool.ctl("preempt", first)
    pool.wait_for(lambda: pool.run_state(first).status == "PREEMPTED", 30, "the preemption")
    log = pool.run_log(first)
    assert "exiting program after receiving SIGTERM." in log, "SIGUSR2 must reach the rank"
    saved = max(_iterations(log))
    second = pool.wait_for(
        lambda: [r for r in pool.runs_of("dummy_u") if r != first], 30, "the requeue"
    )[0]
    pool.wait_for(
        lambda: pool.record("dummy_u")["runtime"]["final_state"] == "finished",
        60,
        "the resumed run to finish",
    )
    assert _resumed_from(pool.run_log(second)) == saved


def test_runs_execute_in_the_directory_they_were_enqueued_from(pool, tmp_path):
    """MareNostrum runs from synced snapshots: a job enqueued from a new snapshot must run
    there, not in the directory the host was submitted from (job.metadata.pool_workdir,
    default the enqueuing process's working directory)."""
    import shlex

    snapshot = tmp_path / "autoexp_new_snapshot"
    snapshot.mkdir()
    pool.create_pool(*FAST)
    pool.enqueue("w", "aux.iters=3", "aux.tick=0.1", f"+job.metadata.pool_workdir={snapshot}")
    pool.start_monitor()
    rid = pool.wait_for(lambda: pool.runs_of("dummy_w"), 30, "run")[0]
    step = (pool.pool_dir / "runs" / rid / "step.sh").read_text()
    assert f"cd {shlex.quote(str(snapshot))} ||" in step
    meta = (pool.pool_dir / "runs" / rid / "meta.env").read_text()
    assert f"WORKDIR={shlex.quote(str(snapshot))}" in meta
    pool.wait_for(
        lambda: pool.record("dummy_w")["runtime"]["final_state"] == "finished", 60, "the run"
    )


def test_host_accounts_race_and_the_first_account_keeps_its_successor(pool):
    """host_accounts: the first account queues its usual hosts, a fallback one insurance host;
    whichever starts takes the run, and the first account keeps a successor queued (aging)."""
    pool.create_pool(
        *FAST,
        "job.pool_host.keep_pending=1",
        "job.pool_host.handover=true",
        "job.pool_host.handover_min_age_s=3",
        'job.pool_host.host_accounts="dev,accB"',
    )
    pool.enqueue("acc", "aux.iters=300", "aux.tick=0.1")
    pool.start_monitor(hold_new_jobs=True)

    def pending(account):
        return [
            j["id"] for j in pool.fake_jobs() if j["state"] == "PENDING" and j["account"] == account
        ]

    # nothing runs: the run's host + its successor = one queued host in each account
    pool.wait_for(
        lambda: len(pending("dev")) == 1 and len(pending("accB")) == 1, 40, "1 dev + 1 accB"
    )
    time.sleep(3)
    assert len(pending("dev")) == 1 and len(pending("accB")) == 1, "no more hosts than that"
    # the fallback starts first: it takes the run
    b1 = pending("accB")[0]
    pool.release(b1)
    first = pool.wait_for(lambda: pool.runs_of("dummy_acc"), 30, "run")[0]
    pool.wait_for(lambda: pool.run_state(first).status == "RUNNING", 30, "running")
    assert pool.run_state(first).started["job"] == b1
    # one host is up: dev keeps exactly one queued successor, accB a new insurance host
    pool.wait_for(
        lambda: len(pending("dev")) == 1 and len(pending("accB")) == 1, 40, "1 dev + 1 accB queued"
    )
    # the dev successor starts: the run moves over to it
    a1 = pending("dev")[0]
    pool.release(a1)
    second = pool.wait_for(
        lambda: [r for r in pool.runs_of("dummy_acc") if r != first], 60, "handover requeue"
    )[0]
    pool.wait_for(lambda: pool.run_state(second).status == "RUNNING", 60, "run on the dev host")
    assert pool.run_state(second).started["job"] == a1


USER_JOB = """#!/bin/bash
#SBATCH --job-name=user_job
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --account=dev
export PYTHONPATH=.
srun --nodes=1 --ntasks=1 python3 -c "import time; print('USER TRAINING RAN', flush=True); time.sleep(${USER_TRAIN_S:-0})"
echo "user batch: step rc $?"
"""


def _submit_user_job(pool, workdir, train_s=0):
    import subprocess

    script = workdir / "user_job.sbatch"
    script.write_text(USER_JOB)
    env = {**pool.env, "FAKESLURM_HOLD": "1", "USER_TRAIN_S": str(train_s)}
    out = subprocess.run(
        ["sbatch", str(script)], cwd=workdir, env=env, capture_output=True, text=True, timeout=30
    ).stdout
    return out.split()[-1]


def _fake_job(pool, jid):
    return next(j for j in pool.fake_jobs() if j["id"] == jid)


def test_an_adopted_queued_job_runs_pool_work_and_never_its_own(pool, tmp_path):
    """pool_ctl adopt: a queued job's own step is held by the hook; when it runs, the pool's
    supervisor is started into the allocation, runs the pool's job there, and on release the
    job ends COMPLETED -- its own training never ran."""
    wd = tmp_path / "user_snapshot"
    wd.mkdir()
    pool.create_pool(*FAST, "job.pool_host.adopt_stop_margin_s=30")
    jid = _submit_user_job(pool, wd)
    print(pool.ctl("adopt", jid, "--hold-max-s", "120", "--no-wait"))
    assert (wd / "sitecustomize.py").exists()
    pool.enqueue("ad", "aux.iters=15", "aux.tick=0.1")
    pool.start_monitor(hold_new_jobs=True)  # the pool's own hosts stay queued
    pool.release(jid)
    rid = pool.wait_for(lambda: pool.runs_of("dummy_ad"), 60, "the run")[0]
    pool.wait_for(
        lambda: pool.run_state(rid).status in ("RUNNING", "COMPLETED"), 60, "the run to start"
    )
    assert pool.run_state(rid).started["job"] == jid, "the run went to the adopted allocation"
    pool.wait_for(
        lambda: pool.record("dummy_ad")["runtime"]["final_state"] == "finished",
        90,
        "the run to finish",
    )
    # nothing left to do: the adopted host is released, its held step leaves, the job completes
    pool.wait_for(
        lambda: _fake_job(pool, jid)["state"] in ("COMPLETED", "FAILED", "CANCELLED"),
        90,
        "the adopted job to end",
    )
    job = _fake_job(pool, jid)
    log = Path(job["output"]).read_text()
    assert job["state"] == "COMPLETED", log
    assert "USER TRAINING RAN" not in log
    assert "adopted by a pool" in log and "released" in log
    assert (pool.pool_dir / "adopt" / f"{jid}.started").exists()


IDLE_ALLOC = """#!/bin/bash
#SBATCH --job-name=idle_alloc
#SBATCH --nodes=2
#SBATCH --time=00:10:00
#SBATCH --account=dev
sleep 600
"""


def test_attach_packs_runs_into_a_running_allocation_and_leaves_it_alive(pool, tmp_path):
    """pool_ctl attach: a RUNNING allocation with nothing of its own on its nodes (here a sleeping
    batch job, on MN5 an idle host of another pool) becomes a host of this pool without a hook; the
    pool's runs go there side by side, and when the pool is done the allocation is left running."""
    import subprocess

    wd = tmp_path / "alloc"
    wd.mkdir()
    (wd / "idle.sbatch").write_text(IDLE_ALLOC)
    jid = subprocess.run(
        ["sbatch", str(wd / "idle.sbatch")],
        cwd=wd,
        env=pool.env,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.split()[-1]
    pool.create_pool(
        *FAST,
        "job.pool_host.max_runs_per_host=2",
        "job.pool_host.max_own_hosts=0",
        "job.pool_host.adopt_stop_margin_s=30",
    )
    pool.wait_for(lambda: _fake_job(pool, jid)["state"] == "RUNNING", 30, "the allocation runs")
    print(pool.ctl("attach", jid, "--no-wait"))
    for tag in ("x", "y"):
        pool.enqueue(tag, "aux.iters=40", "aux.tick=0.1", "slurm.sbatch.nodes=1")
    pool.start_monitor()
    rx, ry = (pool.wait_for(lambda t=t: pool.runs_of(f"dummy_{t}"), 60, t)[0] for t in ("x", "y"))
    pool.wait_for(
        lambda: all(pool.run_state(r).status == "RUNNING" for r in (rx, ry)), 60, "both run at once"
    )
    sx, sy = pool.run_state(rx).started, pool.run_state(ry).started
    assert sx["job"] == sy["job"] == jid, "the runs went to the attached allocation"
    assert sx["nodes"] != sy["nodes"], "side by side on their own nodes"
    for name in ("dummy_x", "dummy_y"):
        pool.wait_for(
            lambda n=name: pool.record(n)["runtime"]["final_state"] == "finished", 90, name
        )
    assert [j["id"] for j in pool.fake_jobs()] == [jid], "the pool submitted no host of its own"
    time.sleep(3)
    assert _fake_job(pool, jid)["state"] == "RUNNING", "an attached allocation is never cancelled"


def test_replacing_an_adopted_supervisor_keeps_the_allocation(pool, tmp_path):
    """Swapping the supervisor of an idle adopted / attached host for new code: SIGKILL its step
    and drop its info / alive / state files. The monitor starts a new supervisor into the SAME
    allocation (never releases or cancels it), and the next run goes there."""
    import subprocess

    wd = tmp_path / "alloc"
    wd.mkdir()
    (wd / "idle.sbatch").write_text(IDLE_ALLOC)
    jid = subprocess.run(
        ["sbatch", str(wd / "idle.sbatch")],
        cwd=wd,
        env=pool.env,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.split()[-1]
    pool.create_pool(
        *FAST,
        "job.pool_host.max_own_hosts=0",
        "job.pool_host.adopt_stop_margin_s=30",
        "job.pool_host.idle_exit_s=120",
    )
    pool.wait_for(lambda: _fake_job(pool, jid)["state"] == "RUNNING", 30, "the allocation runs")
    pool.ctl("attach", jid, "--no-wait")
    pool.start_monitor()
    hdir = pool.pool_dir / "hosts" / "host001"
    pool.wait_for(lambda: (hdir / "info.env").exists(), 60, "the first supervisor")
    first = (hdir / "info.env").read_text()
    steps = pool.slurm("squeue", "-h", "-s", "-j", jid, "-o", "%i %j")
    sup = [ln.split()[0] for ln in steps.splitlines() if ln.split()[1:] == ["pool-supervisor"]]
    assert sup, steps
    for name in ("info.env", "alive", "state"):
        (hdir / name).unlink(missing_ok=True)
    pool.slurm("scancel", "--signal=KILL", sup[0])
    pool.wait_for(
        lambda: (hdir / "info.env").exists(), 240, "a new supervisor in the same allocation"
    )
    assert _fake_job(pool, jid)["state"] == "RUNNING"
    assert not (pool.pool_dir / "adopt" / f"{jid}.release").exists(), "the allocation was released"
    assert (hdir / "info.env").read_text() != first
    pool.enqueue("after", "aux.iters=10", "aux.tick=0.1", "slurm.sbatch.nodes=1")
    rid = pool.wait_for(lambda: pool.runs_of("dummy_after"), 60, "the run")[0]
    pool.wait_for(
        lambda: pool.record("dummy_after")["runtime"]["final_state"] == "finished", 90, "run"
    )
    assert pool.run_state(rid).started["job"] == jid


def test_an_adopted_job_whose_hook_does_not_engage_is_cancelled(pool, tmp_path):
    """The second, independent fail-safe: registered for adoption but the hook is missing ->
    the monitor cancels the job adopt_hook_timeout_s after it starts."""
    wd = tmp_path / "user_snapshot"
    wd.mkdir()
    pool.create_pool(*FAST, "job.pool_host.adopt_hook_timeout_s=5")
    jid = _submit_user_job(pool, wd, train_s=120)  # its own "training" would run for minutes
    pool.ctl("adopt", jid, "--no-wait")
    (wd / "sitecustomize.py").unlink()  # the hook "does not load"
    pool.start_monitor(hold_new_jobs=True)
    pool.release(jid)
    pool.wait_for(
        lambda: _fake_job(pool, jid)["state"] in ("CANCELLED", "COMPLETED", "FAILED"),
        60,
        "the job to end",
    )
    assert _fake_job(pool, jid)["state"] == "CANCELLED"
    assert "hook never engaged" in pool.monitor_log.read_text()


def test_a_singleton_job_moves_to_the_larger_pool_and_never_runs_twice(pool):
    """job.metadata.pool_singleton across two pools (two monitors): the job runs on 1 node in
    the small pool; when the large pool gets a host, the small pool's run is superseded (its
    job cancelled after a graceful save) and the job continues on 2 nodes in the large pool from
    that checkpoint. At no time do two runs of the job run."""
    import json as _json
    import subprocess
    import sys

    big = pool.root / "pool_big"
    pool.create_pool(*FAST)
    pool.autoexp(
        "--no-monitor",
        "--config-name",
        "experiments/tests/pool_local_host",
        *FAST,
        "job.name=localpool_big",
        f"job.base_output_dir={big}",
        "slurm.sbatch.nodes=3",
    )
    single = "+job.metadata.pool_singleton=sjob"
    pool.enqueue("s", "aux.iters=400", "aux.tick=0.1", "slurm.sbatch.nodes=1", single)
    pool.autoexp(
        "--no-monitor",
        "--pool",
        str(big),
        "--config-name",
        "experiments/tests/pool_local_dummy",
        "aux.tag=s",
        "aux.iters=400",
        "aux.tick=0.1",
        "slurm.sbatch.nodes=2",
        single,
    )
    pool.start_monitor()
    first = pool.wait_for(lambda: pool.runs_of("dummy_s"), 30, "the small pool's run")[0]
    pool.wait_for(lambda: max(_iterations(pool.run_log(first)) or [0]) >= 20, 60, "20 iterations")
    big_session = _json.loads((big / "pool.json").read_text())["session_dir"]
    env = {**pool.env, "FAKESLURM_HOLD": "1"}
    mon = subprocess.Popen(
        [
            sys.executable,
            "scripts/monitor_autoexp.py",
            "--session-dir",
            big_session,
            "--poll-interval",
            "1",
            "--force",
        ],
        cwd=pool.env["PWD"],
        env=env,
        stdout=open(pool.root / "monitor_big.out", "a"),
        stderr=subprocess.STDOUT,
    )
    try:

        def big_runs():
            return (
                sorted(p.name for p in (big / "runs").iterdir()) if (big / "runs").exists() else []
            )

        pool.wait_for(
            lambda: [j for j in pool.fake_jobs() if j["name"].startswith("localpool_big")],
            30,
            "the large pool's host",
        )
        host = next(j for j in pool.fake_jobs() if j["name"].startswith("localpool_big"))
        pool.release(host["id"])

        def running():
            out = []
            for root in (pool.pool_dir, big):
                for rd in (root / "runs").iterdir() if (root / "runs").exists() else []:
                    if (rd / "started").exists() and not (rd / "rc").exists():
                        out.append(rd)
            return out

        deadline = time.time() + 120
        moved = None
        while time.time() < deadline and moved is None:
            assert len(running()) <= 1, f"two runs of the singleton at once: {running()}"
            for rid in big_runs():
                if (big / "runs" / rid / "started").exists():
                    moved = rid
            time.sleep(0.2)
        assert moved, "the job never moved to the large pool"
        # superseded = preempted (saved), NOT cancelled: the small pool keeps the job queued
        assert pool.run_state(first).status == "PREEMPTED"
        saved = max(_iterations(pool.run_log(first)))
        log = (big / "runs" / moved / "run.json").read_text()
        big_log = Path(_json.loads(log)["log"])
        pool.wait_for(
            lambda: big_log.exists() and _resumed_from(big_log.read_text()) is not None,
            60,
            "resume",
        )
        assert _resumed_from(big_log.read_text()) == saved
        assert pool.record("dummy_s")["runtime"]["final_state"] is None, (
            "the small job stays queued"
        )
        # the large run goes away (cancelled): the small pool takes the job back from its checkpoint
        pool.wait_for(
            lambda: max(_iterations(big_log.read_text()) or [0]) >= saved + 10,
            60,
            "large run trains",
        )
        subprocess.run(
            [
                sys.executable,
                "scripts/pool_ctl.py",
                str(big),
                "cancel",
                "dummy_s",
                "--wait",
                "60",
            ],
            cwd=pool.env["PWD"],
            env=pool.env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        back = pool.wait_for(
            lambda: [
                r
                for r in pool.runs_of("dummy_s")
                if r != first and pool.run_state(r).status == "RUNNING"
            ],
            120,
            "the job back in the small pool",
        )[0]
        big_saved = max(_iterations(big_log.read_text()))
        pool.wait_for(
            lambda: _resumed_from(pool.run_log(back)) is not None, 60, "resume in the small pool"
        )
        assert _resumed_from(pool.run_log(back)) == big_saved
        assert len(running()) == 1
    finally:
        mon.terminate()
        mon.wait(timeout=20)


def test_packing_runs_side_by_side_on_disjoint_nodes(pool):
    """max_runs_per_host: several small jobs share one allocation at the same time, each on its
    own nodes; a run that does not fit waits for capacity; preempting one leaves the others."""
    pool.create_pool(*FAST, "job.pool_host.max_runs_per_host=3", "slurm.sbatch.nodes=3")
    for tag in ("a", "b", "c"):
        pool.enqueue(tag, "aux.iters=250", "aux.tick=0.1", "slurm.sbatch.nodes=1")
    pool.enqueue("d", "aux.iters=5", "aux.tick=0.1", "slurm.sbatch.nodes=1")
    pool.start_monitor()

    def running():
        out = {}
        for rd in sorted((pool.pool_dir / "runs").iterdir()):
            st = rd / "started"
            if st.exists() and not (rd / "rc").exists():
                nodes = [
                    ln.split("=", 1)[1]
                    for ln in st.read_text().splitlines()
                    if ln.startswith("nodes=")
                ]
                out[rd.name] = nodes[0] if nodes else ""
        return out

    three = pool.wait_for(
        lambda: (lambda r: r if len(r) == 3 else None)(running()), 60, "3 runs at once"
    )
    assert len(set(three.values())) == 3, f"the packed runs share a node: {three}"
    assert not pool.runs_of("dummy_d") or all(
        not (pool.pool_dir / "runs" / r / "started").exists() for r in pool.runs_of("dummy_d")
    ), "a 4th run must wait for capacity"
    for _ in range(10):  # never more than 3, never a shared node
        r = running()
        assert len(r) <= 3 and len(set(r.values())) == len(r), r
        time.sleep(0.3)
    b = pool.runs_of("dummy_b")[0]
    pool.wait_for(lambda: max(_iterations(pool.run_log(b)) or [0]) >= 10, 60, "b trains")
    pool.ctl("preempt", b)
    pool.wait_for(lambda: pool.run_state(b).status == "PREEMPTED", 60, "b preempted")
    others = [pool.runs_of(f"dummy_{t}")[0] for t in ("a", "c")]
    states = {
        r: (pool.run_state(r).status, (pool.pool_dir / "runs" / r / "rc").exists()) for r in others
    }
    assert all(not rc for _, rc in states.values()), f"a / c keep running: {states}"
    for name in ("dummy_a", "dummy_b", "dummy_c", "dummy_d"):
        pool.wait_for(
            lambda n=name: pool.record(n)["runtime"]["final_state"] == "finished", 120, name
        )
    assert len(pool.fake_jobs()) == 1, "all of it in ONE allocation"


def test_handover_to_a_successor_allocation(pool):
    pool.create_pool(
        *FAST,
        "job.pool_host.keep_pending=1",
        "job.pool_host.handover=true",
        "job.pool_host.handover_min_age_s=3",
    )
    pool.enqueue("train", "aux.iters=400", "aux.tick=0.1")
    pool.start_monitor(hold_new_jobs=True)
    jobs = pool.wait_for(
        lambda: len(pool.fake_jobs()) >= 2 and pool.fake_jobs(), 30, "two hosts in the queue"
    )
    assert all(j["state"] == "PENDING" for j in jobs)
    old_jid, new_jid = jobs[0]["id"], jobs[1]["id"]
    pool.release(old_jid)
    first = pool.wait_for(lambda: pool.runs_of("dummy_train"), 30, "run")[0]
    pool.wait_for(
        lambda: max(_iterations(pool.run_log(first)) or [0]) >= 30,
        60,
        "30 iterations on the first host",
    )
    assert pool.run_state(first).started["job"] == old_jid
    # the successor gets its allocation: the run moves over, the old allocation is released
    pool.release(new_jid)
    pool.wait_for(
        lambda: pool.run_state(first).status == "PREEMPTED", 60, "the handover preemption"
    )
    saved = max(_iterations(pool.run_log(first)))
    second = pool.wait_for(
        lambda: [r for r in pool.runs_of("dummy_train") if r != first], 30, "requeue"
    )[0]
    pool.wait_for(lambda: pool.run_state(second).status == "RUNNING", 60, "the run on the new host")
    assert pool.run_state(second).started["job"] == new_jid
    pool.wait_for(lambda: _resumed_from(pool.run_log(second)) is not None, 30, "the resume line")
    assert _resumed_from(pool.run_log(second)) == saved
    pool.wait_for(
        lambda: next(j for j in pool.fake_jobs() if j["id"] == old_jid)["state"] == "COMPLETED",
        30,
        "the old allocation to be released",
    )
    # and one successor is waiting again (no dependency: it ages in the queue)
    pool.wait_for(
        lambda: sum(j["state"] == "PENDING" for j in pool.fake_jobs()) == 1,
        30,
        "a new pending host",
    )
    pool.wait_for(
        lambda: pool.record("dummy_train")["runtime"]["final_state"] == "finished",
        90,
        "the run to finish",
    )
    pool.wait_for(lambda: pool.run_state(second).status == "COMPLETED", 30, "the run's rc")
    # closing the pool cancels the queued successor, the monitor then ends
    pool.ctl("close")
    assert _monitor_done(pool, 60) == 0
    assert {j["state"] for j in pool.fake_jobs()} == {"COMPLETED", "CANCELLED"}


def test_wall_clock_end_of_a_host_continues_on_the_next(pool):
    # 30 s allocations, USR1 to the batch shell 12 s before the end
    pool.create_pool(*FAST, "slurm.sbatch.time=00:00:30", "slurm.sbatch.signal=B:USR1@12")
    pool.enqueue("a", "aux.iters=250", "aux.tick=0.1")
    pool.start_monitor()
    pool.wait_for(
        lambda: pool.record("dummy_a")["runtime"]["final_state"] == "finished",
        150,
        "the run to finish",
    )
    runs = pool.runs_of("dummy_a")
    assert len(runs) >= 2 and len(pool.fake_jobs()) >= 2
    for earlier, later in zip(runs, runs[1:]):
        assert pool.run_state(earlier).reason == "preempted"
        assert _resumed_from(pool.run_log(later)) == max(_iterations(pool.run_log(earlier)))
    # the policy finishes the record on '[after training is done]', possibly before rc is written
    pool.wait_for(lambda: pool.run_state(runs[-1]).status == "COMPLETED", 30, "the last run's rc")
    # no host ever hit its hard limit: every one was released by its supervisor
    pool.wait_for(
        lambda: all(j["state"] not in ("PENDING", "RUNNING") for j in pool.fake_jobs()),
        60,
        "the last host to be released",
    )
    assert all(j["state"] == "COMPLETED" for j in pool.fake_jobs())
    assert all(
        "signal USR1" in (pool.layout.host_dir(f"host{i + 1:03d}") / "exited").read_text()
        for i in range(len(runs) - 1)
    )


def test_lost_host_requeues_from_the_last_checkpoint(pool):
    pool.create_pool(*FAST)
    pool.enqueue("a", "aux.iters=200", "aux.tick=0.1")
    pool.start_monitor()
    first = pool.wait_for(lambda: pool.runs_of("dummy_a"), 30, "run")[0]
    pool.wait_for(lambda: max(_iterations(pool.run_log(first)) or [0]) >= 25, 60, "25 iterations")
    pool.slurm("_crash", pool.run_state(first).started["job"])  # the nodes die
    pool.wait_for(lambda: pool.run_state(first).reason == "host_lost", 60, "the loss to be noticed")
    second = pool.wait_for(
        lambda: [r for r in pool.runs_of("dummy_a") if r != first], 60, "requeue"
    )[0]
    pool.wait_for(
        lambda: pool.run_state(second).status == "COMPLETED", 90, "the run to finish elsewhere"
    )
    assert pool.run_state(second).started["job"] != pool.run_state(first).started["job"]
    resumed = _resumed_from(pool.run_log(second))
    assert resumed is not None and resumed % 10 == 0 and resumed >= 20  # the last periodic save


def test_excluded_nodes_are_never_used(pool):
    pool.create_pool(*FAST)
    pool.ctl("exclude", "n1000-0")  # the first fake allocation is job 1000
    pool.enqueue("a")
    pool.start_monitor()
    rid = pool.wait_for(lambda: pool.runs_of("dummy_a"), 30, "run")[0]
    pool.wait_for(lambda: pool.run_state(rid).status == "COMPLETED", 60, "the run")
    assert pool.run_state(rid).started["nodes"] == "n1000-1"


def test_two_node_job_gets_both_nodes_and_a_moving_port(pool):
    pool.create_pool(*FAST)
    for tag in ("a", "b"):
        pool.enqueue(
            tag,
            "slurm.sbatch.nodes=2",
            "+slurm.env.MASTER_PORT=29500",
            "backend.command='echo port=$MASTER_PORT node=$SLURM_NODEID of $SLURM_NNODES on $SLURMD_NODENAME'",
        )
    pool.start_monitor()
    assert _monitor_done(pool, 120) == 0
    ports = []
    for tag in ("a", "b"):
        rid = pool.runs_of(f"dummy_{tag}")[0]
        log = pool.run_log(rid)
        assert sorted(re.findall(r"node=(\d) of 2 on (\S+)", log)) == [
            ("0", "n1000-0"),
            ("1", "n1000-1"),
        ]
        ports.append(set(re.findall(r"port=(\d+)", log)))
    assert ports[0] != ports[1], "consecutive steps must not share a rendezvous port"


def test_job_time_limit_is_enforced_per_run(pool):
    pool.create_pool(*FAST)
    pool.enqueue("slow", "aux.iters=1000", "aux.tick=0.1", "slurm.sbatch.time=00:00:04")
    pool.start_monitor()
    rid = pool.wait_for(lambda: pool.runs_of("dummy_slow"), 30, "run")[0]
    pool.wait_for(lambda: pool.run_state(rid).status == "TIMEOUT", 60, "the run's own time limit")
    pool.wait_for(
        lambda: pool.record("dummy_slow")["runtime"]["final_state"] == "cancelled", 30, "final"
    )


def test_enqueue_rejects_a_job_that_cannot_fit(pool):
    pool.create_pool(*FAST)
    proc = pool.enqueue("huge", "slurm.sbatch.nodes=4", check=False)
    assert proc.returncode != 0
    assert "needs 4 nodes, the pool's hosts have 2" in proc.stdout + proc.stderr
    assert not list(pool.session_dir.glob("*.job.json"))


def test_idle_host_releases_itself_when_the_monitor_dies(pool):
    pool.create_pool(*FAST, "job.pool_host.idle_exit_s=3600", "job.pool_host.monitor_dead_exit_s=4")
    pool.enqueue("a")
    pool.start_monitor()
    rid = pool.wait_for(lambda: pool.runs_of("dummy_a"), 30, "run")[0]
    pool.wait_for(lambda: pool.run_state(rid).status == "COMPLETED", 60, "the run")
    pool.monitor.kill()  # no clean shutdown
    pool.monitor = None
    pool.wait_for(
        lambda: pool.fake_jobs()[0]["state"] == "COMPLETED", 60, "the host to release itself"
    )
    exited = (pool.layout.host_dir("host001") / "exited").read_text()
    assert "no monitor tick" in exited


def test_a_new_monitor_adopts_a_running_pool(pool):
    pool.create_pool(*FAST)
    pool.enqueue("a", "aux.iters=150", "aux.tick=0.1")
    pool.start_monitor()
    rid = pool.wait_for(lambda: pool.runs_of("dummy_a"), 30, "run")[0]
    pool.wait_for(lambda: pool.run_state(rid).status == "RUNNING", 30, "running")
    pool.monitor.kill()
    pool.monitor = None
    pool.start_monitor()  # a different process, same session
    pool.wait_for(
        lambda: pool.record("dummy_a")["runtime"]["final_state"] == "finished", 90, "finish"
    )
    assert pool.runs_of("dummy_a") == [rid] and len(pool.fake_jobs()) == 1


def test_status_lists_hosts_and_runs(pool):
    pool.create_pool(*FAST)
    pool.enqueue("a")
    pool.start_monitor()
    assert _monitor_done(pool, 120) == 0
    out = pool.ctl("status")
    assert "host001" in out and "COMPLETED" in out and "dummy_a" in out
    meta = json.loads(pool.layout.pool_json.read_text())
    assert meta["config"]["poll_s"] == 1
    pool.ctl("set", "keep_pending=1", "handover=true")
    cfg = pool.layout.load_config()
    assert (cfg.keep_pending, cfg.handover) == (1, True)
    assert "POOL_POLL_S=1" in pool.layout.pool_env.read_text()


def test_monitor_resumes_a_pool_that_ran_out_of_work(pool):
    pool.create_pool(*FAST)
    pool.enqueue("a")
    pool.start_monitor()
    assert _monitor_done(pool, 120) == 0  # writes .monitor.stop "all jobs finished"
    pool.enqueue("b")
    pool.start_monitor()  # monitor_autoexp must not refuse the pool session
    pool.wait_for(lambda: pool.record("dummy_b")["runtime"]["final_state"] == "finished", 120, "b")


def test_bad_node_in_a_live_host(pool):
    pool.create_pool(*FAST)
    pool.enqueue("a", "aux.iters=200", "aux.tick=0.1")
    pool.start_monitor()
    first = pool.wait_for(lambda: pool.runs_of("dummy_a"), 30, "run")[0]
    pool.wait_for(lambda: max(_iterations(pool.run_log(first)) or [0]) >= 15, 60, "15 iterations")
    jid, bad = pool.run_state(first).started["job"], pool.run_state(first).started["nodes"]
    # a node goes bad: exclude it, move the run off it -- the host has a spare node
    pool.ctl("exclude", bad)
    pool.ctl("preempt", first)
    second = pool.wait_for(
        lambda: [r for r in pool.runs_of("dummy_a") if r != first], 60, "requeue"
    )[0]
    pool.wait_for(
        lambda: pool.run_state(second).status == "RUNNING", 30, "the run on the spare node"
    )
    state = pool.run_state(second)
    assert (state.started["job"], state.started["nodes"]) == (jid, f"n{jid}-1")
    # a 2-node job cannot run on the degraded host: it is released at once (not after
    # idle_exit_s) and the new allocation is told to avoid the bad node
    pool.enqueue("big", "slurm.sbatch.nodes=2")
    pool.wait_for(lambda: pool.record("dummy_a")["runtime"]["final_state"] == "finished", 90, "a")
    big = pool.wait_for(lambda: pool.runs_of("dummy_big"), 30, "big queued")[0]
    pool.wait_for(lambda: pool.run_state(big).status == "COMPLETED", 90, "big on a fresh host")
    assert pool.run_state(big).started["job"] != jid
    exited = (pool.layout.host_dir("host001") / "exited").read_text()
    assert "cannot run any queued job (needs 2 node(s), 1 usable)" in exited
    assert (
        f"#SBATCH --exclude={bad}" in (pool.layout.host_dir("host002") / "host.sbatch").read_text()
    )


def test_a_node_slurm_reports_down_is_skipped_by_the_next_run(pool):
    # the spare-node model: a host one node larger than its job, with --no-kill, survives a
    # node failure; the supervisor keeps the next run off a node SLURM reports unusable,
    # before any fault scan has named it
    pool.create_pool(*FAST)
    pool.enqueue("a", "aux.iters=300", "aux.tick=0.1")
    pool.start_monitor()
    first = pool.wait_for(lambda: pool.runs_of("dummy_a"), 30, "run")[0]
    pool.wait_for(lambda: max(_iterations(pool.run_log(first)) or [0]) >= 15, 60, "15 iterations")
    jid, node = pool.run_state(first).started["job"], pool.run_state(first).started["nodes"]
    (pool.fs / "node_states").write_text(f"{node} down*\n")
    pool.ctl("preempt", first)
    second = pool.wait_for(
        lambda: [r for r in pool.runs_of("dummy_a") if r != first], 60, "requeue"
    )[0]
    pool.wait_for(lambda: pool.run_state(second).status == "RUNNING", 30, "the run on the spare")
    state = pool.run_state(second)
    assert (state.started["job"], state.started["nodes"]) == (jid, f"n{jid}-1")
    assert pool.layout.exclude_nodes.read_text().strip() == ""  # nobody had to exclude it

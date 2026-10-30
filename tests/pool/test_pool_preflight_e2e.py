"""End-to-end tests of the pool's preflight node check, verified handover and held hosts
(oellm_autoexp/monitor/pool.py, pool_preflight.py, scripts/pool/supervisor.sh) against
tests/fakeslurm. The benchmark is a fake that prints scripts/korbi/pool/fsdp_comm_bench.py's
result lines, one canned output per round."""

from __future__ import annotations

import time
from pathlib import Path

from tests.pool.test_pool_e2e import FAST, _iterations, pool  # noqa: F401  (pool: fixture)

GOOD = """[fsdp_bench] group gpu0 (4 ranks, 256 MB): all_gather 12.0 ms (22.2 GB/s), reduce_scatter 11.7 ms (22.8 GB/s)
[fsdp_bench] group gpu1 (4 ranks, 256 MB): all_gather 11.9 ms (22.3 GB/s), reduce_scatter 11.7 ms (22.8 GB/s)
[fsdp_bench] group gpu2 (4 ranks, 256 MB): all_gather 11.8 ms (22.6 GB/s), reduce_scatter 11.7 ms (22.7 GB/s)
[fsdp_bench] group gpu3 (4 ranks, 256 MB): all_gather 11.9 ms (22.3 GB/s), reduce_scatter 11.6 ms (22.9 GB/s)
[fsdp_bench] slowest receivers (from -> to, GB/s): n1000-0->n1000-2:1=12.1, n1000-2->n1000-0:3=12.2"""

# rail 2 degraded on n1000-3: it is in both slowest GPU-2 pairs (receiver of one, sender of the other)
SLOW_N3 = GOOD.replace(
    "(22.6 GB/s), reduce_scatter 11.7 ms (22.7 GB/s)",
    "(11.9 GB/s), reduce_scatter 22.6 ms (11.8 GB/s)",
).replace(
    "n1000-0->n1000-2:1=12.1, n1000-2->n1000-0:3=12.2",
    "n1000-2->n1000-3:2=8.1, n1000-3->n1000-0:2=8.1, n1000-0->n1000-2:2=9.0",
)


def _fake_bench(tmp_path: Path, outputs: list[str]) -> Path:
    """A 'benchmark' printing outputs[k] in its k-th run (the last one from then on)."""
    for k, text in enumerate(outputs):
        (tmp_path / f"bench_{k}.txt").write_text(text + "\n")
    script = tmp_path / "fake_bench.sh"
    script.write_text(
        "#!/bin/bash\n"
        '[ "${SLURM_PROCID:-0}" = 0 ] || exit 0  # the real benchmark prints on rank 0 only\n'
        f"c={tmp_path}/bench_count; n=$(( $(cat $c 2>/dev/null || echo 0) )); echo $((n + 1)) > $c\n"
        f"[ -e {tmp_path}/bench_$n.txt ] || n={len(outputs) - 1}\n"
        f"cat {tmp_path}/bench_$n.txt\n"
    )
    script.chmod(0o755)
    return script


def _check(script: Path | None = None) -> tuple[str, ...]:
    """Host-config overrides: the pool's preflight check is config/experiments/tests/node_health_local
    (the real benchmark on CPU), with ``script`` as its launcher if given (canned outputs)."""
    args = (
        "job.pool_host.preflight=true",
        "job.pool_host.preflight_check=experiments/tests/node_health_local",
    )
    if script is not None:
        args += (f'job.pool_host.preflight_check_overrides=["backend.launcher=bash {script}"]',)
    return args


def _host_exit(pool) -> str:  # noqa: F811
    for name in ("exited", "control/exit"):
        p = pool.pool_dir / "hosts" / "host001" / name
        if p.exists():
            return p.read_text()
    return ""


def test_preflight_excludes_bad_nodes_before_any_run(pool, tmp_path):  # noqa: F811
    """GPU probe: n1000-1 shows 2 of 4 GPUs -> excluded at once. Benchmark round 1: rail 2 slow,
    n1000-3 common to the two slowest pairs -> excluded; round 2 clean -> the host takes runs,
    only on n1000-0 / n1000-2."""
    script = _fake_bench(tmp_path, [SLOW_N3, GOOD])
    pool.create_pool(
        *FAST,
        "slurm.sbatch.nodes=4",
        *_check(script),
        "job.pool_host.preflight_gpus_per_node=4",
        "job.pool_host.preflight_gpu_cmd='[ \"$SLURMD_NODENAME\" = n1000-1 ] && echo 2 || echo 4'",
        "job.pool_host.max_runs_per_host=2",
    )
    pool.enqueue("work", "aux.iters=10", "aux.tick=0.1", "slurm.sbatch.nodes=2")
    pool.start_monitor()
    rid = pool.wait_for(lambda: pool.runs_of("dummy_work"), 60, "the run")[0]
    pool.wait_for(
        lambda: pool.run_state(rid).status in ("RUNNING", "COMPLETED"), 120, "the run starts"
    )
    nodes = pool.run_state(rid).started["nodes"].split(",")
    excluded = set((pool.pool_dir / "exclude_nodes").read_text().split())
    assert {"n1000-1", "n1000-3"} <= excluded, excluded
    assert sorted(nodes) == ["n1000-0", "n1000-2"], nodes
    assert (tmp_path / "bench_count").read_text().strip() == "2", "two benchmark rounds before it"
    log = pool.monitor_log.read_text()
    assert "excluding n1000-3" in log and "preflight host001: passed" in log
    assert not _host_exit(pool)


def test_preflight_failure_releases_the_host(pool, tmp_path):  # noqa: F811
    script = _fake_bench(tmp_path, ["(no result)"])
    pool.create_pool(*FAST, "slurm.sbatch.nodes=2", *_check(script))
    pool.enqueue("work", "aux.iters=10", "aux.tick=0.1")
    pool.start_monitor()
    pool.wait_for(lambda: "preflight failed" in _host_exit(pool), 120, "the host released")
    rid = pool.runs_of("dummy_work")[0]
    assert not pool.run_state(rid).started, "no run on a host that failed its preflight"


def test_preflight_with_the_real_benchmark(pool):  # noqa: F811
    """The composed check end to end: run_autoexp enqueues node_health_local as the template, the
    pool runs the real benchmark (CPU / gloo, two agents) on the new host, passes it, then the run."""
    pool.create_pool(*FAST, "slurm.sbatch.nodes=2", *_check(), "job.pool_host.preflight_ratio=0.0")
    pool.enqueue("work", "aux.iters=5", "aux.tick=0.1", "slurm.sbatch.nodes=2")
    pool.start_monitor()
    rid = pool.wait_for(lambda: pool.runs_of("dummy_work"), 60, "the run")[0]
    pool.wait_for(
        lambda: pool.run_state(rid).status in ("RUNNING", "COMPLETED"), 180, "the run starts"
    )
    log = pool.monitor_log.read_text()
    assert "preflight template set" in log and "preflight host001: passed" in log, log[-3000:]
    template = (pool.pool_dir / "preflight_template" / "rid").read_text().strip()
    assert pool.run_state(template).status == "COMPLETED", pool.run_state(template)
    bench_logs = list((pool.root / "runs" / "node_health").glob("slurm-*.log"))
    assert any("[node_health] {" in p.read_text() for p in bench_logs), bench_logs


def test_verified_handover_keeps_the_old_host_until_healthy(pool):  # noqa: F811
    """The run moves to the new host; the old allocation stays until 20 healthy steps there."""
    pool.create_pool(
        *FAST,
        "job.pool_host.keep_pending=1",
        "job.pool_host.handover=true",
        "job.pool_host.handover_min_age_s=3",
        "job.pool_host.handover_verify=true",
        "job.pool_host.handover_healthy_steps=20",
    )
    pool.enqueue("train", "aux.iters=600", "aux.tick=0.2")
    pool.start_monitor(hold_new_jobs=True)
    jobs = pool.wait_for(lambda: len(pool.fake_jobs()) >= 2 and pool.fake_jobs(), 30, "two hosts")
    old_jid, new_jid = jobs[0]["id"], jobs[1]["id"]
    pool.release(old_jid)
    first = pool.wait_for(lambda: pool.runs_of("dummy_train"), 30, "run")[0]
    pool.wait_for(
        lambda: max(_iterations(pool.run_log(first)) or [0]) >= 15, 60, "training on the old host"
    )
    pool.release(new_jid)
    second = pool.wait_for(
        lambda: [r for r in pool.runs_of("dummy_train") if r != first], 90, "the moved run"
    )[0]
    pool.wait_for(lambda: pool.run_state(second).status == "RUNNING", 60, "running on the new host")
    assert pool.run_state(second).started["job"] == new_jid
    old = next(j for j in pool.fake_jobs() if j["id"] == old_jid)
    assert old["state"] == "RUNNING", "the old allocation is kept until the moved run is healthy"
    pool.wait_for(
        lambda: next(j for j in pool.fake_jobs() if j["id"] == old_jid)["state"] == "COMPLETED",
        90,
        "the old allocation released once healthy",
    )
    assert len(_iterations(pool.run_log(second))) >= 20
    assert "verified" in pool.monitor_log.read_text()


def test_verified_handover_falls_back_when_the_new_host_does_not_get_healthy(pool):  # noqa: F811
    pool.create_pool(
        *FAST,
        "job.pool_host.keep_pending=1",
        "job.pool_host.handover=true",
        "job.pool_host.handover_min_age_s=3",
        "job.pool_host.handover_verify=true",
        "job.pool_host.handover_healthy_pattern=NEVER-PRINTED",
        "job.pool_host.handover_verify_timeout_s=15",
    )
    pool.enqueue("train", "aux.iters=2000", "aux.tick=0.1")
    pool.start_monitor(hold_new_jobs=True)
    jobs = pool.wait_for(lambda: len(pool.fake_jobs()) >= 2 and pool.fake_jobs(), 30, "two hosts")
    old_jid, new_jid = jobs[0]["id"], jobs[1]["id"]
    pool.release(old_jid)
    first = pool.wait_for(lambda: pool.runs_of("dummy_train"), 30, "run")[0]
    pool.wait_for(
        lambda: max(_iterations(pool.run_log(first)) or [0]) >= 15, 60, "training on the old host"
    )
    pool.release(new_jid)
    pool.wait_for(
        lambda: next(j for j in pool.fake_jobs() if j["id"] == new_jid)["state"]
        in ("COMPLETED", "CANCELLED"),
        120,
        "the NEW host released",
    )
    back = pool.wait_for(
        lambda: [
            r
            for r in pool.runs_of("dummy_train")
            if pool.run_state(r).status == "RUNNING"
            and pool.run_state(r).started.get("job") == old_jid
            and r != first
        ],
        120,
        "the run back on the old host",
    )
    assert back
    assert next(j for j in pool.fake_jobs() if j["id"] == old_jid)["state"] == "RUNNING"
    assert "FAILED" in pool.monitor_log.read_text()


def test_a_held_host_is_not_released_when_idle(pool):  # noqa: F811
    pool.create_pool(*FAST, "job.pool_host.hold_hosts=true", "job.pool_host.keep_monitor=true")
    pool.enqueue("a", "aux.iters=5", "aux.tick=0.1")
    pool.start_monitor()
    pool.wait_for(
        lambda: pool.record("dummy_a")["runtime"]["final_state"] == "finished", 60, "the run"
    )
    time.sleep(12)  # idle_exit_s is 5
    assert not _host_exit(pool), _host_exit(pool)
    assert pool.fake_jobs()[0]["state"] == "RUNNING"


def test_switching_preflight_on_in_a_running_pool_keeps_the_busy_host(pool):  # noqa: F811
    """preflight enabled live while a host works (its supervisor started without the GPU probe):
    the host counts as checked -- the next run goes straight onto it, no benchmark, no release."""
    pool.create_pool(*FAST)
    pool.enqueue("a", "aux.iters=40", "aux.tick=0.1")
    pool.start_monitor()
    first = pool.wait_for(lambda: pool.runs_of("dummy_a"), 30, "run a")[0]
    pool.wait_for(lambda: pool.run_state(first).status == "RUNNING", 60, "a running")
    pool.ctl("set", "preflight=true", "preflight_gpus_per_node=4")
    pool.enqueue("b", "aux.iters=5", "aux.tick=0.1")
    second = pool.wait_for(lambda: pool.runs_of("dummy_b"), 30, "run b")[0]
    pool.wait_for(lambda: pool.run_state(second).status in ("RUNNING", "COMPLETED"), 90, "b starts")
    assert pool.run_state(second).started["job"] == pool.run_state(first).started["job"]
    assert "had runs before the node check" in pool.monitor_log.read_text()
    assert not _host_exit(pool)


def _held_host_survives_a_dead_monitor(pool, *create, hold_by_ctl: bool):  # noqa: F811
    pool.create_pool(
        *FAST, "job.pool_host.idle_exit_s=3600", "job.pool_host.monitor_dead_exit_s=4", *create
    )
    pool.enqueue("a", "aux.iters=5", "aux.tick=0.1")
    pool.start_monitor()
    rid = pool.wait_for(lambda: pool.runs_of("dummy_a"), 30, "run")[0]
    pool.wait_for(lambda: pool.run_state(rid).status == "COMPLETED", 60, "the run")
    if hold_by_ctl:
        pool.ctl("hold", "host001")
    pool.monitor.kill()  # no clean shutdown: the supervisor's dead-man switch is armed
    pool.monitor = None
    time.sleep(15)  # monitor_dead_exit_s is 4: without the hold it is gone by now
    assert pool.fake_jobs()[0]["state"] == "RUNNING", pool.debug()
    assert not (pool.layout.host_dir("host001") / "exited").exists()
    log = pool.layout.host_dir("host001").parent.parent / f"slurm-{pool.fake_jobs()[0]['id']}.log"
    assert "held: idle and no monitor tick" in log.read_text(), log.read_text()[-2000:]


def test_a_held_host_survives_a_dead_monitor(pool):  # noqa: F811
    """pool_ctl hold: an allocation with no runs stays even while the monitor is away."""
    _held_host_survives_a_dead_monitor(pool, hold_by_ctl=True)


def test_hold_hosts_survives_a_dead_monitor(pool):  # noqa: F811
    _held_host_survives_a_dead_monitor(pool, "job.pool_host.hold_hosts=true", hold_by_ctl=False)


def test_a_held_queued_host_is_not_cancelled_when_the_work_is_gone(pool):  # noqa: F811
    """keep_pending 1 queues a successor while there is work; when the work is done the
    successor is cancelled -- unless it is held."""
    pool.create_pool(*FAST, "job.pool_host.keep_pending=1")
    pool.enqueue("a", "aux.iters=20", "aux.tick=0.2")
    pool.start_monitor(hold_new_jobs=True)
    jobs = pool.wait_for(lambda: len(pool.fake_jobs()) >= 2 and pool.fake_jobs(), 30, "two hosts")
    first, successor = jobs[0]["id"], jobs[1]["id"]
    pool.ctl("hold", "host002")
    pool.release(first)
    rid = pool.wait_for(lambda: pool.runs_of("dummy_a"), 30, "run")[0]
    pool.wait_for(lambda: pool.run_state(rid).status == "COMPLETED", 90, "the run")
    time.sleep(10)  # several ticks with no work
    state = next(j for j in pool.fake_jobs() if j["id"] == successor)["state"]
    assert state == "PENDING", (state, pool.debug())

"""The post-training backend in a pool, against tests/fakeslurm.

The real job config (config/experiments/post_training/launch_mn5.yaml) is enqueued with a fake
post-training checkout. Its prepare command writes a launch script that runs the dummy trainer.
What matters here: the prepare command runs once, in the enqueuing process; the pool's own
requeue after a stop runs the stored launch command again and continues from the checkpoint.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import textwrap

import pytest
import yaml

from tests.pool.harness import REPO, PoolEnv

pytestmark = pytest.mark.skipif(
    not (shutil.which("bash") and shutil.which("pgrep")), reason="needs bash and pgrep"
)

FAST = ("job.pool_host.idle_exit_s=5", "job.pool_host.poll_s=1")

# Stands in for post-training's scripts/prepare_run.py: the same arguments and result line.
_FAKE_PREPARE = textwrap.dedent(
    """\
    import json, os, sys
    from pathlib import Path
    import yaml

    argv = sys.argv[1:]
    config = yaml.safe_load(Path(argv[argv.index("--config") + 1]).read_text())
    run_dir = Path(os.path.realpath(config["paths"]["output_base"])) / config["run_name"]
    new_run = not (run_dir / "config.yaml").exists()
    with open("calls.log", "a") as f:
        f.write(" ".join(argv) + "\\n")
    if "--check" not in argv:
        (run_dir / "slurm").mkdir(parents=True, exist_ok=True)
        (run_dir / "config.yaml").write_text(yaml.safe_dump(config))
        launch = run_dir / "slurm" / "launch.sh"
        launch.write_text(
            "#!/bin/bash\\n"
            f"exec python3 {config['trainer']} --ckpt-dir {run_dir}/checkpoints "
            "--iters 150 --tick 0.1 --save-every 10 --save-seconds 0.5\\n"
        )
    result = {
        "run_dir": str(run_dir),
        "launch_script": str(run_dir / "slurm" / "launch.sh"),
        "new_run": new_run,
    }
    print("PREPARE_RUN_RESULT " + json.dumps(result))
    """
)


@pytest.fixture
def pool(tmp_path):
    env = PoolEnv(tmp_path)
    yield env
    env.teardown()


@pytest.fixture
def checkout(tmp_path):
    """A fake post-training checkout with one experiment, `my-run`."""
    root = tmp_path / "post-training"
    (root / "scripts").mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "scripts" / "prepare_run.py").write_text(_FAKE_PREPARE)
    config = {
        "run_name": "my-run",
        "paths": {"output_base": "outputs"},
        "slurm": {"num_nodes": 1, "gpus_per_node": 4},
        "trainer": str(REPO / "scripts" / "sigterm_dummy_trainer.py"),
    }
    (root / "configs" / "my-run.yaml").write_text(yaml.safe_dump(config))
    return root


def _enqueue(pool: PoolEnv, checkout, *extra: str):
    return pool.autoexp(
        "--no-monitor",
        "--pool",
        str(pool.pool_dir),
        "--config-name",
        "experiments/post_training/launch_mn5",
        f"backend.repo_dir={checkout}",
        f"backend.python={sys.executable}",
        "backend.config=configs/my-run.yaml",
        "slurm.sbatch.nodes=1",
        "slurm.sbatch.time=00:05:00",
        "slurm.sbatch.account=dev",  # the account of the fake pool's hosts
        *extra,
    )


def _iterations(log: str) -> list[int]:
    return [int(m) for m in re.findall(r"iteration\s+(\d+)/", log)]


def _resumed_from(log: str) -> int | None:
    m = re.search(r"loading checkpoint from \S+ at iteration (\d+)", log)
    return int(m.group(1)) if m else None


def test_dry_run_checks_and_prepares_nothing(pool, checkout):
    pool.create_pool(*FAST)

    out = _enqueue(pool, checkout, "--dry-run", "--dry-run-root", str(pool.root / "dry")).stdout

    assert "DRY RUN: my-run_" in out and "CANNOT RUN IN THIS POOL" not in out
    assert "--check" in (checkout / "calls.log").read_text()
    assert not (checkout / "outputs").exists()
    assert pool.records() == []


def test_enqueue_prepares_once_and_a_requeue_continues_the_run(pool, checkout):
    pool.create_pool(*FAST)
    _enqueue(pool, checkout)
    run_dir = checkout / "outputs" / "my-run"
    assert (run_dir / "slurm" / "launch.sh").is_file()
    prepared = (checkout / "calls.log").read_text()
    assert "--num-nodes 1 --gpus-per-node 4" in prepared and "--check" not in prepared

    pool.start_monitor()
    first = pool.wait_for(lambda: pool.runs_of("my-run"), 30, "run")[0]
    # the job's logs are next to the run's own output
    run_json = json.loads((pool.layout.run_dir(first) / "run.json").read_text())
    assert run_json["log"].startswith(str(run_dir / "autoexp"))
    pool.wait_for(lambda: max(_iterations(pool.run_log(first)) or [0]) >= 25, 60, "25 iterations")
    pool.ctl("preempt", first)
    pool.wait_for(lambda: pool.run_state(first).status == "PREEMPTED", 30, "the preemption")
    second = pool.wait_for(
        lambda: [r for r in pool.runs_of("my-run") if r != first], 30, "the requeue"
    )[0]
    pool.wait_for(
        lambda: pool.record("my-run")["runtime"]["final_state"] == "finished",
        90,
        "the continued run to finish",
    )

    resumed = _resumed_from(pool.run_log(second))
    assert resumed is not None and 0 < resumed <= max(_iterations(pool.run_log(first)))
    assert len(pool.runs_of("my-run")) == 2
    # the monitor runs the stored launch command; only the enqueue calls post-training
    assert (checkout / "calls.log").read_text() == prepared

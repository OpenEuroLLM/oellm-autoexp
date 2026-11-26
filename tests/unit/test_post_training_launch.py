"""The post-training job config (config/experiments/post_training/launch_mn5.yaml).

post-training (https://github.com/OpenEuroLLM/post-training) prepares its run directory and its
launch script itself; the job only runs that script through the BashBackend, in a pool step or in
a job of its own. These tests pin what the launch script relies on: one srun task per node, no
container wrapper, and none of the Megatron env exports.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
RUN_DIR = "/scratch/outputs/my-run"


@pytest.fixture(scope="module")
def record():
    from oellm_autoexp.config.loader import load_config_reference
    from oellm_autoexp.config.schema import BackendInterface, ConfigSetup, RootConfig
    from oellm_autoexp.orchestrator import _build_job_record, build_execution_plan

    setup = ConfigSetup(
        pwd=".",
        config_name="experiments/post_training/launch_mn5",
        config_dir=str(REPO_CONFIG_DIR),
        overrides=[
            "aux.output_base=/scratch/outputs",
            "aux.run_name=my-run",
            "slurm.sbatch.account=cns144",
            "slurm.sbatch.qos=acc_xehpc",
        ],
    )
    root = load_config_reference(config_setup=setup, config_class=RootConfig)
    plan = build_execution_plan(root, setup)
    assert len(plan.jobs) == 1
    plan.jobs[0].config.backend.instantiate(BackendInterface).validate()
    return _build_job_record(plan, plan.jobs[0], "test-session")


def test_job_runs_the_launch_script_of_the_run_directory(record):
    definition = record.definition
    assert definition.name == "my-run"
    assert definition.slurm.command == [f"bash {RUN_DIR}/slurm/launch.sh"]
    # logs and resolved configs go next to the run's own output
    assert definition.log_path.startswith(f"{RUN_DIR}/autoexp/")


def test_environment_is_left_to_the_launch_script(record):
    slurm = record.definition.slurm
    assert slurm.launcher_cmd == ""
    # the launch script sources its own frozen environment: an HF_HOME, NCCL_* or OMP_NUM_THREADS
    # export of a cluster group would change what the training runs with
    assert dict(slurm.env) == {"SLURM_CPUS_PER_TASK": "160"}


def test_failed_run_is_restarted_a_bounded_number_of_times(record):
    (event,) = record.definition.state_events
    assert tuple(event.transition) == ("RUNNING", "FAILED")
    assert event.action.class_name == "RestartAction"
    # per event, not MaxAttemptsCondition: the pool's own requeues must not use the budget up
    assert event.condition.class_name == "MaxActionFiresCondition"
    assert event.condition.max_fires == 3
    assert record.definition.log_events == []


def test_pool_step_runs_one_task_per_node(record):
    from oellm_autoexp.monitor.pool import render_step_script

    slurm = record.definition.slurm
    assert slurm.sbatch.nodes == 8
    script = render_step_script(slurm, "pool.0001", log_path=record.definition.log_path)
    (srun,) = [line for line in script.splitlines() if line.startswith("srun ")]

    assert "--kill-on-bad-exit=1" in srun and "--wait=60" in srun
    assert "--nodes=$AUTOEXP_STEP_NUM_NODES" in srun
    assert "--ntasks-per-node=$AUTOEXP_STEP_NTASKS_PER_NODE" in srun
    assert "--cpus-per-task=160" in srun
    assert srun.endswith(f"bash -c ' bash {RUN_DIR}/slurm/launch.sh'")
    # the only export; the job's own sbatch header is a comment in a step script
    exports = [line for line in script.splitlines() if line.startswith("export ")]
    assert exports == ["export SLURM_CPUS_PER_TASK=160"]
    assert not any(line.startswith("#SBATCH") for line in script.splitlines())
    assert "#   #SBATCH --account=cns144" in script

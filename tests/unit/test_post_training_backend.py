"""The post-training backend (oellm_autoexp/backends/post_training_backend.py)
and its job config (config/experiments/post_training/launch_mn5.yaml).

post-training (https://github.com/OpenEuroLLM/post-training) keeps its own config file, which
describes the experiment. The backend calls post-training's prepare command in a subprocess and
runs the launch script that the command writes. These tests use a fake post-training checkout:
its prepare command records how it was called and prints the result that the test gave it.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

from oellm_autoexp.backends.post_training_backend import (
    PostTrainingBackend,
    PostTrainingBackendConfig,
    _read_run_identity,
)
from oellm_autoexp.utils import dry_run

REPO_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"

_FAKE_PREPARE = textwrap.dedent(
    """\
    import json, os, sys
    from pathlib import Path

    here = Path.cwd()
    env = {k: os.environ.get(k) for k in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PYTHONNOUSERSITE")}
    with (here / "calls.jsonl").open("a") as f:
        f.write(json.dumps({"argv": sys.argv[1:], "cwd": str(here), "env": env}) + "\\n")
    if (here / "fail.txt").exists():
        print((here / "fail.txt").read_text(), file=sys.stderr)
        sys.exit(1)
    print("2026-01-01 | INFO | some log line of the prepare command")
    if (here / "result.json").exists():
        print("PREPARE_RUN_RESULT " + (here / "result.json").read_text())
    """
)


class FakeRepo:
    """A post-training checkout with one experiment YAML and a fake prepare
    command."""

    def __init__(self, root: Path, *, run_name: str = "my-run", output_base: str | None = None):
        self.root = root
        (root / "scripts").mkdir(parents=True)
        (root / "configs").mkdir()
        (root / "scripts" / "prepare_run.py").write_text(_FAKE_PREPARE)
        self.config = "configs/my-run.yaml"
        self.run_dir = self.write_config(run_name=run_name, output_base=output_base or "outputs")
        self.set_result()

    def write_config(self, **values) -> str:
        """Write the experiment YAML; return the run directory that post-
        training would use."""
        data: dict = {"method": "sft"}
        if values.get("run_name"):
            data["run_name"] = values["run_name"]
        paths = {key: values[key] for key in ("output_base", "debug_base") if values.get(key)}
        if paths:
            data["paths"] = paths
        if values.get("debug"):
            data["debug"] = {"enabled": True}
        (self.root / self.config).write_text(yaml.safe_dump(data))
        base = values.get("debug_base") if values.get("debug") else values.get("output_base")
        return os.path.join(
            os.path.realpath(self.root / (base or "")), values.get("run_name") or ""
        )

    def set_result(self, run_dir: str | None = None, *, new_run: bool = True) -> None:
        run_dir = run_dir or self.run_dir
        result = {
            "run_dir": run_dir,
            "launch_script": f"{run_dir}/slurm/launch.sh",
            "config": f"{run_dir}/config.yaml",
            "new_run": new_run,
        }
        (self.root / "result.json").write_text(json.dumps(result))

    def fail_with(self, message: str) -> None:
        (self.root / "fail.txt").write_text(message)

    @property
    def calls(self) -> list[dict]:
        path = self.root / "calls.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def backend(self, **overrides) -> PostTrainingBackend:
        values = {
            "repo_dir": str(self.root),
            "python": sys.executable,
            "config": self.config,
            "num_nodes": 8,
            "gpus_per_node": 4,
            "run_dir": self.run_dir,
            **overrides,
        }
        return PostTrainingBackend(PostTrainingBackendConfig(**values))


@pytest.fixture(autouse=True)
def _no_dry_run(monkeypatch):
    monkeypatch.delenv(dry_run.ROOT_ENV, raising=False)
    monkeypatch.delenv(dry_run.ACTIVE_ENV, raising=False)


@pytest.fixture
def repo(tmp_path):
    return FakeRepo(tmp_path / "post-training")


# ---------------------------------------------------------------------------
# The prepare call
# ---------------------------------------------------------------------------


def test_launch_command_runs_the_launch_script_that_prepare_reports(repo):
    assert repo.backend().build_launch_command() == f"bash {repo.run_dir}/slurm/launch.sh"


def test_prepare_gets_the_config_and_the_world_size_of_the_job(repo):
    repo.backend(overrides=["training.learning_rate=1e-5"]).build_launch_command()

    (call,) = repo.calls
    assert call["argv"] == [
        "--config",
        "configs/my-run.yaml",
        "--num-nodes",
        "8",
        "--gpus-per-node",
        "4",
        "training.learning_rate=1e-5",
    ]
    # relative paths of the post-training config resolve against its checkout
    assert call["cwd"] == os.path.realpath(repo.root)


def test_prepare_does_not_get_the_autoexp_environment(repo, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", ".")
    monkeypatch.setenv("PYTHONHOME", "/some/python")
    monkeypatch.setenv("VIRTUAL_ENV", "/some/autoexp_env")
    monkeypatch.delenv("PYTHONNOUSERSITE", raising=False)

    repo.backend().build_launch_command()

    (call,) = repo.calls
    assert call["env"] == {
        "PYTHONPATH": None,
        "PYTHONHOME": None,
        "VIRTUAL_ENV": None,
        "PYTHONNOUSERSITE": "1",
    }


def test_dry_run_only_checks(repo, monkeypatch):
    monkeypatch.setenv(dry_run.ACTIVE_ENV, "1")

    command = repo.backend().build_launch_command()

    assert "--check" in repo.calls[0]["argv"]
    # the record of a dry run still shows the command of the real run
    assert command == f"bash {repo.run_dir}/slurm/launch.sh"


def test_real_run_does_not_pass_check(repo):
    repo.backend().build_launch_command()
    assert "--check" not in repo.calls[0]["argv"]


def test_failed_prepare_stops_with_its_own_message(repo):
    repo.fail_with("prepare_run.py: error: slurm.num_nodes is 4 in x.yaml, but --num-nodes is 8.")

    with pytest.raises(ValueError, match="prepare command failed") as error:
        repo.backend().build_launch_command()

    assert "slurm.num_nodes is 4 in x.yaml" in str(error.value)
    assert "scripts/prepare_run.py --config configs/my-run.yaml" in str(error.value)


def test_prepare_without_a_result_is_an_error(repo):
    (repo.root / "result.json").unlink()
    with pytest.raises(ValueError, match="prepare command failed"):
        repo.backend().build_launch_command()


def test_a_different_run_directory_than_autoexp_derived_is_an_error(repo):
    repo.set_result("/somewhere/else/my-run")
    with pytest.raises(ValueError, match="autoexp derived"):
        repo.backend().build_launch_command()


# ---------------------------------------------------------------------------
# Checks before the prepare call
# ---------------------------------------------------------------------------


def test_interpreter_that_cannot_start_names_the_modules(repo, tmp_path):
    broken = tmp_path / "python"
    broken.write_text(
        "#!/bin/sh\necho 'libpython3.12.so.1.0: cannot open shared object file' >&2\nexit 127\n"
    )
    broken.chmod(0o755)

    with pytest.raises(ValueError, match="module load python") as error:
        repo.backend(python=str(broken)).build_launch_command()

    assert "libpython3.12.so.1.0" in str(error.value)
    assert repo.calls == []


def test_python_must_be_an_executable_file(repo, tmp_path):
    with pytest.raises(ValueError, match="not an executable file"):
        repo.backend(python=str(tmp_path / "missing")).build_launch_command()
    assert repo.calls == []


def test_repo_dir_must_have_the_prepare_command(repo, tmp_path):
    with pytest.raises(ValueError, match="has no scripts/prepare_run.py"):
        repo.backend(repo_dir=str(tmp_path)).build_launch_command()


@pytest.mark.parametrize("name", ["repo_dir", "python", "config"])
def test_link_to_post_training_is_mandatory(repo, name):
    config = PostTrainingBackendConfig(
        **{
            key: value
            for key, value in {
                "repo_dir": str(repo.root),
                "python": sys.executable,
                "config": repo.config,
                "num_nodes": 8,
                "gpus_per_node": 4,
                "run_dir": repo.run_dir,
            }.items()
            if key != name
        }
    )
    with pytest.raises(ValueError, match=f"backend.{name} is mandatory"):
        PostTrainingBackend(config).build_launch_command()
    assert repo.calls == []


@pytest.mark.parametrize(
    "override",
    [
        "run_name=other",
        "slurm.num_nodes=4",
        "slurm.gpus_per_node=2",
        "paths.output_base=/elsewhere",
        "debug.enabled=true",
    ],
)
def test_values_with_one_source_cannot_be_overridden(repo, override):
    with pytest.raises(ValueError, match="cannot be in backend.overrides"):
        repo.backend(overrides=[override]).build_launch_command()
    assert repo.calls == []


# ---------------------------------------------------------------------------
# The run name and the run directory, read from the post-training YAML
# ---------------------------------------------------------------------------


def test_relative_output_base_resolves_against_the_checkout(repo):
    assert _read_run_identity(str(repo.root), repo.config) == (
        "my-run",
        os.path.join(os.path.realpath(repo.root), "outputs", "my-run"),
    )


def test_absolute_output_base_and_absolute_config_path(repo):
    repo.write_config(run_name="my-run", output_base="/scratch/outputs")
    assert _read_run_identity("/unused", str(repo.root / repo.config)) == (
        "my-run",
        "/scratch/outputs/my-run",
    )


def test_debug_run_uses_the_debug_base(repo):
    repo.write_config(
        run_name="my-run", output_base="outputs", debug_base="/scratch/debug", debug=True
    )
    assert _read_run_identity(str(repo.root), repo.config)[1] == "/scratch/debug/my-run"


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"output_base": "outputs"}, "has no run_name"),
        ({"run_name": "my-run"}, "has no paths.output_base"),
        ({"run_name": "${oc.env:USER}-run", "output_base": "outputs"}, "must be a plain value"),
    ],
)
def test_run_identity_must_be_explicit_in_the_file(repo, values, message):
    repo.write_config(**values)
    with pytest.raises(ValueError, match=message):
        _read_run_identity(str(repo.root), repo.config)


def test_missing_config_file_is_an_error(repo):
    with pytest.raises(ValueError, match="not found"):
        _read_run_identity(str(repo.root), "configs/nope.yaml")


# ---------------------------------------------------------------------------
# The job config: config/experiments/post_training/launch_mn5.yaml
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def job(tmp_path_factory):
    """(fake checkout, job record) of launch_mn5 for the fake experiment."""
    from oellm_autoexp.config.loader import load_config_reference
    from oellm_autoexp.config.schema import ConfigSetup, RootConfig
    from oellm_autoexp.orchestrator import _build_job_record, build_execution_plan

    repo = FakeRepo(
        tmp_path_factory.mktemp("job") / "post-training", output_base="/scratch/outputs"
    )
    setup = ConfigSetup(
        pwd=".",
        config_name="experiments/post_training/launch_mn5",
        config_dir=str(REPO_CONFIG_DIR),
        overrides=[
            f"backend.repo_dir={repo.root}",
            f"backend.python={sys.executable}",
            f"backend.config={repo.config}",
            "slurm.sbatch.account=cns144",
            "slurm.sbatch.qos=acc_xehpc",
        ],
    )
    root = load_config_reference(config_setup=setup, config_class=RootConfig)
    plan = build_execution_plan(root, setup)
    assert len(plan.jobs) == 1
    return repo, _build_job_record(plan, plan.jobs[0], "test-session")


def test_job_takes_its_name_and_directory_from_the_post_training_config(job):
    definition = job[1].definition
    assert definition.name == "my-run"
    assert definition.slurm.command == ["bash /scratch/outputs/my-run/slurm/launch.sh"]
    # logs and resolved configs go next to the run's own output
    assert definition.log_path.startswith("/scratch/outputs/my-run/autoexp/")


def test_job_gives_its_world_size_to_post_training(job):
    repo, record = job
    argv = repo.calls[-1]["argv"]
    assert record.definition.slurm.sbatch.nodes == 8
    assert argv[argv.index("--num-nodes") + 1] == "8"
    assert argv[argv.index("--gpus-per-node") + 1] == "4"


def test_environment_is_left_to_the_launch_script(job):
    slurm = job[1].definition.slurm
    assert slurm.launcher_cmd == ""
    # the launch script sources its own frozen environment: an HF_HOME, NCCL_* or OMP_NUM_THREADS
    # export of a cluster group would change what the training runs with
    assert dict(slurm.env) == {"SLURM_CPUS_PER_TASK": "160"}


def test_failed_run_is_restarted_a_bounded_number_of_times(job):
    definition = job[1].definition
    (event,) = definition.state_events
    assert tuple(event.transition) == ("RUNNING", "FAILED")
    assert event.action.class_name == "RestartAction"
    # per event, not MaxAttemptsCondition: the pool's own requeues must not use the budget up
    assert event.condition.class_name == "MaxActionFiresCondition"
    assert event.condition.max_fires == 3
    assert definition.log_events == []


def test_pool_step_runs_one_task_per_node(job):
    from oellm_autoexp.monitor.pool import render_step_script

    definition = job[1].definition
    slurm = definition.slurm
    script = render_step_script(slurm, "pool.0001", log_path=definition.log_path)
    (srun,) = (line for line in script.splitlines() if line.startswith("srun "))

    assert "--kill-on-bad-exit=1" in srun and "--wait=60" in srun
    assert "--nodes=$AUTOEXP_STEP_NUM_NODES" in srun
    assert "--ntasks-per-node=$AUTOEXP_STEP_NTASKS_PER_NODE" in srun
    assert "--cpus-per-task=160" in srun
    assert srun.endswith("bash -c ' bash /scratch/outputs/my-run/slurm/launch.sh'")
    # the only export; the job's own sbatch header is a comment in a step script
    exports = [line for line in script.splitlines() if line.startswith("export ")]
    assert exports == ["export SLURM_CPUS_PER_TASK=160"]
    assert not any(line.startswith("#SBATCH") for line in script.splitlines())
    assert "#   #SBATCH --account=cns144" in script

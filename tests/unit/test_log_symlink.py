"""Tests for log_path_current symlink functionality."""

from pathlib import Path

from oellm_autoexp.config.schema import SlurmConfig
from oellm_autoexp.monitor.watcher import NullMonitor, NullMonitorConfig
from oellm_autoexp.persistence.state_store import MonitorStateStore
from oellm_autoexp.slurm.client import FakeSlurmClient, FakeSlurmClientConfig
from oellm_autoexp.workflow.host import HostRuntime, submit_pending_jobs
from oellm_autoexp.workflow.manifest import (
    ArraySpec,
    ComponentSpec,
    PlanJobSpec,
    PlanManifest,
    RenderedArtifactsSpec,
)
from oellm_autoexp.monitor.controller import MonitorController


def _make_manifest(tmp_path: Path, jobs: list[PlanJobSpec], array: bool) -> PlanManifest:
    rendered = RenderedArtifactsSpec(
        job_scripts=[job.script_path for job in jobs],
        sweep_json=None,
        array=ArraySpec(
            script_path=str(tmp_path / "array.sbatch"), job_name="array", size=len(jobs)
        )
        if array
        else None,
    )
    return PlanManifest(
        version=1,
        plan_id="test",
        created_at=0.0,
        project_name="test",
        config_ref="test",
        config_dir=str(tmp_path),
        overrides=[],
        base_output_dir=str(tmp_path / "outputs"),
        monitoring_state_dir=str(tmp_path / "monitoring"),
        config={"project": {"name": "test"}},
        jobs=jobs,
        rendered=rendered,
        monitor=ComponentSpec(
            module="oellm_autoexp.monitor.watcher",
            class_name="NullMonitor",
            config_module="oellm_autoexp.monitor.watcher",
            config_class="NullMonitorConfig",
            config={"log_path": str(tmp_path / "logs" / "current.log")},
        ),
        slurm_client=ComponentSpec(
            module="oellm_autoexp.slurm.client",
            class_name="FakeSlurmClient",
            config_module="oellm_autoexp.slurm.client",
            config_class="FakeSlurmClientConfig",
            config={},
        ),
        slurm_config_module="oellm_autoexp.config.schema",
        slurm_config_class="SlurmConfig",
        slurm_config={},
        action_queue_dir=str(tmp_path / "actions"),
    )


def _make_runtime(tmp_path: Path, jobs: list[PlanJobSpec], array: bool) -> HostRuntime:
    manifest = _make_manifest(tmp_path, jobs, array=array)
    monitor = NullMonitor(NullMonitorConfig(log_path=str(tmp_path / "logs" / "current.log")))
    slurm_client = FakeSlurmClient(FakeSlurmClientConfig())
    slurm_client.configure(
        SlurmConfig(
            template_path=str(tmp_path / "template.sbatch"),
            script_dir=str(tmp_path / "scripts"),
            log_dir=str(tmp_path / "logs"),
            array=array,
        )
    )
    state_store = MonitorStateStore(str(tmp_path / "monitoring_state"), session_id="test")
    return HostRuntime(
        manifest=manifest,
        monitor=monitor,
        slurm_client=slurm_client,
        state_store=state_store,
        action_queue_dir=tmp_path / "actions",
    )


def test_log_symlink_created_on_submit(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "slurm-%j.out"
    current_path = tmp_path / "logs" / "current.log"
    job = PlanJobSpec(
        name="test_job",
        script_path=str(tmp_path / "job.sbatch"),
        log_path=str(log_path),
        log_path_current=str(current_path),
        output_dir=str(tmp_path / "outputs/test"),
    )

    runtime = _make_runtime(tmp_path, [job], array=False)
    controller = MonitorController(runtime.monitor, runtime.slurm_client, runtime.state_store)
    submit_pending_jobs(runtime, controller, dry_run=False)

    assert current_path.is_symlink()
    assert current_path.readlink() == log_path.with_name("slurm-1.out")


def test_log_symlink_created_for_array_submission(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "slurm-%A_%a.out"
    job1 = PlanJobSpec(
        name="job1",
        script_path=str(tmp_path / "job.sbatch"),
        log_path=str(log_path),
        log_path_current=str(tmp_path / "logs" / "current_0.log"),
        output_dir=str(tmp_path / "outputs/job1"),
    )
    job2 = PlanJobSpec(
        name="job2",
        script_path=str(tmp_path / "job.sbatch"),
        log_path=str(log_path),
        log_path_current=str(tmp_path / "logs" / "current_1.log"),
        output_dir=str(tmp_path / "outputs/job2"),
    )

    runtime = _make_runtime(tmp_path, [job1, job2], array=True)
    controller = MonitorController(runtime.monitor, runtime.slurm_client, runtime.state_store)
    submit_pending_jobs(runtime, controller, dry_run=False)

    assert Path(job1.log_path_current).is_symlink()
    assert Path(job2.log_path_current).is_symlink()
    assert Path(job1.log_path_current).readlink() == log_path.with_name("slurm-1_0.out")
    assert Path(job2.log_path_current).readlink() == log_path.with_name("slurm-1_1.out")

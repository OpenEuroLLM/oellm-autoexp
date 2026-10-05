"""job.array_group: many points of one stage as ONE SLURM job array.

Each point keeps its own record (conditions, events, attempts), the members
share one script that dispatches on $SLURM_ARRAY_TASK_ID, points that become
ready in the same poll go out as one ``sbatch --array``, and a failed point is
resubmitted alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from oellm_autoexp.monitor.actions import StateEventConfig
from oellm_autoexp.monitor.conditions import (
    ConditionContext,
    FileExistsConditionConfig,
    JobsFinishedCondition,
    JobsFinishedConditionConfig,
)
from oellm_autoexp.monitor.loop import JobFileStore, JobRecord, JobRuntime, MonitorLoop
from oellm_autoexp.monitor.slurm_client import SlurmClient, SlurmClientConfig
from oellm_autoexp.monitor.submission import SlurmJobConfig
from oellm_autoexp.orchestrator import _bind_array_groups
from oellm_autoexp.slurm_gen import SlurmConfig
from oellm_autoexp.slurm_gen.client import FakeSlurmClientConfig
from oellm_autoexp.slurm_gen.generator import generate_script

from compoconf import parse_config


def _member(tmp_path: Path, index: int, *, group: str = "evals", **kwargs) -> JobRecord:
    out = tmp_path / "out" / f"evals_{index}"
    slurm_kwargs = {"sbatch": {"time": "01:00:00", "nodes": 1, "output": str(out / "slurm-%j.log")}}
    slurm_kwargs.update(kwargs.pop("slurm", {}))
    slurm = parse_config(
        SlurmConfig,
        {
            "name": f"evals_{index}",
            "template_path": "templates/base.sbatch",
            "script_dir": str(out),
            "log_dir": str(out),
            "command": [f"echo task {index}"],
            "array_concurrency": 2,
            **slurm_kwargs,
        },
    )
    definition = SlurmJobConfig(
        name=f"evals_{index}",
        log_path=str(out / "slurm-%j.log"),
        array_group=group,
        slurm=slurm,
        metadata={"stage": "eval"},
        **kwargs,
    )
    return JobRecord(job_id=f"evals_{index}_abc", definition=definition, runtime=JobRuntime())


def _client() -> SlurmClient:
    return SlurmClient(SlurmClientConfig(base_client=FakeSlurmClientConfig()))


def test_bind_gives_one_script_and_an_index_per_member(tmp_path: Path):
    records = _bind_array_groups([_member(tmp_path, i) for i in range(3)])

    scripts = {r.definition.slurm.script_path for r in records}
    assert len(scripts) == 1
    assert [r.array_idx for r in records] == [0, 1, 2]
    # %j is a task's own numeric id under an array; the monitor tracks <array>_<index>
    assert {r.definition.log_path for r in records} == {
        str(tmp_path / "out" / "evals" / "slurm-%A_%a.log")
    }

    script = Path(generate_script(records[0].definition.slurm)).read_text()
    assert 'case "$SLURM_ARRAY_TASK_ID" in' in script
    for i in range(3):
        assert f"{i})\necho task {i}\n;;" in script
    assert "#SBATCH --job-name=evals" in script
    assert f"#SBATCH --output={tmp_path}/out/evals/slurm-%A_%a.log" in script


def test_members_must_share_the_header(tmp_path: Path):
    records = [_member(tmp_path, 0), _member(tmp_path, 1, slurm={"sbatch": {"time": "02:00:00"}})]
    with pytest.raises(ValueError, match="array group 'evals'"):
        _bind_array_groups(records)


def test_records_without_a_group_are_untouched(tmp_path: Path):
    plain = _member(tmp_path, 0, group=None)
    (bound,) = _bind_array_groups([plain])
    assert bound is plain and bound.array_idx is None


def test_ready_members_go_out_as_one_array(tmp_path: Path):
    store = JobFileStore(tmp_path / "state")
    for record in _bind_array_groups([_member(tmp_path, i) for i in range(3)]):
        store.upsert(record)
    client = _client()

    MonitorLoop(store, slurm_client=client).observe_once()

    ids = sorted(r.runtime.runtime_job_id for r in store.load_all())
    assert ids == ["1_0", "1_1", "1_2"]  # one array, base id 1
    assert all(r.runtime.attempts == 1 for r in store.load_all())


def test_member_waits_for_its_own_start_condition(tmp_path: Path):
    gate = tmp_path / "gate"
    members = [_member(tmp_path, 0), _member(tmp_path, 1)]
    members[1].definition.start_condition = FileExistsConditionConfig(path=str(gate))
    store = JobFileStore(tmp_path / "state")
    for record in _bind_array_groups(members):
        store.upsert(record)
    client = _client()
    loop = MonitorLoop(store, slurm_client=client)

    loop.observe_once()
    gate.touch()
    loop.observe_once()

    by_index = {r.array_idx: r.runtime.runtime_job_id for r in store.load_all()}
    assert by_index == {0: "1_0", 1: "2_1"}  # same script, a second one-task array


def test_failed_member_is_resubmitted_alone_then_given_up(tmp_path: Path):
    retry = StateEventConfig(
        name="retry_failed",
        transition=(None, "FAILED"),
        condition={"class_name": "MaxActionFiresCondition", "max_fires": 1},
        action={"class_name": "RestartAction", "reason": "retry"},
    )
    members = [_member(tmp_path, i, state_events=[retry]) for i in range(3)]
    store = JobFileStore(tmp_path / "state")
    for record in _bind_array_groups(members):
        store.upsert(record)
    client = _client()
    loop = MonitorLoop(store, slurm_client=client)
    loop.observe_once()

    fake = client._client
    fake.set_state("1_1", "FAILED")
    loop.observe_once()
    retried = {r.array_idx: r for r in store.load_all()}[1]
    assert retried.runtime.runtime_job_id == "2_1"  # only index 1, as a new array
    assert retried.runtime.attempts == 2
    assert {r.runtime.runtime_job_id for r in store.load_all() if r.array_idx != 1} == {
        "1_0",
        "1_2",
    }

    fake.set_state("2_1", "FAILED")
    loop.observe_once()
    assert store.load(retried.job_id) is None  # budget spent: closed, not retried
    final = store.load(retried.job_id, include_finished=True)
    assert final.runtime.final_state == "cancelled"


def _write_record(session: Path, name: str, *, group: str | None, stage: str, final: str | None):
    session.mkdir(parents=True, exist_ok=True)
    payload = {
        "definition": {"array_group": group, "metadata": {"stage": stage}},
        "runtime": {"final_state": final},
    }
    (session / f"{name}.job.json").write_text(json.dumps(payload))


def _jobs_finished(session: Path, **kwargs) -> bool:
    condition = JobsFinishedCondition(JobsFinishedConditionConfig(**kwargs))
    return condition.check(ConditionContext(job_metadata={"session_dir": str(session)})).passed


def test_jobs_finished_condition(tmp_path: Path):
    session = tmp_path / "session"
    _write_record(session, "a", group="evals", stage="eval", final="finished")
    _write_record(session, "b", group="evals", stage="eval", final=None)
    _write_record(session, "other", group=None, stage="train", final=None)

    assert not _jobs_finished(session, array_groups=["evals"])
    _write_record(session, "b", group="evals", stage="eval", final="cancelled")
    assert _jobs_finished(session, array_groups=["evals"])
    assert _jobs_finished(session, stages=["eval"])
    assert not _jobs_finished(session, array_groups=["evals"], require_success=True)
    # a misspelt name must not pass vacuously
    assert not _jobs_finished(session, array_groups=["evalz"])

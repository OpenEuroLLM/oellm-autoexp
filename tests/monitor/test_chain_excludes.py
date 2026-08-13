"""Tests for propagating node exclusions across a dependency chain.

Covers ``UpdateChainExcludesAction`` and ``MonitorLoop._update_chain_excludes``:
when a node is recorded in the exclusion file, every *pending* sibling job in
the chain should get ``scontrol update ... ExcludeNodes=`` while the running job
that triggered the event (and any other running job) is left untouched.
"""

from __future__ import annotations

from pathlib import Path

from oellm_autoexp.monitor.actions import (
    ActionContext,
    ActionResult,
    EventRecord,
    UpdateChainExcludesAction,
    UpdateChainExcludesActionConfig,
)
from oellm_autoexp.monitor.loop import (
    JobFileStore,
    JobRecord,
    JobRuntime,
    MonitorLoop,
)
from oellm_autoexp.monitor.submission import SlurmJobConfig
from oellm_autoexp.monitor.slurm_client import SlurmClient as MonitorSlurmClient, SlurmClientConfig
from oellm_autoexp.slurm_gen.client import FakeSlurmClientConfig
from oellm_autoexp.slurm_gen.schema import SlurmConfig


class FakeSlurmClientStub:
    """Minimal slurm client: scriptable squeue + recording update_excludes.

    ``raise_on`` is a set of job ids for which ``update_excludes`` raises, to
    exercise the loop's per-job error handling.
    """

    def __init__(self, statuses: dict[str, str], raise_on: set[str] | None = None) -> None:
        self.statuses = statuses
        self.raise_on = raise_on or set()
        self.excludes: dict[str, str] = {}

    def squeue(self) -> dict[str, str]:
        return dict(self.statuses)

    def update_excludes(self, job_id: str, nodelist: str) -> None:
        if job_id in self.raise_on:
            raise RuntimeError(f"scontrol update failed for {job_id}")
        self.excludes[job_id] = nodelist


def _slurm_job(tmp_path: Path, *, job_id: str, runtime_id: str) -> JobRecord:
    definition = SlurmJobConfig(
        name=job_id,
        log_path=str(tmp_path / f"{job_id}.log"),
        slurm=SlurmConfig(
            name=job_id,
            template_path="templates/base.sbatch",
            script_dir=str(tmp_path),
            log_dir=str(tmp_path),
        ),
    )
    runtime = JobRuntime(submitted=True, runtime_job_id=runtime_id, attempts=1)
    return JobRecord(job_id=job_id, definition=definition, runtime=runtime)


def test_update_chain_excludes_targets_only_pending_siblings(tmp_path: Path):
    exclude_file = tmp_path / "exclude.txt"
    exclude_file.write_text("lrdn0417\nlrdn0001\n")

    store = JobFileStore(tmp_path / "state")
    store.upsert(_slurm_job(tmp_path, job_id="j_cur", runtime_id="100"))  # running trigger
    store.upsert(_slurm_job(tmp_path, job_id="j_p1", runtime_id="101"))  # pending sibling
    store.upsert(_slurm_job(tmp_path, job_id="j_p2", runtime_id="102"))  # pending sibling
    store.upsert(_slurm_job(tmp_path, job_id="j_run", runtime_id="103"))  # running sibling

    client = FakeSlurmClientStub(
        statuses={"100": "RUNNING", "101": "PENDING", "102": "PENDING", "103": "RUNNING"}
    )
    loop = MonitorLoop(store, slurm_client=client)
    current = store.load("j_cur")

    result = ActionResult(
        action_config=UpdateChainExcludesActionConfig(exclude_file=str(exclude_file)),
        metadata={"exclude_file": str(exclude_file)},
    )
    # Drive through the public dispatch path to also cover _handle_action_result.
    loop._handle_action_result(current, result)

    expected = "lrdn0417,lrdn0001"
    # pending siblings updated; running trigger + running sibling left alone
    assert client.excludes == {"101": expected, "102": expected}


def test_update_chain_excludes_noop_when_list_empty(tmp_path: Path):
    exclude_file = tmp_path / "missing.txt"  # never created -> resolver returns None

    store = JobFileStore(tmp_path / "state")
    store.upsert(_slurm_job(tmp_path, job_id="j_cur", runtime_id="100"))
    store.upsert(_slurm_job(tmp_path, job_id="j_p1", runtime_id="101"))

    client = FakeSlurmClientStub(statuses={"100": "RUNNING", "101": "PENDING"})
    loop = MonitorLoop(store, slurm_client=client)
    current = store.load("j_cur")

    result = ActionResult(
        action_config=UpdateChainExcludesActionConfig(exclude_file=str(exclude_file)),
        metadata={"exclude_file": str(exclude_file)},
    )
    loop._handle_action_result(current, result)
    assert client.excludes == {}


def test_update_chain_excludes_skips_unsubmitted_and_survives_errors(tmp_path: Path):
    exclude_file = tmp_path / "exclude.txt"
    exclude_file.write_text("lrdn0417\n")

    store = JobFileStore(tmp_path / "state")
    store.upsert(_slurm_job(tmp_path, job_id="j_cur", runtime_id="100"))  # running trigger
    store.upsert(_slurm_job(tmp_path, job_id="j_p1", runtime_id="101"))  # pending, errors
    store.upsert(_slurm_job(tmp_path, job_id="j_p2", runtime_id="102"))  # pending, ok
    # A not-yet-submitted sibling (no runtime id) must be skipped, not crash.
    unsub = _slurm_job(tmp_path, job_id="j_new", runtime_id="103")
    unsub.runtime.submitted = False
    unsub.runtime.runtime_job_id = None
    store.upsert(unsub)

    client = FakeSlurmClientStub(
        statuses={"100": "RUNNING", "101": "PENDING", "102": "PENDING"},
        raise_on={"101"},  # scontrol fails for this job
    )
    loop = MonitorLoop(store, slurm_client=client)
    result = ActionResult(
        action_config=UpdateChainExcludesActionConfig(exclude_file=str(exclude_file)),
        metadata={"exclude_file": str(exclude_file)},
    )
    # j_p1 raising must not stop j_p2 from being updated.
    loop._handle_action_result(store.load("j_cur"), result)
    assert client.excludes == {"102": "lrdn0417"}


def test_update_chain_excludes_noop_without_slurm_client(tmp_path: Path):
    exclude_file = tmp_path / "exclude.txt"
    exclude_file.write_text("lrdn0417\n")
    store = JobFileStore(tmp_path / "state")
    store.upsert(_slurm_job(tmp_path, job_id="j_cur", runtime_id="100"))
    loop = MonitorLoop(store, slurm_client=None)  # e.g. local-only monitor
    result = ActionResult(
        action_config=UpdateChainExcludesActionConfig(exclude_file=str(exclude_file)),
        metadata={"exclude_file": str(exclude_file)},
    )
    # Must not raise when there is no SLURM client to talk to.
    loop._handle_action_result(store.load("j_cur"), result)


def test_update_chain_excludes_via_real_monitor_client(tmp_path: Path):
    """Exercise the real monitor SlurmClient wrapper (delegates to
    scontrol)."""
    exclude_file = tmp_path / "exclude.txt"
    exclude_file.write_text("lrdn0417\nlrdn0001\n")

    store = JobFileStore(tmp_path / "state")
    store.upsert(_slurm_job(tmp_path, job_id="j_cur", runtime_id="100"))
    store.upsert(_slurm_job(tmp_path, job_id="j_p1", runtime_id="101"))

    monitor_client = MonitorSlurmClient(SlurmClientConfig(base_client=FakeSlurmClientConfig()))
    base = monitor_client._client
    # Register the jobs in the underlying fake so squeue reports their states.
    base.register_job(
        "100",
        _slurm_job(tmp_path, job_id="j_cur", runtime_id="100").definition.slurm,
        state="RUNNING",
    )
    base.register_job(
        "101",
        _slurm_job(tmp_path, job_id="j_p1", runtime_id="101").definition.slurm,
        state="PENDING",
    )

    loop = MonitorLoop(store, slurm_client=monitor_client)
    result = ActionResult(
        action_config=UpdateChainExcludesActionConfig(exclude_file=str(exclude_file)),
        metadata={"exclude_file": str(exclude_file)},
    )
    loop._handle_action_result(store.load("j_cur"), result)
    # The pending job's exclude list was set through the real wrapper -> fake base.
    assert base._excludes == {"101": "lrdn0417,lrdn0001"}


def test_update_chain_excludes_action_carries_resolved_path():
    """The action renders {var} in exclude_file and signals via
    action_config."""
    config = UpdateChainExcludesActionConfig(exclude_file="{exclude_dir}/exclude.txt")
    event = EventRecord(event_id="e", name="propagate", source="log")
    ctx = ActionContext(event=event, job_metadata={"exclude_dir": "/tmp/excl"})

    result = UpdateChainExcludesAction(config).execute(ctx)
    assert result.action_config is config
    assert result.metadata["exclude_file"] == "/tmp/excl/exclude.txt"

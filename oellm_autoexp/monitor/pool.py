"""Sub-SLURM jobs: one allocation ("host") runs a sequence of sub-jobs ("runs").

SLURM only controls the allocation. What runs inside it is decided by the
monitor on the login node, through files on the shared filesystem:

* a **host** is an ordinary SLURM job whose batch shell runs
  ``scripts/pool/supervisor.sh``. The supervisor does nothing on its own: it
  runs the runs the monitor ASSIGNS to it, one ``srun`` step at a time, and
  reports what happened in files;
* a **run** is one attempt of an ordinary autoexp job (one resolved config,
  its own slurm settings, its own log). It is rendered on the login node, like
  any sbatch script, into a *step script*: the job's own template with the
  ``#SBATCH`` header commented out and ``srun`` pinned to the nodes the
  supervisor hands it;
* the :class:`PoolManager` is the monitor-side half. It implements the
  ``JobClientProtocol`` for runs (``submit`` = enqueue, ``cancel`` = stop file,
  ``squeue`` = status from the run's files), so a run is a normal
  :class:`~oellm_autoexp.monitor.loop.JobRecord` and every existing log event,
  condition and RestartAction keeps working. It also schedules: assigns
  queued runs to idle hosts, keeps hosts submitted, hands a run over from an
  old host to a freshly started one, and retires idle hosts.

Pool directory layout (everything a human needs is ``ls`` + ``cat``)::

    <pool>/pool.json             name, session dir, PoolHostConfig (written by run_autoexp / pool_ctl)
    <pool>/pool.env              the same config as shell variables, sourced by the supervisors
    <pool>/host_definition.json  the host job (SlurmJobConfig) every host is cloned from
    <pool>/state.json            monitor-owned: closed flag, host counter
    <pool>/monitor.alive         mtime = last monitor tick (supervisors exit when idle and this is stale)
    <pool>/exclude_nodes         pool-wide node exclusions, honoured at every step launch
    <pool>/ledger.tsv            one line per run start / end, appended by the supervisors
    <pool>/requests/*.json       pool_ctl -> monitor (cancel, close, open, drain)
    <pool>/slurm-<jobid>.log     a host's main log (supervisor lines)
    <pool>/slurm-<jobid>.<k>-<name>.log -> the k-th run of that host's own log (symlink)
    <pool>/hosts/<hid>/          info.env, alive, state, assign/<rid>, control/{exit,drain}, exited
    <pool>/runs/<rid>/           step.sh, meta.env, run.json, assigned, started, rc, stop, preempt

A run's status is a pure function of its directory:

    rc present        -> by its reason: COMPLETED / FAILED / CANCELLED / PREEMPTED / TIMEOUT
    started, no rc    -> RUNNING (the manager writes rc "-1 host_lost" once the host is gone)
    otherwise         -> PENDING (queued, or assigned and not yet launched)
"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
import os
import re
import shlex
import shutil
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from compoconf import asdict, parse_config

from oellm_autoexp.monitor.submission import SlurmJobConfig, dump_job_config
from oellm_autoexp.monitor.utils.paths import expand_log_path

if TYPE_CHECKING:  # pragma: no cover
    from oellm_autoexp.monitor.loop import JobFileStore, JobRecord, MonitorLoop
    from oellm_autoexp.slurm_gen.schema import SlurmConfig

LOGGER = logging.getLogger(__name__)

# how long the preflight waits for a host's GPU probe (supervisor.sh) before it goes on without
PREFLIGHT_PROBE_WAIT_S = 900.0

# The rank processes Megatron's --exit-signal-handler listens on are the DIRECT
# CHILDREN of the torchrun / ft_launcher elastic agents. Signalling the agents
# instead makes them tear their workers down ~30 s later, before the exit
# checkpoint is written (templates/jupiter_relay.sbatch, smoke 1691088). The
# pattern anchors on the python executable because the srun task's bash and the
# apptainer wrapper carry the same words in their command lines.
DEFAULT_RANK_PATTERN = r"^[^ ]*python[^ ]*( +-[A-Za-z]+)* +(-m +torch\.distributed\.run|[^ ]*/(torchrun|ft_launcher))( |$)"

POOL_ROLE_HOST = "host"

# rc reason -> monitor status
_REASON_STATUS = {
    "completed": "COMPLETED",
    "failed": "FAILED",
    "launch_failed": "FAILED",
    "cancelled": "CANCELLED",
    "preempted": "PREEMPTED",
    "host_lost": "PREEMPTED",
    "insufficient_nodes": "PREEMPTED",
    "timeout": "TIMEOUT",
    # a job kept as the pool's preflight template: done with, nothing ran
    "preflight_template": "COMPLETED",
}

ACTIVE_HOST_STATES = frozenset(
    {"PENDING", "RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED", "REQUEUED", "RESIZING"}
)


@dataclass
class PoolHostConfig:
    """How a pool behaves. Lives in ``job.pool_host`` of a HOST config.

    Two typical shapes:

    * speed ablations / interactive queue: ``max_hosts_running: 1``,
      ``keep_pending: 0``, ``min_hosts: 1`` -- one allocation that works through
      the queue and stays reachable while you look at results and enqueue more;
    * continuous training without queue gaps: ``max_hosts_running: 1``,
      ``keep_pending: 1``, ``handover: true`` -- a successor allocation always
      waits in the queue (accumulating age priority, no dependency); when it
      starts, the current run is stopped gracefully on the old host (checkpoint
      save) and continues on the new one, and the old allocation is released.
    """

    name: str = ""
    # hosts allowed to run at the same time; with handover, the oldest beyond
    # this number is retired as soon as a newer one is up
    max_hosts_running: int = 1
    # extra hosts kept waiting in the SLURM queue while there is work
    keep_pending: int = 0
    # hosts kept submitted even without work (until `pool_ctl close`)
    min_hosts: int = 0
    handover: bool = False
    # Handover is worth one save + one startup only if the old host has already
    # used a good part of its allocation: a host younger than this is not retired
    # (the newer host waits for it, and is not released as idle meanwhile).
    # Without it a short queue makes runs ping-pong between fresh allocations
    # (JUPITER 2026-10-26: successors started every 3-4 min).
    handover_min_age_s: float = 1800.0
    # Queue the keep_pending successors only once every running host is this old.
    # 0 = right away (maximum age priority; right for long queues). For short
    # queues set it to roughly (wall time - expected queue wait).
    successor_delay_s: float = 0.0
    # an idle host with nothing to do is released after this long
    idle_exit_s: float = 1200.0
    # a supervisor releases an idle host on its own once the monitor has been
    # silent this long (dead man's switch, so a dead monitor costs no GPU-hours)
    monitor_dead_exit_s: float = 1800.0
    # do not start a run on a host with less allocation time left than this
    min_remaining_s: float = 900.0
    # SIGTERM -> hard kill grace for a cancelled run, and for a preempted one
    # (preemption saves a checkpoint, which takes minutes at scale)
    stop_grace_s: float = 120.0
    preempt_grace_s: float = 900.0
    # UNHEALTHY requeues of one job (lost host, host without enough nodes) before
    # it is given up; healthy ones (handover, wall clock, pool_ctl preempt) are
    # unlimited, a continuous run has one per segment
    max_attempts: int = 20
    # a RUNNING host whose supervisor heartbeat is older than this is cancelled
    host_stale_s: float = 900.0
    # supervisor poll period
    poll_s: float = 10.0
    rank_pattern: str = DEFAULT_RANK_PATTERN
    # Several SLURM accounts, in order of preference ("cns144,ehpc533"; empty = the host
    # definition's own). Each host is submitted under one of them; a run's steps simply use
    # the allocation they land in. The FIRST account keeps the usual queued hosts
    # (keep_pending successors age in its queue); every other one gets queued hosts only
    # while nothing runs, or once the running hosts have less than `fallback_lead_s` left
    # (0: always -- the accounts race, and the first host to start takes the run).
    host_accounts: str = ""
    fallback_lead_s: float = 0.0
    # ADOPTED hosts (pool_ctl adopt: an ordinary queued job whose own step is held by the
    # adoption hook): stop the runs this long before the job's own end time (it has no
    # B:USR1), and cancel the job when it runs but the hook's start marker has not
    # appeared within adopt_hook_timeout_s (the hook did not engage: its own training must
    # not proceed)
    adopt_stop_margin_s: float = 1800.0
    adopt_hook_timeout_s: float = 90.0
    # hosts the pool submits itself (-1: as many as the settings above ask for; 0: none, the
    # pool only uses adopted jobs)
    max_own_hosts: int = -1
    # the monitor stays up while the pool is open (not `pool_ctl close`d), also with no work:
    # a pool that is used in bursts (speed tests) otherwise loses its monitor when it runs empty,
    # and the next `run_autoexp --pool --no-monitor` enqueue sits unscheduled
    keep_monitor: bool = False
    # runs a host executes side by side on disjoint nodes (packing small jobs into one
    # allocation, e.g. three 8-node runs in a 33-node host); 1 = one run per host
    max_runs_per_host: int = 1
    # PREFLIGHT node check of every new host, before any run is assigned to it (user 2026-10-30;
    # one dead-GPU node and four degraded NICs cost 128-node runs 27-50 % for hours, 10-27..29):
    # 1. preflight_gpus_per_node > 0: the supervisor counts each node's GPUs (preflight_gpu_cmd,
    #    run there by srun) and excludes a node with fewer;
    # 2. preflight: the node-health benchmark (oellm_autoexp/node_health/bench.py) on all usable
    #    nodes, run from the pool's preflight template: the job of the config preflight_check
    #    names (+ preflight_check_overrides), enqueued by run_autoexp when it creates / updates
    #    the pool (by hand: any job enqueued with +job.metadata.pool_preflight_template=true).
    #    A slow GPU or GPU group (below preflight_ratio x the median) excludes the node behind it
    #    (oellm_autoexp.node_health.verdict) and the benchmark runs again, up to
    #    preflight_max_rounds. Without a template only the GPU count is checked.
    #    Pass: the host takes runs. Fail (no result twice, or still slow after the last round): the
    #    host is released (kept if held, without runs).
    preflight: bool = False
    preflight_gpus_per_node: int = 0
    preflight_gpu_cmd: str = 'nvidia-smi -L 2>/dev/null | grep -c "^GPU"'
    preflight_ratio: float = 0.75
    preflight_max_rounds: int = 3
    preflight_check: str = ""
    preflight_check_overrides: list[str] = field(default_factory=list)
    # VERIFIED handover (with `handover`): the runs on the old host are stopped (checkpoint) and
    # restart on the new, verified host while the old allocation stays; it is released once every
    # moved run is healthy there (handover_healthy_steps matches of handover_healthy_pattern in
    # its log), unless it still has other runs. A moved run that fails there, or is not healthy
    # within handover_verify_timeout_s, releases the NEW host instead: the runs go back.
    handover_verify: bool = False
    handover_healthy_steps: int = 3
    handover_healthy_pattern: str = r"iteration\s+\d+/"
    handover_verify_timeout_s: float = 3600.0
    # HOLD: the monitor never releases a host -- not idle, not for queued work it cannot run, not
    # after a handover; only its wall clock or `pool_ctl exit-host` ends it (per host:
    # `pool_ctl hold <host>`, a <host>/control/hold file). For allocations that are hard to get
    # back (adopted cns144 jobs: 47998967 was released with 15 h left, 2026-10-30).
    hold_hosts: bool = False


def host_accounts(cfg: PoolHostConfig) -> list[str]:
    return [a.strip() for a in (cfg.host_accounts or "").split(",") if a.strip()]


# ---------------------------------------------------------------------------
# small file helpers
# ---------------------------------------------------------------------------


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:6]}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def read_env_file(path: Path) -> dict[str, str]:
    """Parse the ``KEY=value`` files the supervisor writes (values may be
    shell-quoted)."""
    text = _read(path)
    out: dict[str, str] = {}
    if not text:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        try:
            parts = shlex.split(value)
            out[key.strip()] = parts[0] if parts else ""
        except ValueError:
            out[key.strip()] = value
    return out


def write_env_file(path: Path, values: dict[str, Any]) -> None:
    lines = [f"{k}={shlex.quote(str(v))}" for k, v in values.items()]
    _atomic_write(path, "\n".join(lines) + "\n")


def parse_slurm_time(value: str | None) -> float | None:
    """SLURM time string -> seconds. ``None``/"UNLIMITED" -> None.

    Accepts M, M:S, H:M:S, D-H, D-H:M, D-H:M:S.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in ("UNLIMITED", "INFINITE", "NONE"):
        return None
    days = 0
    if "-" in text:
        d, text = text.split("-", 1)
        days = int(d)
        parts = [int(p) for p in text.split(":")]
        while len(parts) < 3:
            parts.append(0)
        h, m, s = parts[:3]
    else:
        parts = [int(p) for p in text.split(":")]
        if len(parts) == 1:
            h, m, s = 0, parts[0], 0
        elif len(parts) == 2:
            h, m, s = 0, parts[0], parts[1]
        else:
            h, m, s = parts[:3]
    return float(days * 86400 + h * 3600 + m * 60 + s)


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]+", "-", name).strip("-") or "run"


def pool_tag(name: str) -> str:
    """Run ids must not contain ``_``: ``expand_log_path`` reads ``A_B`` as an
    array id."""
    return _sanitize(name).replace(".", "-").replace("_", "-")


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------


class PoolLayout:
    def __init__(self, pool_dir: str | Path) -> None:
        self.root = Path(pool_dir).absolute()

    pool_json = property(lambda self: self.root / "pool.json")
    pool_env = property(lambda self: self.root / "pool.env")
    host_definition = property(lambda self: self.root / "host_definition.json")
    state_json = property(lambda self: self.root / "state.json")
    monitor_alive = property(lambda self: self.root / "monitor.alive")
    exclude_nodes = property(lambda self: self.root / "exclude_nodes")
    ledger = property(lambda self: self.root / "ledger.tsv")
    requests_dir = property(lambda self: self.root / "requests")
    manifests_dir = property(lambda self: self.root / "manifests")
    runs_dir = property(lambda self: self.root / "runs")
    hosts_dir = property(lambda self: self.root / "hosts")

    def run_dir(self, rid: str) -> Path:
        return self.runs_dir / rid

    def host_dir(self, hid: str) -> Path:
        return self.hosts_dir / hid

    def exists(self) -> bool:
        return self.pool_json.exists()

    # --- pool metadata ----------------------------------------------------
    def load_meta(self) -> dict[str, Any]:
        text = _read(self.pool_json)
        if text is None:
            raise FileNotFoundError(f"not a pool (no pool.json): {self.root}")
        return json.loads(text)

    def load_config(self) -> PoolHostConfig:
        meta = self.load_meta()
        values = dict(meta.get("config") or {})
        known = {f for f in PoolHostConfig.__dataclass_fields__}
        cfg = PoolHostConfig(**{k: v for k, v in values.items() if k in known})
        if not cfg.name:
            cfg.name = meta.get("name") or self.root.name
        return cfg

    def write_meta(self, meta: dict[str, Any]) -> None:
        _atomic_write(self.pool_json, json.dumps(meta, indent=2, sort_keys=True))
        cfg = self.load_config()
        write_env_file(
            self.pool_env,
            {
                # whole seconds: the supervisor compares them in bash arithmetic
                "POOL_NAME": cfg.name,
                "POOL_POLL_S": max(1, int(round(cfg.poll_s))),
                "POOL_IDLE_EXIT_S": int(round(cfg.idle_exit_s)),
                "POOL_MONITOR_DEAD_EXIT_S": int(round(cfg.monitor_dead_exit_s)),
                "POOL_STOP_GRACE_S": int(round(cfg.stop_grace_s)),
                "POOL_PREEMPT_GRACE_S": int(round(cfg.preempt_grace_s)),
                "POOL_RANK_PATTERN": cfg.rank_pattern,
                "POOL_ADOPT_STOP_MARGIN_S": int(round(cfg.adopt_stop_margin_s)),
                "POOL_MAX_RUNS_PER_HOST": max(1, int(cfg.max_runs_per_host)),
                "POOL_PREFLIGHT_GPUS": max(0, int(cfg.preflight_gpus_per_node)),
                "POOL_PREFLIGHT_GPU_CMD": cfg.preflight_gpu_cmd,
                "POOL_HOLD_HOSTS": int(bool(cfg.hold_hosts)),
            },
        )

    def load_state(self) -> dict[str, Any]:
        text = _read(self.state_json)
        try:
            return json.loads(text) if text else {}
        except json.JSONDecodeError:
            return {}

    def write_state(self, state: dict[str, Any]) -> None:
        _atomic_write(self.state_json, json.dumps(state, indent=2, sort_keys=True))

    def load_host_definition(self) -> SlurmJobConfig:
        payload = json.loads(self.host_definition.read_text(encoding="utf-8"))
        _import_registry()
        return parse_config(SlurmJobConfig, payload)

    # --- ids --------------------------------------------------------------
    def allocate_run_id(self, tag: str) -> str:
        """``<tag>.<seq>``; ``mkdir`` is the lock, so two allocators never
        collide."""
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        existing = [p.name.rsplit(".", 1)[-1] for p in self.runs_dir.iterdir()]
        seq = max([int(s) for s in existing if s.isdigit()] or [0]) + 1
        while True:
            rid = f"{tag}.{seq:04d}"
            try:
                (self.runs_dir / rid).mkdir()
                return rid
            except FileExistsError:
                seq += 1

    def allocate_host_id(self) -> str:
        self.hosts_dir.mkdir(parents=True, exist_ok=True)
        existing = [p.name[4:] for p in self.hosts_dir.iterdir() if p.name.startswith("host")]
        seq = max([int(s) for s in existing if s.isdigit()] or [0]) + 1
        while True:
            hid = f"host{seq:03d}"
            try:
                (self.hosts_dir / hid).mkdir()
                return hid
            except FileExistsError:
                seq += 1


# ---------------------------------------------------------------------------
# run state
# ---------------------------------------------------------------------------


@dataclass
class RunState:
    rid: str
    status: str  # PENDING / RUNNING / COMPLETED / FAILED / CANCELLED / PREEMPTED / TIMEOUT
    reason: str = ""
    code: int | None = None
    host: str | None = None  # assigned host
    started: dict[str, str] = field(default_factory=dict)


def read_run_state(layout: PoolLayout, rid: str) -> RunState:
    rd = layout.run_dir(rid)
    assigned = (_read(rd / "assigned") or "").strip() or None
    rc_text = _read(rd / "rc")
    started = read_env_file(rd / "started")
    host = started.get("host") or assigned
    if rc_text is not None:
        parts = rc_text.split()
        code = int(parts[0]) if parts and re.fullmatch(r"-?\d+", parts[0]) else -1
        reason = parts[1] if len(parts) > 1 else ("completed" if code == 0 else "failed")
        status = _REASON_STATUS.get(reason, "FAILED")
        if status == "COMPLETED" and code != 0:
            status = "FAILED"
        return RunState(rid, status, reason, code, host, started)
    if started:
        return RunState(rid, "RUNNING", host=host, started=started)
    return RunState(rid, "PENDING", host=assigned)


def write_rc(layout: PoolLayout, rid: str, code: int, reason: str) -> None:
    rc = layout.run_dir(rid) / "rc"
    if rc.exists():
        return
    _atomic_write(rc, f"{code} {reason}\n")


# ---------------------------------------------------------------------------
# step script rendering
# ---------------------------------------------------------------------------

# The batch-shell env exports of a cluster config describe the WHOLE allocation
# (MASTER_ADDR = first node of SLURM_JOB_NODELIST, NUM_NODES = SLURM_JOB_NUM_NODES).
# A run uses the nodes the supervisor picked for its step, which may be fewer.
_STEP_ENV_SUBSTITUTIONS = (
    ("SLURM_JOB_NODELIST", "AUTOEXP_STEP_NODELIST"),
    ("SLURM_JOB_NUM_NODES", "AUTOEXP_STEP_NUM_NODES"),
    ("SLURM_NNODES", "AUTOEXP_STEP_NUM_NODES"),
)


_EXIT_SIGNAL = re.compile(r"--exit[-_]signal[\s=]+['\"]?(?:SIG)?([A-Z0-9]+)")


def stop_signal_of(step_text: str) -> str:
    """The signal that stops a run gracefully: the job's own Megatron ``--exit-signal``
    (SIGUSR2 in our MareNostrum production runs: SIGTERM would kill it without a save),
    else TERM (Megatron's default ``--exit-signal-handler`` signal)."""
    m = _EXIT_SIGNAL.search(step_text)
    return m.group(1) if m else "TERM"


def render_step_script(
    slurm: SlurmConfig, run_id: str, *, log_path: str, workdir: str | None = None
) -> str:
    """The job's own sbatch template, turned into a script for one srun step.

    * the ``#SBATCH`` header is kept, commented out: it documents what the job
      asked for, but the allocation belongs to the host;
    * node-list variables in the env exports point at the step's nodes;
    * ``MASTER_PORT`` moves with the step index, so a rendezvous never meets
      the TCPStore of the step before it;
    * ``srun`` is pinned to the supervisor's nodes, named after the run (that
      is how the supervisor finds the step to cancel it).
    """
    from oellm_autoexp.slurm_gen.generator import build_replacements
    from oellm_autoexp.slurm_gen.template_renderer import render_template

    replacements = build_replacements(
        slurm, job_name=slurm.name, log_path=log_path, command=list(slurm.command)
    )
    directives = replacements["sbatch_directives"].splitlines()
    replacements["sbatch_directives"] = "\n".join(
        ["# [pool] the sub-job's own sbatch header (the allocation belongs to the pool host):"]
        + [f"#   {line}" for line in directives]
    )
    env = replacements["env_exports"]
    for old, new in _STEP_ENV_SUBSTITUTIONS:
        env = env.replace(old, new)
    if "MASTER_PORT" in (slurm.env or {}):
        env += "\nexport MASTER_PORT=$(( MASTER_PORT + AUTOEXP_STEP_INDEX % 64 ))"
    replacements["env_exports"] = env

    sbatch = slurm.sbatch
    srun_flags = [
        "--nodes=$AUTOEXP_STEP_NUM_NODES",
        "--ntasks=$AUTOEXP_STEP_NTASKS",
        "--ntasks-per-node=$AUTOEXP_STEP_NTASKS_PER_NODE",
        '--nodelist="$AUTOEXP_STEP_NODELIST"',
        '--job-name="$AUTOEXP_RUN_ID"',
    ]
    cpus = getattr(sbatch, "cpus_per_task", None)
    if cpus:
        srun_flags.append(f"--cpus-per-task={cpus}")
    replacements["srun_opts"] = replacements["srun_opts"] + " ".join(srun_flags) + " "

    text = render_template(Path(slurm.template_path).read_text(), replacements)
    header = (
        f"# [pool] step script of run {run_id} (job {slurm.name}), rendered "
        f"{time.strftime('%Y-%m-%dT%H:%M:%S')} from {slurm.template_path}\n"
        "# [pool] the supervisor exports AUTOEXP_RUN_ID, AUTOEXP_STEP_INDEX, AUTOEXP_STEP_NODELIST,\n"
        "# [pool] AUTOEXP_STEP_NUM_NODES, AUTOEXP_STEP_NTASKS(_PER_NODE), AUTOEXP_POOL_DIR, AUTOEXP_POOL_HOST\n"
    )
    if workdir:
        # the job's relative paths (PYTHONPATH=., ./submodules/...) resolve against the
        # directory it was enqueued from, not the host's
        header += (
            f'cd {shlex.quote(workdir)} || {{ echo "[pool] no workdir {workdir}" >&2; exit 97; }}\n'
        )
    if text.startswith("#!"):
        first, _, rest = text.partition("\n")
        return f"{first}\n{header}{rest}"
    return "#!/bin/bash\n" + header + text


def sbatch_get(sbatch, key: str):
    """A value of a (non-strict) SbatchConfig, declared field or not.

    Keys the dataclass does not declare (``exclude``, ``output``, ``signal``)
    live in ``_extras``, which plain ``getattr`` does NOT see; an attribute set
    later with ``setattr`` (the restart hook) shadows the extra and wins.
    """
    if key in vars(sbatch):
        return vars(sbatch)[key]
    return (getattr(sbatch, "_extras", None) or {}).get(key)


def sbatch_set(sbatch, key: str, value) -> None:
    """Set a SbatchConfig value so that the rendered ``#SBATCH`` header sees
    it."""
    if key in {f.name for f in dataclasses.fields(sbatch)}:
        setattr(sbatch, key, value)
        return
    sbatch._extras[key] = value
    if key in vars(sbatch):  # drop a shadowing attribute, the extra is what renders
        delattr(sbatch, key)


def _split_nodes(text: str | None) -> list[str]:
    """Comma/whitespace separated node list, hostlist ranges kept as they
    are."""
    return [t for t in re.split(r"[\s,]+(?![^\[]*\])", text or "") if t]


def _exclude_list(slurm: SlurmConfig) -> str:
    """The job's own exclusion list, re-read from exclude_file like
    build_sbatch_directives."""
    current = sbatch_get(slurm.sbatch, "exclude")
    exclude_file = getattr(slurm, "exclude_file", None)
    if exclude_file:
        from oellm_autoexp.hydra_staged_sweep.config.resolvers import oc_exclude_nodes

        fresh = oc_exclude_nodes(str(exclude_file))
        if isinstance(fresh, (list, tuple)):
            fresh = ",".join(fresh)
        if fresh:
            current = fresh
    return str(current or "")


def run_fault_context(layout: PoolLayout, rid: str) -> dict | None:
    """What the restart hooks need to know about a pool run, or None.

    A pool run is not a SLURM job: ``runtime_job_id`` is its run id, so a hook
    templated with ``{runtime_job_id}`` would hand ``sacct`` an id it does not
    know. This gives the HOST's SLURM job id, the run's own nodes (the step's,
    not the allocation's) and its time window.
    """
    state = read_run_state(layout, rid)
    started = state.started
    if not started.get("job"):
        return None
    rd = layout.run_dir(rid)
    try:
        start = float(started.get("ts", "0"))
    except ValueError:
        start = 0.0
    end = _mtime(rd / "rc") or time.time()
    nodes = [n for n in (started.get("nodes") or "").split(",") if n]
    return {
        "slurm_job_id": started["job"],
        "run_id": rid,
        "host": started.get("host", ""),
        "nodes": nodes,
        "start": start,
        "end": end,
        "status": state.status,
        "reason": state.reason,
    }


def scan_pool_run(
    context: dict, *, log_root: str | Path, job_log: str | Path, exclude_file: str | Path
) -> dict:
    """The NCCL / node-fault attribution of monitor/nccl_peers.py for ONE pool
    run.

    nccl_peers.py reads a FAILED SLURM ALLOCATION from sacct (state,
    node list, time window) and stays silent for a live one -- but a
    pool run fails inside an allocation that keeps running. Its rules
    only need five facts about the failed attempt, so they are applied
    here to the RUN: its host's job id, its step's nodes in task order,
    its own window, and a failed state. Exclusions are applied exactly
    as nccl_peers.py --apply does (append-only, locked, audited next to
    the job's NCCL sidecars).
    """
    from oellm_autoexp.monitor import nccl_peers

    state = context["status"] if context["status"] in nccl_peers.FAILED else "FAILED"
    job = {
        "job_id": context["slurm_job_id"],
        "state": state,
        "start": context["start"],
        "end": context["end"],
        "hosts": list(context["nodes"]),
    }
    nodes: dict = {}
    try:
        for line in nccl_peers.command(["scontrol", "show", "nodes", "--oneliner"]).splitlines():
            fields = dict(re.findall(r"(\w+)=([^\s]+)", line))
            if fields.get("NodeName") in job["hosts"]:
                nodes[fields["NodeName"]] = fields
    except Exception as exc:  # scheduler view is corroboration only
        LOGGER.warning("[pool] node scan without scheduler state (%s)", exc)
    result = nccl_peers.scan(Path(log_root), job, nodes, job_log=Path(job_log))
    result["pool_run"] = context["run_id"]
    result["newly_excluded"] = nccl_peers.apply(
        result, Path(exclude_file), Path(log_root) / "nccl-peer-events.jsonl"
    )
    return result


# ---------------------------------------------------------------------------
# pool creation (login node, run_autoexp)
# ---------------------------------------------------------------------------


def create_or_update_pool(
    pool_dir: str | Path,
    *,
    host_definition: SlurmJobConfig,
    config: PoolHostConfig,
    session_dir: str | Path,
) -> PoolLayout:
    layout = PoolLayout(pool_dir)
    layout.root.mkdir(parents=True, exist_ok=True)
    for sub in (layout.runs_dir, layout.hosts_dir, layout.requests_dir, layout.manifests_dir):
        sub.mkdir(parents=True, exist_ok=True)
    meta = layout.load_meta() if layout.exists() else {"created": time.time()}
    cfg = replace(config, name=config.name or host_definition.name)
    meta.update(
        {
            "name": cfg.name,
            "pool_dir": str(layout.root),
            "session_dir": str(Path(session_dir).absolute()),
            "config": asdict(cfg),
            "updated": time.time(),
        }
    )
    _atomic_write(layout.host_definition, json.dumps(asdict(host_definition), indent=2))
    layout.write_meta(meta)
    if not layout.exclude_nodes.exists():
        layout.exclude_nodes.write_text("", encoding="utf-8")
    register_pool_in_session(session_dir, layout.root)
    return layout


def register_pool_in_session(session_dir: str | Path, pool_dir: str | Path) -> None:
    path = Path(session_dir) / "pools.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    pools = set((_read(path) or "").split())
    if str(Path(pool_dir).absolute()) not in pools:
        pools.add(str(Path(pool_dir).absolute()))
        _atomic_write(path, "\n".join(sorted(pools)) + "\n")


def _truthy(value: Any) -> bool:
    """A metadata flag given as a bool, a number or a string ("true", "yes", "1")."""
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def submit_request(pool_dir: str | Path, op: str, **payload: Any) -> Path:
    layout = PoolLayout(pool_dir)
    path = layout.requests_dir / f"{time.time():.6f}-{uuid.uuid4().hex[:6]}.json"
    _atomic_write(path, json.dumps({"op": op, **payload}))
    return path


def pools_have_pending_requests(session_dir: str | Path) -> bool:
    """Whether a pool of this monitor session still needs its monitor although no job record
    is active: unprocessed pool_ctl requests (an `adopt` creates the first record), or a pool
    that keeps hosts without work (min_hosts > 0, not closed) and has none yet."""
    for pool_dir in (_read(Path(session_dir) / "pools.txt") or "").split():
        layout = PoolLayout(pool_dir)
        try:
            if any(layout.requests_dir.glob("*.json")):
                return True
            if layout.exists() and not layout.load_state().get("closed"):
                cfg = layout.load_config()
                if cfg.min_hosts > 0 or cfg.keep_monitor:
                    return True
        except (OSError, ValueError):
            continue
    return False


def check_fits_host(
    job: SlurmJobConfig, host: SlurmJobConfig, accounts: list[str] | None = None
) -> list[str]:
    """Reasons a job cannot run inside this pool's hosts (empty = fine). ``accounts``:
    the pool's ``host_accounts``; a job of any of them fits."""
    problems: list[str] = []
    jsb, hsb = job.slurm.sbatch, host.slurm.sbatch
    if int(jsb.nodes or 1) > int(hsb.nodes or 1):
        problems.append(f"needs {jsb.nodes} nodes, the pool's hosts have {hsb.nodes}")
    for key in ("account", "partition", "qos", "reservation"):
        mine, theirs = getattr(jsb, key, None), getattr(hsb, key, None)
        if key == "account" and accounts:
            if mine and str(mine) not in accounts:
                problems.append(f"sbatch.account={mine} but the pool's hosts use {accounts}")
            continue
        if mine and theirs and str(mine) != str(theirs):
            problems.append(f"sbatch.{key}={mine} but the pool's hosts use {theirs}")
    if job.slurm.array or job.array_group:
        problems.append("array jobs cannot run in a pool")
    return problems


# ---------------------------------------------------------------------------
# the monitor-side manager
# ---------------------------------------------------------------------------


@dataclass
class HostView:
    hid: str
    record: Any
    status: str | None
    layout: PoolLayout
    info: dict[str, str] = field(default_factory=dict)
    alive_age: float | None = None
    state: list[str] = field(default_factory=list)
    retiring: bool = False
    exited: bool = False
    nodes: int = 1
    time_limit_s: float | None = None

    @property
    def submitted(self) -> bool:
        return bool(self.record.runtime.submitted)

    @property
    def adopted(self) -> str | None:
        """The SLURM job id of an adopted host (pool_ctl adopt), else None."""
        meta = getattr(self.record.definition, "metadata", None) or {}
        jid = meta.get("pool_host_adopted")
        return str(jid) if jid else None

    @property
    def account(self) -> str | None:
        meta = getattr(self.record.definition, "metadata", None) or {}
        acc = meta.get("pool_host_account")
        if acc:
            return str(acc)
        acc = sbatch_get(self.record.definition.slurm.sbatch, "account")
        return str(acc) if acc else None

    @property
    def dead(self) -> bool:
        if self.exited:
            return True
        return self.status is not None and self.status not in ACTIVE_HOST_STATES

    @property
    def start_ts(self) -> float:
        try:
            return float(self.info.get("START_TS", "0"))
        except ValueError:
            return 0.0

    def live(self, stale_s: float) -> bool:
        return (
            self.status == "RUNNING"
            and bool(self.info)
            and not self.exited
            and self.alive_age is not None
            and self.alive_age < stale_s
        )

    @property
    def usable_nodes(self) -> int:
        # state: "idle <ts> <usable>" / "busy <rid> <ts> <usable>"
        try:
            return int(self.state[-1]) if len(self.state) >= 3 else self.nodes
        except ValueError:
            return self.nodes

    @property
    def state_ts(self) -> float:
        try:
            return float(self.state[-2]) if len(self.state) >= 3 else self.start_ts
        except ValueError:
            return self.start_ts

    def remaining_s(self, now: float) -> float:
        if self.time_limit_s is None or not self.start_ts:
            return float("inf")
        # an adopted host's allocation started before its supervisor (JOB_START_TS)
        try:
            start = float(self.info.get("JOB_START_TS") or self.start_ts)
        except ValueError:
            start = self.start_ts
        return start + self.time_limit_s - now

    def assignments(self) -> list[str]:
        d = self.layout.host_dir(self.hid) / "assign"
        try:
            return sorted(p.name for p in d.iterdir() if not p.name.startswith("."))
        except OSError:
            return []


class PoolManager:
    """``JobClientProtocol`` for pool runs + the pool scheduler (see module
    doc)."""

    def __init__(self) -> None:
        # rid -> pool dir
        self._runs: dict[str, str] = {}
        # hid views of the current poll, by pool dir
        self._hosts: dict[str, dict[str, HostView]] = {}

    # --- JobClientProtocol --------------------------------------------------
    def submit(self, job: SlurmJobConfig) -> str:
        if not job.pool_dir:
            raise ValueError(f"job {job.name} has no pool_dir")
        layout = PoolLayout(job.pool_dir)
        cfg = layout.load_config()
        rid = layout.allocate_run_id(pool_tag(cfg.name))
        rd = layout.run_dir(rid)
        log_path = str(expand_log_path(job.log_path, rid).absolute())
        sbatch = job.slurm.sbatch
        nodes = int(sbatch.nodes or 1)
        tpn = int(getattr(sbatch, "ntasks_per_node", None) or 1)
        max_runtime = parse_slurm_time(getattr(sbatch, "time", None)) or 0
        metadata = dict(job.metadata or {})
        priority = int(metadata.get("pool_priority", 0) or 0)
        enqueued_ts = float(metadata.get("pool_enqueued_ts") or time.time())

        workdir = metadata.get("pool_workdir") or ""
        step_text = render_step_script(job.slurm, rid, log_path=log_path, workdir=workdir or None)
        (rd / "step.sh").write_text(step_text, encoding="utf-8")
        write_env_file(
            rd / "meta.env",
            {
                "STOP_SIGNAL": metadata.get("pool_stop_signal") or stop_signal_of(step_text),
                "WORKDIR": workdir,
                "NODES": nodes,
                "NTASKS_PER_NODE": tpn,
                "MAX_RUNTIME_S": int(max_runtime),
                "LOG": log_path,
                "NAME": _sanitize(job.name),
                "EXCLUDE": _exclude_list(job.slurm),
                # CPU-only work may also use the pool's excluded (bad GPU / NIC) nodes, those first
                "USE_EXCLUDED_NODES": "1"
                if _truthy(metadata.get("pool_use_excluded_nodes"))
                else "",
            },
        )
        _atomic_write(
            rd / "run.json",
            json.dumps(
                {
                    "rid": rid,
                    "job_id": metadata.get("pool_job_id", ""),
                    "name": job.name,
                    "log": log_path,
                    "nodes": nodes,
                    "priority": priority,
                    "enqueued_ts": enqueued_ts,
                    "submitted_ts": time.time(),
                    # the code it was planned with; the supervisor compares the code it
                    # launches with against this (launch_provenance.json)
                    "provenance": metadata.get("provenance"),
                },
                indent=2,
            ),
        )
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        if job.config_path:
            config_path = expand_log_path(job.config_path, rid)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            dump_job_config(config_path, job)
        self._runs[rid] = str(layout.root)
        if _truthy(metadata.get("pool_preflight_template")):
            # the pool's preflight node check: kept as a template, cloned for every new host
            # (its node count from the host); the record itself ends at once
            tdir = layout.root / "preflight_template"
            tdir.mkdir(exist_ok=True)
            for name in ("step.sh", "meta.env", "run.json"):
                _atomic_write(tdir / name, (rd / name).read_text(encoding="utf-8"))
            _atomic_write(tdir / "rid", rid + "\n")
            write_rc(layout, rid, 0, "preflight_template")
            LOGGER.info("[pool %s] preflight template set from %s (%s)", cfg.name, rid, job.name)
            return rid
        LOGGER.info("[pool %s] queued run %s (%s, %d node(s))", cfg.name, rid, job.name, nodes)
        return rid

    def submit_array(self, job: SlurmJobConfig, indices: list[int]) -> list[str]:
        raise ValueError("array jobs cannot run in a pool")

    def cancel(self, job_id: str) -> None:
        pool_dir = self._runs.get(job_id)
        if pool_dir is None:
            return
        layout = PoolLayout(pool_dir)
        state = read_run_state(layout, job_id)
        if state.status not in ("PENDING", "RUNNING"):
            return
        _atomic_write(layout.run_dir(job_id) / "stop", f"{time.time()}\n")
        if state.status == "PENDING" and state.host is None:
            # never handed to a supervisor: nobody else will close it
            write_rc(layout, job_id, 0, "cancelled")
        LOGGER.info("[pool] cancel requested for run %s (%s)", job_id, state.status)

    def remove(self, job_id: str) -> None:
        # the record is done with this run; scheduling reads the run's files, not this map
        self._runs.pop(job_id, None)

    def register_job(self, job: SlurmJobConfig, job_id: str, state: str | None = None) -> None:
        if job.pool_dir:
            self._runs[job_id] = str(PoolLayout(job.pool_dir).root)

    def update_excludes(self, job_id: str, nodelist: str) -> None:
        return

    def squeue(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for rid, pool_dir in self._runs.items():
            out[rid] = read_run_state(PoolLayout(pool_dir), rid).status
        return out

    # --- per-poll host view -------------------------------------------------
    @staticmethod
    def is_host(record) -> bool:
        return (record.definition.metadata or {}).get("pool_role") == POOL_ROLE_HOST

    @staticmethod
    def is_run(record) -> bool:
        return isinstance(record.definition, SlurmJobConfig) and bool(
            getattr(record.definition, "pool_dir", None)
        )

    def refresh(self, statuses: dict[str, str], records: list[JobRecord]) -> None:
        """Build the host views and close runs whose host is gone.

        Called by the monitor after the SLURM statuses are known and before
        :meth:`squeue`, so a run on a dead host reads PREEMPTED in the same poll.
        """
        self._hosts = {}
        for record in records:
            if not self.is_host(record):
                continue
            pool_dir = str(PoolLayout(record.definition.metadata["pool_dir"]).root)
            hid = record.definition.metadata["pool_host_id"]
            layout = PoolLayout(pool_dir)
            hdir = layout.host_dir(hid)
            rid = record.runtime.runtime_job_id
            alive = _mtime(hdir / "alive")
            view = HostView(
                hid=hid,
                record=record,
                status=statuses.get(rid) if rid else None,
                layout=layout,
                info=read_env_file(hdir / "info.env"),
                alive_age=(time.time() - alive) if alive else None,
                state=(_read(hdir / "state") or "").split(),
                retiring=(hdir / "control" / "exit").exists()
                or (hdir / "control" / "drain").exists(),
                exited=(hdir / "exited").exists(),
                nodes=int(record.definition.slurm.sbatch.nodes or 1),
                time_limit_s=parse_slurm_time(
                    getattr(record.definition.slurm.sbatch, "time", None)
                ),
            )
            self._hosts.setdefault(pool_dir, {})[hid] = view

        for rid, pool_dir in list(self._runs.items()):
            layout = PoolLayout(pool_dir)
            state = read_run_state(layout, rid)
            if state.status not in ("PENDING", "RUNNING") or state.host is None:
                continue
            host = self._hosts.get(pool_dir, {}).get(state.host)
            gone = host is None or host.dead
            if host is not None and not gone and host.status == "RUNNING" and host.info:
                cfg = layout.load_config()
                if (host.alive_age is None or host.alive_age > cfg.host_stale_s) and host.adopted:
                    LOGGER.warning(
                        "[pool] adopted host %s (job %s): supervisor heartbeat %s s old -- "
                        "replacing the supervisor (the allocation is kept)",
                        host.hid,
                        host.adopted,
                        "?" if host.alive_age is None else f"{host.alive_age:.0f}",
                    )
                    self._kill_adopted_supervisor(host)
                    for name in ("info.env", "alive", "state"):
                        (layout.host_dir(host.hid) / name).unlink(missing_ok=True)
                    host.info = {}
                    gone = True
                elif host.alive_age is None or host.alive_age > cfg.host_stale_s:
                    LOGGER.warning(
                        "[pool] host %s (job %s) heartbeat is %s s old: cancelling it",
                        host.hid,
                        host.record.runtime.runtime_job_id,
                        "?" if host.alive_age is None else f"{host.alive_age:.0f}",
                    )
                    _atomic_write(layout.host_dir(host.hid) / "exited", "stale heartbeat\n")
                    host.exited = True
                    gone = True
                    self._cancel_host_job(host)
            if not gone:
                continue
            if state.status == "RUNNING":
                LOGGER.warning("[pool] run %s lost its host %s", rid, state.host)
                write_rc(layout, rid, -1, "host_lost")
            else:
                # assigned but never launched: back into the queue
                self._unassign(layout, rid, state.host)

    _slurm_cancel = None  # set by the monitor loop: callable(job_id)

    def _cancel_host_job(self, host: HostView) -> None:
        rid = host.record.runtime.runtime_job_id
        if rid and self._slurm_cancel is not None:
            try:
                self._slurm_cancel(rid)
            except Exception as exc:  # pragma: no cover - best effort
                LOGGER.warning("[pool] could not cancel host job %s: %s", rid, exc)

    def _cancel_host_record(self, loop: MonitorLoop, host: HostView) -> None:
        if host.adopted:  # never cancel an adopted job just because it is not needed
            return
        LOGGER.info("[pool] cancelling queued host %s", host.hid)
        self._cancel_host_job(host)
        _atomic_write(host.layout.host_dir(host.hid) / "exited", "cancelled while queued\n")
        loop._store.mark_finished(host.record.job_id, "cancelled")

    @staticmethod
    def _unassign(layout: PoolLayout, rid: str, hid: str | None) -> None:
        for path in (
            layout.run_dir(rid) / "assigned",
            layout.host_dir(hid) / "assign" / rid if hid else None,
        ):
            if path is not None:
                try:
                    path.unlink()
                except OSError:
                    pass

    # --- scheduling ---------------------------------------------------------
    def pool_dirs(self, store: JobFileStore, records: list[JobRecord]) -> list[str]:
        pools = set((_read(Path(store.root) / "pools.txt") or "").split())
        for record in records:
            if self.is_run(record):
                pools.add(str(PoolLayout(record.definition.pool_dir).root))
            elif self.is_host(record):
                pools.add(str(PoolLayout(record.definition.metadata["pool_dir"]).root))
        return sorted(p for p in pools if PoolLayout(p).exists())

    def tick(self, loop: MonitorLoop) -> None:
        records = loop._store.load_all()
        for pool_dir in self.pool_dirs(loop._store, records):
            try:
                self._tick_pool(loop, PoolLayout(pool_dir), records)
            except Exception:
                if loop.no_error_catching:
                    raise
                LOGGER.exception("[pool] scheduling of %s failed", pool_dir)

    def _tick_pool(self, loop: MonitorLoop, layout: PoolLayout, records: list[JobRecord]) -> None:
        now = time.time()
        cfg = layout.load_config()
        _atomic_write(layout.monitor_alive, f"{os.getpid()} {now:.0f}\n")
        state = layout.load_state()
        self._process_requests(loop, layout, state)

        runs = [
            r
            for r in records
            if self.is_run(r)
            and str(PoolLayout(r.definition.pool_dir).root) == str(layout.root)
            and r.runtime.final_state is None
        ]
        hosts = list(self._hosts.get(str(layout.root), {}).values())
        self._manage_adopted(loop, layout, cfg, hosts, state, now)

        # 1. close assignments of finished runs a dead supervisor left behind
        for host in hosts:
            for rid in host.assignments():
                if (layout.run_dir(rid) / "rc").exists() and host.dead:
                    self._unassign(layout, rid, host.hid)

        live = [h for h in hosts if h.live(cfg.host_stale_s) and not h.retiring]
        busy = {
            h.hid
            for h in hosts
            if any(not (layout.run_dir(r) / "rc").exists() for r in h.assignments())
        }

        # 1b. hosts the monitor never releases, and the preflight node check of new hosts:
        #     an unverified host takes no run, and is not released as idle meanwhile
        held = {
            h.hid
            for h in hosts
            if cfg.hold_hosts or (layout.host_dir(h.hid) / "control" / "hold").exists()
        }
        unverified = self._preflight(layout, cfg, state, live, held, now)
        live = [h for h in live if not h.retiring]

        # 2. handover: retire the oldest hosts beyond max_hosts_running once a
        #    newer one is up. `exit` makes its supervisor preempt the run
        #    (graceful stop, checkpoint) and release the allocation; the run comes
        #    back PREEMPTED, is requeued by the monitor, and goes to the new host.
        #    With handover_verify the old allocation is kept until the moved runs
        #    are healthy on the new one (or they go back: see _handover_check).
        handover_waiting = False
        handovers = state.setdefault("handover", {})
        if cfg.handover and len(live) > cfg.max_hosts_running:
            excess = len(live) - cfg.max_hosts_running
            verified = [h for h in live if h.hid not in unverified]
            newest = max(verified, key=lambda h: h.start_ts, default=None)
            for host in sorted(live, key=lambda h: h.start_ts)[:excess]:
                if host.hid in handovers:
                    continue
                if now - host.start_ts < cfg.handover_min_age_s:
                    handover_waiting = True
                    continue
                if cfg.handover_verify:
                    if newest is None or newest.start_ts <= host.start_ts:
                        handover_waiting = True  # a newer host is still in its preflight
                        continue
                    jobs = []
                    for rid in host.assignments():
                        rd = layout.run_dir(rid)
                        if (rd / "rc").exists():
                            continue
                        job_id = json.loads(_read(rd / "run.json") or "{}").get("job_id", "")
                        if not job_id:
                            continue  # a preflight run
                        _atomic_write(rd / "preempt", f"handover to {newest.hid}\n")
                        jobs.append(job_id)
                    handovers[host.hid] = {"new": newest.hid, "jobs": jobs, "ts": now}
                    LOGGER.info(
                        "[pool %s] handover: moving %s from %s to %s; %s kept until they are healthy",
                        cfg.name,
                        jobs or "nothing",
                        host.hid,
                        newest.hid,
                        host.hid,
                    )
                    continue
                if host.hid in held:
                    continue
                self._control(layout, host.hid, "exit", "handover to a newer host")
                host.retiring = True
                LOGGER.info(
                    "[pool %s] handover: retiring %s (job %s)",
                    cfg.name,
                    host.hid,
                    host.record.runtime.runtime_job_id,
                )
            live = [h for h in live if not h.retiring]
        self._handover_check(loop, layout, cfg, handovers, hosts, held, now)
        live = [h for h in live if not h.retiring]
        handover_old = set(handovers)
        handover_in_progress = (
            handover_waiting
            or bool(handovers)
            or any(h.retiring and h.hid in busy and not h.dead for h in hosts)
        )

        # 3. queue
        queued = []
        for record in runs:
            rid = record.runtime.runtime_job_id
            if not rid or not record.runtime.submitted:
                continue
            rstate = read_run_state(layout, rid)
            if rstate.status == "PENDING" and rstate.host is None:
                meta = read_env_file(layout.run_dir(rid) / "meta.env")
                md = record.definition.metadata or {}
                queued.append(
                    (
                        -int(md.get("pool_priority", 0) or 0),
                        float(md.get("pool_enqueued_ts") or 0.0),
                        rid,
                        int(meta.get("NODES", "1") or 1),
                        str(md.get("pool_singleton") or ""),
                        record.job_id,
                    )
                )
        queued.sort()

        def run_meta(rid: str) -> tuple[int, bool]:
            """(nodes, may it use the pool's excluded nodes) of a run, from its meta.env."""
            meta = read_env_file(layout.run_dir(rid) / "meta.env")
            try:
                n = int(meta.get("NODES", "1") or 1)
            except ValueError:
                n = 1
            return n, bool(meta.get("USE_EXCLUDED_NODES"))

        def host_total(host: HostView) -> int:
            try:
                return int(host.info.get("NUM_NODES") or host.nodes)
            except (TypeError, ValueError):
                return host.nodes

        def fits(host: HostView, nodes: int, rid: str | None = None) -> bool:
            cap = host_total(host) if rid is not None and run_meta(rid)[1] else host.usable_nodes
            return nodes <= cap and host.remaining_s(now) >= cfg.min_remaining_s

        # 4. assign: newest host first (after a handover that is the new one), as many runs
        #    as its free nodes and max_runs_per_host allow (packing; default one per host)
        max_runs = max(1, int(cfg.max_runs_per_host))
        for host in sorted(live, key=lambda h: -h.start_ts):
            if host.hid in unverified or host.hid in handover_old:
                continue  # in its preflight / being handed over: no new run
            active = [r for r in host.assignments() if not (layout.run_dir(r) / "rc").exists()]
            # the supervisor puts a USE_EXCLUDED_NODES run on the excluded nodes first: free
            # healthy nodes, and free excluded ones only such runs may take
            excluded = max(0, host_total(host) - host.usable_nodes)
            n_plain = sum(run_meta(r)[0] for r in active if not run_meta(r)[1])
            n_excl = sum(run_meta(r)[0] for r in active if run_meta(r)[1])
            free_excl = excluded - min(n_excl, excluded)
            free = host.usable_nodes - n_plain - max(0, n_excl - excluded)
            for item in list(queued):
                if len(active) >= max_runs:
                    break
                _, _, rid, nodes, singleton, job_id = item
                use_excl = run_meta(rid)[1]
                cap = free + (free_excl if use_excl else 0)
                if nodes <= cap and host.remaining_s(now) >= cfg.min_remaining_s:
                    if singleton and not self._claim_singleton(
                        layout, singleton, rid, nodes, job_id
                    ):
                        continue
                    self._assign(layout, rid, host.hid)
                    busy.add(host.hid)
                    active.append(rid)
                    if use_excl:
                        take = min(nodes, free_excl)
                        free_excl -= take
                        free -= nodes - take
                    else:
                        free -= nodes
                    queued.remove(item)
                    LOGGER.info(
                        "[pool %s] assigned %s to %s (job %s)",
                        cfg.name,
                        rid,
                        host.hid,
                        host.record.runtime.runtime_job_id,
                    )

        # 5. release idle hosts that have had nothing to do for idle_exit_s -- or
        #    at once when there IS queued work and this host can run none of it:
        #    too few usable nodes (excluded bad nodes) or too little time left only
        #    get worse, so waiting idle_exit_s would buy nothing. The freed slot
        #    is refilled by a fresh allocation in step 6.
        for host in live:
            if host.hid in busy or handover_in_progress:
                continue
            if host.hid in held or host.hid in unverified:
                continue  # held: never released by the monitor; unverified: in its preflight
            if any(fits(host, item[3], item[2]) for item in queued):
                continue
            # only if a FRESH host would do better: a job bigger than every host of
            # this pool (host definition changed after enqueue) must not turn into
            # a loop of allocations that start and are released at once
            fresh_fit = [item for item in queued if item[3] <= host.nodes]
            if queued and not fresh_fit:
                LOGGER.warning(
                    "[pool %s] queued job(s) %s need more nodes than any host of this pool (%d)",
                    cfg.name,
                    [item[2] for item in queued],
                    host.nodes,
                )
            if fresh_fit and host.adopted:
                # an adopted / attached allocation (pool_ctl adopt / attach) is one we took
                # over and cannot get back: never give it up at once because the queued work
                # does not fit it (cns144 47998967, 2026-10-30: a 128-node run requeued
                # against its 123 usable nodes released it with 15 h left); only idle_exit_s
                warned = self.__dict__.setdefault("_unfit_warned", set())
                if (cfg.name, host.hid) not in warned:
                    LOGGER.warning(
                        "[pool %s] adopted host %s cannot run the queued job(s) %s "
                        "(%d usable nodes): kept; released only after idle_exit_s",
                        cfg.name,
                        host.hid,
                        [item[2] for item in fresh_fit],
                        host.usable_nodes,
                    )
                    warned.add((cfg.name, host.hid))
            elif fresh_fit:
                need = min(item[3] for item in fresh_fit)
                why = (
                    f"needs {need} node(s), {host.usable_nodes} usable"
                    if need > host.usable_nodes
                    else f"{host.remaining_s(now):.0f}s left, {cfg.min_remaining_s:.0f}s needed"
                )
                self._control(layout, host.hid, "exit", f"cannot run any queued job ({why})")
                host.retiring = True
                LOGGER.info(
                    "[pool %s] releasing host %s: cannot run any queued job (%s)",
                    cfg.name,
                    host.hid,
                    why,
                )
                continue
            idle_for = now - host.state_ts
            if idle_for >= cfg.idle_exit_s:
                self._control(layout, host.hid, "exit", f"idle for {idle_for:.0f}s")
                host.retiring = True
                LOGGER.info("[pool %s] releasing idle host %s", cfg.name, host.hid)

        # 6. keep hosts submitted, and their exclusion lists current while queued
        self._refresh_queued_host_excludes(loop, layout, hosts)
        work = bool(runs)
        running = [h for h in hosts if h.live(cfg.host_stale_s) and not h.retiring]
        successors = cfg.keep_pending
        if cfg.successor_delay_s and (
            not running or any(now - h.start_ts < cfg.successor_delay_s for h in running)
        ):
            successors = 0
        desired = (cfg.max_hosts_running + successors) if work else 0
        desired = max(desired, cfg.min_hosts)
        if state.get("closed"):
            desired = 0
            for host in hosts:
                if host.dead or host.retiring:
                    continue
                if host.info:
                    # up: finish what it runs, then release
                    self._control(layout, host.hid, "drain", "pool closed")
                else:
                    # still waiting in the queue: it would only start to sit idle
                    self._cancel_host_record(loop, host)
                host.retiring = True
        # adopted jobs still in their queue are not the pool's to count or cancel
        current = [
            h for h in hosts if not h.retiring and not h.dead and not (h.adopted and not h.info)
        ]
        accounts = host_accounts(cfg)
        create_account: str | None = None
        if accounts:
            missing, create_account = self._account_targets(
                loop, cfg, accounts, current, running, desired, now, held
            )
            current = [h for h in current if not h.retiring]
        else:
            # hosts still waiting in the queue beyond what is needed (keep_pending
            # lowered, work gone): cancel the newest; an up host is left to idle_exit
            # (a held one is never cancelled: it counts, and the newest unheld go first)
            queued_hosts = sorted(
                (h for h in current if not h.info and h.hid not in held),
                key=lambda h: h.record.job_id,
                reverse=True,
            )
            for host in queued_hosts[: max(0, len(current) - desired)]:
                self._cancel_host_record(loop, host)
                host.retiring = True
            current = [h for h in current if not h.retiring]
            missing = desired - len(current)
        # A host that ended without its supervisor ever starting -- sbatch failed,
        # or SLURM killed it before the batch shell ran -- and that the pool did not
        # cancel on purpose (those carry an `exited` file) is a FAILED START. Back
        # off exponentially and say so loudly: retrying every poll would only fill
        # the session with dead host records (a tutorial monitor without sbatch on
        # its PATH created ten in twenty seconds).
        last = state.get("last_host_id")
        if last and state.get("last_host_judged") != last:
            record = loop._store.load(f"{pool_tag(cfg.name)}.{last}", include_finished=True)
            hdir = layout.host_dir(last)
            if (hdir / "info.env").exists():
                state["host_failures"] = 0
                state["last_host_judged"] = last
            elif (
                record is not None and record.runtime.final_state and not (hdir / "exited").exists()
            ):
                state["host_failures"] = int(state.get("host_failures", 0)) + 1
                state["last_host_judged"] = last
                LOGGER.error(
                    "[pool %s] host %s ended without ever starting (sbatch failed, or the job "
                    "died before its batch shell ran; see the monitor log above). "
                    "Failed starts in a row: %d",
                    cfg.name,
                    last,
                    state["host_failures"],
                )
        failures = int(state.get("host_failures", 0))
        backoff = 0.0 if failures == 0 else min(600.0, 30.0 * 2 ** (failures - 1))
        waited = now - float(state.get("last_host_created_ts", 0.0))
        if cfg.max_own_hosts >= 0:
            own = [h for h in current if not h.adopted]
            missing = min(missing, max(0, cfg.max_own_hosts - len(own)))
        # One host at a time while one is still waiting for its sbatch.
        if missing > 0 and not any(not h.submitted for h in current):
            if waited >= backoff:
                state["last_host_id"] = self._create_host(loop, layout, cfg, create_account)
                state["last_host_created_ts"] = now
            elif failures:
                LOGGER.info(
                    "[pool %s] next host in %.0f s (backing off after %d failed start(s))",
                    cfg.name,
                    backoff - waited,
                    failures,
                )
        layout.write_state(state)

    def _account_targets(
        self,
        loop: MonitorLoop,
        cfg: PoolHostConfig,
        accounts: list[str],
        current: list[HostView],
        running: list[HostView],
        desired: int,
        now: float,
        held: set[str] = frozenset(),
    ) -> tuple[int, str | None]:
        """host_accounts: the queued hosts wanted per account, surplus ones cancelled
        (newest first). Returns (hosts missing, the account to create the next one in).

        ``desired`` (max_hosts_running + successors, at least min_hosts) is the total the
        single-account pool keeps. The hosts that are up count once. Of the queued rest,
        every other account holds one insurance host while nothing runs or once the running
        hosts have less than ``fallback_lead_s`` left (0 = always); the FIRST account the
        remainder, and always at least one (its queue keeps an aging host). Whichever starts
        first takes the run (handover)."""
        up = [h for h in current if h.info]
        wanted_queued = max(0, desired - len(up))
        left = min((h.remaining_s(now) for h in running), default=None)
        fallback_on = cfg.fallback_lead_s <= 0 or left is None or left < cfg.fallback_lead_s
        missing, create_account = 0, None
        # a fallback account holds one insurance allocation, not a full set of successors;
        # the first account the rest, and always at least one while any is wanted (it ages)
        fallback = min(1, wanted_queued) if fallback_on else 0
        primary = max(min(1, wanted_queued), wanted_queued - fallback * (len(accounts) - 1))
        for i, account in enumerate(accounts):
            target = primary if i == 0 else fallback
            queued = sorted(
                (h for h in current if not h.info and (h.account or accounts[0]) == account),
                key=lambda h: h.record.job_id,
                reverse=True,
            )
            surplus = [h for h in queued if h.hid not in held]  # a held host is never cancelled
            for host in surplus[: max(0, len(queued) - target)]:
                self._cancel_host_record(loop, host)
                host.retiring = True
            gap = target - min(len(queued), target)
            if gap > 0 and create_account is None:
                create_account = account
            missing += gap
        # queued hosts of an account no longer in the list
        for host in current:
            if (
                not host.info
                and not host.retiring
                and host.hid not in held
                and host.account
                and host.account not in accounts
            ):
                self._cancel_host_record(loop, host)
                host.retiring = True
        return missing, create_account

    # --- one run per job across pools (job.metadata.pool_singleton) ----------
    @staticmethod
    def _singleton_dir(layout: PoolLayout, key: str) -> Path:
        # next to the pools (every pool of a cluster lives under one root): shared by the
        # monitors of all of them
        return layout.root.parent / "_singletons" / _sanitize(key)

    @staticmethod
    def _singleton_holder(d: Path) -> dict | None:
        """The run that holds the key, while it has not ended (no rc), else None."""
        try:
            holder = json.loads((d / "holder.json").read_text())
        except (OSError, json.JSONDecodeError):
            return None
        rd = Path(holder.get("pool", "")) / "runs" / str(holder.get("rid", ""))
        if not rd.is_dir() or (rd / "rc").exists():
            return None
        return holder

    @staticmethod
    def _singleton_reservation(d: Path, max_age_s: float = 3 * 3600) -> dict | None:
        """The run the key is reserved for (it superseded the holder), while that run has not
        ended and the reservation is younger than max_age_s, else None."""
        try:
            res = json.loads((d / "reserved.json").read_text())
        except (OSError, json.JSONDecodeError):
            return None
        rd = Path(res.get("pool", "")) / "runs" / str(res.get("rid", ""))
        if (
            not rd.is_dir()
            or (rd / "rc").exists()
            or time.time() - float(res.get("ts", 0)) > max_age_s
        ):
            return None
        return res

    def _claim_singleton(
        self, layout: PoolLayout, key: str, rid: str, nodes: int, job_id: str
    ) -> bool:
        """May run `rid` (`nodes` nodes) of the singleton job `key` start in this pool now?

        One run per key across every pool: the key is free (no holder, or its run ended, and no
        reservation for a larger run), or held by this very run. A holder in ANOTHER pool on
        fewer nodes is superseded -- its run is preempted there (graceful stop with a save, its
        job requeued in that pool) and the key reserved for this run, which starts once the
        holder's run has ended; on as many or more nodes the holder keeps it.
        Checked and claimed under a mkdir lock (atomic on GPFS), so two monitors never both
        start the job."""
        d = self._singleton_dir(layout, key)
        d.mkdir(parents=True, exist_ok=True)
        lock = d / "lock"
        for _ in range(50):
            try:
                lock.mkdir()
                break
            except FileExistsError:
                if time.time() - (_mtime(lock) or time.time()) > 120:  # a monitor died holding it
                    shutil.rmtree(lock, ignore_errors=True)
                time.sleep(0.2)
        else:
            return False
        try:
            holder = self._singleton_holder(d)
            me = {"pool": str(layout.root), "rid": rid}
            reserved = self._singleton_reservation(d)
            if (
                holder is None
                and reserved is not None
                and {k: reserved.get(k) for k in me} != me
                and int(reserved.get("nodes", 0)) > nodes
            ):
                return False  # the key is promised to a larger run that superseded the holder
            if holder is None or {k: holder.get(k) for k in me} == me:
                _atomic_write(
                    d / "holder.json",
                    json.dumps({**me, "nodes": nodes, "job_id": job_id, "ts": time.time()}),
                )
                return True
            if holder.get("pool") != str(layout.root) and nodes > int(holder.get("nodes", 0)):
                if not holder.get("superseded"):
                    LOGGER.info(
                        "[pool %s] singleton %s: superseding run %s in %s (%s nodes) with %s (%d nodes)",
                        layout.root.name,
                        key,
                        holder["rid"],
                        holder["pool"],
                        holder.get("nodes"),
                        rid,
                        nodes,
                    )
                    # PREEMPT, not cancel: the smaller pool's run saves and its job is requeued
                    # there; the reservation keeps the key for this run, so the requeued run waits.
                    # If this run never starts or ends, the reservation lapses and the job
                    # continues in the smaller pool on its own.
                    hold_rd = Path(holder["pool"]) / "runs" / str(holder["rid"])
                    _atomic_write(hold_rd / "preempt", f"superseded by {layout.root.name} {rid}\n")
                    holder["superseded"] = time.time()
                    _atomic_write(d / "holder.json", json.dumps(holder))
                    _atomic_write(
                        d / "reserved.json",
                        json.dumps({**me, "nodes": nodes, "job_id": job_id, "ts": time.time()}),
                    )
            return False
        finally:
            shutil.rmtree(lock, ignore_errors=True)

    # --- adopted hosts ------------------------------------------------------
    def _adopt(self, loop: MonitorLoop, layout: PoolLayout, state: dict, request: dict) -> str:
        """Register a queued SLURM job (its hook already installed by pool_ctl adopt) as a
        host of this pool."""
        from oellm_autoexp.monitor.loop import JobRecord, JobRuntime

        jid = str(request["job_id"])
        for view in self._hosts.get(str(layout.root), {}).values():
            if view.adopted == jid:
                return f"already adopted as {view.hid}"
        base = layout.load_host_definition()
        hid = layout.allocate_host_id()
        sbatch = copy.deepcopy(base.slurm.sbatch)
        sbatch.nodes = int(request.get("nodes") or sbatch.nodes or 1)
        if request.get("time"):
            sbatch.time = str(request["time"])
        if request.get("account"):
            sbatch_set(sbatch, "account", str(request["account"]))
        slurm = replace(base.slurm, name=f"adopted_{jid}", sbatch=sbatch, exclude_file=None)
        definition = replace(
            base,
            name=f"adopted_{jid}",
            slurm=slurm,
            pool_dir=None,
            metadata={
                **dict(base.metadata or {}),
                "pool_role": POOL_ROLE_HOST,
                "pool_dir": str(layout.root),
                "pool_host_id": hid,
                "pool_host_adopted": jid,
                "pool_host_account": request.get("account"),
                "pool_adopt_workdir": request.get("workdir"),
            },
        )
        record = JobRecord(
            job_id=f"{pool_tag(layout.load_config().name)}.{hid}",
            definition=definition,
            runtime=JobRuntime(submitted=True, runtime_job_id=jid),
        )
        loop._store.upsert(record)
        # the SLURM client only tracks what it submitted (or rehydrated at startup)
        register = getattr(loop._get_client(record), "register_job", None)
        if register is not None:
            register(record.definition, jid, "PENDING")
        layout.host_dir(hid).mkdir(parents=True, exist_ok=True)
        LOGGER.info("[pool] adopted SLURM job %s as host %s (%s nodes)", jid, hid, sbatch.nodes)
        return f"adopted {jid} as {hid}"

    def _manage_adopted(
        self, loop, layout: PoolLayout, cfg: PoolHostConfig, hosts, state, now
    ) -> None:
        """Adopted jobs: when one runs, check that its hook engaged (else cancel it: its own
        training must not proceed) and start the supervisor into it; release one the pool
        no longer wants before a supervisor is up."""
        adopt_dir = layout.root / "adopt"
        seen = state.setdefault("adopt", {})
        for host in hosts:
            jid = host.adopted
            if not jid or host.exited or host.dead:
                continue
            entry = seen.setdefault(jid, {})
            if host.status != "RUNNING":
                continue
            entry.setdefault("running_seen", now)
            started = adopt_dir / f"{jid}.started"
            if not started.exists():
                if now - entry["running_seen"] >= cfg.adopt_hook_timeout_s:
                    LOGGER.error(
                        "[pool] adopted job %s RUNS but its hook never engaged (no %s after %.0f s):"
                        " cancelling it so its own training cannot proceed",
                        jid,
                        started,
                        now - entry["running_seen"],
                    )
                    _atomic_write(
                        layout.host_dir(host.hid) / "exited", "adoption hook did not engage\n"
                    )
                    host.exited = True
                    self._cancel_host_job(host)
                    loop._store.mark_finished(host.record.job_id, "cancelled")
                continue
            if "start_logged" not in entry:
                entry["start_logged"] = now
                line = started.read_text().strip() if started.exists() else ""
                ledger_line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{host.hid}\t{jid}\t-\tadopted job started: {line}\n"
                with open(layout.root / "ledger.tsv", "a") as fh:
                    fh.write(ledger_line)
                log = os.environ.get("POOL_ADOPT_START_LOG")
                if log:
                    try:
                        with open(log, "a") as fh:
                            fh.write(
                                f"{time.strftime('%F %T')} {jid} started={line} (adopted by pool {cfg.name})\n"
                            )
                    except OSError:
                        pass
            if host.retiring and not host.info:
                _atomic_write(
                    adopt_dir / f"{jid}.release", f"{now:.0f} retired before its supervisor\n"
                )
                continue
            if host.info:
                continue
            last = float(entry.get("supervisor_launch_ts", 0))
            if now - last < max(120.0, cfg.poll_s * 4):
                continue
            entry["supervisor_launch_ts"] = now
            entry["supervisor_launches"] = int(entry.get("supervisor_launches", 0)) + 1
            self._launch_adopted_supervisor(layout, host, jid)

    def _launch_adopted_supervisor(self, layout: PoolLayout, host: HostView, jid: str) -> None:
        import subprocess

        nodes = subprocess.run(
            ["squeue", "-h", "-j", jid, "-o", "%N"], capture_output=True, text=True, timeout=60
        ).stdout.strip()
        first = subprocess.run(
            ["scontrol", "show", "hostnames", nodes], capture_output=True, text=True, timeout=60
        ).stdout.split()
        repo = Path(__file__).resolve().parents[2]
        env = dict(os.environ)
        env.update(
            AUTOEXP_POOL_DIR=str(layout.root),
            AUTOEXP_POOL_HOST=host.hid,
            AUTOEXP_POOL_ADOPTED_JOB=jid,
        )
        env.pop("PYTHONPATH", None)
        cmd = [
            "srun",
            f"--jobid={jid}",
            "--overlap",
            "--mem=0",
            "--nodes=1",
            "--ntasks=1",
            "--ntasks-per-node=1",
            "--cpus-per-task=1",
            "--job-name=pool-supervisor",
        ]
        if first:
            cmd.append(f"--nodelist={first[0]}")
        cmd += [
            "bash",
            str(repo / "scripts" / "pool" / "supervisor.sh"),
            str(layout.root),
            host.hid,
        ]
        out = open(layout.root / f"slurm-{jid}.log", "a")
        LOGGER.info(
            "[pool] starting the supervisor of adopted host %s (job %s): %s",
            host.hid,
            jid,
            " ".join(cmd),
        )
        subprocess.Popen(
            cmd,
            cwd=str(repo),
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )

    @staticmethod
    def _kill_adopted_supervisor(host: HostView) -> None:
        import subprocess

        jid = host.adopted
        try:
            rows = subprocess.run(
                ["squeue", "-h", "-s", "-j", jid, "-o", "%i %j"],
                capture_output=True,
                text=True,
                timeout=60,
            ).stdout.splitlines()
            for row in rows:
                parts = row.split()
                if len(parts) == 2 and parts[1] == "pool-supervisor":
                    # KILL, never TERM: the supervisor traps TERM as a host-wide stop and
                    # then releases the adopted job (its held step 0 leaves, the job ends)
                    subprocess.run(["scancel", "--signal=KILL", parts[0]], timeout=60)
        except Exception as exc:  # pragma: no cover - best effort
            LOGGER.warning("[pool] could not stop the old supervisor of %s: %s", jid, exc)

    # --- verified handover ---------------------------------------------------
    def _handover_check(
        self,
        loop: MonitorLoop,
        layout: PoolLayout,
        cfg: PoolHostConfig,
        handovers: dict,
        hosts: list[HostView],
        held: set[str],
        now: float,
    ) -> None:
        """Finish or undo the verified handovers in progress: release the old host once every
        moved job runs healthy on the new one; release the NEW host (the jobs go back) when one
        of them fails there or the verification times out."""
        by_hid = {h.hid: h for h in hosts}
        pattern = re.compile(cfg.handover_healthy_pattern)
        for old_hid, ho in list(handovers.items()):
            old, new = by_hid.get(old_hid), by_hid.get(ho["new"])
            if old is None or old.dead:
                handovers.pop(old_hid)
                continue
            if new is None or new.dead or new.retiring:
                LOGGER.warning(
                    "[pool %s] handover %s -> %s: the new host is gone; the jobs stay on %s",
                    cfg.name,
                    old_hid,
                    ho["new"],
                    old_hid,
                )
                handovers.pop(old_hid)
                continue
            healthy, failed = True, ""
            for job_id in ho["jobs"]:
                record = loop._store.load(job_id, include_finished=True)
                if record is None or record.runtime.final_state in ("finished", "cancelled"):
                    continue
                if record.runtime.final_state == "failed":
                    failed = f"{job_id} failed"
                    break
                rid = record.runtime.runtime_job_id
                rs = read_run_state(layout, rid) if rid else None
                if rs is None or rs.started.get("host") != new.hid:
                    healthy = False  # not (yet) started on the new host
                    continue
                if rs.code not in (None, 0):
                    failed = f"{job_id} ended {rs.code} {rs.reason} on {new.hid}"
                    break
                log = read_env_file(layout.run_dir(rid) / "meta.env").get("LOG", "")
                try:
                    text = Path(log).read_text(errors="replace")[:20_000_000] if log else ""
                except OSError:
                    text = ""
                if len(pattern.findall(text)) < cfg.handover_healthy_steps:
                    healthy = False
            if not failed and not healthy and now - ho["ts"] > cfg.handover_verify_timeout_s:
                failed = f"not healthy within {cfg.handover_verify_timeout_s:.0f}s"
            if failed:
                LOGGER.error(
                    "[pool %s] handover %s -> %s FAILED (%s): releasing %s, the jobs go back to %s",
                    cfg.name,
                    old_hid,
                    new.hid,
                    failed,
                    new.hid,
                    old_hid,
                )
                if new.hid not in held:
                    self._control(
                        layout, new.hid, "exit", f"handover verification failed: {failed}"
                    )
                    new.retiring = True
                handovers.pop(old_hid)
                continue
            if healthy:
                others = [r for r in old.assignments() if not (layout.run_dir(r) / "rc").exists()]
                if not others and old_hid not in held:
                    self._control(layout, old_hid, "exit", f"handover verified on {new.hid}")
                    old.retiring = True
                LOGGER.info(
                    "[pool %s] handover %s -> %s verified (%s healthy)%s",
                    cfg.name,
                    old_hid,
                    new.hid,
                    ho["jobs"],
                    "" if not others and old_hid not in held else f"; {old_hid} kept",
                )
                handovers.pop(old_hid)

    # --- preflight node check -------------------------------------------------
    @staticmethod
    def _add_pool_excludes(layout: PoolLayout, nodes: list[str]) -> list[str]:
        """Append nodes to <pool>/exclude_nodes (one per line); returns the new ones."""
        existing = set(_split_nodes(_read(layout.exclude_nodes)))
        new = [n for n in nodes if n and n not in existing]
        if new:
            with layout.exclude_nodes.open("a") as fh:
                for node in new:
                    fh.write(node + "\n")
        return new

    def _host_free_nodes(self, layout: PoolLayout, host: HostView) -> list[str]:
        """The host's nodes minus the pool's and its own exclusions."""
        nodes = _split_nodes(host.info.get("NODELIST", ""))
        excluded = set(_split_nodes(_read(layout.exclude_nodes))) | set(
            _split_nodes(_read(layout.host_dir(host.hid) / "exclude_nodes"))
        )
        return [n for n in nodes if n not in excluded]

    def _start_preflight(self, layout: PoolLayout, cfg: PoolHostConfig, host: HostView) -> str:
        """A run cloned from <pool>/preflight_template on all of the host's usable nodes,
        assigned to it directly (no queue, no job record: the monitor reads its rc)."""
        tdir = layout.root / "preflight_template"
        trid = (_read(tdir / "rid") or "").strip()
        rid = layout.allocate_run_id(pool_tag(cfg.name))
        rd = layout.run_dir(rid)
        nodes = len(self._host_free_nodes(layout, host)) or host.usable_nodes
        (rd / "step.sh").write_text(
            (_read(tdir / "step.sh") or "").replace(trid, rid), encoding="utf-8"
        )
        meta = read_env_file(tdir / "meta.env")
        meta["NODES"] = str(nodes)
        meta["LOG"] = meta.get("LOG", "").replace(trid, rid) or str(rd / "run.log")
        meta["NAME"] = f"{meta.get('NAME', 'preflight')}_{host.hid}"
        write_env_file(rd / "meta.env", meta)
        info = json.loads(_read(tdir / "run.json") or "{}")
        info.update(
            rid=rid,
            nodes=nodes,
            name=meta["NAME"],
            log=meta["LOG"],
            preflight=host.hid,
            submitted_ts=time.time(),
        )
        _atomic_write(rd / "run.json", json.dumps(info, indent=2))
        Path(meta["LOG"]).parent.mkdir(parents=True, exist_ok=True)
        self._assign(layout, rid, host.hid)
        return rid

    def _preflight(
        self,
        layout: PoolLayout,
        cfg: PoolHostConfig,
        state: dict,
        live: list[HostView],
        held: set[str],
        now: float,
    ) -> set[str]:
        """Drive the preflight node check of the live hosts; returns the hids not (yet) verified.

        Per host (state["preflight"][hid]): pending -> (GPU probe done) -> running (a benchmark
        round) -> passed | pending (nodes excluded, next round) | failed (released unless held).
        """
        if not cfg.preflight and cfg.preflight_gpus_per_node <= 0:
            return set()
        pf = state.setdefault("preflight", {})
        unverified: set[str] = set()
        for host in live:
            if host.hid not in pf and (host.assignments() or host.state[:1] == ["busy"]):
                # already working when the check first sees it (preflight switched on for a
                # running pool, or a monitor of older code): it ran fine so far; never block a
                # restart of its runs in the same allocation
                pf[host.hid] = {"status": "passed", "round": 0, "errors": 0, "why": "had runs"}
                LOGGER.info(
                    "[pool %s] preflight %s: passed (had runs before the node check)",
                    cfg.name,
                    host.hid,
                )
                continue
            st = pf.setdefault(host.hid, {"status": "pending", "round": 0, "errors": 0})
            if st["status"] == "passed":
                continue
            unverified.add(host.hid)
            if st["status"] == "failed":
                continue
            hdir = layout.host_dir(host.hid)
            # the supervisor's GPU probe: its exclusions go into the pool's list
            probed = _split_nodes(_read(hdir / "exclude_nodes"))
            for node in self._add_pool_excludes(layout, probed):
                LOGGER.warning(
                    "[pool %s] preflight %s: %s excluded (GPU probe)", cfg.name, host.hid, node
                )
            if cfg.preflight_gpus_per_node > 0 and not (hdir / "probe_done").exists():
                # a supervisor of older code never probes: do not wait for it forever
                if not host.start_ts or now - host.start_ts < PREFLIGHT_PROBE_WAIT_S:
                    continue
                if not st.get("probe_skipped"):
                    st["probe_skipped"] = True
                    LOGGER.warning(
                        "[pool %s] preflight %s: no GPU probe after %.0f s (supervisor too old?): "
                        "skipping it",
                        cfg.name,
                        host.hid,
                        now - host.start_ts,
                    )
            if not cfg.preflight or not (layout.root / "preflight_template" / "rid").exists():
                if cfg.preflight:
                    LOGGER.warning(
                        "[pool %s] preflight %s: no preflight template (enqueue the benchmark with "
                        "+job.metadata.pool_preflight_template=true): GPU probe only",
                        cfg.name,
                        host.hid,
                    )
                st["status"] = "passed"
                unverified.discard(host.hid)
                LOGGER.info("[pool %s] preflight %s: passed (GPU probe)", cfg.name, host.hid)
                continue
            if st["status"] == "pending":
                st["rid"] = self._start_preflight(layout, cfg, host)
                st["round"] = int(st.get("round", 0)) + 1
                st["status"], st["ts"] = "running", now
                LOGGER.info(
                    "[pool %s] preflight %s: round %d (%s)",
                    cfg.name,
                    host.hid,
                    st["round"],
                    st["rid"],
                )
                continue
            rd = layout.run_dir(st["rid"])
            if not (rd / "rc").exists():
                continue
            from oellm_autoexp.node_health.verdict import judge

            log = read_env_file(rd / "meta.env").get("LOG", "")
            try:
                text = Path(log).read_text(errors="replace")[-2_000_000:] if log else ""
            except OSError:
                text = ""
            verdict = judge(text, cfg.preflight_ratio)
            if verdict.status == "pass":
                st["status"] = "passed"
                unverified.discard(host.hid)
                LOGGER.info("[pool %s] preflight %s: passed (%s)", cfg.name, host.hid, verdict.why)
            elif verdict.status == "exclude" and st["round"] < cfg.preflight_max_rounds:
                new = self._add_pool_excludes(layout, verdict.exclude)
                LOGGER.warning(
                    "[pool %s] preflight %s: excluding %s (%s); next round",
                    cfg.name,
                    host.hid,
                    ", ".join(verdict.exclude),
                    verdict.why,
                )
                st["status"], st["why"] = ("pending" if new else "failed"), verdict.why
            elif verdict.status == "error" and int(st.get("errors", 0)) < 1:
                st["errors"] = int(st.get("errors", 0)) + 1
                st["status"] = "pending"
                LOGGER.warning(
                    "[pool %s] preflight %s: %s; retrying", cfg.name, host.hid, verdict.why
                )
            else:
                st["status"], st["why"] = "failed", verdict.why
            if st["status"] == "failed":
                if host.hid in held:
                    LOGGER.error(
                        "[pool %s] preflight %s FAILED (%s): held, kept without runs",
                        cfg.name,
                        host.hid,
                        st.get("why"),
                    )
                else:
                    self._control(layout, host.hid, "exit", f"preflight failed: {st.get('why')}")
                    host.retiring = True
                    LOGGER.error(
                        "[pool %s] preflight %s FAILED (%s): releasing it",
                        cfg.name,
                        host.hid,
                        st.get("why"),
                    )
        return unverified

    @staticmethod
    def _control(layout: PoolLayout, hid: str, name: str, reason: str) -> None:
        path = layout.host_dir(hid) / "control" / name
        if not path.exists():
            _atomic_write(path, f"{reason}\n")

    @staticmethod
    def _assign(layout: PoolLayout, rid: str, hid: str) -> None:
        _atomic_write(layout.run_dir(rid) / "assigned", f"{hid}\n")
        _atomic_write(layout.host_dir(hid) / "assign" / rid, f"{rid}\n")

    @staticmethod
    def _host_excludes(layout: PoolLayout, base: SlurmJobConfig) -> list[str]:
        """Nodes a host of this pool must avoid NOW: the host config's own list
        (its exclude_file re-read) plus the pool's exclude_nodes."""
        excluded = _split_nodes(_exclude_list(base.slurm)) + _split_nodes(
            _read(layout.exclude_nodes)
        )
        return list(dict.fromkeys(excluded))

    def _refresh_queued_host_excludes(
        self, loop: MonitorLoop, layout: PoolLayout, hosts: list[HostView]
    ) -> None:
        """Push exclusions found since a host was submitted to it while it
        waits.

        A successor (keep_pending) can sit in the queue for hours; a
        node that failed meanwhile -- appended to the fleet list by a
        fault scan, or to the pool's exclude_nodes -- must not end up in
        it when it starts. SLURM lets a PENDING job's ExcNodeList be
        edited (scontrol update), so the full, current list is set on
        every queued host that lacks part of it.
        """
        queued = [h for h in hosts if h.submitted and not h.info and not h.dead and not h.retiring]
        queued = [h for h in queued if h.status in (None, "PENDING")]
        if not queued or loop._slurm_client is None:
            return
        try:
            target = self._host_excludes(layout, layout.load_host_definition())
        except Exception as exc:  # pragma: no cover - never block the poll
            LOGGER.warning("[pool] could not read the exclusion lists: %s", exc)
            return
        for host in queued:
            sbatch = host.record.definition.slurm.sbatch
            current = _split_nodes(sbatch_get(sbatch, "exclude"))
            missing = [n for n in target if n not in current]
            if not missing:
                continue
            nodelist = ",".join(dict.fromkeys(current + target))
            try:
                loop._slurm_client.update_excludes(host.record.runtime.runtime_job_id, nodelist)
            except Exception as exc:  # it may have just started; next poll sees info
                LOGGER.warning(
                    "[pool] could not update the exclusions of queued host %s: %s", host.hid, exc
                )
                continue
            sbatch_set(sbatch, "exclude", nodelist)
            loop._store.upsert(host.record)
            LOGGER.info(
                "[pool %s] queued host %s (job %s): %d more node(s) excluded: %s",
                layout.root.name,
                host.hid,
                host.record.runtime.runtime_job_id,
                len(missing),
                ",".join(missing[:10]),
            )

    def _create_host(
        self, loop: MonitorLoop, layout: PoolLayout, cfg: PoolHostConfig, account: str | None = None
    ) -> str:
        from oellm_autoexp.monitor.loop import JobRecord, JobRuntime

        base = layout.load_host_definition()
        hid = layout.allocate_host_id()
        hdir = layout.host_dir(hid)
        name = f"{cfg.name}_{hid}"
        env = dict(base.slurm.env or {})
        env["AUTOEXP_POOL_DIR"] = str(layout.root)
        env["AUTOEXP_POOL_HOST"] = hid
        sbatch = copy.deepcopy(base.slurm.sbatch)
        # The new allocation avoids every node known to be bad NOW: the host
        # config's own list (its exclude_file re-read, e.g. the JUPITER fleet list
        # of a *_autoexclude_pool_host config) plus the pool's exclude_nodes
        # (pool_ctl exclude, which every pool has, autoexclude or not). The union
        # is baked into the clone and its exclude_file dropped, because rendering
        # would otherwise REPLACE the list with the file's content alone.
        excluded = self._host_excludes(layout, base)
        if excluded:
            sbatch_set(sbatch, "exclude", ",".join(excluded))
        if account:
            sbatch_set(sbatch, "account", account)
        slurm = replace(
            base.slurm,
            name=name,
            script_path=str(hdir / "host.sbatch"),
            env=env,
            sbatch=sbatch,
            exclude_file=None,
        )
        definition = replace(
            base,
            name=name,
            slurm=slurm,
            pool_dir=None,
            metadata={
                **dict(base.metadata or {}),
                "pool_role": POOL_ROLE_HOST,
                "pool_dir": str(layout.root),
                "pool_host_id": hid,
                **({"pool_host_account": account} if account else {}),
            },
        )
        record = JobRecord(
            job_id=f"{pool_tag(cfg.name)}.{hid}",
            definition=definition,
            runtime=JobRuntime(submitted=False),
        )
        loop._store.upsert(record)
        LOGGER.info(
            "[pool %s] created host %s (%s nodes%s%s)",
            cfg.name,
            hid,
            sbatch.nodes,
            f", account {account}" if account else "",
            f", {len(excluded)} node(s) excluded" if excluded else "",
        )
        return hid

    # --- requests from pool_ctl ---------------------------------------------
    def _process_requests(self, loop: MonitorLoop, layout: PoolLayout, state: dict) -> None:
        try:
            paths = sorted(p for p in layout.requests_dir.glob("*.json"))
        except OSError:
            return
        for path in paths:
            try:
                request = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            op = request.get("op")
            result = "ok"
            try:
                if op == "close":
                    state["closed"] = True
                elif op == "open":
                    state["closed"] = False
                elif op == "cancel":
                    targets = request.get("targets") or [str(request.get("target", ""))]
                    result = self._cancel_targets(loop, layout, [str(t) for t in targets])
                elif op == "adopt":
                    result = self._adopt(loop, layout, state, request)
                elif op == "drain":
                    for hid in request.get("hosts") or [
                        h.hid for h in self._hosts.get(str(layout.root), {}).values()
                    ]:
                        self._control(layout, hid, "drain", "pool_ctl drain")
                else:
                    result = f"unknown op {op!r}"
            except Exception as exc:  # pragma: no cover - never let a request kill the poll
                result = f"error: {exc}"
            LOGGER.info("[pool] request %s: %s -> %s", path.name, request, result)
            done = layout.requests_dir / "done"
            done.mkdir(exist_ok=True)
            request["result"] = result
            request["processed_ts"] = time.time()
            _atomic_write(done / path.name, json.dumps(request))
            path.unlink(missing_ok=True)

    def _cancel_targets(self, loop: MonitorLoop, layout: PoolLayout, targets: list[str]) -> str:
        matched = []
        for record in loop._store.load_all():
            if not self.is_run(record):
                continue
            if str(PoolLayout(record.definition.pool_dir).root) != str(layout.root):
                continue
            rid = record.runtime.runtime_job_id
            if {record.job_id, rid, record.definition.name} & set(targets):
                if rid:
                    self.cancel(rid)
                loop._store.mark_finished(record.job_id, "cancelled")
                matched.append(record.job_id)
        return f"cancelled {matched}" if matched else f"no active job matches {targets}"


def _import_registry() -> None:
    import oellm_autoexp.monitor.actions  # noqa: F401
    import oellm_autoexp.monitor.conditions  # noqa: F401
    import oellm_autoexp.monitor.submission  # noqa: F401
    import oellm_autoexp.config.conditions  # noqa: F401
    import oellm_autoexp.slurm_gen  # noqa: F401


__all__ = [
    "DEFAULT_RANK_PATTERN",
    "PoolHostConfig",
    "PoolLayout",
    "PoolManager",
    "RunState",
    "check_fits_host",
    "create_or_update_pool",
    "parse_slurm_time",
    "read_run_state",
    "register_pool_in_session",
    "render_step_script",
    "submit_request",
]

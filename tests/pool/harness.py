"""End-to-end harness for sub-SLURM-job pools against tests/fakeslurm.

Everything runs as on a cluster, only local: ``run_autoexp.py`` creates the pool
and enqueues jobs, ``monitor_autoexp.py`` runs the real monitor loop with the
real ``SlurmClient`` (whose sbatch/squeue/... are the fakeslurm shims on PATH),
hosts are real batch shells running ``scripts/pool/supervisor.sh``, and runs are
``scripts/sigterm_dummy_trainer.py`` under a fake torchrun agent.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from oellm_autoexp.monitor.pool import PoolLayout, read_run_state

REPO = Path(__file__).resolve().parents[2]
FAKE_BIN = REPO / "tests" / "fakeslurm" / "bin"
FAKESLURM = REPO / "tests" / "fakeslurm" / "fakeslurm.py"


class PoolEnv:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.fs = root / "fs"
        self.fs.mkdir(parents=True, exist_ok=True)
        self.state_dir = root / "monitor_state"
        self.pool_dir = root / "pool"
        self.layout = PoolLayout(self.pool_dir)
        self.env = {
            **os.environ,
            "PATH": f"{FAKE_BIN}:{os.environ.get('PATH', '')}",
            "FAKESLURM_DIR": str(self.fs),
            "POOL_TEST_ROOT": str(root),
            "PYTHONPATH": str(REPO),
            "PWD": str(REPO),
            "OUTPUT_DIR": str(root / "output"),
        }
        self.monitor: subprocess.Popen | None = None
        self.monitor_log = root / "monitor.out"

    # --- driving ------------------------------------------------------------
    def autoexp(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        proc = subprocess.run(
            [
                sys.executable,
                "scripts/run_autoexp.py",
                "--monitor-state-dir",
                str(self.state_dir),
                *args,
            ],
            cwd=REPO,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if check and proc.returncode != 0:
            raise AssertionError(
                f"run_autoexp {args} failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
            )
        return proc

    def create_pool(self, *overrides: str) -> None:
        self.autoexp(
            "--no-monitor", "--config-name", "experiments/tests/pool_local_host", *overrides
        )

    def enqueue(self, tag: str, *overrides: str, check: bool = True) -> subprocess.CompletedProcess:
        return self.autoexp(
            "--no-monitor",
            "--pool",
            str(self.pool_dir),
            "--config-name",
            "experiments/tests/pool_local_dummy",
            f"aux.tag={tag}",
            *overrides,
            check=check,
        )

    def ctl(self, *args: str) -> str:
        proc = subprocess.run(
            [sys.executable, "scripts/pool_ctl.py", str(self.pool_dir), *args],
            cwd=REPO,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        return proc.stdout

    def slurm(self, *args: str) -> str:
        proc = subprocess.run(
            [sys.executable, str(FAKESLURM), *args],
            env=self.env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return proc.stdout

    @property
    def session_dir(self) -> Path:
        return Path(self.layout.load_meta()["session_dir"])

    def start_monitor(self, poll: float = 1.0, *, hold_new_jobs: bool = False) -> None:
        env = dict(self.env)
        if hold_new_jobs:
            env["FAKESLURM_HOLD"] = "1"
        self.monitor = subprocess.Popen(
            [
                sys.executable,
                "scripts/monitor_autoexp.py",
                "--session-dir",
                str(self.session_dir),
                "--poll-interval",
                str(poll),
                "--force",
            ],
            cwd=REPO,
            env=env,
            stdout=open(self.monitor_log, "a"),
            stderr=subprocess.STDOUT,
        )

    def stop_monitor(self) -> None:
        if self.monitor is not None and self.monitor.poll() is None:
            self.monitor.terminate()
            try:
                self.monitor.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.monitor.kill()
        self.monitor = None

    def teardown(self) -> None:
        self.stop_monitor()
        self.slurm("_killall")
        time.sleep(0.5)

    # --- observing ------------------------------------------------------------
    def fake_jobs(self) -> list[dict]:
        out = []
        for path in sorted((self.fs / "jobs").glob("*/job.json")):
            try:
                out.append(json.loads(path.read_text()))
            except (OSError, json.JSONDecodeError):
                pass
        return out

    def release(self, jid: str) -> None:
        (self.fs / "jobs" / jid / "hold").unlink(missing_ok=True)

    def run_ids(self) -> list[str]:
        return sorted(p.name for p in self.layout.runs_dir.iterdir() if (p / "run.json").exists())

    def run_state(self, rid: str):
        return read_run_state(self.layout, rid)

    def runs_of(self, name: str) -> list[str]:
        out = []
        for rid in self.run_ids():
            if json.loads((self.layout.run_dir(rid) / "run.json").read_text())["name"] == name:
                out.append(rid)
        return out

    def records(self) -> list[dict]:
        out = []
        for path in sorted(self.session_dir.glob("*.job.json")):
            try:
                out.append(json.loads(path.read_text()))
            except (OSError, json.JSONDecodeError):
                pass
        return out

    def record(self, name: str) -> dict:
        matches = [r for r in self.records() if r["definition"]["name"] == name]
        assert len(matches) == 1, [r["job_id"] for r in self.records()]
        return matches[0]

    def run_log(self, rid: str) -> str:
        log = json.loads((self.layout.run_dir(rid) / "run.json").read_text())["log"]
        try:
            return Path(log).read_text()
        except OSError:
            return ""

    def debug(self) -> str:
        parts = [f"--- monitor log ({self.monitor_log}) tail:"]
        try:
            lines = [
                ln
                for ln in self.monitor_log.read_text().splitlines()
                if "Monitor Polling" not in ln
            ]
            parts += lines[-60:]
        except OSError:
            pass
        for log in sorted(self.pool_dir.glob("slurm-*.log")):
            if log.is_symlink():
                continue
            parts.append(f"--- {log.name}:")
            parts.append(log.read_text()[-3000:])
        try:
            parts.append(self.ctl("status"))
        except AssertionError as exc:  # pragma: no cover
            parts.append(str(exc))
        return "\n".join(parts)

    def wait_for(self, predicate, timeout: float = 60.0, what: str = "condition"):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                value = predicate()
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                value = None
            if value:
                return value
            if self.monitor is not None and self.monitor.poll() is not None:
                raise AssertionError(
                    f"monitor exited ({self.monitor.returncode}) while waiting for {what}\n{self.debug()}"
                )
            time.sleep(0.25)
        raise AssertionError(f"timed out after {timeout}s waiting for {what}\n{self.debug()}")

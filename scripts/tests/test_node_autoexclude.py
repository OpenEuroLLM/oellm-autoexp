#!/usr/bin/env python3
"""End-to-end cluster test for the node auto-exclusion flow.

IMPORTANT: run this from a SLURM login node (needs `sbatch`/`squeue`/`scontrol`),
NOT inside a container. It submits real (short, bash-only) jobs on the cluster.

What it verifies
----------------
Using `config/experiments/tests/leonardo_autoexclude_chain.yaml` (BashBackend, no
trainer) it pre-queues a 3-deep dependency chain (chain_repeat=3) and drives the
whole node-exclusion mechanism end to end:

  1. append      - when a job logs "Communication connection failure", the node
                   it ran on is appended to $LEONARDO_EXCLUDE_NODES
                   (AppendToFileAction).
  2. propagate   - the monitor runs `scontrol update .. ExcNodeList=` on the
                   still-PENDING chained siblings (UpdateChainExcludesAction);
                   verified both in the monitor log and via `scontrol show job`.
  3. resolve     - a fresh `--dry-run` render now bakes `#SBATCH --exclude=<node>`
                   into the script (oc.exclude_nodes resolver) -> future launches
                   avoid the bad node.

The harness starts `run_autoexp.py --chain` in the background (it submits the
chain and then runs the monitor loop in the foreground), polls SLURM itself, and
scancels everything on exit.

Usage
-----
    python scripts/tests/test_node_autoexclude.py
    python scripts/tests/test_node_autoexclude.py --timeout 900 --keep-exclude-file
    python scripts/tests/test_node_autoexclude.py --config-name experiments/tests/leonardo_autoexclude_chain
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_NAME = "experiments/tests/leonardo_autoexclude_chain"
JOB_NAME_SUBSTR = "autoexclude"  # slurm job-name is autoexclude_<timestring>
SBATCH_PATH_RE = re.compile(r"(/\S+\.sbatch)")
EXCLUDE_DIRECTIVE_RE = re.compile(r"^#SBATCH --exclude=(\S+)", re.MULTILINE)
# `scontrol show job` reports the exclusion list as ExcNodeList=... on most SLURM
# builds (older/newer ones may say ExcludeNodes); accept either.
SCONTROL_EXCLUDE_RE = re.compile(r"(?:ExcNodeList|ExcludeNodes)=(\S+)")
PROPAGATE_LOG_RE = re.compile(r"UpdateChainExcludes: set ExcNodeList=(\S+) on (\d+) pending")
# A real SLURM short node name: letters then alnum/dash (e.g. lrdn0387). Must NOT
# be an unexpanded shell token like "$SLURMD_NODENAME" - guards the class of bug
# where the monitor records literal command text from a stale/misresolved file.
NODE_NAME_RE = re.compile(r"^[a-z][a-z0-9-]+$")
PROPAGATE_ERROR_RE = re.compile(
    r"UpdateChainExcludes: failed to update|Invalid node name|not supported"
)


class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def log(msg: str, color: str = "") -> None:
    print(f"{color}[{time.strftime('%H:%M:%S')}] {msg}{Colors.END}", flush=True)


def log_success(msg: str) -> None:
    log(f"✓ {msg}", Colors.GREEN)


def log_error(msg: str) -> None:
    log(f"✗ {msg}", Colors.RED)


def log_info(msg: str) -> None:
    log(f"ℹ {msg}", Colors.BLUE)


def log_warning(msg: str) -> None:
    log(f"⚠ {msg}", Colors.YELLOW)


def run(cmd: list[str], *, env: dict[str, str] | None = None, timeout: float = 60.0):
    """Run a command, tolerating transient fork failures.

    Shared SLURM login nodes enforce a per-user process/thread limit
    (RLIMIT_NPROC); under contention `fork()` intermittently fails with
    BlockingIOError/OSError "Resource temporarily unavailable". A single failed
    `squeue` poll must not abort a 20-minute test, so retry a few times with
    backoff and only then surface a non-zero result.
    """
    last_exc: Exception | None = None
    for attempt in range(5):
        try:
            return subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                env=env,
                timeout=timeout,
                check=False,
            )
        except (BlockingIOError, OSError) as exc:
            last_exc = exc
            time.sleep(2.0 * (attempt + 1))
    log_warning(f"Command failed to spawn after retries ({cmd[0]}): {last_exc}")
    return subprocess.CompletedProcess(cmd, returncode=255, stdout="", stderr=str(last_exc))


def check_environment() -> None:
    if run(["which", "sbatch"]).returncode != 0:
        log_error("`sbatch` not found - run this on a SLURM login node, not in a container.")
        sys.exit(2)
    if os.environ.get("SINGULARITY_NAME") or os.environ.get("APPTAINER_NAME"):
        log_error("Detected a container environment; run from the login node instead.")
        sys.exit(2)
    if not (REPO_ROOT / "pyproject.toml").exists():
        log_error(f"Cannot find repo root at {REPO_ROOT}")
        sys.exit(2)
    os.chdir(REPO_ROOT)
    log_info(f"Working directory: {REPO_ROOT}")


def squeue_me() -> dict[str, tuple[str, str]]:
    """Return {job_id: (state, name)} for the current user's jobs."""
    user = os.environ.get("USER", "")
    proc = run(["squeue", "-u", user, "-h", "-o", "%i|%T|%j"])
    jobs: dict[str, tuple[str, str]] = {}
    if proc.returncode != 0:
        return jobs
    for line in proc.stdout.strip().splitlines():
        parts = line.split("|")
        if len(parts) >= 3:
            jobs[parts[0].strip()] = (parts[1].strip(), parts[2].strip())
    return jobs


def our_jobs() -> dict[str, tuple[str, str]]:
    return {jid: v for jid, v in squeue_me().items() if JOB_NAME_SUBSTR in v[1]}


def wait_until(predicate, *, timeout: float, interval: float, desc: str):
    """Poll ``predicate`` until it returns a truthy value or timeout.

    Returns the value (or None on timeout).
    """
    log_info(f"Waiting (<= {int(timeout)}s) for: {desc}")
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    return None


def read_nodes(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def scontrol_excludes(job_id: str) -> str:
    proc = run(["scontrol", "show", "job", job_id])
    if proc.returncode != 0:
        return ""
    match = SCONTROL_EXCLUDE_RE.search(proc.stdout)
    return match.group(1) if match else ""


def resolver_bakes_exclude(env: dict[str, str]) -> tuple[bool, str]:
    """Run a `--dry-run` render and check the generated script carries an
    `#SBATCH --exclude=` directive (proves the oc.exclude_nodes resolver picked
    up the freshly-appended node).

    Returns (ok, detail).
    """
    proc = run(
        [
            sys.executable,
            "scripts/run_autoexp.py",
            "--config-name",
            DEFAULT_CONFIG_NAME,
            "--dry-run",
        ],
        env=env,
        timeout=300,
    )
    output = proc.stdout + proc.stderr
    paths = SBATCH_PATH_RE.findall(output)
    for script in paths:
        try:
            text = Path(script).read_text()
        except OSError:
            continue
        m = EXCLUDE_DIRECTIVE_RE.search(text)
        if m:
            return True, f"{Path(script).name}: #SBATCH --exclude={m.group(1)}"
    return False, "no '#SBATCH --exclude=' directive found in rendered script(s)"


def scancel_all(job_ids) -> None:
    ids = [jid.split("_")[0] for jid in job_ids]  # base ids cover array/chain tasks
    for jid in sorted(set(ids)):
        run(["scancel", jid])
    if job_ids:
        log_info(f"scancel issued for: {sorted(set(ids))}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument(
        "--exclude-file",
        type=Path,
        default=REPO_ROOT / "output" / "e2e_autoexclude" / "leonardo_exclude_nodes.txt",
        help="Node-exclusion file to use for this run (reset empty at start).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Per-step wait timeout for action detection (seconds). Monitor polls ~60s.",
    )
    parser.add_argument(
        "--queue-timeout",
        type=float,
        default=1800.0,
        help="How long to wait for a chained job to start RUNNING (queue wait).",
    )
    parser.add_argument("--poll-interval", type=float, default=10.0)
    parser.add_argument(
        "--keep-exclude-file",
        action="store_true",
        help="Do not delete the exclusion file on exit (inspect what was recorded).",
    )
    args = parser.parse_args()

    log(f"\n{'=' * 64}", Colors.BOLD)
    log("NODE AUTO-EXCLUSION END-TO-END TEST", Colors.BOLD)
    log(f"{'=' * 64}\n", Colors.BOLD)

    check_environment()

    exclude_file = args.exclude_file.resolve()
    exclude_file.parent.mkdir(parents=True, exist_ok=True)
    exclude_file.write_text("")  # start from a clean, empty list
    log_info(f"Exclusion file (LEONARDO_EXCLUDE_NODES): {exclude_file}")

    state_dir = REPO_ROOT / "output" / "e2e_autoexclude" / f"monitor_state_{int(time.time())}"
    monitor_log = REPO_ROOT / "output" / "e2e_autoexclude" / "monitor.log"
    monitor_log.parent.mkdir(parents=True, exist_ok=True)

    env = {**os.environ, "LEONARDO_EXCLUDE_NODES": str(exclude_file), "PYTHONUNBUFFERED": "1"}

    launch_cmd = [
        sys.executable,
        "-u",
        "scripts/run_autoexp.py",
        "--config-name",
        args.config_name,
        "--chain",
        "--monitor-state-dir",
        str(state_dir),
    ]
    log_info(f"Launching (background): {' '.join(launch_cmd)}")
    log_info(f"Monitor log: {monitor_log}")

    results: dict[str, bool] = {}
    discovered: set[str] = set()
    monitor_fp = monitor_log.open("w", encoding="utf-8")
    monitor_proc = subprocess.Popen(
        launch_cmd, stdout=monitor_fp, stderr=subprocess.STDOUT, text=True, env=env
    )

    try:
        # 1) The dependency chain shows up in the queue (>=2 tasks; r1 runs, rest pending).
        def _chain_visible():
            jobs = our_jobs()
            discovered.update(jobs)
            return jobs if len(jobs) >= 2 else None

        jobs = wait_until(
            _chain_visible,
            timeout=args.timeout,
            interval=args.poll_interval,
            desc="dependency chain to appear in squeue",
        )
        if not jobs:
            log_error("Chain never appeared in squeue (submission failed?). See monitor log.")
            results["chain_submitted"] = False
            return _finish(results, monitor_proc, monitor_fp, discovered, exclude_file, args)
        results["chain_submitted"] = True
        pending = [jid for jid, (st, _) in jobs.items() if st == "PENDING"]
        log_success(f"Chain visible: { {jid: st for jid, (st, _) in jobs.items()} }")
        log_info(f"Pending siblings at start: {sorted(pending)}")

        # 1b) Wait for the head job to actually start RUNNING before arming the
        # action-detection timers - otherwise a long queue wait eats their budget.
        running = wait_until(
            lambda: next((jid for jid, (st, _) in our_jobs().items() if st == "RUNNING"), None),
            timeout=args.queue_timeout,
            interval=args.poll_interval,
            desc="a chained job to reach RUNNING (queue wait)",
        )
        if not running:
            log_error("No chained job started RUNNING within the queue timeout.")
            results["job_running"] = False
            return _finish(results, monitor_proc, monitor_fp, discovered, exclude_file, args)
        results["job_running"] = True
        log_success(f"Job {running} is RUNNING - arming action-detection checks.")

        # 2) A node gets appended to the exclusion file (LogEvent -> AppendToFileAction).
        node = wait_until(
            lambda: (read_nodes(exclude_file) or [None])[0],
            timeout=args.timeout,
            interval=args.poll_interval,
            desc="a failing node to be appended to the exclusion file",
        )
        # The appended value must be a REAL node name, not unexpanded command
        # text like "$SLURMD_NODENAME" (that would mean the monitor read the wrong
        # file). Reject anything that does not look like a short SLURM node name.
        node_valid = bool(node) and bool(NODE_NAME_RE.match(node))
        results["node_appended"] = node_valid
        if node_valid:
            log_success(f"Exclusion file now lists node(s): {read_nodes(exclude_file)}")
        elif node:
            log_error(f"Appended value {node!r} is not a valid node name (unexpanded?).")
            node = None  # don't use garbage for the downstream scontrol check
        else:
            log_error("No node was appended - the append action did not fire.")

        # 3) The monitor logged the chain-propagation action (UpdateChainExcludesAction).
        def _propagated():
            try:
                text = monitor_log.read_text(errors="replace")
            except OSError:
                return None
            m = PROPAGATE_LOG_RE.search(text)
            return m if m else None

        prop = wait_until(
            _propagated,
            timeout=args.timeout,
            interval=args.poll_interval,
            desc="monitor to run UpdateChainExcludes",
        )
        results["propagate_logged"] = bool(prop)
        if prop:
            log_success(
                f"Monitor propagated excludes={prop.group(1)} to {prop.group(2)} pending job(s)"
            )
        else:
            log_error("Monitor never logged 'UpdateChainExcludes' - propagate action didn't run.")

        # 3b) The scontrol updates must not have errored (wrong param name, invalid
        # node, unsupported) - these are the failure modes seen on Leonardo.
        monitor_text = monitor_log.read_text(errors="replace") if monitor_log.exists() else ""
        propagate_errors = PROPAGATE_ERROR_RE.findall(monitor_text)
        results["no_propagate_errors"] = not propagate_errors
        if propagate_errors:
            log_error(f"Monitor logged scontrol propagate errors: {set(propagate_errors)}")
        else:
            log_success("No scontrol propagate errors in monitor log.")

        # 4) Best-effort: a still-pending sibling actually carries the node in SLURM.
        if node:

            def _sibling_has_exclude():
                for jid, (st, _) in our_jobs().items():
                    if st == "PENDING" and node in scontrol_excludes(jid):
                        return jid
                return None

            sib = wait_until(
                _sibling_has_exclude,
                timeout=min(args.timeout, 180.0),
                interval=args.poll_interval,
                desc=f"a PENDING sibling to carry ExcludeNodes={node}",
            )
            if sib:
                log_success(f"scontrol confirms job {sib} excludes {node}")
                results["scontrol_updated"] = True
            else:
                log_warning(
                    "No PENDING sibling carries the node (they may have already advanced "
                    "or the SLURM field name differs). Non-fatal."
                )

        # 5) The resolver bakes the node into freshly generated scripts (future runs).
        if node:
            ok, detail = resolver_bakes_exclude(env)
            results["resolver_bakes"] = ok
            (log_success if ok else log_error)(f"Resolver dry-run: {detail}")

        return _finish(results, monitor_proc, monitor_fp, discovered, exclude_file, args)
    except KeyboardInterrupt:  # pragma: no cover - interactive
        log_warning("Interrupted - cleaning up...")
        return _finish(results, monitor_proc, monitor_fp, discovered, exclude_file, args)


def _finish(results, monitor_proc, monitor_fp, discovered, exclude_file, args) -> int:
    # Stop the monitor and cancel every job we saw.
    log_info("Cleaning up (stopping monitor, scancelling jobs)...")
    if monitor_proc.poll() is None:
        monitor_proc.terminate()
        try:
            monitor_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            monitor_proc.kill()
    monitor_fp.close()
    discovered.update(our_jobs())
    scancel_all(discovered)
    if not args.keep_exclude_file and exclude_file.exists():
        exclude_file.unlink()

    log(f"\n{'=' * 64}", Colors.BOLD)
    log("SUMMARY", Colors.BOLD)
    log(f"{'=' * 64}", Colors.BOLD)
    # Deterministic checks that must pass; scontrol_updated is best-effort.
    required = [
        "chain_submitted",
        "job_running",
        "node_appended",
        "propagate_logged",
        "no_propagate_errors",
        "resolver_bakes",
    ]
    for key, ok in results.items():
        tag = "(required)" if key in required else "(best-effort)"
        (log_success if ok else log_error)(f"{key} {tag}: {'PASS' if ok else 'FAIL'}")

    all_required_pass = all(results.get(k) for k in required)
    if all_required_pass:
        log(f"\n{'=' * 64}", Colors.GREEN + Colors.BOLD)
        log("END-TO-END TEST PASSED ✓", Colors.GREEN + Colors.BOLD)
        log(f"{'=' * 64}\n", Colors.GREEN + Colors.BOLD)
        return 0
    log(f"\n{'=' * 64}", Colors.RED + Colors.BOLD)
    log("END-TO-END TEST FAILED ✗", Colors.RED + Colors.BOLD)
    log(f"{'=' * 64}\n", Colors.RED + Colors.BOLD)
    return 1


if __name__ == "__main__":
    sys.exit(main())

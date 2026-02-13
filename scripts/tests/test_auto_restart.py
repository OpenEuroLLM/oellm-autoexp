#!/usr/bin/env python3
"""Test script for automatic restart functionality.

IMPORTANT: This script must be run from the SLURM login node, NOT from inside a container.

Architecture Overview:
  The oellm-autoexp system separates concerns between the login node and compute nodes:

  Login Node (where THIS script runs):
    - This test script (test_auto_restart.py)
    - Calls plan_autoexp.py to render manifests/scripts and submit_autoexp.py to submit/monitor
    - submit_autoexp.py generates sbatch scripts (from the manifest) and submits them via `sbatch`
    - The `sbatch` command is available on the login node, NOT inside containers

  Compute Nodes (where training runs):
    - sbatch scripts execute on compute nodes
    - Scripts load required modules (CUDA, NCCL, etc.)
    - Scripts use `srun ... singularity exec ...` to launch containers
    - Megatron and training code run INSIDE the Singularity container
    - Container path: $CONTAINER_CACHE_DIR/MegatronTraining_*.sif

  Why this split?
    - sbatch must be called from outside containers (SLURM client tools)
    - Megatron needs GPU drivers, CUDA, NCCL from inside containers (reproducible env)
    - This approach mirrors the architecture described in SPEC.md and README.md

  Schema-only validation mode:
    - This allows oellm-autoexp to validate configs WITHOUT importing Megatron-LM
    - Uses pre-generated config_schema.py instead of megatron_args.py
    - The full Megatron parser validation happens inside the container on compute nodes
    - This avoids the "sbatch outside container, megatron inside container" conflict

Test Workflow:
  1. Generate the plan manifest (`plan_autoexp.py`) and submit/monitor via `submit_autoexp.py`
  2. Simulate various failure scenarios (cancel, hang, OOM, NCCL errors)
  3. Verify that the monitoring system detects failures
  4. Verify that restart bindings fire correctly (restart on transient errors, not on OOM)
  5. Verify that retry limits are respected

Prerequisites:
  - Must be run on a SLURM login node with sbatch access
  - Container image must be available at $CONTAINER_CACHE_DIR
  - oellm-autoexp must be installed: pip install -e .
  - Must have access to the config files in config/

Usage:
    # Run from the repo root
    python scripts/test_auto_restart.py --scenario scancel
    python scripts/test_auto_restart.py --scenario hang
    python scripts/test_auto_restart.py --scenario oom
    python scripts/test_auto_restart.py --scenario all
    python scripts/test_auto_restart.py --scenario scancel --iterations 3
"""

import argparse
import json
import os
import subprocess
import time
import sys
from pathlib import Path
import threading

from oellm_autoexp.utils.run import run_with_tee

# Global list to track background monitor processes
_monitor_processes: list[subprocess.Popen] = []  # noqa


class Colors:
    """ANSI color codes for terminal output."""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def log(msg: str, color: str = "") -> None:
    """Print a log message with optional color."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"{color}[{timestamp}] {msg}{Colors.END}")


def log_success(msg: str) -> None:
    log(f"✓ {msg}", Colors.GREEN)


def log_error(msg: str) -> None:
    log(f"✗ {msg}", Colors.RED)


def log_info(msg: str) -> None:
    log(f"ℹ {msg}", Colors.BLUE)


def log_warning(msg: str) -> None:
    log(f"⚠ {msg}", Colors.YELLOW)


def cleanup_monitors() -> None:
    """Kill all background monitoring processes."""
    for proc in _monitor_processes:
        try:
            if proc.poll() is None:  # Still running
                log_info(f"Terminating monitor process {proc.pid}...")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    log_warning(f"Monitor {proc.pid} didn't terminate, killing...")
                    proc.kill()
        except Exception as e:
            log_warning(f"Error cleaning up monitor {proc.pid}: {e}")
    _monitor_processes.clear()


def check_environment() -> None:
    """Check that we're running in the correct environment.

    This script must run on the SLURM login node (not inside a container) because:
    - sbatch must be called from outside containers
    - The sbatch script will launch containers on compute nodes
    """
    # Check if sbatch is available
    result = run_with_tee(["which", "sbatch"], capture_output=True, text=True)
    if result.returncode != 0:
        log_error("sbatch command not found!")
        log_error("This script must be run on a SLURM login node, not inside a container.")
        sys.exit(1)

    # Warn if SINGULARITY_NAME or APPTAINER_NAME is set (indicates we're inside a container)
    if os.environ.get("SINGULARITY_NAME") or os.environ.get("APPTAINER_NAME"):
        log_error("Detected running inside a container (SINGULARITY_NAME or APPTAINER_NAME set)!")
        log_error("This script must be run from the SLURM login node, NOT from inside a container.")
        sys.exit(1)

    # Check if oellm-autoexp is importable
    try:
        import oellm_autoexp  # noqa: F401
    except ImportError:
        log_error("oellm_autoexp package not found!")
        log_error("Please install: pip install -e .")
        sys.exit(1)

    # Check if we're in the repo root
    repo_root = Path(__file__).parent.parent.parent
    if not (repo_root / "pyproject.toml").exists():
        log_error("Cannot find pyproject.toml in expected location!")
        log_error(f"Expected repo root: {repo_root}")
        log_error("Please run this script from the oellm-autoexp repository root.")
        sys.exit(1)

    # Change to repo root for consistent paths
    os.chdir(repo_root)
    log_info(f"Working directory: {repo_root}")

    # Set required SLURM environment variables if not already set
    # These are needed by the base config but may not be set in test environment
    if "SLURM_ACCOUNT" not in os.environ:
        os.environ["SLURM_ACCOUNT"] = os.environ.get("USER", "test_user")
        log_info(f"Set SLURM_ACCOUNT={os.environ['SLURM_ACCOUNT']}")
    if "SLURM_PARTITION" not in os.environ:
        os.environ["SLURM_PARTITION"] = "batch"
        log_info(f"Set SLURM_PARTITION={os.environ['SLURM_PARTITION']}")
    if "SLURM_QOS" not in os.environ:
        os.environ["SLURM_QOS"] = "normal"
        log_info(f"Set SLURM_QOS={os.environ['SLURM_QOS']}")

    # Clean up any stale state from previous runs
    state_dirs = [
        Path("output/.oellm-autoexp"),
        Path(".oellm-autoexp"),
    ]
    for state_dir in state_dirs:
        state_file = state_dir / "state.json"
        if state_file.exists():
            log_info(f"Removing stale state file: {state_file}")
            state_file.unlink()
            # Also remove the directory if it's empty
            try:
                state_dir.rmdir()
            except OSError:
                pass  # Directory not empty, that's fine

    log_success("Environment checks passed")


def run_command(cmd: list[str], capture: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    log_info(f"Running: {' '.join(cmd)}")
    if capture:
        return run_with_tee(cmd, capture_output=True, text=True, check=False)
    else:
        return run_with_tee(cmd, check=False)


def wait_for_job_state(
    job_id: str, expected_state: str, timeout: int = 120, poll_interval: int = 30
) -> bool:
    """Wait for a job to reach a specific SLURM state."""
    log_info(f"Waiting for job {job_id} to reach state {expected_state}...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        result = run_command(["squeue", "-j", str(job_id), "-h", "-o", "%T"])
        if result.returncode == 0 and result.stdout.strip():
            current_state = result.stdout.strip()
            if current_state == expected_state:
                log_success(f"Job {job_id} reached state {expected_state}")
                return True
            log_info(f"Current state: {current_state}")
        time.sleep(poll_interval)

    log_error(f"Timeout waiting for job {job_id} to reach {expected_state}")
    return False


def get_job_state(job_id: str) -> str | None:
    """Get the current SLURM state of a job."""
    result = run_command(["squeue", "-j", str(job_id), "-h", "-o", "%T"])
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    return None


def submit_test_job(
    micro_batch_size: int = 8,
    train_iters: int = 100,
    array_mode: bool = False,
    overrides: list[str] = [],
) -> int | None:
    """Submit a test job and return the job ID.

    This function runs on the login node and invokes plan/submit helpers:
    1. Generates sbatch scripts inside the manifest via `plan_autoexp.py`
    2. Submits the scripts via `submit_autoexp.py` (calls sbatch from login node)
    3. The sbatch scripts run on compute nodes and launch Megatron in containers
    4. Starts monitoring in the background to enable auto-restart

    Args:
        micro_batch_size: Batch size (8 works, 16 causes OOM)
        train_iters: Number of training iterations
        array_mode: If True, use SLURM array job submission; otherwise use single job mode

    Returns:
        Job ID if successful, None otherwise
    """
    log_info(f"Submitting test job (micro_bs={micro_batch_size}, iters={train_iters})...")

    # Ensure logs directory exists
    Path("logs").mkdir(parents=True, exist_ok=True)

    overrides = [
        f"backend.megatron.micro_batch_size={micro_batch_size}",
        f"backend.megatron.train_iters={train_iters}",
        "slurm.sbatch.time=00:30:00",
        f"job.name=restart_test_mbs{micro_batch_size}",
        "slurm.log_dir=logs",
        f"slurm.array={str(array_mode).lower()}",
    ] + overrides

    manifest_path = Path("logs") / f"plan_mbs{micro_batch_size}.json"
    plan_cmd = [
        sys.executable,
        "scripts/plan_autoexp.py",
        "--config-ref",
        "experiments/megatron_with_auto_restart",
        "-C",
        "config",
        "--manifest",
        str(manifest_path),
    ] + overrides

    log_info(f"Generating plan manifest: {' '.join(plan_cmd)}")
    plan_result = run_command(plan_cmd)
    if plan_result.returncode != 0:
        log_error("Plan generation failed")
        return None

    try:
        manifest_json = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log_warning(f"Unable to read manifest for monitor config: {exc}")
    else:
        monitor_spec = manifest_json.get("monitor", {})
        log_info("Monitor component spec:")
        log_info(f"  class: {monitor_spec.get('class_name')} ({monitor_spec.get('module')})")
        cfg_text = json.dumps(monitor_spec.get("config", {}), indent=2)
        for line in cfg_text.splitlines():
            log_info(f"    {line}")

    submit_cmd = [
        sys.executable,
        "-u",
        "scripts/submit_autoexp.py",
        "--manifest",
        str(manifest_path),
        "--verbose",
    ]

    # Start the monitoring process in background (this will submit AND monitor)
    log_info(f"Starting job submission and monitoring in background: {' '.join(submit_cmd)}")
    monitor_log = Path(f"logs/monitor_test_mbs{micro_batch_size}.log")
    monitor_proc = subprocess.Popen(
        submit_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # Line buffered
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    job_ready = threading.Event()
    job_info: dict[str, str | None] = {"job_id": None}

    def _drain_monitor_output() -> None:
        assert monitor_proc.stdout is not None  # for mypy
        monitor_log.parent.mkdir(parents=True, exist_ok=True)
        with monitor_log.open("w", encoding="utf-8") as log_fp:
            for line in monitor_proc.stdout:
                log_fp.write(line)
                log_fp.flush()
                log_info(f"Monitor output: {line.strip()}")
                if job_info["job_id"] is None and "submitted" in line and "-> job" in line:
                    parts = line.split("-> job")
                    if len(parts) > 1:
                        job_id_str = parts[1].split("->")[0].strip()
                        job_info["job_id"] = job_id_str
                        job_ready.set()

    reader_thread = threading.Thread(
        target=_drain_monitor_output, name="monitor-output", daemon=True
    )
    reader_thread.start()

    if not job_ready.wait(timeout=120):
        log_error("Timed out waiting for job submission output")
        monitor_proc.terminate()
        return None

    job_id = job_info["job_id"]
    if job_id is None:
        log_error("Failed to parse job ID from output")
        monitor_proc.terminate()
        return None

    # Store the monitor PID so we can clean it up later
    _monitor_processes.append(monitor_proc)
    log_info(f"Monitoring running (PID: {monitor_proc.pid}, log: {monitor_log})")

    return job_id


def inject_error_pattern(job_id: str, pattern: str, description: str) -> bool:
    """Inject an error pattern into a job's log file.

    Args:
        job_id: SLURM job ID
        pattern: Error message to inject
        description: Human-readable description

    Returns:
        True if successful, False otherwise
    """
    log_info(f"Injecting {description} into job {job_id} log...")

    # Find the log file - try multiple patterns
    # Pattern 1: logs/restart_test_mbs*.out (based on job name)
    # Pattern 2: logs/*{job_id}*.out (if job ID is in filename)
    # Pattern 3: output/{job_id}.out (fallback to SLURM default)
    patterns = [
        "logs/restart_test_*.out",
        f"logs/*{job_id}*.out",
        f"output/{job_id}.out",
        f"logs/{job_id}.out",
    ]

    log_file = None
    for log_pattern in patterns:
        result = run_command(["bash", "-c", f"ls -t {log_pattern} 2>/dev/null | head -1"])
        if result.returncode == 0 and result.stdout.strip():
            log_file = result.stdout.strip()
            log_info(f"Found log file: {log_file}")
            break

    if not log_file:
        log_error(f"Could not find log file for job {job_id}")
        log_error(f"Tried patterns: {', '.join(patterns)}")
        return False

    # Append error pattern
    try:
        with open(log_file, "a") as f:
            f.write(f"\n{pattern}\n")
        log_success(f"Injected {description}")
        return True
    except Exception as e:
        log_error(f"Failed to inject pattern: {e}")
        return False


def cancel_job(job_id: str, array_id: int | None = None) -> bool:
    """Cancel a SLURM job."""
    job_id_full = str(job_id) + (("_" + str(array_id)) if array_id is not None else "")
    log_info(f"Cancelling job {job_id_full}...")
    result = run_command(["scancel", job_id_full])
    if result.returncode == 0:
        log_success(f"Cancelled job {job_id_full}")
        return True
    else:
        log_error(f"Failed to cancel job {job_id_full}")
        return False


def check_for_restart(original_job_id: str, timeout: int = 200) -> int | None:
    """Check if a job was restarted and return new job ID.

    Args:
        original_job_id: Original job ID
        timeout: Max time to wait for restart (seconds)

    Returns:
        New job ID if restarted, None if not restarted
    """
    log_info(f"Checking for restart of job {original_job_id}...")
    start_time = time.time()

    # Wait a bit for the monitor to detect and restart
    time.sleep(10)

    while time.time() - start_time < timeout:
        # Get all jobs for current user
        result = run_command(["squeue", "-u", os.environ["USER"], "-h", "-o", "%i %j"])

        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            for line in lines:
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    job_id_str = parts[0]
                    job_name = parts[1]

                    try:
                        job_id = job_id_str
                        # Look for a job with the same name but different ID
                        if job_id != original_job_id and "restart_test" in job_name:
                            log_success(f"Found restarted job: {job_id}")
                            return job_id
                    except ValueError:
                        continue

        time.sleep(10)  # Check every 10 seconds for restart

    log_warning(f"No restart detected within {timeout}s")
    return None


def test_scenario_scancel(
    iterations: int = 2, array_mode: bool = False, overrides: list[str] = []
) -> bool:
    """Test restart on manual scancel.

    Args:
        iterations: Number of cancel/restart cycles to test

    Returns:
        True if test passed, False otherwise
    """
    log(f"\n{'=' * 60}", Colors.BOLD)
    log("TEST: Manual scancel (should restart)", Colors.BOLD)
    log(f"{'=' * 60}\n", Colors.BOLD)

    # Submit a job that will run for a while
    job_id = submit_test_job(
        micro_batch_size=8, train_iters=1000, array_mode=array_mode, overrides=overrides
    )
    if job_id is None:
        return False

    # Wait for job to start running (30s intervals for initial start, as log file may not exist yet)
    if not wait_for_job_state(job_id, "RUNNING", timeout=600, poll_interval=30):
        log_error("Job never started running")
        cancel_job(job_id)
        return False

    for i in range(iterations):
        log_info(f"\n--- Iteration {i + 1}/{iterations} ---")

        # Cancel the job
        if not cancel_job(job_id):
            return False

        # Check for restart (10s intervals, 20 tries = 200s)
        new_job_id = check_for_restart(job_id, timeout=200)

        if new_job_id is None:
            log_error(f"Iteration {i + 1}: No restart detected!")
            return False

        log_success(f"Iteration {i + 1}: Job restarted ({job_id} -> {new_job_id})")

        # Wait for new job to start (10s intervals for restart checks)
        if not wait_for_job_state(new_job_id, "RUNNING", timeout=200, poll_interval=10):
            log_error("Restarted job never started running")
            cancel_job(new_job_id)
            return False

        job_id = new_job_id

    # Clean up
    cancel_job(job_id)
    cleanup_monitors()
    log_success("Test passed: scancel restart works!")
    return True


def test_scenario_hang(array_mode: bool = False, overrides: list[str] = []) -> bool:
    """Test restart on CUDA hang detection."""
    log(f"\n{'=' * 60}", Colors.BOLD)
    log("TEST: CUDA hang (should restart)", Colors.BOLD)
    log(f"{'=' * 60}\n", Colors.BOLD)

    job_id = submit_test_job(
        micro_batch_size=8, train_iters=1000, array_mode=array_mode, overrides=overrides
    )
    if job_id is None:
        return False

    # Wait for job to start (30s intervals for initial start)
    if not wait_for_job_state(job_id, "RUNNING", timeout=600, poll_interval=30):
        cancel_job(job_id)
        return False

    # Let it run a bit
    time.sleep(30)

    # Inject hang error
    if not inject_error_pattern(
        job_id,
        "[default0]:CUDA device-side assert detected in kernel execution",
        "CUDA hang pattern",
    ):
        cancel_job(job_id)
        return False

    # Check for restart
    new_job_id = check_for_restart(job_id, timeout=180)

    # Clean up
    if new_job_id:
        cancel_job(new_job_id)
    else:
        cancel_job(job_id)

    cleanup_monitors()

    if new_job_id is None:
        log_error("No restart detected after CUDA hang!")
        return False

    log_success("Test passed: CUDA hang restart works!")
    return True


def test_scenario_nccl_error(array_mode: bool = False, overrides: list[str] = []) -> bool:
    """Test restart on NCCL error detection."""
    log(f"\n{'=' * 60}", Colors.BOLD)
    log("TEST: NCCL error (should restart)", Colors.BOLD)
    log(f"{'=' * 60}\n", Colors.BOLD)

    job_id = submit_test_job(
        micro_batch_size=8, train_iters=1000, array_mode=array_mode, overrides=overrides
    )
    if job_id is None:
        return False

    # Wait for initial start (30s intervals)
    if not wait_for_job_state(job_id, "RUNNING", timeout=600, poll_interval=30):
        cancel_job(job_id)
        return False

    time.sleep(30)

    # Inject NCCL error
    if not inject_error_pattern(
        job_id,
        "[default0]:NCCL ERROR Remote process has terminated or communication failure",
        "NCCL error pattern",
    ):
        cancel_job(job_id)
        return False

    new_job_id = check_for_restart(job_id, timeout=180)

    if new_job_id:
        cancel_job(new_job_id)
    else:
        cancel_job(job_id)

    cleanup_monitors()

    if new_job_id is None:
        log_error("No restart detected after NCCL error!")
        return False

    log_success("Test passed: NCCL error restart works!")
    return True


def test_scenario_oom(array_mode: bool = False, overrides: list[str] = []) -> bool:
    """Test NO restart on OOM (excluded error type).

    This submits a job with micro_batch_size=16 which should OOM.
    """
    log(f"\n{'=' * 60}", Colors.BOLD)
    log("TEST: OOM error (should NOT restart)", Colors.BOLD)
    log(f"{'=' * 60}\n", Colors.BOLD)

    # Submit with batch size that causes OOM
    job_id = submit_test_job(
        micro_batch_size=16, train_iters=100, array_mode=array_mode, overrides=overrides
    )
    if job_id is None:
        return False

    # Wait for job to start and likely OOM
    log_info("Waiting for job to start and hit OOM...")
    time.sleep(120)

    # Check if job is still running or failed
    state = get_job_state(job_id)
    log_info(f"Job state after 120s: {state}")

    # Check for restart (should not happen)
    new_job_id = check_for_restart(job_id, timeout=120)

    # Clean up
    if new_job_id:
        cancel_job(new_job_id)
        cleanup_monitors()
        log_error("Job was restarted after OOM - this is incorrect!")
        return False

    if state == "RUNNING":
        cancel_job(job_id)

    cleanup_monitors()
    log_success("Test passed: OOM did not trigger restart (as expected)!")
    return True


def test_scenario_max_retries(array_mode: bool = False, overrides: list[str] = []) -> bool:
    """Test that max_retries is respected."""
    log(f"\n{'=' * 60}", Colors.BOLD)
    log("TEST: Max retries (should stop after 3 restarts)", Colors.BOLD)
    log(f"{'=' * 60}\n", Colors.BOLD)

    job_id = submit_test_job(
        micro_batch_size=8,
        train_iters=1000,
        array_mode=array_mode,
        overrides=["monitoring.state_events.1.actions.0.conditions.1.max_attempts=2"] + overrides,
    )
    if job_id is None:
        return False

    # Initial start with 30s intervals
    if not wait_for_job_state(job_id, "RUNNING", timeout=600, poll_interval=30):
        cancel_job(job_id)
        return False

    max_retries = 1
    for attempt in range(max_retries + 1):  # Try one more than max
        log_info(f"\n--- Attempt {attempt + 1}/{max_retries + 1} ---")

        cancel_job(job_id)
        # Check for restart (10s intervals, 20 tries = 200s)
        new_job_id = check_for_restart(job_id, timeout=200)

        if attempt < max_retries:
            # Should restart
            if new_job_id is None:
                log_error(f"Attempt {attempt + 1}: Should have restarted but didn't!")
                return False
            log_success(f"Attempt {attempt + 1}: Restarted as expected")
            job_id = new_job_id

            # Wait for restarted job (10s intervals)
            if not wait_for_job_state(job_id, "RUNNING", timeout=200, poll_interval=10):
                cancel_job(job_id)
                return False
        else:
            # Should NOT restart (budget exhausted)
            if new_job_id is not None:
                log_error(f"Attempt {attempt + 1}: Should have stopped but restarted!")
                cancel_job(new_job_id)
                return False
            log_success(f"Attempt {attempt + 1}: Stopped as expected (retry budget exhausted)")

    cleanup_monitors()
    log_success("Test passed: max_retries is respected!")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Test automatic restart functionality")
    parser.add_argument(
        "--scenario",
        choices=["scancel", "hang", "nccl", "oom", "max_retries", "all"],
        default="scancel",
        help="Which test scenario to run",
    )
    parser.add_argument(
        "--iterations", type=int, default=2, help="Number of iterations for scancel test"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging for detailed diagnostics"
    )
    parser.add_argument(
        "--array-mode",
        action="store_true",
        default=False,
        help="Use SLURM array job submission (default: single job mode)",
    )
    parser.add_argument("overrides", nargs="*", default=[], help="Overrides")

    args = parser.parse_args()

    log(f"\n{'=' * 60}", Colors.BOLD)
    log("AUTO-RESTART TEST SUITE", Colors.BOLD)
    log(f"{'=' * 60}\n", Colors.BOLD)

    # Register cleanup handler
    import atexit

    atexit.register(cleanup_monitors)

    # Verify environment before running tests
    check_environment()

    results = {}

    if args.scenario == "all":
        scenarios = ["scancel", "hang", "nccl", "oom", "max_retries"]
    else:
        scenarios = [args.scenario]

    # Log array mode setting
    mode_str = "ARRAY MODE" if args.array_mode else "SINGLE JOB MODE"
    log_info(f"Running tests in {mode_str}")

    for scenario in scenarios:
        if scenario == "scancel":
            results[scenario] = test_scenario_scancel(
                args.iterations, args.array_mode, overrides=args.overrides
            )
        elif scenario == "hang":
            results[scenario] = test_scenario_hang(args.array_mode, overrides=args.overrides)
        elif scenario == "nccl":
            results[scenario] = test_scenario_nccl_error(args.array_mode, overrides=args.overrides)
        elif scenario == "oom":
            results[scenario] = test_scenario_oom(args.array_mode, overrides=args.overrides)
        elif scenario == "max_retries":
            results[scenario] = test_scenario_max_retries(args.array_mode, overrides=args.overrides)

        # Wait between tests
        if len(scenarios) > 1:
            log_info("Waiting 30s before next test...")
            time.sleep(30)

    # Print summary
    log(f"\n{'=' * 60}", Colors.BOLD)
    log("TEST SUMMARY", Colors.BOLD)
    log(f"{'=' * 60}\n", Colors.BOLD)

    all_passed = True
    for scenario, passed in results.items():
        if passed:
            log_success(f"{scenario}: PASSED")
        else:
            log_error(f"{scenario}: FAILED")
            all_passed = False

    if all_passed:
        log(f"\n{'=' * 60}", Colors.GREEN + Colors.BOLD)
        log("ALL TESTS PASSED! ✓", Colors.GREEN + Colors.BOLD)
        log(f"{'=' * 60}\n", Colors.GREEN + Colors.BOLD)
        return 0
    else:
        log(f"\n{'=' * 60}", Colors.RED + Colors.BOLD)
        log("SOME TESTS FAILED! ✗", Colors.RED + Colors.BOLD)
        log(f"{'=' * 60}\n", Colors.RED + Colors.BOLD)
        return 1


if __name__ == "__main__":
    sys.exit(main())

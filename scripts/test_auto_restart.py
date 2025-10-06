#!/usr/bin/env python3
"""Test script for automatic restart functionality.

This script tests the monitoring and restart system by:
1. Submitting test jobs
2. Simulating various failure scenarios
3. Verifying restart behavior

Usage:
    python scripts/test_auto_restart.py --scenario scancel
    python scripts/test_auto_restart.py --scenario hang
    python scripts/test_auto_restart.py --scenario oom
    python scripts/test_auto_restart.py --scenario all
"""

import argparse
import subprocess
import time
import sys
from pathlib import Path
from typing import Optional


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


def run_command(cmd: list[str], capture: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    log_info(f"Running: {' '.join(cmd)}")
    if capture:
        return subprocess.run(cmd, capture_output=True, text=True, check=False)
    else:
        return subprocess.run(cmd, check=False)


def wait_for_job_state(job_id: int, expected_state: str, timeout: int = 120) -> bool:
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
        time.sleep(5)

    log_error(f"Timeout waiting for job {job_id} to reach {expected_state}")
    return False


def get_job_state(job_id: int) -> Optional[str]:
    """Get the current SLURM state of a job."""
    result = run_command(["squeue", "-j", str(job_id), "-h", "-o", "%T"])
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    return None


def submit_test_job(micro_batch_size: int = 8, train_iters: int = 100) -> Optional[int]:
    """Submit a test job and return the job ID.

    Args:
        micro_batch_size: Batch size (8 works, 16 causes OOM)
        train_iters: Number of training iterations

    Returns:
        Job ID if successful, None otherwise
    """
    log_info(f"Submitting test job (micro_bs={micro_batch_size}, iters={train_iters})...")

    cmd = [
        "python",
        "scripts/run_autoexp.py",
        "--config-name",
        "experiments/megatron_with_auto_restart",
        "slurm=juwels",
        "monitoring=megatron_production",
        "restart=megatron_transient",
        f"backend.megatron.micro_batch_size={micro_batch_size}",
        f"backend.megatron.train_iters={train_iters}",
        "slurm.sbatch.time=00:30:00",
        f"project.name=restart_test_mbs{micro_batch_size}",
    ]

    result = run_command(cmd)

    # Parse job ID from output
    # Expected format: "submitted <name> -> job <job_id> -> log: <path>"
    for line in result.stdout.splitlines():
        if "submitted" in line and "-> job" in line:
            parts = line.split("-> job")
            if len(parts) > 1:
                job_id_str = parts[1].split("->")[0].strip()
                try:
                    job_id = int(job_id_str)
                    log_success(f"Submitted job {job_id}")
                    return job_id
                except ValueError:
                    pass

    log_error("Failed to parse job ID from output")
    log_info(f"Output: {result.stdout}")
    log_error(f"Error: {result.stderr}")
    return None


def inject_error_pattern(job_id: int, pattern: str, description: str) -> bool:
    """Inject an error pattern into a job's log file.

    Args:
        job_id: SLURM job ID
        pattern: Error message to inject
        description: Human-readable description

    Returns:
        True if successful, False otherwise
    """
    log_info(f"Injecting {description} into job {job_id} log...")

    # Find the log file
    log_pattern = f"logs/*{job_id}*.out"
    result = run_command(["bash", "-c", f"ls {log_pattern} 2>/dev/null"])

    if result.returncode != 0 or not result.stdout.strip():
        log_error(f"Could not find log file for job {job_id}")
        return False

    log_file = result.stdout.strip().split("\n")[0]
    log_info(f"Found log file: {log_file}")

    # Append error pattern
    try:
        with open(log_file, "a") as f:
            f.write(f"\n{pattern}\n")
        log_success(f"Injected {description}")
        return True
    except Exception as e:
        log_error(f"Failed to inject pattern: {e}")
        return False


def cancel_job(job_id: int) -> bool:
    """Cancel a SLURM job."""
    log_info(f"Cancelling job {job_id}...")
    result = run_command(["scancel", str(job_id)])
    if result.returncode == 0:
        log_success(f"Cancelled job {job_id}")
        return True
    else:
        log_error(f"Failed to cancel job {job_id}")
        return False


def check_for_restart(original_job_id: int, timeout: int = 300) -> Optional[int]:
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
        result = run_command(["squeue", "-u", "$USER", "-h", "-o", "%i %j"])

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
                        job_id = int(job_id_str)
                        # Look for a job with the same name but different ID
                        if job_id != original_job_id and "restart_test" in job_name:
                            log_success(f"Found restarted job: {job_id}")
                            return job_id
                    except ValueError:
                        continue

        time.sleep(10)

    log_warning(f"No restart detected within {timeout}s")
    return None


def test_scenario_scancel(iterations: int = 2) -> bool:
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
    job_id = submit_test_job(micro_batch_size=8, train_iters=1000)
    if job_id is None:
        return False

    # Wait for job to start running
    if not wait_for_job_state(job_id, "RUNNING", timeout=300):
        log_error("Job never started running")
        cancel_job(job_id)
        return False

    for i in range(iterations):
        log_info(f"\n--- Iteration {i + 1}/{iterations} ---")

        # Cancel the job
        if not cancel_job(job_id):
            return False

        # Check for restart
        new_job_id = check_for_restart(job_id, timeout=120)

        if new_job_id is None:
            log_error(f"Iteration {i + 1}: No restart detected!")
            return False

        log_success(f"Iteration {i + 1}: Job restarted ({job_id} -> {new_job_id})")

        # Wait for new job to start
        if not wait_for_job_state(new_job_id, "RUNNING", timeout=300):
            log_error("Restarted job never started running")
            cancel_job(new_job_id)
            return False

        job_id = new_job_id

    # Clean up
    cancel_job(job_id)
    log_success("Test passed: scancel restart works!")
    return True


def test_scenario_hang() -> bool:
    """Test restart on CUDA hang detection."""
    log(f"\n{'=' * 60}", Colors.BOLD)
    log("TEST: CUDA hang (should restart)", Colors.BOLD)
    log(f"{'=' * 60}\n", Colors.BOLD)

    job_id = submit_test_job(micro_batch_size=8, train_iters=1000)
    if job_id is None:
        return False

    # Wait for job to start
    if not wait_for_job_state(job_id, "RUNNING", timeout=300):
        cancel_job(job_id)
        return False

    # Let it run a bit
    time.sleep(30)

    # Inject hang error
    if not inject_error_pattern(
        job_id, "[default0]:CUDA device-side assert detected in kernel execution", "CUDA hang pattern"
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

    if new_job_id is None:
        log_error("No restart detected after CUDA hang!")
        return False

    log_success("Test passed: CUDA hang restart works!")
    return True


def test_scenario_nccl_error() -> bool:
    """Test restart on NCCL error detection."""
    log(f"\n{'=' * 60}", Colors.BOLD)
    log("TEST: NCCL error (should restart)", Colors.BOLD)
    log(f"{'=' * 60}\n", Colors.BOLD)

    job_id = submit_test_job(micro_batch_size=8, train_iters=1000)
    if job_id is None:
        return False

    if not wait_for_job_state(job_id, "RUNNING", timeout=300):
        cancel_job(job_id)
        return False

    time.sleep(30)

    # Inject NCCL error
    if not inject_error_pattern(
        job_id, "[default0]:NCCL ERROR Remote process has terminated or communication failure", "NCCL error pattern"
    ):
        cancel_job(job_id)
        return False

    new_job_id = check_for_restart(job_id, timeout=180)

    if new_job_id:
        cancel_job(new_job_id)
    else:
        cancel_job(job_id)

    if new_job_id is None:
        log_error("No restart detected after NCCL error!")
        return False

    log_success("Test passed: NCCL error restart works!")
    return True


def test_scenario_oom() -> bool:
    """Test NO restart on OOM (excluded error type).

    This submits a job with micro_batch_size=16 which should OOM.
    """
    log(f"\n{'=' * 60}", Colors.BOLD)
    log("TEST: OOM error (should NOT restart)", Colors.BOLD)
    log(f"{'=' * 60}\n", Colors.BOLD)

    # Submit with batch size that causes OOM
    job_id = submit_test_job(micro_batch_size=16, train_iters=100)
    if job_id is None:
        return False

    # Wait for job to start and likely OOM
    log_info("Waiting for job to start and hit OOM...")
    time.sleep(60)

    # Check if job is still running or failed
    state = get_job_state(job_id)
    log_info(f"Job state after 60s: {state}")

    # Check for restart (should not happen)
    new_job_id = check_for_restart(job_id, timeout=120)

    # Clean up
    if new_job_id:
        cancel_job(new_job_id)
        log_error("Job was restarted after OOM - this is incorrect!")
        return False

    if state == "RUNNING":
        cancel_job(job_id)

    log_success("Test passed: OOM did not trigger restart (as expected)!")
    return True


def test_scenario_max_retries() -> bool:
    """Test that max_retries is respected."""
    log(f"\n{'=' * 60}", Colors.BOLD)
    log("TEST: Max retries (should stop after 3 restarts)", Colors.BOLD)
    log(f"{'=' * 60}\n", Colors.BOLD)

    job_id = submit_test_job(micro_batch_size=8, train_iters=1000)
    if job_id is None:
        return False

    if not wait_for_job_state(job_id, "RUNNING", timeout=300):
        cancel_job(job_id)
        return False

    max_retries = 3
    for attempt in range(max_retries + 1):  # Try one more than max
        log_info(f"\n--- Attempt {attempt + 1}/{max_retries + 1} ---")

        cancel_job(job_id)
        new_job_id = check_for_restart(job_id, timeout=120)

        if attempt < max_retries:
            # Should restart
            if new_job_id is None:
                log_error(f"Attempt {attempt + 1}: Should have restarted but didn't!")
                return False
            log_success(f"Attempt {attempt + 1}: Restarted as expected")
            job_id = new_job_id

            if not wait_for_job_state(job_id, "RUNNING", timeout=300):
                cancel_job(job_id)
                return False
        else:
            # Should NOT restart (budget exhausted)
            if new_job_id is not None:
                log_error(f"Attempt {attempt + 1}: Should have stopped but restarted!")
                cancel_job(new_job_id)
                return False
            log_success(f"Attempt {attempt + 1}: Stopped as expected (retry budget exhausted)")

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
    parser.add_argument("--iterations", type=int, default=2, help="Number of iterations for scancel test")

    args = parser.parse_args()

    log(f"\n{'=' * 60}", Colors.BOLD)
    log("AUTO-RESTART TEST SUITE", Colors.BOLD)
    log(f"{'=' * 60}\n", Colors.BOLD)

    results = {}

    if args.scenario == "all":
        scenarios = ["scancel", "hang", "nccl", "oom", "max_retries"]
    else:
        scenarios = [args.scenario]

    for scenario in scenarios:
        if scenario == "scancel":
            results[scenario] = test_scenario_scancel(args.iterations)
        elif scenario == "hang":
            results[scenario] = test_scenario_hang()
        elif scenario == "nccl":
            results[scenario] = test_scenario_nccl_error()
        elif scenario == "oom":
            results[scenario] = test_scenario_oom()
        elif scenario == "max_retries":
            results[scenario] = test_scenario_max_retries()

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

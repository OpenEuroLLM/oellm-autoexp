#!/usr/bin/env python3
"""Demo script showing monitor logging in action."""

import asyncio
import logging
from pathlib import Path
import sys

from oellm_autoexp.monitor.controller import JobRegistration, MonitorController
from oellm_autoexp.monitor.policy import (
    AlwaysRestartPolicy,
    AlwaysRestartPolicyConfig,
    NoRestartPolicy,
    NoRestartPolicyConfig,
    SelectiveRestartPolicy,
    SelectiveRestartPolicyConfig,
)
from oellm_autoexp.monitor.watcher import SlurmLogMonitor, SlurmLogMonitorConfig
from oellm_autoexp.slurm.client import FakeSlurmClient, FakeSlurmClientConfig


def setup_logging():
    """Configure logging to show monitor events."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    # Create temp directory for test logs
    tmp_dir = Path("/tmp/oellm_monitor_demo")
    tmp_dir.mkdir(exist_ok=True)

    logger.info("=== Monitor Logging Demo ===")
    logger.info("This demonstrates the comprehensive logging added to MonitorController")

    # Set up monitor with selective restart policy
    monitor = SlurmLogMonitor(SlurmLogMonitorConfig(check_interval_seconds=1))
    slurm = FakeSlurmClient(FakeSlurmClientConfig())
    policies = {
        "crash": SelectiveRestartPolicy(
            SelectiveRestartPolicyConfig(
                max_retries=3,
                default_action="restart",
                restart_on_error_types=["cancelled", "hang", "nccl"],
                restart_on_subsystems=["slurm"],
                exclude_error_types=["oom"],
            )
        ),
        "success": NoRestartPolicy(NoRestartPolicyConfig()),
        "stall": AlwaysRestartPolicy(AlwaysRestartPolicyConfig(max_retries=2)),
    }
    controller = MonitorController(monitor, slurm, policies)

    # Create test job files
    script = tmp_dir / "test.sh"
    script.write_text("#!/bin/bash\necho 'Training...'")
    log = tmp_dir / "job.log"
    log.write_text("Starting job\n")

    # Submit job
    job_id = slurm.submit("demo_job", script, log)
    controller.register_job(
        job_id,
        JobRegistration(name="demo_job", script_path=script, log_path=log),
    )
    logger.info(f"\n--- Submitted job {job_id} ---\n")

    # Scenario 1: Job starts running
    logger.info("\n=== SCENARIO 1: Job starts running ===")
    slurm.set_state(job_id, "RUNNING")
    result = controller.observe_once_sync()
    logger.info(f"Actions: {[a.action for a in result.actions]}")

    # Scenario 2: Job is cancelled (should restart)
    logger.info("\n=== SCENARIO 2: Job is cancelled (should restart) ===")
    slurm.set_state(job_id, "CANCELLED")
    result = controller.observe_once_sync()
    logger.info(f"Decisions: {[(k, v.action, v.reason) for k, v in result.decisions.items()]}")
    logger.info(f"Actions: {[a.action for a in result.actions]}")

    # Check if job was restarted
    job_states = list(controller.jobs())
    if job_states:
        new_job_id = job_states[0].job_id
        if new_job_id != job_id:
            logger.info(f"\n✓ Job successfully restarted: {job_id} -> {new_job_id}")
            logger.info(f"  Attempt count: {job_states[0].attempts}")
        else:
            logger.error(f"\n✗ Job was NOT restarted (still {job_id})")
    else:
        logger.error("\n✗ No jobs in controller after restart attempt")

    logger.info("\n=== Demo Complete ===")
    logger.info("Review the log output above to see all monitoring events:")
    logger.info("  - SLURM state transitions")
    logger.info("  - Mode classification")
    logger.info("  - Policy decisions")
    logger.info("  - Restart operations")


if __name__ == "__main__":
    main()

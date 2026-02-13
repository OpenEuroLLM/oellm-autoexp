# #!/usr/bin/env python3
# """End-to-end monitoring test harness for real SLURM clusters.

# This script automates the workflow previously covered by shell scripts such
# as ``test_scancel_full.sh`` but adds a richer CLI, supports OmegaConf overrides,
# and can be adapted to different clusters (JUWELS, LUMI, JUPITER, …).

# Basic flow (scenario=``scancel``):
# 1. Run ``run_autoexp_container.py`` with the provided config + overrides,
#    generating a manifest but skipping monitoring (``--no-monitor``).
# 2. Start ``monitor_autoexp.py`` in the background.
# 3. Wait for the submitted job to reach RUNNING, then cancel it via ``scancel``.
# 4. Wait for the monitor to detect the cancellation and restart the job.
# 5. Summarise the outcome (original job id, new job id, monitor log location).
# """

# from __future__ import annotations

# import argparse
# import os
# import signal
# import subprocess
# import sys
# import time
# from pathlib import Path
# from collections.abc import Sequence

# from oellm_autoexp.workflow.manifest import read_manifest


# def _now_token() -> str:
#     return time.strftime("%Y%m%d_%H%M%S")


# def run_command(
#     cmd: Sequence[str],
#     *,
#     cwd: Path | None = None,
#     capture: bool = False,
#     env: dict[str, str] | None = None,
# ) -> subprocess.CompletedProcess[str]:
#     """Run a subprocess, raising on failure."""

#     text_env = os.environ.copy()
#     if env:
#         text_env.update(env)
#     return subprocess.run(
#         list(cmd),
#         cwd=str(cwd) if cwd else None,
#         env=text_env,
#         check=True,
#         text=True,
#         capture_output=capture,
#     )


# def wait_for_slurm_state(job_id: str, target: str, timeout: float, poll_interval: float) -> str:
#     """Poll ``squeue`` until the job reaches ``target`` or timeout expires."""

#     deadline = time.time() + timeout
#     target_upper = target.upper()
#     last_state = "UNKNOWN"
#     while time.time() < deadline:
#         try:
#             proc = run_command(["squeue", "-j", job_id, "-h", "-o", "%T"], capture=True)
#             state = proc.stdout.strip() or "NOT_FOUND"
#         except subprocess.CalledProcessError:
#             state = "NOT_FOUND"
#         last_state = state
#         if state.upper() == target_upper:
#             return state
#         time.sleep(poll_interval)
#     raise TimeoutError(f"Job {job_id} did not reach state {target} (last={last_state})")


# def wait_for_restarted_job(
#     store: MonitorStateStore,
#     original_job_id: str,
#     timeout: float,
#     poll_interval: float,
# ) -> str:
#     """Wait until the MonitorStateStore records a different job id."""

#     deadline = time.time() + timeout
#     while time.time() < deadline:
#         jobs = store.load()
#         ids = list(jobs)
#         if ids:
#             current = ids[0]
#             if current != original_job_id:
#                 return current
#         time.sleep(poll_interval)
#     raise TimeoutError("Monitor did not register a restarted job within timeout")


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description=__doc__)
#     parser.add_argument(
#         "--scenario",
#         choices=["scancel"],
#         default="scancel",
#         help="Scenario to run (default: scancel).",
#     )
#     parser.add_argument(
#         "--config-ref",
#         default="experiments/megatron_with_auto_restart",
#         help="Hydra config reference passed to run_autoexp_container.py",
#     )
#     parser.add_argument(
#         "--config-dir",
#         default="config",
#         help="Config root (forwarded via -C/--config-path).",
#     )
#     parser.add_argument(
#         "--manifest",
#         type=Path,
#         help="Manifest path to use. Defaults to output/e2e_<scenario>_<timestamp>.json",
#     )
#     parser.add_argument(
#         "--container",
#         help="Convenience shortcut for the common 'container=<name>' override (optional).",
#     )
#     parser.add_argument(
#         "--overrides",
#         action="append",
#         default=[],
#         help="OmegaConf override (repeatable). Example: --override slurm.partition=develbooster",
#     )
#     parser.add_argument(
#         "--run-arg",
#         action="append",
#         default=[],
#         help="Extra argument forwarded verbatim to run_autoexp_container.py (repeatable).",
#     )
#     parser.add_argument(
#         "--monitor-arg",
#         action="append",
#         default=[],
#         help="Extra argument forwarded to monitor_autoexp.py (repeatable).",
#     )
#     parser.add_argument(
#         "--no-monitor",
#         action="store_true",
#         help="Do not spawn monitor_autoexp.py automatically (allows manual runs).",
#     )
#     parser.add_argument(
#         "--monitor-log",
#         type=Path,
#         default=None,
#         help="File to capture monitor stdout/stderr (default: output/e2e_<plan_id>_monitor.log).",
#     )
#     parser.add_argument(
#         "--poll-interval",
#         type=float,
#         default=5.0,
#         help="Polling interval (seconds) for SLURM/job-state checks.",
#     )
#     parser.add_argument(
#         "--timeout",
#         type=float,
#         default=600.0,
#         help="Overall timeout per wait step (seconds).",
#     )
#     parser.add_argument(
#         "--run-script",
#         default="scripts/run_autoexp_container.py",
#         help="Path to the planning/submission helper (default: scripts/run_autoexp_container.py).",
#     )
#     parser.add_argument(
#         "--monitor-script",
#         default="scripts/monitor_autoexp.py",
#         help="Path to the monitoring helper (default: scripts/monitor_autoexp.py).",
#     )
#     parser.add_argument(
#         "--dry-run",
#         action="store_true",
#         help="Print planned commands but do not execute them.",
#     )
#     return parser.parse_args()


# def main() -> None:
#     args = parse_args()
#     manifest_path = args.manifest or Path("output") / f"e2e_{args.scenario}_{_now_token()}.json"
#     manifest_path.parent.mkdir(parents=True, exist_ok=True)

#     run_cmd = [
#         sys.executable,
#         args.run_script,
#         "--manifest",
#         str(manifest_path),
#         "--no-monitor",
#         "-C",
#         args.config_dir,
#         "--config-ref",
#         args.config_ref,
#     ]
#     if args.container:
#         run_cmd.append(f"container={args.container}")
#     if args.run_arg:
#         run_cmd.extend(args.run_arg)
#     if args.overrides:
#         run_cmd.extend(args.overrides)

#     print(f"[run_autoexp_container] {' '.join(run_cmd)}")
#     if args.dry_run:
#         return

#     run_command(run_cmd)

#     if not manifest_path.exists():
#         raise RuntimeError(f"Manifest not found at {manifest_path}")
#     manifest = read_manifest(manifest_path)
#     print(
#         f"[manifest] plan_id={manifest.plan_id}, monitor_state_dir={manifest.monitoring_state_dir}"
#     )

#     state_store = MonitorStateStore(manifest.monitor_state_dir, session_id=manifest.plan_id)

#     jobs = state_store.load()
#     if not jobs:
#         raise RuntimeError("No jobs registered in monitoring state (submission failed?)")
#     initial_job_id = next(iter(jobs))
#     print(f"[state] initial job: {initial_job_id}")

#     monitor_proc: subprocess.Popen[str] | None = None
#     monitor_log_path = args.monitor_log or (Path("output") / f"e2e_{manifest.plan_id}_monitor.log")
#     if not args.no_monitor:
#         monitor_log_path.parent.mkdir(parents=True, exist_ok=True)
#         monitor_cmd = [
#             sys.executable,
#             args.monitor_script,
#             "--manifest",
#             str(manifest_path),
#         ]
#         if args.monitor_arg:
#             monitor_cmd.extend(args.monitor_arg)
#         print(f"[monitor] {' '.join(monitor_cmd)}")
#         monitor_log = open(monitor_log_path, "w", encoding="utf-8")
#         monitor_proc = subprocess.Popen(
#             monitor_cmd,
#             stdout=monitor_log,
#             stderr=subprocess.STDOUT,
#             text=True,
#         )

#     try:
#         if args.scenario == "scancel":
#             print(f"[scenario:scancel] waiting for job {initial_job_id} to reach RUNNING …")
#             wait_for_slurm_state(initial_job_id, "RUNNING", args.timeout, args.poll_interval)
#             print(f"[scenario:scancel] job {initial_job_id} is RUNNING. Issuing scancel …")
#             run_command(["scancel", initial_job_id])
#             print("[scenario:scancel] waiting for monitor to submit replacement job …")
#             new_job_id = wait_for_restarted_job(
#                 state_store,
#                 initial_job_id,
#                 timeout=args.timeout,
#                 poll_interval=args.poll_interval,
#             )
#             print(f"[scenario:scancel] restart detected: {initial_job_id} -> {new_job_id}")
#         else:
#             raise NotImplementedError(f"Scenario {args.scenario!r} not implemented")
#     finally:
#         if monitor_proc is not None:
#             monitor_proc.send_signal(signal.SIGINT)
#             try:
#                 monitor_proc.wait(timeout=10)
#             except subprocess.TimeoutExpired:
#                 monitor_proc.kill()
#         print(f"[monitor] log written to {monitor_log_path}")


# if __name__ == "__main__":
#     main()

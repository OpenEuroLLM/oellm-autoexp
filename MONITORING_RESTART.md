# Monitoring Restart & Persistence – Current State and Restructure Plan

## 1. Current Monitoring Pipeline

1. **Planning & Submission**
   - `scripts/plan_autoexp.py` renders SBATCH scripts and writes a manifest (`PlanManifest`) containing job metadata (script path, log template, monitoring config, restart policies, etc.).
   - `scripts/submit_autoexp.py` loads the manifest, instantiates a monitor/controller by way of `build_host_runtime()` and `instantiate_controller()`, submits jobs (or an array) through the SLURM client, and registers them with the `MonitorController`.
   - During registration the controller persists each job in `MonitorStateStore` (`StoredJob` entries keyed by `job_id`).

2. **Monitoring Loop**
   - `run_monitoring()` drives `_monitor_loop()`, which periodically calls `MonitorController.observe_once()`.
   - The controller calls `monitor.watch(...)` (async), merges the result with the latest `squeue()` snapshot, evaluates events, and, if required, calls the restart policy (by way of `_apply_policy()`).
   - Action payloads are written as JSON files to `<monitoring_state_dir>/actions/…`.
   - Whenever job state changes (`_set_state`, `_finalize_job`, `_restart_job`) the controller updates `MonitorStateStore` so the session can be resumed.

3. **Persistence Layout**
   - `MonitorStateStore` writes `<monitoring_state_dir>/<plan_id>.json`, containing:
     ```json
     {
       "session_id": "...",
       "project_name": "...",
       "config": {... original manifest config ...},
       "jobs": [
         {
           "job_id": "123456",
           "name": "demo_0",
           "script_path": "...sbatch",
           "log_path": "...%A_%a.log",
           "attempts": 1,
           "metadata": {...},
           ...
         }
       ]
     }
     ```
   - Each restart (triggered by policies) rewrites the store with the new `job_id` (for example `123456 -> 123789`).

4. **Monitoring Resume (`scripts/monitor_autoexp.py`)**
   - Reads the session JSON (by way of `--session <plan_id>` or `--session path/to/session.json`) to locate the manifest, then loads the monitor runtime, instantiates a controller, and calls `restore_jobs()`.
   - `restore_jobs()` pulls all `StoredJob` entries and re-registers them verbatim (`controller.register_job(job_id, registration, attempts)`). Sessions with no active jobs are skipped by default when running `--all`; add `--include-completed` to reprocess archived sessions.
   - If the monitor is run with `--monitor-override …`, Hydra-style overrides are merged into the monitor config before instantiation (for example `debug_sync=true`).

## 2. Observed Issues When Monitoring Is Interrupted

### 2.1. Running Jobs Are Not Re-captured Reliably
- The state store persists the **exact SLURM job ID** active at the time of persistence.
- If monitoring is stopped and the job continues under the same ID, resuming works — the controller finds the job by way of `squeue()` and proceeds.
- **Problem:** If the job is resubmitted while monitoring is offline (for example manual restart, scheduler requeue), the new SLURM job ID differs from the stored one. On resume:
  - `restore_jobs()` registers the stale job ID.
  - `observe_once()` sees `squeue()` return `NOT_FOUND`, classifies it as `timeout`, and applies the `timeout` policy. Net result: the controller may immediately attempt a new restart, overriding the in-flight job.
- There is no mechanism to reconcile the stored job ID with the current queue state (for example lookup by job name or metadata).

### 2.2. Finished Jobs Remain Visible After Resume
- When monitoring stops before a job has been finalized (`_finalize_job()`), the `StoredJob` entry remains.
- On resume, the job is re-registered:
  - If the log contains a terminal marker (`termination_string`) the monitor emits a `SuccessState` event and `_apply_policy()` (for mode `success`) finalizes the job.
  - If no termination marker exists and the job is missing from `squeue()`, the controller interprets the situation as `timeout`. Depending on policies, this may either stop the job with a “timeout” reason or attempt another restart — even though the job is already done.
- The UI/CLI therefore shows “undefined/pending” jobs until a manual intervention removes them from the state store.

### 2.3. Resumed Sessions Do Not Sync Default Monitoring for All Jobs
- `monitor_autoexp.py` restores only the jobs persisted in the state store. Jobs defined in the manifest but never registered previously are not added.
- There is no flag or workflow to “monitor everything from the manifest” after a stop/restart; operators must manually re-register jobs or re-run submission.

### 2.4. Lack of Attempt History in `StoredJob`
- The persistence layer stores the current attempt count but not the mapping of attempt → job ID.
- When resuming, the user cannot tell which SLURM job ID is the active attempt, nor can the controller match a new job ID back to the job it belongs to.

### 2.5. Log Tails Depend on Templates
- Log paths now include `%j`/`%A_%a` so restarts write to unique files, but the state store persists the template rather than an expanded path.
- If external tooling truncates or rotates logs differently per attempt, the monitor replays entire files on resume, potentially re-firing events.

## 3. Implemented Improvements (April 2024)

Several of the pain points above are now addressed in code and tooling:

1. **Persist resolved log paths and last-known states.** The monitor writes
   `resolved_log_path`, `last_monitor_state`, `last_slurm_state`, and a
   `last_updated` timestamp for every tracked job. Restarted monitors no
   longer have to expand `%j`/`%A_%a` placeholders and can immediately point
   operators at concrete log files.

2. **Re-register SLURM state on resume.** `restore_jobs()` and the container
   orchestrator both call `slurm_client.register_job(...)` when rehydrating a
   session. Fake and real clients now have the job in their internal
   registries, making `squeue()` snapshots work again after a pause.

3. **State store validation script.** `scripts/tests/test_monitor_resume.py`
   automates a plan → submit → interrupt → resume workflow on a login node and
   asserts that the state file contains resolved log paths. It also checks that
   the monitor prints a restore message on the second run, providing an
   end-to-end smoke test for the resume path.

4. **Documentation updates.** README/SPEC now describe the resume flow, the new
   state fields, and how to use the monitoring test harness on a cluster.

## 4. Outstanding Items

The longer-term restructure ideas remain valid and would further harden the
resume path:

- Track logical job instances (name → attempt history) instead of persisting a
  single job ID.
- Reconcile SLURM job IDs that change while monitoring is offline by querying
  `squeue`/`sacct` by job name.
- Surface completion state for jobs that finished while the monitor was down
  without relying solely on log markers.
- Provide a `--monitor-all` option to (re)register jobs defined in the manifest
  even if they were never tracked before the pause.
- Expand attempt history in the state store so operators can inspect previous
  retries after the job finalises.

# Nested Monitoring Loop Solution

## Problem

When `RunAutoexpAction` executes (for example, for cooldown runs), it starts a new monitoring loop inside the already-running monitoring loop:

```
Monitoring Loop (outer)
  └─> Action execution: RunAutoexpAction
      └─> run_autoexp_container.py
          └─> submit_autoexp.py
              └─> run_monitoring()  ← NEW MONITORING LOOP (nested)
```

This creates nested loops where:
1. The outer loop monitors the original training job
2. The inner loop monitors the cooldown job
3. The inner loop blocks the outer loop from continuing
4. Multiple cooldowns would create deeply nested loops

## Current Flow

### 1. RunAutoexpAction Execution (actions.py:162-178)
```python
class RunAutoexpAction(BaseMonitorAction):
    config: RunAutoexpActionConfig  # contains script, config_path, overrides

    def execute(self, context: ActionContext) -> ActionResult:
        cmd = [sys.executable, self.config.script]  # for example, scripts/run_autoexp_container.py
        if self.config.config_path:
            cmd.extend(["--config-ref", context.render(self.config.config_path)])
        cmd.extend(context.render(arg) for arg in self.config.overrides)
        env = {**context.env} if context.env else None
        proc = _run_command(cmd, cwd=context.workspace, env=env)
        ...
```

### 2. run_autoexp_container.py Execution
- Runs `plan_autoexp.py` to generate manifest
- Runs `submit_autoexp.py` to submit jobs
- `submit_autoexp.py` starts monitoring by default (unless `--no-monitor` flag is passed)

### 3. submit_autoexp.py Monitoring Logic (submit_autoexp.py:87-96)
```python
if args.no_monitor:
    print("Skipping monitoring (--no-monitor).", flush=True)
    print(f"To monitor later run: {cmd}", flush=True)
    return

try:
    run_monitoring(runtime, controller)  ← STARTS NEW MONITORING LOOP
except KeyboardInterrupt:
    ...
```

## Desired Behavior

Instead of nested monitoring loops:
1. **Submit cooldown job** without starting a new monitoring loop
2. **Register cooldown job** with the existing (outer) monitoring controller
3. **Monitor both jobs** in the same monitoring session

## Solution

### Option 1: Use --no-monitor with Manual Registration (Recommended)

Modify the workflow to:
1. Pass `--no-monitor` flag from `RunAutoexpAction`
2. Pass `--plan-id <current_session_id>` to reuse the same session
3. After action completes, restore new jobs into the current controller

#### Changes Required

**1. Modify RunAutoexpActionConfig to support no_monitor flag (actions.py)**
```python
@dataclass
class RunAutoexpActionConfig(ConfigInterface):
    class_name: str = "RunAutoexpAction"
    script: str = "scripts/run_autoexp.py"
    overrides: list[str] = field(default_factory=list)
    config_path: str | None = None
    no_monitor: bool = True  # NEW: Skip nested monitoring by default
```

**2. Modify RunAutoexpAction to pass --no-monitor (actions.py)**
```python
class RunAutoexpAction(BaseMonitorAction):
    def execute(self, context: ActionContext) -> ActionResult:
        cmd = [sys.executable, self.config.script]
        if self.config.config_path:
            cmd.extend(["--config-ref", context.render(self.config.config_path)])
        cmd.extend(context.render(arg) for arg in self.config.overrides)

        # NEW: Pass --no-monitor flag
        if self.config.no_monitor:
            cmd.append("--no-monitor")

        # NEW: Pass current session ID to reuse the same monitoring session
        session_id = context.job_metadata.get("session_id")
        if session_id:
            cmd.extend(["--plan-id", session_id])

        env = {**context.env} if context.env else None
        proc = _run_command(cmd, cwd=context.workspace, env=env)

        if proc.returncode == 0:
            # NEW: Parse output for submitted job IDs
            submitted_jobs = self._parse_submitted_jobs(proc.stdout)
            return ActionResult(
                status="success",
                message="run_autoexp completed",
                metadata={"submitted_jobs": submitted_jobs, "session_id": session_id}
            )
        ...
```

**3. Modify _process_action_queue to restore new jobs (host.py)**
```python
def _process_action_queue(
    state_store: MonitorStateStore,
    controller: MonitorController,  # NEW parameter
    runtime: HostRuntime,  # NEW parameter
) -> int:
    """Process all pending actions in the queue and return count of processed actions."""
    queue = ActionQueue(state_store.session_path.with_suffix(".actions"))
    processed = 0

    # NEW: Track existing jobs before processing actions
    existing_job_ids = {state.job_id for state in controller.jobs()}

    while True:
        record = queue.claim_next()
        if record is None:
            break

        # ... existing action processing logic ...

        try:
            result = action.execute(context)
        except Exception as exc:
            # ... error handling ...

        action.update_event(event_record, result)
        state_store.upsert_event(event_record)
        queue.mark_done(
            record.queue_id,
            status="done" if result.status == "success" else "failed",
            result={"message": result.message, "metadata": result.metadata},
        )
        processed += 1

    # NEW: After processing all actions, check if new jobs were submitted
    restored = restore_jobs(runtime, controller)
    new_jobs = [name for name in restored if name not in existing_job_ids]
    if new_jobs:
        print(f"[worker] Registered {len(new_jobs)} new job(s) from actions: {', '.join(new_jobs)}", flush=True)

    return processed
```

**4. Update monitoring loop calls to pass controller and runtime (host.py)**
```python
async def _monitor_loop(controller, monitor, action_queue_dir, state_store, runtime):  # Added runtime
    ...
    while list(controller.jobs()):
        ...
        # Process queued actions automatically
        processed = _process_action_queue(state_store, controller, runtime)  # Pass new params
        ...
```

**5. Add session_id to job metadata when registering (host.py)**
```python
def _register_job(controller, job_id, job, config=None, attempts=1, session_id=None):  # Added session_id
    metadata = {"parameters": dict(job.parameters), "output_dir": job.output_dir}

    # NEW: Add session_id to metadata so actions can reuse it
    if session_id:
        metadata["session_id"] = session_id

    # Add flattened config to metadata
    if config:
        flattened = _flatten_config(config)
        metadata.update(flattened)

    registration = JobRegistration(...)
    controller.register_job(job_id, registration, attempts=attempts)
```

### Option 2: Shared Monitoring Session (Alternative)

Instead of starting/stopping monitoring, use a single monitoring session:
1. All jobs (original + cooldown) share the same session ID
2. Jobs are registered with the controller by way of `state_store.upsert_job()`
3. Monitoring loop periodically calls `restore_jobs()` to pick up new jobs

This requires less changes to the action execution but adds periodic restore overhead.

## How Job Registration Works

When a job is registered by way of `controller.register_job()`:
1. Creates a `JobRuntimeState` object
2. Adds it to `controller._jobs` dict
3. **Automatically calls `_persist_job()`** which saves to state store by way of `state_store.upsert_job()`

When `restore_jobs()` is called:
1. Loads all jobs from `state_store.load()`
2. Registers any new jobs with the controller by way of `controller.register_job()`
3. Returns the set of restored job names

This means:
- Jobs persisted to the session file are automatically available for restoration
- Multiple processes can write to the same session file (by way of upsert)
- The monitoring loop can periodically restore new jobs

## Implementation Steps

1. ✅ Modify `RunAutoexpActionConfig` to add `no_monitor` field (default True)
2. ✅ Modify `RunAutoexpAction.execute()` to pass `--no-monitor` and `--plan-id`
3. ✅ Modify `_register_job()` to include `session_id` in metadata
4. ✅ Modify `_process_action_queue()` to accept controller and runtime parameters
5. ✅ Add logic to restore new jobs after processing actions
6. ✅ Update monitoring loop calls to pass controller and runtime
7. ✅ Test with cooldown configuration

## Testing

```bash
# Run a training job with cooldown monitoring
python scripts/run_autoexp_container.py \
  project=test_cooldown \
  backend=megatron \
  slurm=lumi \
  monitoring=megatron_cooldown_single \
  backend.args.train_iters=200 \
  backend.args.save_interval=100

# When checkpoint 100 is detected:
# 1. Cooldown action executes with --no-monitor --plan-id <session_id>
# 2. Cooldown job gets submitted to the same session
# 3. Action worker restores the cooldown job into the controller
# 4. Monitoring loop picks up both jobs in the same session

# Check monitoring session
python scripts/monitor_autoexp.py --session <session_id> --cmd queue

# Expected: Single monitoring session with 2 jobs (original + cooldown)
```

## Benefits

1. **No nested loops**: Actions submit jobs without blocking
2. **Single monitoring session**: All related jobs monitored together
3. **Crash recovery**: If monitoring crashes, both jobs can be resumed
4. **Visibility**: All jobs visible in the same session
5. **Scalability**: Multiple cooldowns don't create nested loops

## Considerations

- **Race conditions**: Multiple actions might restore jobs concurrently → restore_jobs() is idempotent (only registers new jobs)
- **Session file locking**: State store uses JSON with atomic writes → safe for concurrent access
- **Job name collisions**: Each cooldown has unique name by way of `{checkpoint_iteration}` → no collisions

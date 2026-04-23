# Provenance Writing Fix Summary

## Problem

Provenance files (`_provenance/` directory with config files) were not being written when using `run_autoexp_container.py`, which is the typical workflow for container-based execution.

## Root Cause

The provenance writing logic was only in `scripts/run_autoexp.py`, but the container workflow uses:
```
run_autoexp_container.py → plan_autoexp.py → manifest
```

So `run_autoexp.py` was never called, and no provenance files were written.

## Solution

Added provenance writing to **`scripts/plan_autoexp.py`** (the script actually used in container workflows):

### Changes Made

1. **`scripts/plan_autoexp.py`**:
   - Added `_write_job_provenance()` function
   - Calls it after building the plan (line 119-125)
   - Writes provenance files to `{output_dir}/provenance/`

2. **`oellm_autoexp/config/loader.py`**:
   - Modified `load_config_reference()` to use Hydra compose for group overrides
   - When loading `unresolved_config.yaml` with overrides:
     - Loads `config_reference.json` from same directory
     - Combines original overrides + new overrides
     - Uses Hydra's `compose()` instead of `OmegaConf.update()`
   - This properly handles group overrides like `monitoring=megatron_basic`

3. **`oellm_autoexp/workflow/host.py`**:
   - Added `_flatten_config()` to flatten nested config dict
   - Modified `_register_job()` to include flattened config in job metadata
   - Modified `restore_jobs()` to include flattened config when restoring
   - This makes config values available for template rendering in actions
   - Example: `{project.name}` in cooldown config now resolves correctly

4. **Files Created** (for each job in `{output_dir}/provenance/`):
   - `resolved_config.json` - Fully resolved config (JSON, existing)
   - **`resolved_config.yaml`** - Fully resolved config (YAML, NEW)
   - **`unresolved_config.yaml`** - Config with `${...}` interpolations intact (NEW)
   - `config_reference.json` - Hydra reconstruction metadata (existing)

3. **Error Handling**:
   - Clears Hydra global state to avoid re-initialization errors
   - Gracefully handles failures (prints warnings but doesn't crash)
   - Falls back to JSON-only if YAML generation fails

## Cooldown Configuration Impact

The cooldown configs now correctly reference:
```yaml
config_path: "{output_dir}/provenance/unresolved_config.yaml"
```

This file will now exist because `plan_autoexp.py` writes it during the planning phase.

### How Hydra Group Overrides Work

When a cooldown action uses `monitoring=megatron_basic`:

1. **Original config** (from `config_reference.json`):
   ```json
   {
     "config_ref": "autoexp",
     "config_dir": "/path/to/config",
     "overrides": ["job=test", "monitoring=megatron_cooldown_single", ...]
   }
   ```

2. **Cooldown overrides**:
   ```yaml
   - monitoring=megatron_basic
   - backend.megatron.load=/path/to/checkpoint
   ```

3. **Combined overrides** (passed to Hydra compose):
   ```python
   ["job=test", "monitoring=megatron_cooldown_single", ..., "monitoring=megatron_basic", "backend.megatron.load=/path/to/checkpoint"]
   ```

4. **Result**: The later `monitoring=megatron_basic` override **replaces** the earlier `monitoring=megatron_cooldown_single` in the Hydra composition, giving the cooldown job the simplified monitoring config.

### How Template Rendering Works

When cooldown configs use templates like `{project.name}`:

1. **Config is flattened** when jobs are registered:
   ```python
   {
     "project": {"name": "test", "base_output_dir": "/path"},
     "backend": {"megatron": {"train_iters": 1000}}
   }
   ```
   Becomes:
   ```python
   {
     "project.name": "test",
     "project.base_output_dir": "/path",
     "backend.megatron.train_iters": 1000,
     ...
   }
   ```

2. **Flattened config is added to job metadata** during registration

3. **ActionContext.render()** merges:
   - `job_metadata` (includes flattened config)
   - `event.metadata` (includes `checkpoint_iteration`, `checkpoint_path`, etc.)
   - `event.payload`

4. **Template rendering** uses Python's `.format()`:
   ```yaml
   - project.name={project.name}_cooldown_{checkpoint_iteration}
   ```
   Becomes:
   ```bash
   project.name=test_cooldown_100
   ```

## Testing

### Quick Test (Local, No Container)

```bash
# Test provenance writing by way of plan_autoexp.py
bash scripts/test_plan_provenance.sh
```

Expected output:
```
✓ Found _provenance directory
✓ Found resolved_config.json
✓ Found resolved_config.yaml
✓ Found unresolved_config.yaml
✓ Found config_reference.json
=== ✓ All tests passed! ===
```

### Full Test (With Container)

```bash
# Test with actual container workflow
python scripts/run_autoexp_container.py \
  job=test_cooldown \
  backend=megatron \
  slurm=lumi \
  monitoring=megatron_cooldown_single \
  backend.args.train_iters=200 \
  backend.args.save_interval=100 \
  --no-submit

# Check provenance was created
ls -la output/test_cooldown*/provenance/
```

Expected files:
```
provenance/
├── resolved_config.json
├── resolved_config.yaml
├── unresolved_config.yaml
└── config_reference.json
```

### Verify Cooldown Can Use Provenance

After a run completes and hits the cooldown iteration (for example, 100), the monitoring system will try to execute:

```bash
python scripts/run_autoexp_container.py \
  --config-ref /path/to/output/provenance/unresolved_config.yaml \
  backend.megatron.load=/path/to/checkpoint \
  project.base_output_dir=/path/to/output_cooldown_100_${oc.timestring:} \
  ...
```

This will now work because:
1. The `unresolved_config.yaml` file exists
2. The overrides are properly applied by way of `load_config_reference()`
3. The `${oc.timestring:}` interpolation is resolved at cooldown run time

## Files Modified

1. `scripts/plan_autoexp.py` - Added provenance writing
2. `scripts/run_autoexp.py` - Already had provenance writing (for non-container workflow)
3. `oellm_autoexp/config/loader.py` - Added override support for YAML file paths
4. `config/monitoring/megatron_cooldown_single.yaml` - Uses `unresolved_config.yaml`
5. `config/monitoring/megatron_cooldown_multi.yaml` - Uses `unresolved_config.yaml`

## Debugging Tips

If provenance files still don't appear:

1. **Check for warnings** during planning:
   ```bash
   python scripts/plan_autoexp.py ... 2>&1 | grep Warning
   ```

2. **Check output directory exists**:
   ```bash
   ls -la output/
   ```

3. **Check permissions**:
   ```bash
   ls -ld output/your_project/
   ```

4. **Run with verbose logging**:
   ```bash
   python scripts/plan_autoexp.py --verbose ...
   ```

## Automatic Action Execution

When monitoring detects events that trigger queued actions (like cooldown runs), the actions are **automatically executed** during each monitoring cycle. The monitoring loop now includes an integrated action worker that processes all pending actions after each check.

### How It Works

1. Monitoring detects an event (for example, checkpoint saved at iteration 100)
2. Event triggers actions based on conditions (for example, `checkpoint_iteration == 100`)
3. Actions are queued to `{monitoring_state_dir}/actions/`
4. **Automatically**, in the same monitoring cycle, the action worker executes all queued actions
5. Results are logged and event records are updated

### Manual Action Execution (Optional)

If monitoring is not running, you can manually execute queued actions:

```bash
# Execute all pending actions in the queue
python scripts/monitor_autoexp.py --cmd actions --manifest /path/to/manifest.json
```

### Check Action Queue Status

```bash
# List all queued actions (pending, running, done, failed)
python scripts/monitor_autoexp.py --cmd queue --manifest /path/to/manifest.json

# Show details of a specific action
python scripts/monitor_autoexp.py --cmd queue --queue-id <queue_id> --queue-show --manifest /path/to/manifest.json

# Retry a failed action
python scripts/monitor_autoexp.py --cmd queue --queue-id <queue_id> --queue-retry --manifest /path/to/manifest.json
```

### What This Means for You

- **No separate worker needed**: Just run monitoring as usual by way of `run_autoexp_container.py`
- **Actions execute immediately**: Cooldown runs start as soon as checkpoints are detected
- **Transparent**: You'll see "Processed N queued action(s)." messages in the monitoring output

## Avoiding Nested Monitoring Loops

### Problem

When `RunAutoexpAction` executed (for example, for cooldown runs), it started a new monitoring loop inside the already-running monitoring loop, creating nested loops that blocked the outer loop from continuing.

### Solution

Modified the workflow to submit cooldown jobs without starting nested monitoring:

1. **Added `no_monitor` flag to `RunAutoexpActionConfig`** (default: True)
   - Prevents nested monitoring loops by passing `--no-monitor` to the script

2. **Pass `session_id` in job metadata**
   - Allows cooldown actions to submit jobs to the same monitoring session
   - Pass `--plan-id <session_id>` to reuse the existing session

3. **Auto-restore new jobs in action worker**
   - After processing actions, `_process_action_queue()` calls `restore_jobs()`
   - Newly submitted jobs are automatically registered with the existing controller
   - All jobs monitored in a single session

### How It Works

```
Monitoring Loop (single)
  ├─> Original training job
  └─> Action execution: RunAutoexpAction
      ├─> run_autoexp_container.py --no-monitor --plan-id <session_id>
      ├─> Submits cooldown job to same session
      └─> Returns immediately (no nested loop)
  ├─> Action worker calls restore_jobs()
  ├─> Cooldown job registered with existing controller
  └─> Both jobs monitored together
```

### Files Modified

1. **`oellm_autoexp/monitor/actions.py`**:
   - Added `no_monitor: bool = True` to `RunAutoexpActionConfig`
   - Modified `RunAutoexpAction.execute()` to pass `--no-monitor` and `--plan-id` flags

2. **`oellm_autoexp/workflow/host.py`**:
   - Modified `_register_job()` to accept and store `session_id` in metadata
   - Modified `_process_action_queue()` to accept controller and runtime parameters
   - Added logic to restore new jobs after processing actions
   - Updated monitoring loop signatures to pass runtime parameter

## Next Steps

Once provenance is confirmed working:

1. Run a full training job with cooldown monitoring
2. Wait for cooldown to trigger (checkpoint save detected)
3. Actions execute automatically during the next monitoring cycle
4. Cooldown job submitted to same session (no nested monitoring)
5. Action worker restores cooldown job into existing controller
6. Both jobs monitored in single monitoring session
7. Verify paths are correct (load from original, save to new location)
8. Monitor both jobs' progress in the same monitoring output

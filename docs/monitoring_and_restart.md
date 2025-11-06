# Monitoring and Automatic Restart

This document describes the monitoring and automatic restart capabilities for handling training failures.

## Overview

The system provides:
1. **Log-based monitoring** - Detect errors, progress, and completion from SLURM logs
2. **Selective restart policies** - Automatically restart jobs on transient errors
3. **Signal extraction** - Capture error details (loss values, error messages, etc.)

## Monitoring Configurations

### Megatron-specific Monitors

Located in `config/monitoring/`:

#### `megatron_basic.yaml`
Basic monitoring for Megatron training runs:
- **Success detection**: validation completion, checkpoint saves
- **Progress tracking**: iteration updates with loss values
- **Error detection**: OOM, torch errors, Python exceptions
- **Use for**: Simple monitoring, testing

#### `megatron_nccl.yaml`
Extends `megatron_basic` with NCCL debugging:
- NCCL ERROR/WARN messages
- NCCL timeouts
- Communication errors
- Collective operation failures
- **Use for**: Debugging distributed training, runs with `NCCL_DEBUG=info`

#### `megatron_stall_detection.yaml`
Extends `megatron_basic` with aggressive stall detection:
- Reduced inactivity threshold (15 minutes)
- CUDA hang detection
- Deadlock detection
- ProcessGroup timeouts
- Gradient/loss NaN detection
- **Use for**: Detecting hangs between training iterations

#### `megatron_production.yaml` (Recommended)
Comprehensive production monitoring combining all features:
- Success: validation, checkpoints, progress
- Memory: CUDA/CPU OOM
- Distributed: NCCL errors, timeouts, communication
- Numerical: NaN/Inf in loss/gradients
- Hangs: CUDA hangs, deadlocks
- Generic: torch/Python errors
- **Use for**: Production training runs

## Restart Policies

### Policy Types

Located in `config/restart/`:

#### `default.yaml`
Simple always-restart policy:
- Restarts on all crashes (max 2 retries)
- Restarts on all stalls (max 3 retries)
- No restart on success

#### `megatron_transient.yaml` (Recommended)
Selective restart on transient errors only:

**Will restart on:**
- CUDA hangs, deadlocks
- NCCL errors and timeouts
- Distributed communication failures
- Watchdog timeouts

**Will NOT restart on:**
- OOM errors (requires config change)
- Python exceptions (code bugs)
- NaN/Inf (hyperparameter issue)

**Limits:**
- Max 3 crash restarts
- Max 2 stall restarts
- Max 1 timeout restart

#### `megatron_aggressive.yaml`
Restarts on most errors for maximum uptime:
- Restarts on almost everything except OOM and code bugs
- Max 5 crash restarts
- Max 3 stall restarts
- **Use for**: Long training runs where uptime is critical

#### `megatron_conservative.yaml`
Minimal restarts, requires human intervention:
- Only restarts on clear transient network/hang issues
- Max 2 crash restarts, 1 stall restart
- No restart on timeouts
- **Use for**: Production with strict control

## How It Works

### 1. Signal Detection

Monitors parse logs using regex patterns and detect signals:

```yaml
- name: cuda_oom
  pattern: '(?:CUDA out of memory).*Tried to allocate (?P<size>[\d.]+\s*[KMGT]iB)'
  pattern_type: regex
  state:
    class_name: CrashState  # Triggers crash policy
  actions:
    - class_name: ErrorNoteAction
      note: error
  metadata:
    severity: critical
    error_type: oom  # Used by SelectiveRestartPolicy
  extract_groups:
    allocation_size: size
```

### 2. SLURM State Monitoring

The monitor also tracks SLURM job state transitions and enriches them with metadata:

- **PENDING → RUNNING**: Emits `run_started` action
- **RUNNING → CANCELLED**: Classified as `crash` mode with `error_type: cancelled`, `subsystem: slurm`
- **RUNNING → FAILED**: Classified as `crash` mode with `error_type: slurm_failure`
- **RUNNING → COMPLETED**: Classified as `success` mode
- **Job not in queue**: Classified as `timeout` mode with `error_type: timeout`

All transitions are logged at INFO level with old and new states.

### 3. Policy Decision

When an event carries a `CrashState` (or SLURM classifies the job into a restartable mode), the restart policy decides:

```python
# SelectiveRestartPolicy checks:
if error_type in restart_on_error_types:
    return restart
elif error_type in exclude_error_types:
    return stop
else:
    return default_action
```

The policy logs its decision with action, reason, and attempt count.

### 4. Job Restart

If policy returns `restart`:
1. Log restart decision with reason
2. Cancel current job (`scancel <job_id>`)
3. Submit new job using same script
4. Log old job_id → new job_id
5. Increment attempt counter
6. Track new job_id

All operations are logged at INFO level for full visibility.

## Usage Examples

### Basic Usage

```yaml
defaults:
  - /monitoring: megatron_production
  - /restart: megatron_transient
```

### Custom Selective Policy

```yaml
# config/restart/my_policy.yaml
policies:
  - mode: crash
    implementation:
      class_name: SelectiveRestartPolicy
      max_retries: 5
      default_action: stop

      # Only restart on NCCL issues
      restart_on_error_types:
        - nccl
        - nccl_timeout

      # Never restart OOM
      exclude_error_types:
        - oom
```

### Monitor-Only (No Restart)

```yaml
defaults:
  - /monitoring: megatron_production
  # No restart config = NoRestartPolicy by default
```

## Monitored Error Types

The system detects and categorizes these error types:

| Error Type | Restartable? | Description |
|------------|--------------|-------------|
| `oom` | ❌ No | CUDA/CPU out of memory |
| `hang` | ✅ Yes | CUDA hang, training stuck |
| `deadlock` | ✅ Yes | Distributed deadlock |
| `nccl` | ✅ Yes | NCCL errors |
| `nccl_timeout` | ✅ Yes | NCCL timeout |
| `distributed_timeout` | ✅ Yes | ProcessGroup timeout |
| `communication` | ✅ Yes | Socket/connection errors |
| `watchdog_timeout` | ✅ Yes | Watchdog timeout |
| `cancelled` | ✅ Yes | Job cancelled (scancel, preemption) |
| `slurm_failure` | ✅ Yes | SLURM job failures |
| `numerical_instability` | ❌ No | NaN/Inf in loss/gradients |
| `exception` | ❌ No | Python exceptions (code bugs) |
| `torch` | ❌ No | PyTorch errors |

## Configuration Reference

### Monitoring Config Schema

```yaml
class_name: SlurmLogMonitor
poll_interval_seconds: 180           # How often to check logs
check_interval_seconds: 120          # Interval for inactivity detection
inactivity_threshold_seconds: 1200   # Max time without updates before "stall"

log_signals:
  - name: signal_name
    pattern: 'regex pattern with (?P<name>capture groups)'
    pattern_type: regex
    state:
      class_name: CrashState | SuccessState | StalledState | TimeoutState
    actions:
      - class_name: ErrorNoteAction | ExecutionAction | RestartAction | TerminationAction
        note: optional_tag
    metadata:
      error_type: oom | hang | nccl | ...
      subsystem: cuda | distributed | ...
      severity: critical | error | warning
    extract_groups:
      field_name: capture_group_name
```

### Restart Policy Schema

```yaml
class_name: SelectiveRestartPolicy
max_retries: 3
default_action: restart | stop

# Restart if any match
restart_on_error_types: [hang, nccl, ...]
restart_on_subsystems: [distributed, ...]
restart_on_signals: [cuda_hang, ...]

# Never restart if match
exclude_error_types: [oom, exception, ...]
```

## Best Practices

1. **Start conservative**: Use `megatron_transient` or `megatron_conservative` initially
2. **Monitor first**: Run with monitoring only, no restart, to understand failure patterns
3. **Check logs**: Review extracted metadata to tune restart rules
4. **Set max_retries**: Prevent infinite restart loops (recommended: 2-5)
5. **Exclude permanent errors**: Always exclude OOM and code bugs from restart
6. **Automate evaluations**: Select `monitoring=megatron_checkpoint_eval` to reuse the packaged preset that enqueues checkpoint evaluation commands by way of `ExecutionAction`.
7. **Separate policy from actions**: `RestartAction` augments the record of *why* a restart happened, but only the configured restart policy decides whether a job is resubmitted. Use the action for logging/automation, and let the policy enforce retry limits and exclusions.

## Troubleshooting

### Job keeps restarting on same error
- Check `max_retries` is set
- Verify error is in `exclude_error_types` if it's permanent
- Review logs to confirm error classification

### Job not restarting when expected
- Verify the event attaches a restartable state (`CrashState`, `StalledState`, `TimeoutState`)
- Check error_type matches `restart_on_error_types`
- Ensure `default_action: restart` if using catch-all

### False positive stall detection
- Increase `inactivity_threshold_seconds`
- Check training iterations aren't legitimately slow
- Verify log updates are being written regularly

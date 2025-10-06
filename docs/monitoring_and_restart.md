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
  action: error
  mode: crash  # Triggers crash policy
  metadata:
    severity: critical
    error_type: oom  # Used by SelectiveRestartPolicy
  extract_groups:
    allocation_size: size
```

### 2. Policy Decision

When a signal with `mode: crash` is detected, the restart policy decides:

```python
# SelectiveRestartPolicy checks:
if error_type in restart_on_error_types:
    return restart
elif error_type in exclude_error_types:
    return stop
else:
    return default_action
```

### 3. Job Restart

If policy returns `restart`:
1. Cancel current job (`scancel <job_id>`)
2. Submit new job using same script
3. Increment attempt counter
4. Track new job_id

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
    action: progress_update | error | warning | new_checkpoint | run_finished
    mode: crash | stall | success | null
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

## Troubleshooting

### Job keeps restarting on same error
- Check `max_retries` is set
- Verify error is in `exclude_error_types` if it's permanent
- Review logs to confirm error classification

### Job not restarting when expected
- Verify signal has `mode: crash` (or stall/timeout)
- Check error_type matches `restart_on_error_types`
- Ensure `default_action: restart` if using catch-all

### False positive stall detection
- Increase `inactivity_threshold_seconds`
- Check training iterations aren't legitimately slow
- Verify log updates are being written regularly

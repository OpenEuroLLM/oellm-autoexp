# Testing Guide for oellm-autoexp

## Integration Tests

The `scripts/test_auto_restart.py` script provides comprehensive integration tests for the auto-restart functionality.

### Running Tests

#### Basic Usage

```bash
# Test single job mode (default)
python scripts/test_auto_restart.py --scenario scancel

# Test array job mode
python scripts/test_auto_restart.py --scenario scancel --array-mode

# Run all scenarios
python scripts/test_auto_restart.py --scenario all

# Enable debug logging
python scripts/test_auto_restart.py --scenario scancel --debug
```

#### Test Scenarios

1. **scancel**: Tests restart after manual job cancellation
   ```bash
   python scripts/test_auto_restart.py --scenario scancel --iterations 2
   ```

2. **hang**: Tests restart on CUDA hang detection
   ```bash
   python scripts/test_auto_restart.py --scenario hang
   ```

3. **nccl**: Tests restart on NCCL error detection
   ```bash
   python scripts/test_auto_restart.py --scenario nccl
   ```

4. **oom**: Tests that OOM errors do NOT trigger restart (excluded error type)
   ```bash
   python scripts/test_auto_restart.py --scenario oom
   ```

5. **max_retries**: Tests that retry budget is respected
   ```bash
   python scripts/test_auto_restart.py --scenario max_retries
   ```

### Array Mode vs Single Job Mode

The `--array-mode` flag controls whether jobs are submitted as SLURM array jobs or individual jobs.

| Mode | Flag | SLURM Submission | Log Path Template |
|------|------|------------------|-------------------|
| **Single Job** (default) | None | Individual `sbatch` calls | `logs/{name}-%j.out` |
| **Array Job** | `--array-mode` | `sbatch --array=0-N` | `logs/{name}-%A_%a.out` |

**Key Differences:**

- **Single job mode**: Each job is submitted separately, easier to debug, simpler log paths
- **Array job mode**: All jobs submitted as one array, more efficient for large sweeps, requires proper array ID handling

**Recommendation**: Start with single job mode for testing, use array mode for production sweeps.

### Expected Output

When tests run successfully, you should see:

```
[11:23:45] ℹ Running tests in SINGLE JOB MODE

============================================================
TEST: Manual scancel (should restart)
============================================================

[11:23:45] ℹ Submitting test job (micro_bs=8, iters=1000)...
2025-10-08 11:23:56 [INFO] oellm_autoexp.monitor.controller: [job 12345] registered for monitoring: name=restart_test_mbs8, log_path=logs/restart_test_mbs8-12345.out, attempts=1
2025-10-08 11:24:06 [INFO] oellm_autoexp.monitor.controller: [job 12345] SLURM state transition: NONE -> RUNNING
[11:24:15] ✓ Job 12345 is RUNNING
[11:24:15] ℹ Cancelling job 12345...
2025-10-08 11:24:26 [INFO] oellm_autoexp.monitor.controller: [job 12345] SLURM state transition: RUNNING -> CANCELLED
2025-10-08 11:24:26 [INFO] oellm_autoexp.monitor.controller: [job 12345] classified as mode 'crash' (slurm_state=CANCELLED)
2025-10-08 11:24:26 [INFO] oellm_autoexp.monitor.controller: [job 12345] detected event 'crash' with state 'crash'
2025-10-08 11:24:26 [INFO] oellm_autoexp.monitor.controller: [job 12345] restarting job due to event 'crash' (attempt 1 -> 2)
[11:24:36] ✓ Job 12346 started (restart #1)
[11:24:36] ✓ Test passed: scancel restart works!
```

### Troubleshooting

#### No Monitor Events Logged

**Symptom**: Test runs but no monitor events appear in output

**Solution**: Ensure `--verbose` flag is passed (should be automatic in test script)

#### Job Not Found in squeue

**Symptom**: `classified as mode 'timeout' (slurm_state=None)`

**Causes**:
1. **Array mode mismatch**: Running with `--array-mode` but base config has `array: false` (or the other way around)
2. **Job completed too quickly**: Job finished before monitor could poll

**Solution**: Ensure array mode flag matches config, or increase `train_iters`

#### Log File Path Mismatch

**Symptom**: Monitor doesn't detect errors in log file

**Check**:
```bash
# Find actual log file location
find logs/ -name "*.log" -o -name "*.out"

# Check what path monitor is watching (look for this log line)
grep "registered for monitoring" <monitor_output>
```

**Solution**: Ensure log path template matches SLURM's output path:
- Single job: `logs/{name}-%j.out`
- Array job: `logs/{name}-%A_%a.out`

#### Environment Variable Not Expanded

**Symptom**: Error like `invalid int value: '$SLURM_NODEID'`

**Cause**: Shell variables in command arguments are being escaped/quoted, preventing expansion

**Solution**: This is fixed in the orchestrator (only array jobs need unescaped commands). If you see this error:
1. Make sure you're running the latest code
2. Verify the generated sbatch script has variables like `$SLURM_NODEID` without surrounding single quotes
3. Check that `escape_str=False` is used for array job script generation

## Unit Tests

Run unit tests to verify core functionality:

```bash
# Run all monitor tests
python -m pytest tests/unit/test_monitor.py -v

# Run specific test
python -m pytest tests/unit/test_monitor.py::test_cancelled_job_restarts_with_metadata_condition -v

# Run with coverage
python -m pytest tests/unit/test_monitor.py --cov=oellm_autoexp/monitor
```

## Cluster E2E Harness (real SLURM)

Use the Python harness under `scripts/tests/` to exercise the monitoring stack directly on your cluster while still benefiting from OmegaConf overrides:

```bash
# Basic scancel/restart check on JUWELS
python scripts/tests/run_cluster_monitoring.py \
  --scenario scancel \
  --config-ref experiments/megatron_with_auto_restart \
  --override container=juwels \
  --override slurm.sbatch.partition=$SLURM_PARTITION_DEBUG \
  --override project.name=juwels_monitor_e2e
```

Arguments:
- `--override` can be repeated to adapt for different clusters (LUMI/JUPITER/etc.).
- `--run-arg` / `--monitor-arg` forward extra CLI flags to `run_autoexp_container.py` or `monitor_autoexp.py`.
- `--no-monitor` skips automatic monitor startup if you want to attach manually.
- `--dry-run` prints the planned commands without executing them.

## Quick Verification

To quickly verify the monitoring system is working:

```bash
# Run the demo script
PYTHONPATH=. python scripts/demo_monitor_logging.py
```

This demonstrates the logging in a controlled environment without requiring SLURM.

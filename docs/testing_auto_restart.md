# Testing Auto-Restart System

## Quick Start

```bash
# Make sure you're in the project root
cd /home/korbip/Work/OELLM/Projects/oellm-autoexp

# Run a simple test (manual scancel)
python scripts/test_auto_restart.py --scenario scancel

# Run all tests
python scripts/test_auto_restart.py --scenario all
```

## Available Test Scenarios

### 1. `scancel` - Manual Job Cancellation
Tests basic restart on `scancel`:
- Submits a job with monitoring enabled
- Waits for it to start running
- Cancels it manually
- Verifies a new job is automatically submitted
- Repeats multiple times to test reliability

```bash
python scripts/test_auto_restart.py --scenario scancel --iterations 2
```

**Expected behavior:**
- ✅ Job restarts after each `scancel`
- ✅ New job ID is returned
- ✅ Restart happens within 2 minutes

### 2. `hang` - CUDA Hang Detection
Tests restart on CUDA hang:
- Submits a job
- Injects a CUDA device-side assert message into the log
- Verifies the monitor detects it and restarts

```bash
python scripts/test_auto_restart.py --scenario hang
```

**Expected behavior:**
- ✅ Monitor detects CUDA hang pattern
- ✅ Classified as `error_type: hang`
- ✅ Job automatically restarts
- ✅ Restart reason: "Transient failure detected (error_type=hang)"

### 3. `nccl` - NCCL Error Detection
Tests restart on NCCL errors:
- Submits a job
- Injects an NCCL error message into the log
- Verifies restart behavior

```bash
python scripts/test_auto_restart.py --scenario nccl
```

**Expected behavior:**
- ✅ Monitor detects NCCL error
- ✅ Job automatically restarts
- ✅ Restart reason includes "error_type=nccl"

### 4. `oom` - OOM Should NOT Restart
Tests that OOM errors do NOT trigger restart:
- Submits a job with `micro_batch_size=16` (known to OOM)
- Job should fail with OOM
- Verifies NO restart happens

```bash
python scripts/test_auto_restart.py --scenario oom
```

**Expected behavior:**
- ❌ NO restart (OOM is in exclude list)
- ✅ Job stops permanently
- ✅ Stop reason: "Permanent failure detected (excluded error_type: oom)"

### 5. `max_retries` - Retry Budget Limit
Tests that restart count is limited:
- Submits a job
- Cancels it 4 times
- First 3 times should restart
- 4th time should NOT restart (budget exhausted)

```bash
python scripts/test_auto_restart.py --scenario max_retries
```

**Expected behavior:**
- ✅ Restarts attempts 1-3
- ❌ Does NOT restart attempt 4
- ✅ Stop reason: "retry budget exhausted (attempt 4/3)"

### 6. `all` - Run All Tests
Runs all scenarios sequentially:

```bash
python scripts/test_auto_restart.py --scenario all
```

## Test Configuration Used

The test script uses this configuration:
```yaml
backend: megatron_torchrun
backend.megatron: llama1_8b_qkln
slurm: juwels
monitoring: megatron_production
restart: megatron_transient
```

Key parameters:
- `micro_batch_size=8`: Works fine
- `micro_batch_size=16`: Causes OOM (for OOM test)
- `train_iters=100-1000`: Varies by test
- `max_retries=3`: For crash mode

## Understanding the Output

### Success Example
```
[10:30:15] ℹ Running: python scripts/submit_autoexp.py ...
[10:30:20] ✓ Submitted job 12345
[10:30:25] ℹ Waiting for job 12345 to reach state RUNNING...
[10:30:40] ✓ Job 12345 reached state RUNNING
[10:30:45] ℹ Cancelling job 12345...
[10:30:46] ✓ Cancelled job 12345
[10:30:50] ℹ Checking for restart of job 12345...
[10:31:10] ✓ Found restarted job: 12346
[10:31:15] ✓ Test passed: scancel restart works!
```

### Failure Example
```
[10:30:45] ℹ Checking for restart of job 12345...
[10:32:45] ⚠ No restart detected within 120s
[10:32:45] ✗ No restart detected!
[10:32:45] ✗ scancel: FAILED
```

## Troubleshooting

### Test hangs at "Waiting for job to start"
**Cause:** Job stuck in queue or pending state
**Solution:**
- Check SLURM queue: `squeue -u $USER`
- Check partition availability: `sinfo`
- Verify account/QOS: Check `config/slurm/juwels.yaml`

### No restart detected
**Cause:** Monitor loop not running
**Solution:**
- The test script runs `submit_autoexp.py` without `--no-monitor`; ensure that flag wasn't provided
- If you submitted separately, start `oellm-autoexp monitor-session --session-id <id>` for the recorded session
- Verify the monitoring config is not `NullMonitor`

### Job fails immediately
**Cause:** Config error or missing dependencies
**Solution:**
- Check the job log: `cat logs/*<job_id>*.out`
- Verify container path in `config/slurm/juwels.yaml`
- Check that Megatron config is valid

### OOM test shows restart (incorrect!)
**Cause:** OOM not being detected or not in exclude list
**Solution:**
- Check monitoring config has `cuda_oom` signal
- Verify the `crash` state-event binding excludes `error_type: oom` (or limits retries by way of `MaxAttemptsCondition`)
- Review monitor logs to see what was detected

### "Could not find log file"
**Cause:** Log path doesn't match pattern or job hasn't written logs yet
**Solution:**
- Wait longer before injecting errors (increase sleep time)
- Check log directory: `ls logs/`

## Manual Testing (Alternative)

If the automated script has issues, you can test manually:

### Manual Test 1: Basic Restart
```bash
# Terminal 1: Plan, submit, and monitor
python scripts/plan_autoexp.py \
  --config-ref experiments/megatron_with_auto_restart \
  --manifest output/manual_plan.json

# In the same terminal (continues monitoring)
python scripts/submit_autoexp.py \
  --manifest output/manual_plan.json

# Terminal 2: Cancel the job
squeue -u $USER  # Get job ID
scancel <job_id>

# Watch Terminal 1 for restart message
```

### Manual Test 2: Inject Error Pattern
```bash
# Terminal 1: Plan, submit, and monitor
python scripts/plan_autoexp.py \
  --config-ref experiments/megatron_with_auto_restart \
  --manifest output/manual_plan.json

python scripts/submit_autoexp.py \
  --manifest output/manual_plan.json

# Terminal 2: Inject error and watch
squeue -u $USER  # Get job ID
echo "[default0]:NCCL ERROR timeout" >> logs/<job_id>.out

# Watch Terminal 1 for restart
```

## Expected Test Duration

- `scancel` (2 iterations): ~10-15 minutes
- `hang`: ~5-8 minutes
- `nccl`: ~5-8 minutes
- `oom`: ~3-5 minutes (waits 60s for OOM to occur)
- `max_retries`: ~20-25 minutes (4 restarts)
- `all`: ~45-60 minutes

## Checking Results

After tests complete, you can verify:

```bash
# Check state store (tracks restarts)
cat .oellm_state/monitor_state.json | jq

# Check SLURM accounting
sacct -u $USER --starttime today --format=JobID,JobName,State,ExitCode

# Check logs
ls -lht logs/
tail -100 logs/<latest>.out
```

## CI/CD Integration

To run in CI:

```bash
# Quick smoke test (just scancel once)
python scripts/test_auto_restart.py --scenario scancel --iterations 1

# Full test suite
python scripts/test_auto_restart.py --scenario all
```

Exit codes:
- `0`: All tests passed
- `1`: One or more tests failed

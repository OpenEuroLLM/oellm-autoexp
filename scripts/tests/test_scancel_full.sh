#!/bin/bash
# Complete test of scancel detection and monitoring

set -e

echo "========================================="
echo "Full scancel detection test"
echo "========================================="
echo ""

# Clean up any previous state
echo "1. Cleaning up previous test artifacts..."
rm -rf output/monitoring_state/
rm -f logs/run_autoexp_submit.log
rm -f logs/monitor_output.log
rm -f logs/monitor_test_*.log
echo "   Done"
echo ""

# Submit a simple test job (without monitoring)
echo "2. Submitting test job..."
MANIFEST_PATH="output/scancel_test_plan.json"
python scripts/run_autoexp_container.py \
  --verbose \
  --no-monitor \
  --manifest "$MANIFEST_PATH" \
  --config-ref experiments/megatron_with_auto_restart \
  -C config \
  container=juwels \
  backend.megatron.micro_batch_size=1 \
  backend.megatron.train_iters=1000 \
  backend.env.PYTHONPATH=.:submodules/Megatron-LM \
  slurm.sbatch.time=00:30:00 \
  slurm.sbatch.partition=$SLURM_PARTITION_DEBUG \
  project.name=scancel_test \
  slurm.log_dir=logs \
  slurm.array=false \
  monitoring.poll_interval_seconds=5 \
  monitoring.check_interval_seconds=5 \
  > logs/run_autoexp_submit.log 2>&1

echo "   Submission complete"
echo ""

# Parse job ID from submission output
echo "3. Extracting job ID from submission..."

JOB_ID=""
if grep -q "submitted.*" logs/run_autoexp_submit.log 2>/dev/null; then
  JOB_ID=$(grep "submitted.*" logs/run_autoexp_submit.log | head -1 | sed 's/.* job \([0-9_]*\).*/\1/')
fi

if [ -z "$JOB_ID" ]; then
  echo "   ERROR: Could not find job ID in output"
  echo "   Output:"
  cat logs/run_autoexp_submit.log
  exit 1
fi

echo "   Job submitted: $JOB_ID"
echo ""

# Start monitoring in background using plan manifest
echo "4. Starting monitoring process (manifest)..."
python scripts/monitor_autoexp.py \
  --manifest "$MANIFEST_PATH" \
  --verbose \
  > logs/monitor_output.log 2>&1 &

MONITOR_PID=$!
echo "   Monitor started (PID: $MONITOR_PID)"
echo ""

# Wait for job to start running
echo "5. Waiting for job to start running..."
for i in {1..60}; do
  STATE=$(squeue -j "$JOB_ID" -h -o "%T" 2>/dev/null || echo "NOT_FOUND")
  echo "   Job state: $STATE"
  if [ "$STATE" = "RUNNING" ]; then
    break
  fi
  sleep 2
done

if [ "$STATE" != "RUNNING" ]; then
  echo "   WARNING: Job never reached RUNNING state"
fi
echo ""

# Give monitor a chance to detect RUNNING state
echo "6. Waiting 10 seconds for monitor to poll..."
sleep 10
echo "   Done"
echo ""

# Cancel the job
echo "7. Cancelling job $JOB_ID..."
scancel "$JOB_ID"
SCANCEL_EXIT=$?
echo "   scancel exit code: $SCANCEL_EXIT"
echo ""

# Check job state immediately and over time
echo "8. Checking job state after cancellation:"
for i in 0 2 5 10; do
  if [ $i -gt 0 ]; then
    sleep $i
  fi
  STATE=$(squeue -j "$JOB_ID" -h -o "%T" 2>&1)
  echo "   After ${i}s: $STATE"
done
echo ""

# Check sacct
echo "9. Checking sacct for job state:"
sacct -j "$JOB_ID" --format=JobID,State,ExitCode --parsable2 2>&1 | head -10
echo ""

# Wait for monitor to detect the cancellation (should poll every 5 seconds)
echo "10. Waiting 20 seconds for monitor to detect cancellation and potentially restart..."
sleep 20
echo ""

# Check if a new job was submitted
echo "11. Checking for restart (looking for new jobs with name 'scancel_test'):"
squeue -u $USER -h -o "%i %j" | grep scancel_test || echo "   No jobs found"
echo ""

# Check monitor logs
echo "12. Checking monitor output (last 30 lines):"
if [ -f "logs/monitor_output.log" ]; then
  tail -30 logs/monitor_output.log
else
  echo "   No monitor log found"
fi
echo ""

# Check for CANCELLED detection in logs
echo "13. Searching for CANCELLED in monitor output:"
grep -i "cancelled\|slurm_state" logs/monitor_output.log 2>/dev/null | tail -10 || echo "   No matches found"
echo ""

# Check for restart attempts
echo "14. Searching for restart attempts in monitor output:"
grep -i "restart\|resubmit" logs/monitor_output.log 2>/dev/null | tail -10 || echo "   No matches found"
echo ""

# Check state persistence
echo "15. Checking persisted state:"
echo "   Listing monitoring sessions:"
python scripts/manage_monitoring.py --monitoring-state-dir output/monitoring_state list
echo ""

if [ -d "output/monitoring_state" ]; then
  echo "   Session files:"
  ls -la output/monitoring_state/
else
  echo "   No monitoring_state directory found"
fi
echo ""

# Clean up
echo "16. Cleaning up monitor process..."
kill $MONITOR_PID 2>/dev/null || true
echo "   Done"
echo ""

# Cancel any remaining jobs
echo "17. Cancelling any remaining scancel_test jobs..."
squeue -u $USER -h -o "%i %j" | grep scancel_test | awk '{print $1}' | xargs -r scancel 2>/dev/null || true
echo "   Done"
echo ""

echo "========================================="
echo "Test complete!"
echo "========================================="
echo ""
echo "Summary:"
echo "  - Original job: $JOB_ID"
echo "  - Check logs/run_autoexp_submit.log for submission output"
echo "  - Check logs/monitor_output.log for full monitor output"
echo "  - Look for 'classified as mode' to see how the job was classified"
echo "  - Look for \"restarting job due to event\" to see restart logic"

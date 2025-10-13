#!/bin/bash
# Diagnostic script to test scancel behavior and monitor detection

JOB_ID="$1"
if [ -z "$JOB_ID" ]; then
    echo "Usage: $0 <job_id>"
    echo "Example: $0 12583749"
    exit 1
fi

echo "========================================="
echo "Diagnostic: Job $JOB_ID"
echo "========================================="
echo ""

echo "1. Job state BEFORE scancel:"
squeue -j $JOB_ID -h -o "%i %T" 2>&1
echo ""

echo "2. Running: scancel $JOB_ID"
scancel $JOB_ID
SCANCEL_EXIT=$?
echo "   scancel exit code: $SCANCEL_EXIT"
echo ""

echo "3. Job state IMMEDIATELY after scancel:"
squeue -j $JOB_ID -h -o "%i %T" 2>&1
echo ""

sleep 2
echo "4. Job state 2 seconds after scancel:"
squeue -j $JOB_ID -h -o "%i %T" 2>&1
echo ""

sleep 3
echo "5. Job state 5 seconds after scancel:"
squeue -j $JOB_ID -h -o "%i %T" 2>&1
echo ""

sleep 5
echo "6. Job state 10 seconds after scancel:"
squeue -j $JOB_ID -h -o "%i %T" 2>&1
echo ""

echo "7. Checking sacct for job history:"
sacct -j $JO_ID --format=JobID,State,ExitCode --parsable2 2>&1
echo ""

echo "8. Checking sacct with brief format:"
sacct -j $JOB_ID -b 2>&1
echo ""

echo "9. Looking for log files:"
find logs -name "*${JOB_ID}*" -type f 2>/dev/null | head -5
echo ""

echo "10. Checking if monitor is running:"
ps aux | grep -i "run_autoexp\|monitor" | grep -v grep
echo ""

echo "========================================="
echo "Diagnostic complete"
echo "========================================="

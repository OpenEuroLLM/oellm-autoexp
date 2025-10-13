#!/bin/bash
# Check monitor log files to see what the monitor is actually detecting

echo "========================================="
echo "Monitor Log Analysis"
echo "========================================="
echo ""

echo "1. Finding monitor log files:"
find logs -name "monitor_*.log" -type f 2>/dev/null
echo ""

echo "2. Last 50 lines of most recent monitor log:"
LATEST_LOG=$(find logs -name "monitor_*.log" -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
if [ -n "$LATEST_LOG" ]; then
    echo "   Log file: $LATEST_LOG"
    echo "   ----------------------------------------"
    tail -50 "$LATEST_LOG"
else
    echo "   No monitor log files found"
fi
echo ""

echo "3. Searching for CANCELLED in monitor logs:"
grep -h "CANCELLED" logs/monitor_*.log 2>/dev/null | tail -10
echo ""

echo "4. Searching for 'restart' in monitor logs:"
grep -h "restart" logs/monitor_*.log 2>/dev/null | tail -10
echo ""

echo "5. Searching for 'classified as mode' in monitor logs:"
grep -h "classified as mode" logs/monitor_*.log 2>/dev/null | tail -10
echo ""

echo "6. Searching for 'policy decision' in monitor logs:"
grep -h "policy decision" logs/monitor_*.log 2>/dev/null | tail -10
echo ""

echo "7. Searching for 'sacct' in monitor logs:"
grep -h "sacct" logs/monitor_*.log 2>/dev/null | tail -10
echo ""

echo "8. Checking state.json for persisted jobs:"
if [ -f "output/.oellm-autoexp/state.json" ]; then
    echo "   Found: output/.oellm-autoexp/state.json"
    cat output/.oellm-autoexp/state.json
elif [ -f ".oellm-autoexp/state.json" ]; then
    echo "   Found: .oellm-autoexp/state.json"
    cat .oellm-autoexp/state.json
else
    echo "   No state.json found"
fi
echo ""

echo "========================================="
echo "Analysis complete"
echo "========================================="

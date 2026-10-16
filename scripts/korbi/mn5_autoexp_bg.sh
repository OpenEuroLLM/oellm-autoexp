#!/bin/bash
# scripts/korbi/mn5_autoexp.sh, detached from the terminal (survives the ssh session), for
# the long-running monitor on the MareNostrum login node:
#
#   scripts/korbi/mn5_autoexp_bg.sh <name> scripts/run_autoexp.py --config-name ... [overrides]
#   scripts/korbi/mn5_autoexp_bg.sh <name> scripts/monitor_autoexp.py --session <id>
#
# Output: $R/monitor_logs/<name>.log (pid in <name>.pid). Stop a monitor with kill <pid>.
set -euo pipefail
R=${R:-/gpfs/projects/ehpc533/users/korbip/hybrid_v019}
name=$1; shift
mkdir -p "$R/monitor_logs"
cd "$(dirname "$0")/../.."
setsid nohup scripts/korbi/mn5_autoexp.sh "$@" > "$R/monitor_logs/$name.log" 2>&1 < /dev/null &
echo $! > "$R/monitor_logs/$name.pid"
echo "started $name (pid $!), log $R/monitor_logs/$name.log"

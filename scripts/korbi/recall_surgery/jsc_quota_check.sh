#!/bin/bash
# Save-path pre-flight of the 84B JUPITER runs (docs/recall_surgery_ambitious.md §19.1): exit 0 only if the file system
# holding DIR has at least NEED_TB free INSIDE ITS QUOTA. Used as a pool start condition (ShellCommandCondition, run by
# the pool monitor on the login node) so a run does not start, train for an hour and die at its first save with
# "[Errno 122] Disk quota exceeded" (smoke_b, 2026-11-15: fscratch full after 738 of 1081 GB).
#
# Why df: on JUPITER the project directories are GPFS filesets with a quota, and `df` on a path inside one reports the
# FILESET (size = quota, avail = what is left: /e/fscratch/e-sta-openeurollm showed 110T / 110T / 5.4G at the failure),
# while `jutil project dataquota` showed fscratch "0 used" (stale / not accounted). df is the number that bites.
#   jsc_quota_check.sh DIR NEED_TB
set -uo pipefail
dir=${1:?dir}; need_tb=${2:?need TB}
d=$dir; while [ ! -e "$d" ]; do d=$(dirname "$d"); done
avail_kb=$(df -Pk "$d" | awk 'NR == 2 {print $4}')
[ -n "$avail_kb" ] || { echo "df failed on $d" >&2; exit 2; }
awk -v a="$avail_kb" -v n="$need_tb" -v d="$d" 'BEGIN {
  tb = a / 1024 / 1024 / 1024
  printf "%s: %.2f TiB free in quota, need %.2f\n", d, tb, n
  exit (tb >= n) ? 0 : 1 }'

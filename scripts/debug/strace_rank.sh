#!/bin/bash
# Trace file-related syscalls of one rank; all other ranks run unchanged.
#
# Usage (as the Megatron python command):
#   backend.python_cmd="bash scripts/debug/strace_rank.sh python"
#
# Env:
#   STRACE_RANK  rank to trace (default 0)
#   STRACE_DIR   where the final trace is copied (default: $PWD/strace)
#
# The trace is written to node-local /tmp while running (so tracing itself does
# not add load to Lustre) and copied to STRACE_DIR when the process exits.

rank="${SLURM_PROCID:-${RANK:-0}}"
if [ "$rank" != "${STRACE_RANK:-0}" ]; then
  exec "$@"
fi
if ! command -v strace >/dev/null 2>&1; then
  echo "strace_rank: strace not found in container; running untraced" >&2
  exec "$@"
fi

tmp_out="/tmp/strace_${SLURM_JOB_ID:-local}_rank${rank}.txt"
dest="${STRACE_DIR:-$PWD/strace}"

save_trace() {
  mkdir -p "$dest" && cp "$tmp_out" "$dest/" && rm -f "$tmp_out"
  echo "strace_rank: trace saved to $dest/$(basename "$tmp_out")" >&2
}
# Slurm sends SIGTERM on time limit / scancel; keep the trace in that case too.
trap 'wait; save_trace; exit 143' TERM INT

strace -f -tt -e trace=%file -o "$tmp_out" "$@" &
wait $!
status=$?
trap - TERM INT
save_trace
exit $status

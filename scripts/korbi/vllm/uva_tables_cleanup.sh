# Source from a vLLM eval runner. With OELLM_ENGRAM_TABLES=uva the Engram tables are staged into
# shared memory (~160 GB RAM per node for the ambitious v4 exports) and stay there after the
# processes end: remove stale ones of a killed job now and this job's at exit (pool steps run
# inside a long host job, so no Slurm epilog cleans them before the next run on the node).
if [ "${OELLM_ENGRAM_TABLES:-}" = uva ]; then
  _oellm_uva_dir=${OELLM_ENGRAM_UVA_DIR:-/dev/shm}
  rm -f "$_oellm_uva_dir"/oellm_engram_* 2>/dev/null || true
  trap 'rm -f "$_oellm_uva_dir"/oellm_engram_* 2>/dev/null || true' EXIT
  trap 'exit 143' TERM
fi

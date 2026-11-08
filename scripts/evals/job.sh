#!/bin/bash
# One job of a typed eval launch (scripts/evals/launch.py) inside the MN5 image: the runtime, then
# scripts/evals/run_job.py with the job's JobSpec JSON (all its settings; nothing else is passed).
#
#   job.sh <runtime> <job spec json>
#
#   vllm   the node-local vLLM install of the eval engines (setup_vllm_uv.sh: INFER_PY, DEST,
#          PYTHONPATH, the offline HF_HOME / NLTK data)
#   plain  the image's python (the export job)
set -euo pipefail
runtime=$1 spec=$2
cd "$(dirname "$0")/../.."
case "$runtime" in
  vllm)
    source scripts/korbi/vllm/setup_vllm_uv.sh
    exec "$INFER_PY" scripts/evals/run_job.py "$spec" ;;
  plain)
    exec python3 scripts/evals/run_job.py "$spec" ;;
  *) echo "[job.sh] unknown runtime $runtime" >&2; exit 2 ;;
esac

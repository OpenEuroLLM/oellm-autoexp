#!/bin/bash
# One pool job of a typed eval launch (scripts/evals/launch.py) inside the MN5 image: the runtime,
# then the job's program with the paths the launch rendered (nothing else is passed).
#
#   job.sh worker <plan.json> <unit id>   an oellm-eval unit (oellm.runner.worker) on the node-local
#                                         vLLM install (setup_vllm_uv.sh: INFER_PY, DEST, PYTHONPATH,
#                                         the offline HF_HOME / NLTK data); runtime.json beside plan.json
#   job.sh plain <export.json> -          the export job (scripts/evals/run_job.py) on the image's python
set -euo pipefail
runtime=$1 spec=$2
cd "$(dirname "$0")/../.."
case "$runtime" in
  worker)
    unit=${3:?job.sh worker <plan.json> <unit id>}
    source scripts/korbi/vllm/setup_vllm_uv.sh
    export PYTHONPATH="$PWD/submodules/oellm-eval:$PYTHONPATH"
    # the code identity in every provenance record (the snapshot's commits + uncommitted-diff hashes)
    [ -f SYNC_STAMP ] && export OELLM_CODE_STAMP="$(cat SYNC_STAMP)"
    # --missing-ok: the pool would retry a failed job, and the same call fails the same way; rows
    # without a result keep no done marker (a relaunch reruns them, oellm.reports.check lists them)
    exec "$INFER_PY" -m oellm.runner.worker --plan "$spec" --runtime "$(dirname "$spec")/runtime.json" \
      --unit "$unit" --missing-ok ;;
  plain)
    exec python3 scripts/evals/run_job.py "$spec" ;;
  vllm)
    echo "[job.sh] runtime vllm (a JobSpec eval job) is gone: relaunch with scripts/evals/launch.py" >&2; exit 2 ;;
  *) echo "[job.sh] unknown runtime $runtime" >&2; exit 2 ;;
esac

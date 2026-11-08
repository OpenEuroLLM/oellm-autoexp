#!/bin/bash
# An HF-export probe of the midtraining campaign (docs/campaign_3b_hybrid_midtraining.md) on one MN5 node,
# with the fixed validation sets (v8ml-valid) as data: router_probe.py (R0 / R2), ced_probe.py (CED retrofit).
# Launched by config/experiments/korbi/campaign_3b_hybrid_midtraining/{r0_r2_router,ced}_probe_mn5.yaml.
#
#   run_hf_probe_mn5.sh <probe script> <export dir> <out dir> [probe args...]
set -uo pipefail
PROBE=$1
BASE=$2
OUT=$3
shift 3
source scripts/korbi/vllm/setup_vllm_uv.sh
export CC=${CC:-gcc}
mkdir -p "$OUT"
{ hostname; nvidia-smi -L; free -g; cat SYNC_STAMP 2>/dev/null; } > "$OUT/node.txt" 2>&1
cat "$OUT/node.txt"
# an export view: hard links to the immutable weights; this checkout's model code (atomic replace)
VIEW=${BASE}_probe
[ -d "$VIEW" ] || cp -al "$BASE" "$VIEW"
for f in oellm_ambitious/configuration_oellm_ambitious.py oellm_ambitious/modeling_oellm_ambitious.py \
         oellm_ambitious/mhc_kernels.py oellm_ambitious/oellm_kernels.py \
         oellm_hybrid/configuration_oellm_hybrid.py oellm_hybrid/modeling_oellm_hybrid.py; do
  cp "oellm_autoexp/hf_export/$f" "$VIEW/$(basename $f).new" && mv "$VIEW/$(basename $f).new" "$VIEW/$(basename $f)"
done
D=/gpfs/projects/ehpc533/users/korbip/data/v8ml-valid
DATA_ARGS=()
case "$PROBE" in
  router_probe.py|ced_probe.py)  # the fixed validation sets as data
    DATA=""
    for s in web_en multilingual code math reasoning long_64k; do DATA="$DATA${DATA:+,}$s=$D/${s}_text_document"; done
    DATA_ARGS=(--data "$DATA" --long long_64k) ;;
esac
"$INFER_PY" "scripts/korbi/campaign_midtraining/$PROBE" --model "$VIEW" "${DATA_ARGS[@]}" \
    --out "$OUT" "$@" 2>&1 | grep -v "UserWarning\|warnings.warn"
status=${PIPESTATUS[0]}
echo "PROBE_DONE $PROBE status=$status"
exit $status

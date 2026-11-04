#!/bin/bash
# R0 + R2 of the midtraining campaign (docs/campaign_3b_hybrid_midtraining.md §5) on one MN5 node:
# router predictability and the expert skip policy on an HF export of v4_2ph (router_probe.py), with the
# fixed validation sets (v8ml-valid) as data. Launched by
# config/experiments/korbi/campaign_3b_hybrid_midtraining/r0_r2_router_probe_mn5.yaml.
#
#   run_r0_r2_mn5.sh <export dir> <out dir> [router_probe.py args...]
set -uo pipefail
BASE=$1
OUT=$2
shift 2
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
DATA=""
for s in web_en multilingual code math reasoning long_64k; do DATA="$DATA${DATA:+,}$s=$D/${s}_text_document"; done
"$INFER_PY" scripts/korbi/campaign_midtraining/router_probe.py --model "$VIEW" --data "$DATA" --long long_64k \
    --out "$OUT" "$@" 2>&1 | grep -v "UserWarning\|warnings.warn"
status=${PIPESTATUS[0]}
echo "R0R2_DONE status=$status"
exit $status

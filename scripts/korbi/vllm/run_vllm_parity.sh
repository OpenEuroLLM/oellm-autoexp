#!/bin/bash
# vLLM port of oellm_ambitious against its HF export on MareNostrum 5 (config/experiments/korbi/
# production/vllm_parity_mn5.yaml): installs vLLM into node-local /tmp/pylibs_vllm_<sha> first, then
#   run_vllm_parity.sh tiny [<out dir>]            tiny random models (mHC / LDM), one GPU
#   run_vllm_parity.sh export <export dir> [<tp>]  an existing export
set -euo pipefail
R=${R:-/gpfs/projects/ehpc533/users/korbip/hybrid_v019}
cd "$(dirname "$0")/../../.."
SHA=$(grep -m1 "^submodules/vllm " SYNC_STAMP | awk '{print $3}')
export DEST=/tmp/pylibs_vllm_$SHA
bash scripts/korbi/vllm/build_pylibs_vllm.sh
export PYTHONPATH="$DEST:$PWD:$PWD/flash-linear-attention"
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-WARNING} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mode=$1; shift
case $mode in
  tiny)   python3 -m oellm_autoexp.hf_export.vllm_parity tiny --out "${1:-/tmp/vllm_parity_tiny}" ;;
  export) python3 -m oellm_autoexp.hf_export.vllm_parity export "$1" --tp "${2:-1}" ;;
  *) echo "unknown mode $mode" >&2; exit 2 ;;
esac

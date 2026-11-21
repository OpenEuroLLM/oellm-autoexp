#!/bin/bash
# eval_pack_mn5.sh after refreshing the given exports' model code once (refresh_export needs torch: the job's container,
# not the login node). The first node to take the lock refreshes, the others wait for it; a marker per export and
# Megatron commit makes reruns skip it.
#
#   REFRESH_EXPORTS=<export dir>[:<export dir>...] refresh_then_pack_mn5.sh <queue dir> <arm:task[:step]> ...
# (an env variable, colon-separated: commas in a Hydra override are a sweep, and a space breaks the container's
# --env argument)
set -u
exports=${REFRESH_EXPORTS//:/ }
Q=$1
mkdir -p "$Q"
rev=$(git -C submodules/Megatron-LM rev-parse --short HEAD 2>/dev/null || head -c 12 SYNC_STAMP)
(
    flock 9
    for d in $exports; do
        if [ ! -f "$d/.refreshed_$rev" ]; then
            python -m oellm_autoexp.hf_export.refresh_export "$d" && touch "$d/.refreshed_$rev"
        fi
    done
) 9>"$Q/.refresh.lock"
exec bash scripts/korbi/campaign_midtraining/eval_pack_mn5.sh "$@"

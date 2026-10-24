#!/bin/bash
# Hard-link a finished run's rolling checkpoint into a branch arm's ckpt_rolling (no copy, no
# space; Megatron only adds new iteration directories), so the branch resumes there. On MN5:
#   link_branch_ckpt.sh <source suite/arm> <branch suite/arm> <iteration>
# paths relative to the campaign_3b run directory.
set -euo pipefail
R=${CAMPAIGN_RUNS:-/gpfs/scratch/ehpc533/users/korbip/hybrid_v019_runs/autoexp/campaign_3b}
src="$R/$1/ckpt_rolling" dst="$R/$2/ckpt_rolling" it=$(printf "iter_%07d" "$3")
[ -d "$src/$it" ] || { echo "missing $src/$it" >&2; exit 1; }
[ -e "$dst/$it" ] && { echo "exists: $dst/$it"; exit 0; }
mkdir -p "$dst"
cp -al "$src/$it" "$dst/$it"
printf "%s" "$3" > "$dst/latest_checkpointed_iteration.txt"
echo "linked $src/$it -> $dst/$it"

#!/bin/bash
# Mirror this checkout (with the Megatron-LM and flash-linear-attention working trees) to
# MareNostrum 5 (ehpc533). All edits happen locally; the cluster copy is overwritten.
#
#   ./sync_to_marenostrum.sh             -> $R/autoexp           (the live copy)
#   ./sync_to_marenostrum.sh m13         -> $R/autoexp_m13       (a frozen snapshot for a job,
#                                          so later syncs don't change code under a running job)
#
# No -u (a newer file on the cluster must not survive) and --delete (removed files go), so
# the remote tree equals the local one. SYNC_STAMP records the commits and the hash of the
# uncommitted diff of every repo; compare it with `cat $R/autoexp*/SYNC_STAMP`.
set -euo pipefail
HOST=${MN_HOST:-marenostrum_new}
R=/gpfs/projects/ehpc533/users/korbip/hybrid_v019
DEST=$R/autoexp${1:+_$1}
cd "$(dirname "$0")"

stamp() {
    for repo in . submodules/Megatron-LM flash-linear-attention submodules/cute-triadic-gdn; do
        printf '%-28s %s %s dirty=%s\n' "$repo" \
            "$(git -C "$repo" rev-parse --abbrev-ref HEAD)" \
            "$(git -C "$repo" rev-parse --short HEAD)" \
            "$( (git -C "$repo" diff HEAD; git -C "$repo" ls-files --others --exclude-standard | grep -vx SYNC_STAMP) | sha1sum | cut -c1-12)"
    done
}
STAMP=$(stamp)
echo "$STAMP" > SYNC_STAMP

rsync -a --delete --stats \
    --exclude .git --exclude __pycache__ --exclude '*.pyc' --exclude .pytest_cache \
    --exclude dump/ --exclude outputs/ --exclude monitor_state --exclude '*.sif' \
    --exclude 'build/' --exclude '*.egg-info' \
    ./ "$HOST:$DEST/" 2>&1 | grep -E '^(Number of (created|deleted|regular) files|Total transferred file size)' || true

echo "synced to $HOST:$DEST"
remote=$(ssh "$HOST" "cat $DEST/SYNC_STAMP" 2>/dev/null | grep -E '^(\.|submodules|flash)')
if [ "$remote" = "$STAMP" ]; then
    echo "$STAMP"
else
    echo "SYNC_STAMP mismatch after sync:" >&2
    diff <(echo "$remote") <(echo "$STAMP") >&2
    exit 1
fi
rm -f SYNC_STAMP

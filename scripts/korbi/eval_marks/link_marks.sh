#!/bin/bash
# Keep the equal-token checkpoints of the main runs (marks.tsv): once a run's checkpoint at a mark
# step is complete (its tracker has reached the step), hard-link it into <run dir>/ckpt_eval/iter_N
# -- no extra space while the original exists, and it survives the rolling checkpoint's deletion.
# Runs on the workstation, acting on MN5 over ssh (one pass per call; sync_loop.sh calls it).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
MARKS="${MARKS:-$HERE/marks.tsv}"
HOST="${HOST:-marenostrum_new}"
grep -v '^\s*#' "$MARKS" | while IFS=$'\t' read -r name dir steps; do
    [ -z "${name:-}" ] && continue
    ssh "$HOST" "bash -s" <<REMOTE 2>&1 | grep -v "^load\|CPLUS\|CUDA\|CUDNN"
set -u
dir='$dir'; out="\$dir/ckpt_eval"
for s in \$(echo '$steps' | tr ',' ' '); do
    it=\$(printf 'iter_%07d' \$s)
    [ -e "\$out/\$it" ] && continue
    for sub in ckpt ckpt_rolling; do
        src="\$dir/\$sub/\$it"; tracker="\$dir/\$sub/latest_checkpointed_iteration.txt"
        [ -d "\$src" ] && [ -f "\$tracker" ] || continue
        latest=\$(cat "\$tracker" 2>/dev/null | tr -dc 0-9)
        [ -n "\$latest" ] && [ "\$latest" -ge "\$s" ] || continue   # the save of step s is complete
        mkdir -p "\$out"
        if cp -al "\$src" "\$out/\$it.tmp" && mv "\$out/\$it.tmp" "\$out/\$it"; then
            echo "$name: kept step \$s (\$sub) -> \$out/\$it"
        else
            rm -rf "\$out/\$it.tmp"; echo "$name: FAILED to link step \$s from \$sub"
        fi
        break
    done
done
REMOTE
done

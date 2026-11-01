#!/bin/bash
# Move old persistent checkpoints of a run from scratch to projects, leaving a symlink behind
# (user 2026-11-01: ehpc533 scratch at 139 / 146 TB). Per checkpoint iter_N of <src run>/ckpt:
#   1. copy its files in parallel (J cp's; a file already there with the right size is skipped,
#      so an interrupted move resumes) into <dst run>/ckpt/iter_N.part
#   2. verify: the same file names and sizes (VERIFY=cmp also compares every byte: reads twice)
#   3. rename .part -> iter_N, recreate the run's ckpt_eval/iter_N hard links (link_marks.sh) at
#      the destination as hard links of the moved copy
#   4. replace the scratch dirs by symlinks to the moved copies, then delete the scratch copies
# Only steps <= MAX_STEP and at least KEEP_STEPS below the run's latest checkpoint; symlinks
# (already moved) are skipped. P checkpoints at a time. Files keep their timestamps; their modes
# come from the destination's default ACL (as hybrid_nucleus on scratch: group ehpc533 rwx).
#
#   move_ckpts.sh [--dry-run] SRC_RUN_DIR DST_RUN_DIR MAX_STEP
#   env: P (default 4), J (16), KEEP_STEPS (5000), VERIFY (size | cmp)
# Not on a login node (it kills processes at 300 s CPU): as a 1-node pool run,
# config/experiments/korbi/storage/move_ckpts_mn5.yaml.
set -u
DRY=0
[ "${1:-}" = "--dry-run" ] && { DRY=1; shift; }
if [ $# -ne 3 ]; then
    sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi
SRC=${1%/} DST=${2%/} MAX_STEP=$3
P=${P:-4} J=${J:-16} KEEP_STEPS=${KEEP_STEPS:-5000} VERIFY=${VERIFY:-size}
export SRC DST J VERIFY DRY

log() { echo "[move_ckpts $(date +%T)] $*"; }
export -f log

latest=$(tr -dc 0-9 <"$SRC/ckpt/latest_checkpointed_iteration.txt")
[ -n "$latest" ] || { log "no tracker in $SRC/ckpt"; exit 1; }
limit=$((latest - KEEP_STEPS))
[ "$MAX_STEP" -lt "$limit" ] && limit=$MAX_STEP
log "src $SRC -> dst $DST: steps <= $limit (latest $latest, keep $KEEP_STEPS, max $MAX_STEP);" \
    "P=$P J=$J verify=$VERIFY dry=$DRY"

# one file: copy unless an equal-sized copy is there already
copy_file() {
    local s=$1 d=$2
    if [ -f "$d" ] && [ "$(stat -c %s "$s")" = "$(stat -c %s "$d")" ]; then return 0; fi
    cp --preserve=timestamps "$s" "$d.tmp" && mv -f "$d.tmp" "$d"
}
export -f copy_file

# names and sizes of the regular files of a dir, sorted ("name size" lines)
listing() { (cd "$1" && find . -maxdepth 1 -type f ! -name '*.tmp' -printf '%f %s\n' | sort); }
export -f listing

move_one() {
    local it=$1 src_dir="$SRC/ckpt/$1" dst_dir="$DST/ckpt/$1" t0 bytes
    t0=$(date +%s)
    bytes=$(du -sb "$src_dir" | cut -f1)
    if [ "$DRY" = 1 ]; then log "$it: would move $((bytes / 1000000000)) GB"; return 0; fi
    if [ -n "$(find "$src_dir" -mindepth 1 -type d -print -quit)" ]; then
        log "$it: has subdirectories, not handled -- skipped"; return 1
    fi
    # copy into .part; a complete dst from an earlier, interrupted run is checked again in place
    local tgt="$dst_dir.part"
    [ -d "$dst_dir" ] && [ ! -d "$tgt" ] && tgt=$dst_dir
    mkdir -p "$tgt" || return 1
    (cd "$src_dir" && find . -maxdepth 1 -type f -printf '%f\n') |
        xargs -P "$J" -I{} bash -c 'copy_file "$0/{}" "$1/{}"' "$src_dir" "$tgt" ||
        { log "$it: copy FAILED"; return 1; }
    # files the source does not have (an interrupted rsync's hidden temp files, *.tmp) go
    local extra
    extra=$(comm -13 <(cd "$src_dir" && find . -maxdepth 1 -type f -printf '%f\n' | sort) \
                     <(cd "$tgt" && find . -maxdepth 1 -type f -printf '%f\n' | sort))
    [ -n "$extra" ] && (cd "$tgt" && printf '%s\n' "$extra" | xargs -r -d '\n' rm -f --)
    # group-writable like the rest of hybrid_nucleus (rsync -a / cp -p copies keep 640)
    chmod -R g+rwX "$tgt"
    if ! diff -q <(listing "$src_dir") <(listing "$tgt") >/dev/null; then
        log "$it: verify FAILED (names / sizes differ); scratch copy kept"; return 1
    fi
    if [ "$VERIFY" = cmp ]; then
        (cd "$src_dir" && find . -maxdepth 1 -type f -printf '%f\n') |
            xargs -P "$J" -I{} cmp -s "$src_dir/{}" "$tgt/{}" ||
            { log "$it: cmp FAILED; scratch copy kept"; return 1; }
    fi
    [ "$tgt" = "$dst_dir" ] || mv -T "$tgt" "$dst_dir" || return 1
    # ckpt_eval/iter_N: a hard-linked copy of ckpt/iter_N (link_marks.sh) -> the same at the dst
    local ev="$SRC/ckpt_eval/$it" first
    first=$(cd "$src_dir" && find . -maxdepth 1 -type f -printf '%f\n' -quit)
    if [ -d "$ev" ] && [ ! -L "$ev" ] && [ "$(stat -c %i "$ev/$first")" = "$(stat -c %i "$src_dir/$first")" ]; then
        mkdir -p "$DST/ckpt_eval"
        [ -e "$DST/ckpt_eval/$it" ] || cp -al "$dst_dir" "$DST/ckpt_eval/$it" || return 1
        mv -T "$ev" "$ev.del.$$" && ln -s "$DST/ckpt_eval/$it" "$ev" && rm -rf "$ev.del.$$"
    fi
    mv -T "$src_dir" "$src_dir.del.$$" && ln -s "$dst_dir" "$src_dir" && rm -rf "$src_dir.del.$$" ||
        { log "$it: replacing the scratch copy by a symlink FAILED"; return 1; }
    local dt=$(( $(date +%s) - t0 ))
    log "$it: moved $((bytes / 1000000000)) GB in ${dt} s ($((bytes / 1000000 / (dt > 0 ? dt : 1))) MB/s)"
}
export -f move_one

[ "$DRY" = 1 ] || mkdir -p "$DST/ckpt" || exit 1
for d in "$SRC"/ckpt/iter_*; do
    [ -d "$d" ] && [ ! -L "$d" ] || continue
    case "$d" in *.del.*) continue ;; esac
    it=$(basename "$d")
    step=$((10#${it#iter_}))
    [ "$step" -le "$limit" ] && echo "$it"
done | xargs -r -P "$P" -I{} bash -c 'move_one {}'
rc=$?
log "done (rc $rc); scratch left in $SRC/ckpt: $(find "$SRC/ckpt/" -maxdepth 1 -name 'iter_*' -type d | wc -l) dirs"
exit $rc

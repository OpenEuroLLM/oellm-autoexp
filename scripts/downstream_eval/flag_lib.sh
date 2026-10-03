# Shared by eval_checkpoints.sh (flag commands) and flag_launch.sh.

# Login-node check that the oellm-eval in use is the pinned submodule; prints its revision.
flag_preflight() {
    local rev
    rev=$(git -C "$EVAL_REPO" rev-parse HEAD 2>/dev/null || echo unknown)
    if [ "$rev" != "$EXPECTED_EVAL_REV" ] && [ "${ALLOW_EVAL_REV_MISMATCH:-0}" != 1 ]; then
        echo "error: $EVAL_REPO is at $rev, the repo pins $EXPECTED_EVAL_REV (ALLOW_EVAL_REV_MISMATCH=1 to override)" >&2
        return 1
    fi
    [ -z "$(git -C "$EVAL_REPO" status --porcelain 2>/dev/null)" ] || [ "${ALLOW_EVAL_REV_MISMATCH:-0}" = 1 ] || {
        echo "error: $EVAL_REPO has uncommitted changes (ALLOW_EVAL_REV_MISMATCH=1 to override)" >&2; return 1; }
    flag_tool_is_repo || return 1
    echo "$rev"
}

# Checked-out commit of a repo or submodule, read without the git binary (absent on compute nodes).
git_head_rev() {
    local repo="$1" gitdir head ref
    if [ -f "$repo/.git" ]; then gitdir=$(sed -n 's/^gitdir: //p' "$repo/.git"); [[ "$gitdir" = /* ]] || gitdir="$repo/$gitdir"
    else gitdir="$repo/.git"; fi
    head=$(cat "$gitdir/HEAD" 2>/dev/null) || { echo unknown; return; }
    case "$head" in
        ref:*) ref=${head#ref: }
               local common="$gitdir" rev=""
               [ -f "$gitdir/commondir" ] && common="$gitdir/$(cat "$gitdir/commondir")"   # worktrees
               for d in "$gitdir" "$common"; do
                   [ -n "$rev" ] || rev=$(cat "$d/$ref" 2>/dev/null || true)
                   [ -n "$rev" ] || rev=$(grep -m1 " $ref\$" "$d/packed-refs" 2>/dev/null | cut -d' ' -f1 || true)
               done
               echo "${rev:-unknown}" ;;
        *) echo "$head" ;;
    esac
}

# The oellm tool that renders the suite must run this checkout's source (the suite definition is
# read from it). By default it does: jupiter_flag_evals.sh runs it on the vLLM image's Python with
# PYTHONPATH=EVAL_REPO (OELLM_TOOL=image); this also guards OELLM_TOOL=installed.
flag_tool_is_repo() {
    local pkg
    pkg=$("$EVAL_REPO/scripts/jupiter_flag_evals.sh" check) || { echo "error: cannot run the oellm tool" >&2; return 1; }
    case "$pkg" in
        "$(realpath "$EVAL_REPO")"/*) ;;
        *) echo "error: the oellm tool runs $pkg, not $EVAL_REPO (OELLM_TOOL=${OELLM_TOOL:-image})" >&2; return 1 ;;
    esac
}

# Compute-node subset of flag_preflight; prints the checked-out revision.
# (Revision pin and cleanliness need git; they are checked on the login node.)
flag_check_installed() {
    flag_tool_is_repo || return 1
    git_head_rev "$EVAL_REPO"
}

# Cross-node lock. flock is not coherent across nodes on the shared filesystems (two stages on
# different nodes both entered the "locked" section, job 2162600); mkdir is atomic. A lock older
# than 10 minutes is taken to be left by a killed holder and broken.
lock_acquire() {
    local lock="$1" waited=0
    until mkdir "$lock" 2>/dev/null; do
        sleep 1; waited=$((waited + 1))
        if [ "$waited" -ge 600 ]; then echo "warning: breaking stale lock $lock" >&2; rm -rf "$lock"; waited=0; fi
    done
}
lock_release() { rmdir "$1" 2>/dev/null || true; }

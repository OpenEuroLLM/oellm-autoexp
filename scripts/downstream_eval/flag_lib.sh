# Shared by eval_checkpoints.sh (flag commands) and flag_launch.sh.

# Login-node check that the oellm-eval in use is the pinned submodule; prints its revision.
flag_preflight() {
    local rev tool pkg
    rev=$(git -C "$EVAL_REPO" rev-parse HEAD 2>/dev/null || echo unknown)
    if [ "$rev" != "$EXPECTED_EVAL_REV" ] && [ "${ALLOW_EVAL_REV_MISMATCH:-0}" != 1 ]; then
        echo "error: $EVAL_REPO is at $rev, the repo pins $EXPECTED_EVAL_REV (ALLOW_EVAL_REV_MISMATCH=1 to override)" >&2
        return 1
    fi
    [ -z "$(git -C "$EVAL_REPO" status --porcelain 2>/dev/null)" ] || [ "${ALLOW_EVAL_REV_MISMATCH:-0}" = 1 ] || {
        echo "error: $EVAL_REPO has uncommitted changes (ALLOW_EVAL_REV_MISMATCH=1 to override)" >&2; return 1; }
    tool=$(command -v oellm-eval) || { echo "error: oellm-eval not on PATH; run: uv tool install -p 3.12 -e $EVAL_REPO" >&2; return 1; }
    pkg=$("$(dirname "$tool")/python" -c 'import oellm, os; print(os.path.dirname(os.path.realpath(oellm.__file__)))')
    case "$pkg" in
        "$(realpath "$EVAL_REPO")"/*) ;;
        *) echo "error: oellm-eval is installed from $pkg, not $EVAL_REPO;" \
                "run: uv tool install -p 3.12 -e $EVAL_REPO --force" >&2; return 1 ;;
    esac
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

# Compute-node subset of flag_preflight: the installed oellm-eval must come from EVAL_REPO.
# (Revision pin and cleanliness need git; they are checked when the chain is submitted.)
flag_check_installed() {
    local tool pkg
    tool=$(command -v oellm-eval) || { echo "error: oellm-eval not on PATH" >&2; return 1; }
    pkg=$("$(dirname "$tool")/python" -c 'import oellm, os; print(os.path.dirname(os.path.realpath(oellm.__file__)))')
    case "$pkg" in
        "$(realpath "$EVAL_REPO")"/*) ;;
        *) echo "error: oellm-eval is installed from $pkg, not $EVAL_REPO" >&2; return 1 ;;
    esac
    git_head_rev "$EVAL_REPO"
}

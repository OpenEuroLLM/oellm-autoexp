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

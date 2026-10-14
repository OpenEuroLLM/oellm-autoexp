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

# Submit the arrays with the environment of a plain login shell. An oellm-autoexp `job.local`
# stage exports slurm.env and backend.env (e.g. TORCHDYNAMO_DISABLE=1, which breaks vLLM's
# torch.compile, and cache paths into a training tree), and sbatch hands its environment to the
# jobs. Drop those and every SLURM_* (stale outside a job).
flag_clean_env() {
    local v
    for v in TORCHDYNAMO_DISABLE TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE TOKENIZERS_PARALLELISM \
             PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF OMP_NUM_THREADS NCCL_SOCKET_IFNAME \
             NCCL_SOCKET_FAMILY GLOO_SOCKET_IFNAME GLOO_SOCKET_FAMILY MASTER_ADDR MASTER_PORT \
             LOCAL_ADDR NUM_NODES NUM_GPUS_PER_NODE NUM_GPUS ARCH WANDB_MODE MACHINE_NAME \
             HF_ALLOW_CODE_EVAL HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE PYTHONUNBUFFERED \
             $(compgen -e | grep '^SLURM_' || true); do
        unset "$v"
    done
}

# The two eval views of an export (identity chat template for vLLM, plain for lighteval) via
# jupiter_flag_evals.sh views. The plain view is `cp -al <export>`, which fails on an export owned
# by another user: we may not hard-link the files that are read-only to us (tokenizer.model,
# special_tokens_map.json of vanosch1's exports). Build it per file first -- hard link where allowed
# (the model shards), copy the rest; `views` then keeps it as it is. Needs FLAG_WORK, EVAL_REPO.
flag_views() {  # <export dir>
    local export=$1 plain f
    plain="$FLAG_WORK/views/plain/$(basename "$(realpath "$export")")"
    if [ ! -e "$plain" ] && [ "$(stat -c %U "$export")" != "$USER" ]; then
        rm -rf "$plain.tmp"            # leftover of a failed `cp -al` (links and copies only)
        mkdir -p "$plain.tmp"
        for f in "$export"/*; do
            ln "$f" "$plain.tmp/" 2>/dev/null || cp -a "$f" "$plain.tmp/"
        done
        mv "$plain.tmp" "$plain"
    fi
    "$EVAL_REPO/scripts/jupiter_flag_evals.sh" views "$export" > /dev/null
}

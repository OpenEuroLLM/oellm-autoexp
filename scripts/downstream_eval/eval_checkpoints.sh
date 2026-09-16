#!/bin/bash
# Downstream evaluation of Megatron checkpoints: convert to HF, then run the eval suites.
#
#   MODEL=32b_dense ./eval_checkpoints.sh convert v2:40000
#   MODEL=32b_dense ./eval_checkpoints.sh hf v2_40k
#
# Model-specific settings live in models/<MODEL>.env, site-specific ones in sites/<SITE>.env.
# Results and exports live under WORK_ROOT; only the scripts live in this repo.
HERE=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
source "$HERE/lib.sh"

usage() {
    cat <<USAGE
Usage: [MODEL=<profile>] [SITE=<site>] eval_checkpoints.sh COMMAND [NAME...]

Commands:
  list [RUN]           Show the profiles and runs, or the iterations of one run
  prepare              One-time: tokenizer for conversion and evaluation, Megatron-Bridge copy
  convert SPEC...      Convert checkpoints: <run>:<iter>, <run>:latest, or an alias
                       (DRY_RUN=1 prints the sbatch command instead of submitting)
  check NAME           Submit a sanity check of one export (NLL + greedy continuation)
  hf NAME...           Render the 12 open-sci lm-eval tasks for those exports
  reasoning [NAME|all] Render GSM8K/MATH500/MBPP etc. via vLLM (never submits; see TASKS=)
  table | plot         Print or draw the result tables (hf and reasoning)

Profiles: MODEL=$MODEL SITE=$SITE   WORK_ROOT=$WORK_ROOT
USAGE
}

# DRY_RUN=1 prints the sbatch command instead of submitting it.
submit() {
    if [ "${DRY_RUN:-0}" = 1 ]; then
        printf 'sbatch'; printf ' %q' "$@"; printf '\n'
    else
        sbatch "$@"
    fi
}

CMD="${1:-}"; shift || true
case "$CMD" in
list)
    echo "site=$SITE  model=$MODEL  account=$ACCOUNT  partition=$PARTITION"
    echo "work root=$WORK_ROOT"
    echo "exports=$EXPORT_ROOT  runs=$RUN_ROOT"
    echo "eval dp=$EVAL_DP tp=$EVAL_TP  gpus/node=$GPUS_PER_NODE"
    if [ $# -gt 0 ]; then                      # iterations of one run
        [ -n "${RUN_PATH[$1]:-}" ] || { echo "error: unknown run '$1'" >&2; exit 1; }
        echo "checkpoints of $1 (${RUN_PATH[$1]}):"
        for dir in "${RUN_PATH[$1]}"/iter_*; do
            [ -d "$dir" ] || continue
            iter=$((10#$(basename "$dir" | sed 's/iter_//')))
            name=$(checkpoint_name "$1" "$iter")
            printf '  %-14s %s\n' "$1:$iter" \
                "$([ -d "$EXPORT_ROOT/$name" ] && echo "[converted as $name]" || echo '')"
        done
        exit 0
    fi
    echo "runs:"
    for run in "${!RUN_PATH[@]}"; do
        mapfile -t iters < <(ls -d "${RUN_PATH[$run]}"/iter_* 2>/dev/null | xargs -r -n1 basename)
        printf '  %-6s %3s checkpoints  %s..%s\n' "$run" "${#iters[@]}" \
            "${iters[0]:-none}" "${iters[${#iters[@]} - 1]:-}"
    done | sort
    echo "aliases:"
    for name in "${!ALIAS_SPEC[@]}"; do
        printf '  %-14s %s %s\n' "$name" "${ALIAS_SPEC[$name]}" \
            "$([ -d "$EXPORT_ROOT/$name" ] && echo '[converted]' || echo '')"
    done | sort
    echo "exports present: $(ls -d "$EXPORT_ROOT"/*/ 2>/dev/null | wc -l)"
    ;;
prepare)
    mkdir -p "$WORK_ROOT" "$EXPORT_ROOT" "$LOG_DIR"
    # Tokenizer: drop an added <pad> outside the embedding, pad = <eos>, and write tokenizer.json
    # for the bridge container (which has no sentencepiece).
    [ -f "$WORK_ROOT/tokenizer/tokenizer.json" ] || \
    apptainer exec --bind "$CONTAINER_BIND" "$TRAIN_SIF" /opt/venv/bin/python - \
        "$TOKENIZER_SRC" "$WORK_ROOT/tokenizer" "$VOCAB_SIZE" <<'PY'
import json, shutil, sys
from pathlib import Path
from transformers import AutoTokenizer
src, dst, vocab = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
tmp = dst.with_name("tokenizer_src"); tmp.mkdir(parents=True, exist_ok=True)
for f in ("tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"):
    shutil.copy(src / f, tmp / f)
for f in ("tokenizer_config.json", "special_tokens_map.json"):
    c = json.loads((tmp / f).read_text())
    c["pad_token"] = c["eos_token"]
    c["added_tokens_decoder"] = {k: v for k, v in c.get("added_tokens_decoder", {}).items() if int(k) < vocab}
    c["additional_special_tokens"] = [t for t in c.get("additional_special_tokens", [])
                                      if (t if isinstance(t, str) else t.get("content")) != "<pad>"]
    (tmp / f).write_text(json.dumps(c, indent=2))
tok = AutoTokenizer.from_pretrained(tmp)
assert len(tok) == vocab, (len(tok), vocab)
tok.save_pretrained(dst)
PY
    # Megatron-Bridge copy with tolerant model imports, so the repos stay untouched.
    [ -d "$WORK_ROOT/bridge/src" ] || { mkdir -p "$WORK_ROOT/bridge"; cp -r "$MEGATRON_REPO/submodules/Megatron-Bridge/src" "$WORK_ROOT/bridge/"; }
    python3 "$MEGATRON_REPO/container/megatron/patch_bridge_lazy_imports.py" "$WORK_ROOT/bridge/src/megatron/bridge/models" | tail -1
    echo "prepared: $WORK_ROOT"
    ;;
convert)
    [ $# -gt 0 ] || { echo "error: name the checkpoints to convert, e.g. v2:40000 (see 'list')" >&2; exit 1; }
    mapfile -t entries < <(select_checkpoints "$@")
    [ ${#entries[@]} -gt 0 ] || { echo "error: no matching checkpoints" >&2; exit 1; }
    for entry in "${entries[@]}"; do   # never silently rewrite a 64 GB export
        [ ! -d "$EXPORT_ROOT/${entry%%:*}" ] || \
            { echo "error: ${entry%%:*} is already converted; FORCE=1 to convert it again" >&2; [ "${FORCE:-0}" = 1 ] || exit 1; }
    done
    mkdir -p "$LOG_DIR" "$EXPORT_ROOT"
    MANIFEST="$WORK_ROOT/convert_manifest_$(date +%Y%m%d_%H%M%S).txt"
    printf '%s\n' "${entries[@]}" > "$MANIFEST"
    echo "converting ${#entries[@]} checkpoint(s):"; cat "$MANIFEST"
    submit --account="$ACCOUNT" --partition="$PARTITION" --time="$CONVERT_TIME" \
        --gpus-per-node="$GPUS_PER_NODE" --array=0-$((${#entries[@]} - 1)) \
        --output="$LOG_DIR/convert_%A_%a.log" \
        --export=ALL,MANIFEST="$MANIFEST",SCRIPT_DIR="$HERE",WORK_ROOT="$WORK_ROOT",EXPORT_ROOT="$EXPORT_ROOT",BRIDGE_SIF="$BRIDGE_SIF",MEGATRON_REPO="$MEGATRON_REPO",HF_MODEL="$HF_MODEL",DERIVE_HF_ARCH="$DERIVE_HF_ARCH",ARCH_YAML="$ARCH_YAML",CONTAINER_BIND="$CONTAINER_BIND" \
        "$HERE/convert.sbatch"
    ;;
check)
    [ $# -eq 1 ] || { echo "usage: eval_checkpoints.sh check NAME" >&2; exit 2; }
    require_exports "$1"
    submit --account="$ACCOUNT" --partition="$PARTITION" --time=00:30:00 --mem=200G \
        --output="$LOG_DIR/check_%j.log" \
        --export=ALL,CHECK_NAME="$1",EXPORT_ROOT="$EXPORT_ROOT",HF_EVAL_SIF="$HF_EVAL_SIF",CONTAINER_BIND="$CONTAINER_BIND" \
        "$HERE/check.sbatch"
    ;;
hf)
    [ $# -gt 0 ] || { echo "error: name the exports to evaluate (see 'list')" >&2; exit 1; }
    mapfile -t entries < <(select_checkpoints "$@")
    names=(); for e in "${entries[@]}"; do names+=("${e%%:*}"); done
    "$HERE/hf_evals.sh" "${names[@]}"
    ;;
reasoning)
    EXPORT_ROOT="$EXPORT_ROOT" PROMPT_VIEW_ROOT="$PROMPT_VIEW_ROOT" RUN_ROOT="$RUN_ROOT" \
    HF_HOME="$REASONING_CACHE" EVAL_REPO="$EVAL_REPO" EVAL_IMAGE="$REASONING_SIF" \
    EXPECTED_EVAL_REV="$EXPECTED_EVAL_REV" EXPECTED_IMAGE_SHA256="$REASONING_SIF_SHA256" \
    OELLM_ACCOUNT="$ACCOUNT" EVAL_DP="$EVAL_DP" EVAL_TP="$EVAL_TP" \
    CPUS_PER_TASK="$CPUS_PER_TASK" SLURM_MEM="$SLURM_MEM" EVAL_TIME="$EVAL_TIME" \
        "$HERE/reasoning_evals.sh" "${1:-all}"
    ;;
table|plot)
    export EVAL_WORK_ROOT="$WORK_ROOT" EVAL_EXPORT_ROOT="$EXPORT_ROOT" EVAL_SHARED_RUNS="$EVAL_SHARED_RUNS"
    if [ "$CMD" = table ]; then
        python3 "$HERE/table_hf.py" "$@"; echo; python3 "$HERE/table_reasoning.py" "$@"
    else
        python3 "$HERE/plot_hf.py" "${1:-$WORK_ROOT/downstream_evals.png}"
        python3 "$HERE/plot_reasoning.py" "${2:-$WORK_ROOT/reasoning_evals.png}"
    fi
    ;;
*)
    usage; [ -z "$CMD" ] && exit 2 || exit 0
    ;;
esac

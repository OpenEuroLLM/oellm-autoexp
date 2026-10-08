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
source "$HERE/flag_lib.sh"
: "${FLAG_STATE_DIR:=$FLAG_WORK/state}"
FLAG_ENV=(EXPORT_ROOT EVAL_REPO EXPECTED_EVAL_REV FLAG_ACCOUNT FLAG_WORK FLAG_CONCURRENCY FLAG_TIME
          FLAG_STATE_DIR FLAG_ARRAY_VLLM FLAG_ARRAY_LIGHTEVAL FLAG_SUBSET COLLECT_RAW FLAG_DATASETS_CSV)

usage() {
    cat <<USAGE
Usage: [MODEL=<profile>] [SITE=<site>] eval_checkpoints.sh COMMAND [NAME...]

Commands:
  list [RUN]           Show the profiles and runs, or the iterations of one run
  prepare              One-time: tokenizer for conversion and evaluation, Megatron-Bridge copy
  convert SPEC...      Convert checkpoints: <run>:<iter>, <run>:latest, or an alias
                       (DRY_RUN=1 prints the sbatch command instead of submitting)
  convert-here NAME CKPT  Convert one checkpoint in THIS allocation (an oellm-autoexp convert stage)
  check NAME           Submit a sanity check of one export (NLL + greedy continuation)
  hf NAME...           Render the 12 open-sci lm-eval tasks for those exports
  reasoning [NAME|all] Render GSM8K/MATH500/MBPP etc. via vLLM (never submits; see TASKS=)
  table | plot         Print or draw the result tables (hf and reasoning)
  publish NAME...      Stage exports for the Hub (upload_hf.py --execute uploads them)
  flag NAME...         FLAG suite (438 evals) for converted exports: launch both halves
                       (vLLM + lighteval) as job arrays on FLAG_ACCOUNT
  flag-status NAME...  Per-task status; failures with their cause; collects the CSV when complete
  flag-rerun NAME...   Resubmit the failed / never-run tasks into the same run directories
  flag-wait NAME...    Poll until done, resubmit failures FLAG_RETRIES (1) times, collect --
                       the body of an oellm-autoexp eval stage (job.local, on the login node)
  flag-prepare NAME    Views + launchers for one export WITHOUT submitting (login node): the
                       stage before oellm-autoexp's eval array stages, whose tasks each run one
                       row (jupiter_flag_evals.sh run-one); idempotent
  flag-collect NAME... Collect the CSV + join check (after the array stages)
  cot-prepare NAME     Views + the launcher of the "cot" half (task group flag-evals-cot: the
                       forced-reasoning _cot and continuation _cont evals) WITHOUT submitting; own
                       state file NAME.cot.env, so it also works for an export prepared before
  cot-collect NAME...  Collect the cot half into \$FLAG_WORK/results/NAME.flag-cot.tasks.csv,
                       then flag-combine NAME
  flag-combine NAME... \$FLAG_WORK/results/NAME.flag-evals-471.tasks.csv = the 438 suite's CSV + the
                       cot half's (both must exist; the release evals stay, the cot rows are added;
                       rows at an n_shot outside the suite are left out)

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

# NAME.flag-evals-471.tasks.csv: the 438 suite's rows and the cot half's, one file per export for
# aggregation (summarize-evals); rows at an n_shot outside the suite are left out (flag_combine.py).
flag_combine() {
    local r="$FLAG_WORK/results" name="$1"
    local suite="$r/$name.flag-evals-438.tasks.csv" cot="$r/$name.flag-cot.tasks.csv"
    for f in "$suite" "$cot"; do
        [ -f "$f" ] || { echo "flag-combine $name: no $f" >&2; return 1; }
    done
    python3 "$HERE/flag_combine.py" "$suite" "$cot" "$r/$name.flag-evals-471.tasks.csv.tmp" \
        "$HERE/../../config/eval_tasks/flag_evals.yaml" "$HERE/../../config/eval_tasks/flag_cot.yaml" &&
        mv "$r/$name.flag-evals-471.tasks.csv.tmp" "$r/$name.flag-evals-471.tasks.csv"
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
convert-here)
    # The conversion INSIDE the current allocation, with this profile's settings: the body of an
    # oellm-autoexp convert stage (compute jobs cannot sbatch, so `convert` cannot be used there).
    [ $# -eq 2 ] || { echo "usage: eval_checkpoints.sh convert-here NAME CHECKPOINT_DIR" >&2; exit 2; }
    [ ! -d "$EXPORT_ROOT/$1" ] || [ "${FORCE:-0}" = 1 ] || { echo "error: $1 is already converted; FORCE=1 to convert it again" >&2; exit 1; }
    mkdir -p "$EXPORT_ROOT"
    export WORK_ROOT EXPORT_ROOT BRIDGE_SIF MEGATRON_REPO HF_MODEL DERIVE_HF_ARCH ARCH_YAML CONTAINER_BIND
    exec "$HERE/convert_export.sh" "$1" "$2"
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
publish)
    [ $# -gt 0 ] || { echo "error: name the exports to publish (see 'list')" >&2; exit 1; }
    EVAL_WORK_ROOT="$WORK_ROOT" EVAL_EXPORT_ROOT="$EXPORT_ROOT" \
        python3 "$HERE/upload_hf.py" "$@"
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
flag)
    [ $# -gt 0 ] || { echo "error: name converted exports, e.g. v1annealC_118k (see 'list')" >&2; exit 1; }
    mapfile -t entries < <(select_checkpoints "$@")
    [ ${#entries[@]} -gt 0 ] || { echo "error: no matching checkpoints" >&2; exit 1; }
    for entry in "${entries[@]}"; do      # compute jobs cannot sbatch, so conversion is not chained here
        [ -f "$EXPORT_ROOT/${entry%%:*}/config.json" ] || {
            echo "error: ${entry%%:*} is not converted; run 'convert ${entry%%:*}' first, or use an" \
                 "oellm-autoexp stage chain (config/experiments/korbi/chain_flag_130M_jupiter.yaml)" >&2; exit 1; }
    done
    flag_clean_env
    rev=$(flag_preflight)                       # on the login node: compute nodes have no git
    export "${FLAG_ENV[@]}" FLAG_EVAL_REV="$rev"
    for entry in "${entries[@]}"; do
        name=${entry%%:*}
        if [ -f "$FLAG_STATE_DIR/$name.env" ] && [ "${FORCE:-0}" != 1 ]; then
            echo "skip $name: already launched (flag-status $name)"
        else
            FLAG_PREFLIGHT_OK=1 "$HERE/flag_launch.sh" "$name"
        fi
    done
    ;;
flag-prepare)
    [ $# -eq 1 ] || { echo "usage: eval_checkpoints.sh flag-prepare NAME" >&2; exit 2; }
    [ -f "$EXPORT_ROOT/$1/config.json" ] || { echo "error: $1 is not converted ($EXPORT_ROOT/$1)" >&2; exit 1; }
    flag_clean_env
    rev=$(flag_preflight)                       # on the login node: compute nodes have no git
    export "${FLAG_ENV[@]}" FLAG_EVAL_REV="$rev"
    FLAG_PREFLIGHT_OK=1 PREPARE_ONLY=1 "$HERE/flag_launch.sh" "$1"
    ;;
flag-collect)
    [ $# -gt 0 ] || { echo "usage: eval_checkpoints.sh flag-collect NAME..." >&2; exit 2; }
    flag_clean_env
    export "${FLAG_ENV[@]}"
    for name in "$@"; do
        python3 "$HERE/flag_status.py" "$FLAG_STATE_DIR/$name.env" --collect-only
    done
    ;;
cot-prepare)
    [ $# -eq 1 ] || { echo "usage: eval_checkpoints.sh cot-prepare NAME" >&2; exit 2; }
    [ -f "$EXPORT_ROOT/$1/config.json" ] || { echo "error: $1 is not converted ($EXPORT_ROOT/$1)" >&2; exit 1; }
    state="$FLAG_STATE_DIR/$1.cot.env"
    # A second render would make a NEW run directory and orphan the rows already done.
    if [ -f "$state" ] && [ "${FORCE:-0}" != 1 ]; then echo "$1 cot half already prepared ($state)"; exit 0; fi
    flag_clean_env
    rev=$(flag_preflight)                       # on the login node: compute nodes have no git
    export ACCOUNT="$FLAG_ACCOUNT" FLAG_WORK CONCURRENCY="$FLAG_CONCURRENCY" TIME="$FLAG_TIME"
    flag_views "$EXPORT_ROOT/$1"
    rendered=$(HALVES=cot "$EVAL_REPO/scripts/jupiter_flag_evals.sh" render "$1")
    sb=$(awk '$1 == "sbatch" && $2 ~ /\/cot\// {print $2}' <<< "$rendered")
    [ -f "$sb" ] || { echo "error: render did not produce the cot launcher" >&2; exit 1; }
    mkdir -p "$FLAG_STATE_DIR"
    printf '# written by eval_checkpoints.sh cot-prepare %s\nNAME=%s\nEXPORT=%s\nEVAL_REV=%s\nCOT_RUN=%s\n' \
        "$(date -Is)" "$1" "$EXPORT_ROOT/$1" "$rev" "$(dirname "$sb")" > "$state"
    echo "prepared the cot half of $1: $(dirname "$sb")  (state: $state)"
    ;;
cot-collect)
    [ $# -gt 0 ] || { echo "usage: eval_checkpoints.sh cot-collect NAME..." >&2; exit 2; }
    flag_clean_env
    for name in "$@"; do
        state="$FLAG_STATE_DIR/$name.cot.env"
        [ -f "$state" ] || { echo "error: $name: no $state (cot-prepare first)" >&2; exit 1; }
        run=$(sed -n 's/^COT_RUN=//p' "$state")
        mkdir -p "$FLAG_WORK/results"
        # collect_raw matches the model directory's basename: the identity view is named after the export.
        python3 "$COLLECT_RAW" --checkpoints "$name" --runs "$run" -o "$FLAG_WORK/results/$name.flag-cot.tasks.csv"
        flag_combine "$name" || true  # the 438 suite may not have run (yet)
    done
    ;;
flag-combine)
    [ $# -gt 0 ] || { echo "usage: eval_checkpoints.sh flag-combine NAME..." >&2; exit 2; }
    for name in "$@"; do flag_combine "$name"; done
    ;;
flag-status|flag-rerun|flag-wait)
    [ $# -gt 0 ] || { echo "usage: eval_checkpoints.sh $CMD NAME..." >&2; exit 2; }
    flag_clean_env
    export "${FLAG_ENV[@]}"
    for name in "$@"; do
        state="$FLAG_STATE_DIR/$name.env"
        [ -f "$state" ] || { echo "error: $name was not launched with 'flag' (no $state)" >&2; exit 1; }
        case "$CMD" in
            flag-status) python3 "$HERE/flag_status.py" "$state" ;;
            flag-rerun)  python3 "$HERE/flag_status.py" "$state" --rerun ;;
            flag-wait)   python3 "$HERE/flag_status.py" "$state" --wait --retries "${FLAG_RETRIES:-1}" --poll "${FLAG_POLL_S:-60}" ;;
        esac
    done
    ;;
*)
    usage; [ -z "$CMD" ] && exit 2 || exit 0
    ;;
esac

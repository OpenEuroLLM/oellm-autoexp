#!/usr/bin/env bash
# export_megatron_runs_to_hf.sh
#
# Batch-submit HuggingFace exports for Megatron-LM training runs whose
# checkpoints live under RUN_DIR/checkpoints/iter_*.
#
# Usage:
#   bash export_megatron_runs_to_hf.sh [OPTIONS] --run-root RUN_DIR [RUN_DIR ...] --hf-model HF --out-base DIR
#   bash export_megatron_runs_to_hf.sh [OPTIONS] --run-list runs.txt --hf-model HF --out-base DIR
#
# Required:
#   --run-root PATH [PATH ...]     Run save directory/directories containing checkpoints/iter_*.
#                                  Accepts multiple paths before the next flag, or repeat the flag.
#   --run-list FILE                File with one run save directory per line. Blank lines and
#                                  lines beginning with # are ignored.
#                                  At least one --run-root or --run-list is required.
#   --hf-model PATH                HuggingFace model id or local snapshot path.
#   --out-base DIR                 Parent directory for per-checkpoint HF export directories.
#
# Optional:
#   --bridge-root DIR              Megatron-Bridge checkout containing tw-tools/
#                                  Default: /home/risto.luukkonen@amd.com/rluukkon/oellm/Megatron-Bridge
#   --latest-only                  Export only the latest checkpoint per run.
#   --iters N [N ...]              Export explicit iteration numbers only, e.g. --iters 150 300.
#   --keep-original-checkpoints MODE
#                                  Source checkpoint retention plan: all, latest, or none.
#                                  Default: all. This script never deletes checkpoints; it writes
#                                  a reviewable deletion manifest.
#   --retention-manifest FILE      Manifest path for checkpoint directories flagged for deletion.
#                                  Default: <out-base>/checkpoints_to_delete.txt
#   --dry-run                      Print sbatch commands and retention manifest contents only.
#   --sequential                   Chain jobs within each run with --dependency=afterok.
#   --keep-megatron-vl             Forward --keep-megatron-vl to the converter.
#   --strict-export                Forward --strict-export to the converter.
#
# Output layout:
#   <out-base>/
#     <run_name>/
#       iter_0000150/
#       iter_0000300/
#     checkpoints_to_delete.txt
#
# Examples:
#   # Produces: exports/qwen3_5_35B_A3B_tw_test_cpt/iter_0000300/
#   bash export_megatron_runs_to_hf.sh \
#       --run-root ./output/qwen3_5_35B_A3B_tw_test_cpt \
#       --hf-model Qwen/Qwen3.5-35B-A3B-Base \
#       --out-base /shared_silo/scratch/rluukkon/oellm/Megatron-Bridge/exports \
#       --keep-original-checkpoints latest
#
#   bash export_megatron_runs_to_hf.sh --dry-run --latest-only \
#       --run-list runs.txt \
#       --hf-model /shared_silo/scratch/rluukkon/oellm/hf_home/hub/models--Qwen--Qwen3.5-35B-A3B-Base/snapshots/0f0813072d2358973511097385626f21fcb6d422 \
#       --out-base /shared_silo/scratch/rluukkon/oellm/Megatron-Bridge/exports

set -euo pipefail

DEFAULT_BRIDGE_ROOT="/home/risto.luukkonen@amd.com/rluukkon/oellm/Megatron-Bridge"

RUN_ROOTS=()
RUN_LISTS=()
HF_MODEL=""
OUT_BASE=""
BRIDGE_ROOT="$DEFAULT_BRIDGE_ROOT"
LATEST_ONLY=0
EXPLICIT_ITERS=()
KEEP_ORIGINAL_CHECKPOINTS="all"
RETENTION_MANIFEST=""
DRY_RUN=0
SEQUENTIAL=0
EXTRA_PY_ARGS=()
RETENTION_CANDIDATES=()

_usage() {
    sed -n '2,/^set -euo/{ /^set -euo/d; s/^# \{0,1\}//; p }' "$0"
}

_expect_value() {
    local flag="$1"
    local val="${2:-}"
    if [[ -z "$val" || "$val" == --* ]]; then
        echo "ERROR: ${flag} requires an argument." >&2
        exit 1
    fi
}

_trim() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s\n' "$value"
}

_print_cmd() {
    local arg
    for arg in "$@"; do
        printf '%q ' "$arg"
    done
    printf '\n'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-root)
            shift
            if [[ $# -eq 0 || "$1" == --* ]]; then
                echo "ERROR: --run-root requires at least one run directory." >&2
                exit 1
            fi
            while [[ $# -gt 0 && "$1" != --* ]]; do
                RUN_ROOTS+=("$1")
                shift
            done
            ;;
        --run-list)
            shift
            _expect_value "--run-list" "${1:-}"
            RUN_LISTS+=("$1")
            shift
            ;;
        --hf-model)
            shift
            _expect_value "--hf-model" "${1:-}"
            HF_MODEL="$1"
            shift
            ;;
        --out-base)
            shift
            _expect_value "--out-base" "${1:-}"
            OUT_BASE="$1"
            shift
            ;;
        --bridge-root)
            shift
            _expect_value "--bridge-root" "${1:-}"
            BRIDGE_ROOT="${1%/}"
            shift
            ;;
        --latest-only)
            LATEST_ONLY=1
            shift
            ;;
        --iters)
            shift
            if [[ $# -eq 0 || "$1" == --* ]]; then
                echo "ERROR: --iters requires at least one iteration number." >&2
                exit 1
            fi
            while [[ $# -gt 0 && "$1" != --* ]]; do
                EXPLICIT_ITERS+=("$1")
                shift
            done
            ;;
        --keep-original-checkpoints)
            shift
            _expect_value "--keep-original-checkpoints" "${1:-}"
            case "$1" in
                all|latest|none) KEEP_ORIGINAL_CHECKPOINTS="$1" ;;
                *)
                    echo "ERROR: --keep-original-checkpoints must be one of: all, latest, none." >&2
                    exit 1
                    ;;
            esac
            shift
            ;;
        --retention-manifest)
            shift
            _expect_value "--retention-manifest" "${1:-}"
            RETENTION_MANIFEST="$1"
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --sequential)
            SEQUENTIAL=1
            shift
            ;;
        --keep-megatron-vl)
            EXTRA_PY_ARGS+=(--keep-megatron-vl)
            shift
            ;;
        --strict-export)
            EXTRA_PY_ARGS+=(--strict-export)
            shift
            ;;
        -h|--help)
            _usage
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

for run_list in "${RUN_LISTS[@]}"; do
    if [[ ! -f "$run_list" ]]; then
        echo "ERROR: Run list does not exist: ${run_list}" >&2
        exit 1
    fi
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line%%#*}"
        line="$(_trim "$line")"
        [[ -z "$line" ]] && continue
        RUN_ROOTS+=("$line")
    done < "$run_list"
done

if [[ ${#RUN_ROOTS[@]} -eq 0 ]]; then
    echo "ERROR: At least one --run-root or --run-list entry is required." >&2
    exit 1
fi
if [[ -z "$HF_MODEL" ]]; then
    echo "ERROR: --hf-model is required." >&2
    exit 1
fi
if [[ -z "$OUT_BASE" ]]; then
    echo "ERROR: --out-base is required." >&2
    exit 1
fi
if [[ "$LATEST_ONLY" -eq 1 && ${#EXPLICIT_ITERS[@]} -gt 0 ]]; then
    echo "ERROR: --latest-only and --iters are mutually exclusive." >&2
    exit 1
fi

SINGLE_CKPT_SCRIPT="${BRIDGE_ROOT}/tw-tools/rewrap_text_to_vl_and_export.sh"
if [[ ! -d "$BRIDGE_ROOT" ]]; then
    echo "ERROR: Bridge root does not exist: ${BRIDGE_ROOT}" >&2
    exit 1
fi
if [[ ! -f "$SINGLE_CKPT_SCRIPT" ]]; then
    echo "ERROR: Single-checkpoint exporter not found: ${SINGLE_CKPT_SCRIPT}" >&2
    exit 1
fi

if [[ -z "$RETENTION_MANIFEST" ]]; then
    RETENTION_MANIFEST="${OUT_BASE%/}/checkpoints_to_delete.txt"
fi

_collect_all_iter_dirs() {
    local ckpt_dir="$1"
    find "$ckpt_dir" -maxdepth 1 -mindepth 1 -type d -name 'iter_*' -printf '%f\t%p\n' \
        | awk -F '\t' '$1 ~ /^iter_[0-9]+$/ { printf "%020d\t%s\n", substr($1, 6) + 0, $2 }' \
        | sort -n \
        | cut -f2-
}

_resolve_latest_iter_dir() {
    local ckpt_dir="$1"
    local tracker="${ckpt_dir}/latest_checkpointed_iteration.txt"
    if [[ -f "$tracker" ]]; then
        local it=""
        IFS= read -r it < "$tracker" || true
        it="${it//[[:space:]]/}"
        if [[ "$it" =~ ^[0-9]+$ ]]; then
            local cand
            cand="${ckpt_dir}/$(printf 'iter_%07d' "$it")"
            if [[ -d "$cand" ]]; then
                echo "$cand"
                return
            fi
            echo "WARNING: latest_checkpointed_iteration.txt points to missing ${cand}; falling back to discovered latest." >&2
        fi
    fi

    local all_iter_dirs=()
    mapfile -t all_iter_dirs < <(_collect_all_iter_dirs "$ckpt_dir")
    if [[ ${#all_iter_dirs[@]} -eq 0 ]]; then
        echo "ERROR: No iter_* directories found under ${ckpt_dir}" >&2
        exit 1
    fi
    echo "${all_iter_dirs[$((${#all_iter_dirs[@]} - 1))]}"
}

_collect_iter_dirs_to_export() {
    local ckpt_dir="$1"

    if [[ "$LATEST_ONLY" -eq 1 ]]; then
        _resolve_latest_iter_dir "$ckpt_dir"
        return
    fi

    local all_iter_dirs=()
    mapfile -t all_iter_dirs < <(_collect_all_iter_dirs "$ckpt_dir")
    if [[ ${#all_iter_dirs[@]} -eq 0 ]]; then
        echo "WARNING: No iter_* directories found under ${ckpt_dir}; skipping." >&2
        return
    fi

    if [[ ${#EXPLICIT_ITERS[@]} -gt 0 ]]; then
        local iter_num tag cand
        for iter_num in "${EXPLICIT_ITERS[@]}"; do
            if [[ ! "$iter_num" =~ ^[0-9]+$ ]]; then
                echo "ERROR: --iters values must be numeric, got: ${iter_num}" >&2
                exit 1
            fi
            tag="$(printf 'iter_%07d' "$iter_num")"
            cand="${ckpt_dir}/${tag}"
            if [[ -d "$cand" ]]; then
                echo "$cand"
            else
                echo "WARNING: Requested iter ${iter_num} (${tag}) not found under ${ckpt_dir}; skipping." >&2
            fi
        done
    else
        printf '%s\n' "${all_iter_dirs[@]}"
    fi
}

_add_retention_candidates() {
    local ckpt_dir="$1"
    local all_iter_dirs=()
    mapfile -t all_iter_dirs < <(_collect_all_iter_dirs "$ckpt_dir")
    [[ ${#all_iter_dirs[@]} -eq 0 ]] && return

    case "$KEEP_ORIGINAL_CHECKPOINTS" in
        all)
            return 0
            ;;
        latest)
            local latest
            latest="$(_resolve_latest_iter_dir "$ckpt_dir")"
            local dir
            for dir in "${all_iter_dirs[@]}"; do
                [[ "$dir" == "$latest" ]] && continue
                RETENTION_CANDIDATES+=("$dir")
            done
            ;;
        none)
            RETENTION_CANDIDATES+=("${all_iter_dirs[@]}")
            ;;
    esac
}

if [[ "$DRY_RUN" -eq 0 ]]; then
    mkdir -p "$OUT_BASE"
    mkdir -p "$(dirname "$RETENTION_MANIFEST")"
fi

echo "========================================"
echo "export_megatron_runs_to_hf.sh - batch checkpoint export"
echo "  HF reference              : $HF_MODEL"
echo "  out-base                  : $OUT_BASE"
echo "  bridge-root               : $BRIDGE_ROOT"
echo "  latest-only               : $LATEST_ONLY"
echo "  keep original checkpoints : $KEEP_ORIGINAL_CHECKPOINTS"
echo "  retention manifest        : $RETENTION_MANIFEST"
echo "  dry-run                   : $DRY_RUN"
echo "  sequential                : $SEQUENTIAL"
[[ ${#EXPLICIT_ITERS[@]} -gt 0 ]] && echo "  iters filter              : ${EXPLICIT_ITERS[*]}"
[[ ${#EXTRA_PY_ARGS[@]} -gt 0 ]] && echo "  extra args                : ${EXTRA_PY_ARGS[*]}"
echo "  run roots:"
for root in "${RUN_ROOTS[@]}"; do
    echo "    $root"
done
echo "========================================"

TOTAL_SUBMITTED=0
TOTAL_SKIPPED=0

for root in "${RUN_ROOTS[@]}"; do
    root="${root%/}"
    if [[ ! -d "$root" ]]; then
        echo "ERROR: Run root does not exist: ${root}" >&2
        exit 1
    fi

    ckpt_dir="${root}/checkpoints"
    if [[ ! -d "$ckpt_dir" ]]; then
        echo "ERROR: Run root does not contain checkpoints/: ${root}" >&2
        exit 1
    fi

    run_name="$(basename "$root")"
    echo ""
    echo "--- Run: ${run_name} (${root})"

    _add_retention_candidates "$ckpt_dir"

    iter_dirs=()
    mapfile -t iter_dirs < <(_collect_iter_dirs_to_export "$ckpt_dir")
    if [[ ${#iter_dirs[@]} -eq 0 ]]; then
        echo "  (no iterations to export)"
        continue
    fi

    echo "  Found ${#iter_dirs[@]} iteration(s) to consider:"
    for dir in "${iter_dirs[@]}"; do
        echo "    $(basename "$dir")"
    done

    dep=""
    for iter_dir in "${iter_dirs[@]}"; do
        iter_tag="$(basename "$iter_dir")"
        out="${OUT_BASE%/}/${run_name}/${iter_tag}"

        # Resolve to absolute paths so the job survives --chdir to BRIDGE_ROOT.
        abs_iter_dir="$(realpath "$iter_dir")"
        abs_hf_model="$(realpath "$HF_MODEL")"
        abs_out="$(realpath -m "$out")"

        if [[ -d "$out" && -n "$(find "$out" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
            echo "  [skip] ${out} already exists and is non-empty"
            (( TOTAL_SKIPPED++ )) || true
            continue
        fi

        cmd=(sbatch --chdir "$BRIDGE_ROOT")
        if [[ "$SEQUENTIAL" -eq 1 && -n "$dep" ]]; then
            cmd+=(--dependency="afterok:${dep}")
        fi
        cmd+=("$SINGLE_CKPT_SCRIPT" "$abs_iter_dir" "$abs_hf_model" "$abs_out")
        cmd+=("${EXTRA_PY_ARGS[@]+"${EXTRA_PY_ARGS[@]}"}")

        printf '  [submit] '
        _print_cmd "${cmd[@]}"

        if [[ "$DRY_RUN" -eq 0 ]]; then
            sbatch_out="$("${cmd[@]}")"
            echo "           -> ${sbatch_out}"
            job_id="$(awk '{print $NF}' <<< "$sbatch_out")"
            dep="$job_id"
            (( TOTAL_SUBMITTED++ )) || true
        fi
    done
done

echo ""
echo "========================================"
echo "Retention manifest"
echo "  mode : ${KEEP_ORIGINAL_CHECKPOINTS}"
echo "  path : ${RETENTION_MANIFEST}"
if [[ ${#RETENTION_CANDIDATES[@]} -eq 0 ]]; then
    echo "  checkpoints flagged for deletion: 0"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        : > "$RETENTION_MANIFEST"
    fi
else
    echo "  checkpoints flagged for deletion: ${#RETENTION_CANDIDATES[@]}"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        printf '%s\n' "${RETENTION_CANDIDATES[@]}" > "$RETENTION_MANIFEST"
        echo "  wrote: ${RETENTION_MANIFEST}"
    else
        echo "  dry-run contents:"
        printf '    %s\n' "${RETENTION_CANDIDATES[@]}"
    fi
fi
echo "========================================"

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Dry-run complete. No jobs were submitted and no manifest was written."
else
    echo "Done. Submitted: ${TOTAL_SUBMITTED}  Skipped: ${TOTAL_SKIPPED}"
fi

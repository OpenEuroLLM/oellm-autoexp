#!/bin/bash

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <training_dir>"
    exit 1
fi
TRAINING_DIR="$1"

printf "%-60s %-30s %s\n" "RUN" "LOG FILE" "SPLIT"
printf "%-60s %-30s %s\n" "$(printf '%0.s-' {1..60})" "$(printf '%0.s-' {1..30})" "----------"

declare -A run_distinct_splits  # run_name -> space-separated distinct splits (excluding NOT FOUND)
declare -A run_all_splits        # run_name -> space-separated all splits including NOT FOUND

for run_dir in "$TRAINING_DIR"/*/; do
    run_name=$(basename "$run_dir")
    logs_dir="$run_dir/logs"

    if [[ ! -d "$logs_dir" ]]; then
        printf "%-60s %-30s %s\n" "$run_name" "-" "NO LOGS DIR"
        continue
    fi

    stdout_files=("$logs_dir"/stdout-*.log)
    if [[ ! -e "${stdout_files[0]}" ]]; then
        printf "%-60s %-30s %s\n" "$run_name" "-" "NO STDOUT FILE"
        continue
    fi

    declare -A seen=()

    for stdout_file in "${stdout_files[@]}"; do
        log_name=$(basename "$stdout_file")
        split_line=$(grep -am1 '^\[default0\]:  split ' "$stdout_file" 2>/dev/null)
        if [[ -n "$split_line" ]]; then
            split_value=$(echo "$split_line" | sed 's/.*split \.*[ \t]*//')
            printf "%-60s %-30s %s\n" "$run_name" "$log_name" "$split_value"
            seen["$split_value"]=1
        else
            printf "%-60s %-30s %s\n" "$run_name" "$log_name" "NOT FOUND"
        fi
    done

    run_distinct_splits["$run_name"]="${!seen[*]}"
    unset seen
done

SEP=$(printf '%0.s-' {1..95})

# ── Summary 1: runs with inconsistent splits across their own logs ───────────
echo ""
echo "SUMMARY 1: Stable runs with differing splits across their own log files"
printf "%-60s %s\n" "$SEP" ""
printf "%-60s %s\n" "RUN" "SPLITS FOUND"
printf "%-60s %s\n" "$SEP" ""

found_any=0
for run_name in "${!run_distinct_splits[@]}"; do
    splits="${run_distinct_splits[$run_name]}"
    count=$(echo "$splits" | wc -w)
    if [[ $count -gt 1 ]]; then
        printf "%-60s %s\n" "$run_name" "$splits"
        found_any=1
    fi
done
[[ $found_any -eq 0 ]] && echo "  (none)"

# ── Summary 2: decay runs whose split differs from their stable counterpart ──
echo ""
echo "SUMMARY 2: Decay runs with a different split than their stable counterpart"
printf "%-60s %-40s %s\n" "$SEP" "" ""
printf "%-60s %-40s %-30s %s\n" "DECAY RUN" "STABLE RUN" "DECAY SPLITS" "STABLE SPLITS"
printf "%-60s %-40s %-30s %s\n" "$SEP" "" "" ""

found_any=0
for run_name in "${!run_distinct_splits[@]}"; do
    # Only process decay runs that have at least one found split
    [[ "$run_name" != *_decay* ]] && continue
    decay_splits="${run_distinct_splits[$run_name]}"
    [[ -z "$decay_splits" ]] && continue

    # Derive base prefix by stripping _decay<anything>
    base="${run_name%_decay*}"

    # Find matching stable run(s) with same prefix
    stable_run=""
    for candidate in "${!run_distinct_splits[@]}"; do
        if [[ "$candidate" == "${base}_stable"* ]]; then
            stable_run="$candidate"
            break
        fi
    done

    [[ -z "$stable_run" ]] && continue
    stable_splits="${run_distinct_splits[$stable_run]}"
    [[ -z "$stable_splits" ]] && continue

    # Compare: flag if the sets differ
    if [[ "$decay_splits" != "$stable_splits" ]]; then
        printf "%-60s %-40s %-30s %s\n" "$run_name" "$stable_run" "$decay_splits" "$stable_splits"
        found_any=1
    fi
done
[[ $found_any -eq 0 ]] && echo "  (none)"

#!/bin/bash

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 <training_dir> [output_csv_path]"
    exit 1
fi
TRAINING_DIR="$1"
CSV_PATH="${2:-}"

printf "%-60s %-30s %s\n" "RUN" "LOG FILE" "SPLIT"
printf "%-60s %-30s %s\n" "$(printf '%0.s-' {1..60})" "$(printf '%0.s-' {1..30})" "----------"

declare -A run_distinct_splits  # run_name -> space-separated distinct splits (excluding NOT FOUND)
declare -A run_all_splits        # run_name -> space-separated all splits including NOT FOUND
declare -A run_100_logs          # run_name -> space-separated log files with split 100,0,0
declare -A run_found_logs        # run_name -> count of log files that reported any split value
declare -A split_counts          # "run_name|split_value" -> count of log files reporting that split

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
            run_found_logs["$run_name"]=$(( ${run_found_logs[$run_name]:-0} + 1 ))
            split_counts["${run_name}|${split_value}"]=$(( ${split_counts["${run_name}|${split_value}"]:-0} + 1 ))

            split_normalized=$(echo "$split_value" | tr -d '[:space:]')
            if [[ "$split_normalized" == "100,0,0" ]]; then
                run_100_logs["$run_name"]="${run_100_logs[$run_name]:-} $log_name"
            fi
        else
            printf "%-60s %-30s %s\n" "$run_name" "$log_name" "NOT FOUND"
        fi
    done

    run_distinct_splits["$run_name"]="${!seen[*]}"
    unset seen
done

SEP=$(printf '%0.s-' {1..95})

# Prints, for a run with inconsistent splits, how many/what % of its found
# logs reported each distinct split value. No-op if the run only has one
# distinct split.
print_split_breakdown() {
    local run_name="$1"
    local label="${2:-}"
    local splits="${run_distinct_splits[$run_name]}"
    local total=${run_found_logs[$run_name]:-0}
    local count
    count=$(echo "$splits" | wc -w)
    [[ "$count" -le 1 ]] && return

    local split_value cnt pct
    for split_value in $splits; do
        cnt=${split_counts["${run_name}|${split_value}"]:-0}
        pct="0.0"
        [[ "$total" -gt 0 ]] && pct=$(awk -v c="$cnt" -v t="$total" 'BEGIN{printf "%.1f", (c/t)*100}')
        printf "      %s%s: %d/%d logs (%s%%)\n" "${label:+$label }" "$split_value" "$cnt" "$total" "$pct"
    done
}

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
        print_split_breakdown "$run_name"
        found_any=1
    fi
done
[[ $found_any -eq 0 ]] && echo "  (none)"

# ── Summary 2/3: decay runs whose split differs from their stable counterpart,
# split into two tables depending on whether the stable counterpart itself had
# a consistent split across its own log files. ───────────────────────────────
echo ""
echo "SUMMARY 2: Decay runs with a different split than their stable counterpart (stable also had different splits)"
printf "%-60s %-40s %s\n" "$SEP" "" ""
printf "%-60s %-40s %-30s %s\n" "DECAY RUN" "STABLE RUN" "DECAY SPLITS" "STABLE SPLITS"
printf "%-60s %-40s %-30s %s\n" "$SEP" "" "" ""

declare -A flagged_in_summary2  # run_name -> 1 if flagged as mismatched decay run in Summary 2 or 3

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
    [[ "$decay_splits" == "$stable_splits" ]] && continue

    stable_split_count=$(echo "$stable_splits" | wc -w)
    [[ "$stable_split_count" -le 1 ]] && continue

    printf "%-60s %-40s %-30s %s\n" "$run_name" "$stable_run" "$decay_splits" "$stable_splits"
    print_split_breakdown "$run_name" "DECAY"
    print_split_breakdown "$stable_run" "STABLE"
    found_any=1
    flagged_in_summary2["$run_name"]=1
done
[[ $found_any -eq 0 ]] && echo "  (none)"

echo ""
echo "SUMMARY 3: Decay runs with a different split than their stable counterpart (stable was trained with consistent split)"
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
    [[ "$decay_splits" == "$stable_splits" ]] && continue

    stable_split_count=$(echo "$stable_splits" | wc -w)
    [[ "$stable_split_count" -gt 1 ]] && continue

    printf "%-60s %-40s %-30s %s\n" "$run_name" "$stable_run" "$decay_splits" "$stable_splits"
    print_split_breakdown "$run_name" "DECAY"
    found_any=1
    flagged_in_summary2["$run_name"]=1
done
[[ $found_any -eq 0 ]] && echo "  (none)"

# ── Summary 4: runs where every log file that reports a split shows 100,0,0, ─
# excluding runs already flagged in Summary 2/3 as a decay run with a ───────
# mismatched split (logs with no split found at all are ignored). ──────────
echo ""
echo "SUMMARY 4: Runs trained with split 100,0,0 (all reported logs, incl if their stable was also fully trained with 100,0,0)"
printf "%-60s %s\n" "$SEP" ""
printf "%-60s %s\n" "RUN" "STABLE RUN"
printf "%-60s %s\n" "$SEP" ""

found_any=0
for run_name in "${!run_100_logs[@]}"; do
    [[ -n "${flagged_in_summary2[$run_name]:-}" ]] && continue
    logs="${run_100_logs[$run_name]# }"
    count_100=$(echo "$logs" | wc -w)
    found=${run_found_logs[$run_name]:-0}
    [[ "$count_100" -eq "$found" && "$found" -gt 0 ]] || continue

    if [[ "$run_name" == *_decay* ]]; then
        base="${run_name%_decay*}"
        stable_run="-"
        for candidate in "${!run_distinct_splits[@]}"; do
            if [[ "$candidate" == "${base}_stable"* ]]; then
                stable_run="$candidate"
                break
            fi
        done
    elif [[ "$run_name" == *_stable* ]]; then
        stable_run="$run_name"
    else
        stable_run="-"
    fi

    printf "%-60s %s\n" "$run_name" "$stable_run"
    found_any=1
done
[[ $found_any -eq 0 ]] && echo "  (none)"

# ── CSV export: stable runs that used split 100,0,0 in some or all logs ──────
if [[ -n "$CSV_PATH" ]]; then
    echo "run,logs_with_100_0_0,total_logs_with_split,coverage" > "$CSV_PATH"

    for run_name in "${!run_100_logs[@]}"; do
        [[ "$run_name" != *_stable* ]] && continue
        logs="${run_100_logs[$run_name]# }"
        count_100=$(echo "$logs" | wc -w)
        found=${run_found_logs[$run_name]:-0}
        [[ "$count_100" -eq 0 ]] && continue

        if [[ "$count_100" -eq "$found" ]]; then
            coverage="all"
        else
            coverage="partial"
        fi

        echo "${run_name},${count_100},${found},${coverage}" >> "$CSV_PATH"
    done

    echo ""
    echo "CSV written to: $CSV_PATH"
fi

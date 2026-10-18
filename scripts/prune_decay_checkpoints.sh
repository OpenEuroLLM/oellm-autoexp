#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Conservative first-pass checkpoint prune: keep ONLY the latest checkpoint in
# each DECAY (cooldown) job directory, delete every older intermediate.
#
# Scope on purpose (safety):
#   * Touches ONLY *_decay*BT dirs (the firstcd/contcd cooldown dirs). The
#     stable dirs — which the cooldowns branch from — are NEVER touched here.
#   * "Latest" = the iteration named in latest_checkpointed_iteration.txt, i.e.
#     exactly what Megatron would load on restart. For a finished cooldown that
#     is the fully-decayed ENDPOINT; for a running one it is the last good save.
#   * We KEEP the tracker iter AND anything strictly NEWER than it (a possibly
#     in-progress save), and delete only iters strictly OLDER. So this is safe
#     to run even while a cooldown is training — a live job never re-reads the
#     old intermediates, and its restart point is always kept.
#   * If the tracker is missing/unreadable, or the iter it names does not exist
#     on disk, the whole dir is SKIPPED (nothing deleted) and a warning printed.
#
# Dry-run by default: prints exactly what it would delete and the reclaimable
# size. Nothing is removed unless you pass --apply.
#
# Usage (run on JUPITER, where the scratch tree is local):
#   scripts/prune_decay_checkpoints.sh                 # dry-run (default)
#   scripts/prune_decay_checkpoints.sh --apply         # actually delete
#   scripts/prune_decay_checkpoints.sh --apply BASE    # override base dir
# ---------------------------------------------------------------------------
set -euo pipefail

APPLY=0
BASE="/e/scratch/e-sta-openeurollm/poeppel1/output/architecture_scaling_variants_multilingual_main_7to1"
for a in "$@"; do
  case "$a" in
    --apply) APPLY=1 ;;
    /*)      BASE="$a" ;;
    *)       echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done

[ -d "$BASE" ] || { echo "base dir not found: $BASE" >&2; exit 1; }
if [ "$APPLY" -eq 1 ]; then echo ">>> APPLY mode: checkpoints WILL be deleted."; else echo ">>> DRY-RUN (no deletions). Re-run with --apply to delete."; fi
echo ">>> base: $BASE"
echo

total_bytes=0
total_del=0
total_dirs=0
skipped=0

# Names of currently queued/running jobs. A firstcd dir is written by BOTH its
# own firstcd job and the corresponding contcd job (contcd saves into the firstcd
# dir), so we skip a dir if EITHER is active — belt-and-suspenders on top of the
# keep-tracker-and-newer rule. Job names carry a trailing _<index>, so match by
# substring on the dir basename.
active="$(squeue -u "${USER}" -h -o '%j' 2>/dev/null || true)"
job_active() { printf '%s\n' "$active" | grep -qF -- "$1"; }

# Only cooldown dirs. (contcd dirs are log-only; the loop no-ops on them.)
shopt -s nullglob
for d in "$BASE"/*_decay*BT/; do
  d="${d%/}"
  iters=( "$d"/iter_* )                           # nullglob -> empty array if none
  [ "${#iters[@]}" -eq 0 ] && continue            # nothing to prune (e.g. contcd log dirs)
  total_dirs=$((total_dirs+1))

  base_job="$(basename "$d")"                     # qwen3_..._firstcd_decayXBT
  if job_active "$base_job" || job_active "${base_job/firstcd/contcd}"; then
    echo "SKIP (job active): $base_job"; skipped=$((skipped+1)); continue
  fi

  tracker_file="$d/latest_checkpointed_iteration.txt"
  if [ ! -r "$tracker_file" ]; then
    echo "SKIP (no tracker): $(basename "$d")"; skipped=$((skipped+1)); continue
  fi
  keep_raw="$(tr -dc '0-9' < "$tracker_file")"
  if [ -z "$keep_raw" ]; then
    echo "SKIP (empty tracker): $(basename "$d")"; skipped=$((skipped+1)); continue
  fi
  keep=$((10#$keep_raw))
  keep_dir="$d/iter_$(printf '%07d' "$keep")"
  if [ ! -d "$keep_dir" ]; then
    echo "SKIP (tracker $keep has no iter dir): $(basename "$d")"; skipped=$((skipped+1)); continue
  fi

  # Report a decision for EVERY iter in this dir, in ascending order, so each
  # DELETE can be checked by eye against the later KEEP that supersedes it.
  # Rule: delete iters strictly older than the tracker; keep the tracker and
  # anything newer. Iter dirs are zero-padded to 7 digits, so a plain lexical
  # sort is already numeric order.
  echo "$(basename "$d")  (latest kept = iter_$(printf '%07d' "$keep"))"
  dir_bytes=0
  ndel=0
  while IFS= read -r it; do
    n=$((10#$(basename "$it" | tr -dc '0-9')))
    if [ "$n" -lt "$keep" ]; then
      b=$(du -sB1 "$it" 2>/dev/null | cut -f1); dir_bytes=$((dir_bytes + ${b:-0})); ndel=$((ndel+1))
      note="superseded by kept iter_$(printf '%07d' "$keep")"
      if [ "$APPLY" -eq 1 ]; then
        rm -rf "$it" && printf '    DELETE  iter_%07d  (%s)  [removed]\n' "$n" "$note"
      else
        printf '    DELETE  iter_%07d  (%s)\n'           "$n" "$note"
      fi
    elif [ "$n" -eq "$keep" ]; then
      printf   '    KEEP    iter_%07d  (latest = tracker / endpoint)\n' "$n"
    else
      printf   '    KEEP    iter_%07d  (newer than tracker; in-progress, kept)\n' "$n"
    fi
  done < <(printf '%s\n' "${iters[@]}" | sort)
  total_bytes=$((total_bytes + dir_bytes))
  total_del=$((total_del + ndel))
  printf '    -> %d delete, %d keep, %s GB freed here\n' \
    "$ndel" "$(( ${#iters[@]} - ndel ))" "$(awk "BEGIN{printf \"%.1f\", $dir_bytes/1e9}")"
done

echo
printf '=== %s: %d decay dirs with checkpoints, %d skipped, %d checkpoints %s, %s GB %s ===\n' \
  "$([ "$APPLY" -eq 1 ] && echo APPLIED || echo DRY-RUN)" \
  "$total_dirs" "$skipped" "$total_del" \
  "$([ "$APPLY" -eq 1 ] && echo deleted || echo deletable)" \
  "$(awk "BEGIN{printf \"%.1f\", $total_bytes/1e9}")" \
  "$([ "$APPLY" -eq 1 ] && echo reclaimed || echo reclaimable)"

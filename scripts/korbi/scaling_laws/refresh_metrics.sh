#!/usr/bin/env bash
# Regenerate the eval val-loss metrics CSV from the eval logs on JUPITER.
# Writes dump/eval_val_metrics.csv (arch,size,gbs,lr,type,budgetBT,val_loss,val_ppl,iter),
# which scripts/korbi/scaling_laws/run.py consumes. Re-run as more evals complete.
#
# Usage: scripts/korbi/scaling_laws/refresh_metrics.sh [SSH_HOST] [OUT_CSV]
set -euo pipefail
HOST="${1:-jupiter}"
OUT="${2:-dump/eval_val_metrics.csv}"
# Eval groups whose cells pair with the main_7to1 ladder (same val blend, same
# (gbs, lr) tiers). matched_0p4B_eval: mlstm7gqa_nope / ckda3_qwen35 at 0.4B.
OUT_ROOT="/e/scratch/e-sta-openeurollm/poeppel1/output"
EBS="$OUT_ROOT/architecture_scaling_variants_multilingual_main_7to1_eval $OUT_ROOT/architecture_scaling_variants_multilingual_matched_0p4B_eval"

# Never redirect straight into $OUT: the shell truncates it before ssh even runs,
# so an auth failure or a dropped connection silently destroys the last good
# snapshot. Stage into a temp file and only replace $OUT once the pull looks sane.
TMP="$(mktemp "${TMPDIR:-/tmp}/eval_val_metrics.XXXXXX.csv")"
trap 'rm -f "$TMP"' EXIT

# `|| true`: a failed ssh must fall through to the row check below, which prints
# something actionable, rather than tripping `set -e` with no explanation.
ssh "$HOST" bash -s <<EOF 2>/dev/null | grep -vE 'GCCcore|reloaded with' > "$TMP" || true
EBS="$EBS"
echo "variant,size,gbs,lr,type,budgetBT,val_loss,val_ppl,iter"
for d in \$(for eb in \$EBS; do ls -d \$eb/*eval*/ 2>/dev/null; done); do
  nm=\$(basename \$d)
  gbs=\$(echo \$nm | grep -oE "gbsz[0-9]+" | grep -oE "[0-9]+")
  typ=\$(echo \$nm | grep -oE "evalcd|evalstable")
  bud=\$(echo \$nm | grep -oE "decay[0-9]+BT|branch[0-9]+BT" | grep -oE "[0-9]+")
  # Checkpoint iteration this eval must report (as extract.expected_iter):
  # the budget's end for a cooldown, 0.8x of it for a stable branch.
  want=\$(( (bud*1000000000 + 4096*gbs - 1) / (4096*gbs) ))
  [ "\$typ" = evalstable ] && want=\$(( want*8/10 ))
  # Newest log whose validation line is AT that iteration; else the newest
  # log with any validation line (extract.py then drops it, loudly). Neither
  # "newest log" nor "newest with a validation line" is safe on its own: a
  # re-submission leaves a newer log with no result, and a bad re-eval (the
  # non-persistent --ckpt-step override) a newer result at the wrong iteration.
  m=""; fb=""
  for L in \$(ls -t \$d/slurm-*.log 2>/dev/null); do
    v=\$(grep -E "validation loss.*validation set" "\$L" | tail -1); [ -z "\$v" ] && continue
    [ -z "\$fb" ] && fb=\$v
    echo "\$v" | grep -qE "iteration \$want " && { m=\$v; break; }
  done
  [ -z "\$m" ] && m=\$fb
  [ -z "\$m" ] && continue
  var=\$(echo \$nm | sed -E "s/qwen3_([a-z0-9_]+)_0\.[0-9]B.*/\1/")
  sz=\$(echo \$nm | grep -oE "0\.[0-9]B" | head -1)
  lr=\$(echo \$nm | grep -oE "lr[0-9.]+" | sed "s/lr//")
  loss=\$(echo "\$m" | grep -oE "lm loss value: [0-9.E+]+" | grep -oE "[0-9.E+]+\$")
  ppl=\$(echo "\$m" | grep -oE "PPL: [0-9.E+]+" | grep -oE "[0-9.E+]+\$")
  it=\$(echo "\$m" | grep -oE "iteration [0-9]+" | grep -oE "[0-9]+")
  echo "\$var,\$sz,\$gbs,\$lr,\$typ,\$bud,\$loss,\$ppl,\$it"
done | sort -t, -k1,1 -k2,2 -k6,6n
EOF

rows=$(( $(wc -l < "$TMP") - 1 ))
if [ "$rows" -lt 1 ]; then
  echo "ERROR: pull from '$HOST' returned no data rows -- keeping the existing $OUT." >&2
  echo "       Check ssh access:  ssh $HOST true" >&2
  echo "       (a passphrase key needs a live agent: eval \"\$(ssh-agent -s)\" && ssh-add)" >&2
  exit 1
fi
if [ -f "$OUT" ]; then
  old=$(( $(wc -l < "$OUT") - 1 ))
  if [ "$rows" -lt "$old" ]; then
    echo "WARNING: pull has $rows rows but $OUT had $old -- evals do not disappear," >&2
    echo "         so this is more likely a partial pull. Previous file kept as $OUT.bak" >&2
    cp "$OUT" "$OUT.bak"
  fi
fi
mv "$TMP" "$OUT"
trap - EXIT
echo "wrote $rows rows to $OUT"

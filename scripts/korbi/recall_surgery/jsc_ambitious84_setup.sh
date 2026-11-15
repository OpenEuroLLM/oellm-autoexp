#!/bin/bash
# JUPITER data layout for the 84B ambitious runs (docs/recall_surgery_ambitious.md §19.1). Run on a JUPITER login node
# after the MN5 -> fscratch transfers (hpctransfer: checkpoints, stage-1 .idx, tokenizers, engram map, datamix/mn5).
# Idempotent; writes only below $A.
#
#   stage 2 (v8ml-stages/s3, the opt81 8k64k stage-2 mix): the JSC mirror oellm_testing (byte-identical to MN5:
#     jsc_data_audit.py, 531 / 531 prefixes, sizes + 16-range .bin hashes + full .idx hashes)
#   stage 1 (v8ml-stages-8k64k/s1): on MN5 its .bin are the SAME inodes as s3's (all 531 checked); only the .idx
#     differ. So: the s1 .idx copied from MN5 + a symlink per .bin to the JSC s3 .bin.
#   datamix files: MN5 paths -> JSC paths, every prefix checked to exist with .bin + .idx.
#   run roots: the main run's 35603 / rolling 35900 hard-linked into runs/main_v4_jsc (Megatron loads the newer),
#     the transferred copies stay untouched (a pool restart re-runs from `load`).
set -euo pipefail
A=${A:-/e/fscratch/e-sta-openeurollm/poeppel1/ambitious84}
T=${T:-/e/data1/datasets/playground/mmlaion/oellm_testing}
MN5_DATA=/gpfs/projects/ehpc533/users/korbip/data

# --- stage-1 tree: .bin symlinks next to the transferred .idx ---
s1=$A/data/v8ml-stages-8k64k/s1
n=0; miss=0
while read -r w p; do
  case $p in "$MN5_DATA"/v8ml-stages-8k64k/s1/*) ;; *) continue ;; esac
  rel=${p#"$MN5_DATA"/v8ml-stages-8k64k/s1/}
  src=$(realpath "$T/v8ml-stages/s3/$rel.bin")
  [ -f "$s1/$rel.idx" ] || { echo "missing s1 idx: $rel" >&2; miss=$((miss + 1)); continue; }
  [ -f "$src" ] || { echo "missing JSC s3 bin: $rel" >&2; miss=$((miss + 1)); continue; }
  [ -e "$s1/$rel.bin" ] || ln -s "$src" "$s1/$rel.bin"
  n=$((n + 1))
done < <(tr -s ' \n' '\n' < "$A/datamix/mn5/datamix_v8ml_opt81_8k64k_stage1_mn5.txt" | paste - -)
echo "stage-1 tree: $n prefixes linked, $miss missing"
[ "$miss" -eq 0 ]

# --- datamix files with JSC paths ---
sed "s#$MN5_DATA/v8ml-stages-8k64k/s1/#$s1/#g" "$A/datamix/mn5/datamix_v8ml_opt81_8k64k_stage1_mn5.txt" \
  > "$A/datamix/datamix_v8ml_opt81_8k64k_stage1_jsc.txt"
sed "s#$MN5_DATA/#$T/#g" "$A/datamix/mn5/datamix_v8ml_opt81_8k64k_stage2_mn5.txt" \
  > "$A/datamix/datamix_v8ml_opt81_8k64k_stage2_jsc.txt"
for f in "$A"/datamix/datamix_v8ml_opt81_8k64k_stage{1,2}_jsc.txt; do
  if grep -q /gpfs/ "$f"; then echo "MN5 path left in $f" >&2; exit 1; fi
  bad=0; cnt=0
  for p in $(tr -s ' \n' '\n' < "$f" | grep '^/'); do
    cnt=$((cnt + 1))
    [ -e "$p.bin" ] && [ -e "$p.idx" ] || { echo "missing: $p" >&2; bad=$((bad + 1)); }
  done
  # same weights / order as the MN5 file (only the path prefixes differ)
  diff <(tr -s ' \n' '\n' < "$f" | grep -v '^/') \
       <(tr -s ' \n' '\n' < "$A/datamix/mn5/$(basename "${f/_jsc/_mn5}")" | grep -v '^/') > /dev/null \
    || { echo "weights differ: $f" >&2; exit 1; }
  echo "$(basename "$f"): $cnt prefixes, $bad missing, weights identical to MN5"
  [ "$bad" -eq 0 ]
done

# --- validation sets, tokenizer, engram map ---
for v in all web_en multilingual code math reasoning long_64k; do
  [ -e "$T/v8ml-valid/${v}_text_document.bin" ] && [ -e "$T/v8ml-valid/${v}_text_document.idx" ]
done
[ -f "$A/data/tokenizers/tokenizer-256k-agent/tokenizer.json" ]
[ -f "$A/data/engram_token_map_256k.npy" ]
echo "valid sets / tokenizer / engram map present"

# --- run roots (hard links; the transferred trees are never written by a run) ---
mkroot() {  # root src_persistent_it src_rolling_it
  local r=$1
  mkdir -p "$r/ckpt" "$r/ckpt_rolling"
  if [ -n "$2" ] && [ -d "$A/ckpt/main/iter_$2" ] && [ ! -e "$r/ckpt/iter_$2" ]; then
    cp -al "$A/ckpt/main/iter_$2" "$r/ckpt/iter_$2"; echo "$((10#$2))" > "$r/ckpt/latest_checkpointed_iteration.txt"
  fi
  if [ -n "$3" ] && [ -d "$A/ckpt/main_rolling/iter_$3" ] && [ ! -e "$r/ckpt_rolling/iter_$3" ]; then
    cp -al "$A/ckpt/main_rolling/iter_$3" "$r/ckpt_rolling/iter_$3"
    echo "$((10#$3))" > "$r/ckpt_rolling/latest_checkpointed_iteration.txt"
  fi
  return 0
}
# the broad-session ctrl copy as a run dir for ds84_surgery.sh's save_of (<CTRL>/ckpt_eval/iter_N)
mkdir -p "$A/ckpt/broad_ctrl"; [ -e "$A/ckpt/broad_ctrl/ckpt_eval" ] || ln -s ../broad_ctrl_eval "$A/ckpt/broad_ctrl/ckpt_eval"
mkroot "$A/runs/main_v4_jsc" 0035603 0035900
mkroot "$A/runs/smoke_v4_jsc" 0035603 0035900
ls -la "$A/runs/main_v4_jsc/ckpt" "$A/runs/main_v4_jsc/ckpt_rolling" 2>/dev/null | grep -v '^total'
echo "setup ok"

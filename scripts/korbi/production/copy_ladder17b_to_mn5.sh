#!/bin/bash
# Copy the 1.7B ladder's HF exports (wsd50B = the stable-phase-end checkpoint of the 1.7B rung) and the
# gpt-neox-20b tokenizer from jupiter to MareNostrum 5, streaming through this host (the two clusters have
# no direct route). Goal 2026-11-11: the three retrieval evals (CoQA / SQuAD v2 / Winogrande) on the
# 1.7B ckda-hybrid vs kda-hybrid vs full-attention arms, in the same FLAG harness as the ambitious v4.
set -u
DST=/gpfs/projects/ehpc533/users/korbip/hybrid_v019/hf_models/ladder17b
J=/e/scratch/e-sta-openeurollm/poeppel1/eval_megatron_ladder/export
ARMS="attn_1.7B_wsd50B skda-shipped-hybrid-lowrank_1.7B_wsd50B kda-sig-hybrid-lowrank_1.7B_wsd50B skda-shipped-lowrank_1.7B_wsd50B"

ssh marenostrum_new "mkdir -p $DST" || exit 1
for arm in $ARMS; do
  echo "=== $arm $(date +%T)"
  ssh marenostrum_new "mkdir -p $DST/$arm" || exit 1
  for f in config.json model.safetensors; do
    if [ "$f" = "config.json" ]; then
      ssh jupiter "cat $J/$arm/$f" 2>/dev/null | ssh marenostrum_new "cat > $DST/$arm/$f"
    else
      ssh jupiter "cat $J/$arm/$f" 2>/dev/null | ssh marenostrum_new "cat > $DST/$arm/$f.part && mv $DST/$arm/$f.part $DST/$arm/$f"
    fi
  done
  ssh marenostrum_new "ls -l $DST/$arm | tail -2"
  echo "=== $arm done $(date +%T)"
done

echo "=== tokenizer $(date +%T)"
ssh marenostrum_new "mkdir -p $DST/tokenizer_gpt_neox_20b"
for f in $(ssh jupiter "ls /e/scratch/jureap59/cache/tokenizer/gpt-neox-20b/" 2>/dev/null); do
  ssh jupiter "cat /e/scratch/jureap59/cache/tokenizer/gpt-neox-20b/$f" | ssh marenostrum_new "cat > $DST/tokenizer_gpt_neox_20b/$f"
done
ssh marenostrum_new "ls -l $DST/tokenizer_gpt_neox_20b"
echo "=== ALL DONE $(date +%T)"

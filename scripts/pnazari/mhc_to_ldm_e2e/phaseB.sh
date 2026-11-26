#!/bin/bash
# mhc_to_ldm e2e, phase B (inside run_in_env.sh): the surgery on ref's iter 20 and on its identity-mixer copy, then
# 5 resumed steps per arm; prints every arm's lm loss per iteration.
set -uo pipefail
E=/fast/pnazari/ldm-work/e2e
MEG=/fast/pnazari/ldm-work/Megatron-LM-rebased
cd $E
mkdir -p logs
IT=iter_0000020
rm -rf identity surgery
python3 identity_mixers.py runs/ref/ckpt/$IT identity/ckpt/$IT || exit 1
for src in ref identity; do
    s=runs/ref/ckpt/$IT; [ $src = identity ] && s=identity/ckpt/$IT
    python3 mhc_to_ldm.py --src $s --dst surgery/${src}_collapse/ckpt/$IT || exit 1
    python3 mhc_to_ldm.py --src $s --dst surgery/$src/ckpt/$IT --ldm || exit 1
done
python3 dump_keys.py surgery/ref/ckpt/$IT > keys_surgery_ldm.txt
python3 compare_keys.py keys_ldm_tpl.txt keys_surgery_ldm.txt || echo "LAYOUT MISMATCH"
for arm in ${ARMS:-ctrl_id collapse_id ldm_id ctrl collapse ldm}; do
    rm -rf runs/$arm
    eval "torchrun --standalone --nproc-per-node 4 $MEG/pretrain_hybrid.py $(cat args/$arm.args)" > logs/$arm.log 2>&1
    echo "$arm exit $?"
done
for arm in ${ARMS:-ctrl_id collapse_id ldm_id ctrl collapse ldm}; do
    echo "== $arm"
    grep -E "iteration +[0-9]+/" logs/$arm.log | grep -oE "iteration +[0-9]+|lm loss: [0-9.E+-]+|grad norm: [0-9.E+-]+" | paste -sd' ' | sed 's/iteration/\niteration/g'
done

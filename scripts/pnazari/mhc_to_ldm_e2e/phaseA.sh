#!/bin/bash
# mhc_to_ldm e2e, phase A (inside run_in_env.sh): the mHC model trained 20 steps and saved, the LDM model's
# checkpoint layout (2 steps, saved), and both key lists.
set -uo pipefail
E=/fast/pnazari/ldm-work/e2e
MEG=/fast/pnazari/ldm-work/Megatron-LM-rebased
cd $E
mkdir -p logs
for arm in ref ldm_tpl; do
    rm -rf runs/$arm
    eval "torchrun --standalone --nproc-per-node 4 $MEG/pretrain_hybrid.py $(cat args/$arm.args)" > logs/$arm.log 2>&1
    echo "$arm exit $?"
    grep -E "iteration +[0-9]+/" logs/$arm.log | tail -3
done
python3 dump_keys.py runs/ref/ckpt/iter_0000020 > keys_ref.txt && echo "ref keys $(wc -l < keys_ref.txt)"
python3 dump_keys.py runs/ldm_tpl/ckpt/iter_0000002 > keys_ldm_tpl.txt && echo "ldm_tpl keys $(wc -l < keys_ldm_tpl.txt)"

#!/bin/bash
# mhc_to_ldm e2e, phase C (inside run_in_env.sh): the negative control. The identity-mixer conversion without the
# V layers' g_v fold must miss ctrl_id at iteration 21 by far more than the converted arms do.
set -uo pipefail
E=/fast/pnazari/ldm-work/e2e
MEG=/fast/pnazari/ldm-work/Megatron-LM-rebased
cd $E
IT=iter_0000020
rm -rf surgery/identity_nofold runs/collapse_nofold
python3 unfold_v.py identity/ckpt/$IT surgery/identity_collapse/ckpt/$IT surgery/identity_nofold/ckpt/$IT || exit 1
sed "s#surgery/identity_collapse/ckpt#surgery/identity_nofold/ckpt#; s#runs/collapse_id/tb#runs/collapse_nofold/tb#" \
    args/collapse_id.args > args/collapse_nofold.args
eval "torchrun --standalone --nproc-per-node 4 $MEG/pretrain_hybrid.py $(cat args/collapse_nofold.args)" > logs/collapse_nofold.log 2>&1
echo "collapse_nofold exit $?"
for arm in ctrl_id collapse_id collapse_nofold; do
    echo "== $arm"
    grep -E "iteration +[0-9]+/" logs/$arm.log | grep -oE "iteration +[0-9]+|lm loss: [0-9.E+-]+" | paste -sd' ' | sed 's/iteration/\niteration/g'
done

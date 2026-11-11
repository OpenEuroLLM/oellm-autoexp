#!/bin/bash
# Continue a native 1.7B ladder checkpoint (signed_kda's own Megatron + lm_scaling.megatron_ext.specs.arm_spec)
# on MareNostrum 5 -- the small-scale testbed of the recall surgery (docs/recall_surgery_ambitious.md, section 8).
# Run once per node under srun (config/experiments/korbi/recall_surgery/ladder17b_continue_mn5.yaml): one torchrun
# per node, rank from SLURM_NODEID, master = the first host of the step.
#
# Environment (all set by the config):
#   SKDA_ARM            skda-shipped-hybrid-lowrank | skda-shipped-lowrank | attn | kda-sig-hybrid-lowrank
#   SKDA_INTERVENTION   intervention name read by a patched megatron_ext (empty: none)
#   CODE_DIR            dir with lm_scaling/ (+ fla/ native_megatron/) -- an intervention's patched copy, or 018's source
#   BASE_CODE_DIR       018's source (fla + native Megatron come from here unless CODE_DIR has them)
#   SRC_CKPT            native checkpoint dir (has latest_checkpointed_iteration.txt) -- the stable-phase end
#   SAVE_DIR            this run's checkpoints; a restart resumes from here once it holds a checkpoint
#   LOAD_MODE           resume      = optimizer + scheduler + iteration from SRC_CKPT (stable continuation)
#                       model_only  = weights only (--finetune --no-load-optim --no-load-rng): iteration 0, fresh Adam,
#                                     linear warmup WARMUP iters; keys absent from the checkpoint (new parameters)
#                                     keep their init (native torch-format loader falls back to strict=False)
#   STEPS               training steps of this run (resume: on top of the source's iteration; model_only: from 0);
#                       the lr is constant after the warmup (WSD, no decay phase: lr-decay-iters past the end)
#   DECAY               > 0: anneal the lr linearly to min-lr over the LAST DECAY steps of this run (WSD tail ending at
#                       its last iteration) -- the decay probe of an arm's checkpoint (LOAD_MODE resume, SRC_CKPT = the
#                       arm's own save dir, so optimizer + data position continue)
#   RESUME_WARMUP       warmup iters a resume declares (2000 = the native ladder's; an arm's checkpoint: its WARMUP)
#   GBS MBS SAVE_INTERVAL LR WARMUP FFN LOG_INTERVAL   (FFN: the arm's ffn hidden size)
#   EXTRA_ARGS          appended verbatim
#   ENTRY / NPROC       another script than pretrain_gpt.py with the same CLI (native_loss.py), processes per node
set -euo pipefail
: "${SKDA_ARM:?}" "${SRC_CKPT:?}" "${SAVE_DIR:?}"
BASE_CODE_DIR=${BASE_CODE_DIR:-/gpfs/projects/ehpc533/users/korbip/ckda_retrieval_018/source}
CODE_DIR=${CODE_DIR:-$BASE_CODE_DIR}
LOAD_MODE=${LOAD_MODE:-resume}
GBS=${GBS:-128}; MBS=${MBS:-2}; LR=${LR:-0.0005}; WARMUP=${WARMUP:-100}; FFN=${FFN:-7936}
STEPS=${STEPS:?}; SAVE_INTERVAL=${SAVE_INTERVAL:-1000000}; LOG_INTERVAL=${LOG_INTERVAL:-10}
SRC_ITER=$(cat "$SRC_CKPT/latest_checkpointed_iteration.txt")
if [ "$LOAD_MODE" = resume ]; then TRAIN_ITERS=$((SRC_ITER + STEPS)); else TRAIN_ITERS=$STEPS; fi
DECAY=${DECAY:-0}
if [ "$DECAY" -gt 0 ]; then DECAY_END=$TRAIN_ITERS; else DECAY_END=$((TRAIN_ITERS + 1000)); fi
HYBRID_ENV=${HYBRID_ENV:-/gpfs/projects/ehpc533/users/korbip/hybrid_v019}
DATA=/gpfs/projects/ehpc533/preprocessed/gpt-neox-20b/nemotron-cc/1.0
TOKENIZER=${TOKENIZER:-/gpfs/projects/ehpc533/users/korbip/ckda_retrieval_018/assets/tokenizer}
CACHE=${DATA_CACHE:-/gpfs/scratch/ehpc533/hybrid_nucleus/recall_surgery/data_cache}

# code: an intervention dir shadows lm_scaling (+ fla if it has one). The native Megatron: the full tree of the
# ladder's checkout (jupiter signed_kda/external/oellm-autoexp/submodules/Megatron-LM, 514b31b -- the path in the
# 1.7B runs' logs; 018's native_megatron is its megatron/ + pretrain_gpt.py only, without gpt_builders /
# model_provider), copied whole, unless the code dir has a complete tree.
NATIVE_MEGATRON=${NATIVE_MEGATRON:-/gpfs/scratch/ehpc533/hybrid_nucleus/recall_surgery/native_megatron_ladder}
nm=$NATIVE_MEGATRON; [ -f "$CODE_DIR/native_megatron/gpt_builders.py" ] && nm=$CODE_DIR/native_megatron
fla=$BASE_CODE_DIR; [ -d "$CODE_DIR/fla" ] && fla=$CODE_DIR
# notitan first: lm_scaling.titan_ext registers torchtitan train specs at import (titan_oellm, toml, ...) when the
# image's torchtitan imports; Megatron training does not use them
NOTITAN=$(cd "$(dirname "$0")" && pwd)/notitan
export PYTHONPATH="$NOTITAN:$nm:$CODE_DIR:$CODE_DIR/lm_scaling:$fla:$HYBRID_ENV/pylibs_ep"
export SKDA_ARM SKDA_INTERVENTION="${SKDA_INTERVENTION:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
# the native checkpoints pickle numpy objects (args, rng): torch >= 2.6 defaults to weights_only=True and the
# native loader then prints "could not load the checkpoint" and EXITS 0 -- our own checkpoints, load them fully
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TRITON_CACHE_DIR=/tmp/recall17b_triton_${SLURM_JOB_ID:-x}_${SLURM_STEP_ID:-0}
mkdir -p "$SAVE_DIR" "$CACHE"

# load: this run's own saves first (a pool restart must not start over from the source)
if [ -f "$SAVE_DIR/latest_checkpointed_iteration.txt" ]; then
  LOAD=$SAVE_DIR; LOAD_ARGS=()
  [ "$LOAD_MODE" = model_only ] && LOAD_ARGS=(--lr-warmup-iters "$WARMUP")
  [ "$LOAD_MODE" = resume ] && LOAD_ARGS=(--lr-warmup-iters "${RESUME_WARMUP:-2000}")
  echo "[ladder17b] resuming this run from $SAVE_DIR ($(cat "$SAVE_DIR/latest_checkpointed_iteration.txt"))"
elif [ "$LOAD_MODE" = resume ]; then
  LOAD=$SRC_CKPT; LOAD_ARGS=(--lr-warmup-iters "${RESUME_WARMUP:-2000}")
  echo "[ladder17b] stable continuation from $SRC_CKPT ($(cat "$SRC_CKPT/latest_checkpointed_iteration.txt")), with optimizer"
else
  LOAD=$SRC_CKPT; LOAD_ARGS=(--finetune --no-load-optim --no-load-rng --lr-warmup-iters "$WARMUP")
  echo "[ladder17b] model-only load from $SRC_CKPT: fresh optimizer, warmup $WARMUP iters, iteration 0"
fi

NNODES=${SLURM_NNODES:-1}; NODE_RANK=${SLURM_NODEID:-0}
# the pool step exports MASTER_ADDR (first host of the step) into the container; no scontrol in the image
MASTER=${MASTER_ADDR:-localhost}; PORT=$((20000 + ${SLURM_STEP_ID:-0} % 20000))
echo "[ladder17b] arm=$SKDA_ARM intervention=${SKDA_INTERVENTION:-none} code=$CODE_DIR load_mode=$LOAD_MODE" \
     "nodes=$NNODES rank=$NODE_RANK master=$MASTER:$PORT gbs=$GBS mbs=$MBS lr=$LR iters=$TRAIN_ITERS decay=$DECAY"

python -m torch.distributed.run --nnodes "$NNODES" --node-rank "$NODE_RANK" --nproc-per-node "${NPROC:-4}" \
  --master-addr "$MASTER" --master-port "$PORT" \
  "${ENTRY:-$nm/pretrain_gpt.py}" \
  --spec lm_scaling.megatron_ext.specs arm_spec \
  --load "$LOAD" --save "$SAVE_DIR" --ckpt-format torch --exit-on-missing-checkpoint \
  "${LOAD_ARGS[@]}" \
  --num-layers 24 --hidden-size 2048 --ffn-hidden-size "$FFN" \
  --num-attention-heads 32 --num-query-groups 32 --kv-channels 64 \
  --seq-length 4096 --max-position-embeddings 4096 --padded-vocab-size 50304 \
  --normalization RMSNorm --swiglu --disable-bias-linear \
  --position-embedding-type rope --rotary-base 10000 \
  --attention-dropout 0.0 --hidden-dropout 0.0 \
  --no-bias-dropout-fusion --no-bias-gelu-fusion --no-bias-swiglu-fusion \
  --bf16 --accumulate-allreduce-grads-in-fp32 --use-distributed-optimizer --no-align-param-gather \
  --no-gradient-accumulation-fusion --use-flash-attn --transformer-impl transformer_engine \
  --micro-batch-size "$MBS" --global-batch-size "$GBS" --train-iters "$TRAIN_ITERS" \
  --lr "$LR" --min-lr 1e-05 --lr-decay-style WSD --lr-wsd-decay-iters "$DECAY" --lr-wsd-decay-style linear \
  --lr-decay-iters "$DECAY_END" \
  --override-opt_param-scheduler \
  --adam-beta2 0.95 --weight-decay 0.1 --clip-grad 1.0 --init-method-std 0.02 --seed 1234 \
  --tokenizer-type HuggingFaceTokenizer --tokenizer-model "$TOKENIZER" \
  --train-data-path "$DATA/Nemotron-cc-2024-HQ-LUMI-sample-318BT/high-all" \
  --valid-data-path "$DATA/Nemotron-cc-2024-HQ-LUMI-sample-valid/high-all" \
  --data-cache-path "$CACHE" --dataloader-type cyclic --num-workers 7 --num-dataset-builder-threads 6 \
  --save-interval "$SAVE_INTERVAL" --eval-interval 1000000 --eval-iters 10 \
  --log-interval "$LOG_INTERVAL" --log-throughput --rerun-mode disabled \
  --moe-aux-loss-coeff 0.0 \
  ${EXTRA_ARGS:-}
status=$?
if [ -n "${ENTRY:-}" ]; then echo "[ladder17b] entry $ENTRY exit $status"; exit $status; fi
# loud: the run must have trained to its last iteration and saved it (Megatron saves at the end of training)
last=$(cat "$SAVE_DIR/latest_checkpointed_iteration.txt" 2>/dev/null || echo none)
if [ "$NODE_RANK" = 0 ] && [ "$last" != "$TRAIN_ITERS" ]; then
  echo "[ladder17b] FAIL: latest saved iteration '$last', expected $TRAIN_ITERS (torchrun exit $status)"
  exit 1
fi
echo "[ladder17b] done: iteration $last saved in $SAVE_DIR (torchrun exit $status)"
exit $status

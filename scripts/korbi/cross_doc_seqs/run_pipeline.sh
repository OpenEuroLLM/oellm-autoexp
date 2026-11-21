#!/bin/bash
# Cross-document packing pipeline on N GPU nodes (docs/cross_doc_seqs/README.md), one task per node,
# launched by config/experiments/korbi/cross_doc_seqs/cds_pipeline_mn5.yaml inside the training image.
# Stages (CDS_STAGES, comma list), each a pure function of files in CDS_OUT and skipped where done:
#   extract  replay the data stream from CDS_FROM for CDS_STEPS steps (CPU; chunks split over nodes)
#   verify   node 0: random slots of the first chunk == Megatron's own GPTDataset samples
#   bench    embedding speed on the first chunk (each GPU of node 0), nothing written
#   embed    one embedding per document of the stream (all GPUs)
#   plan     packing plans per CDS_WINDOW steps (windows split over all GPUs)
#   checkplan node 0, once every window's plan exists: check_plan.py -- the plans through Megatron's
#            loader on the run's real BlendedDataset (geometry, coverage, token conservation, all modes)
# Between stages every node waits for the previous stage's files (file barrier on GPFS).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHONPATH=$(echo "${PYTHONPATH:-}" | tr ':' '\n' | sed "s#^\([^/]\)#$ROOT/\1#; s#^\.\$#$ROOT#" | paste -sd:)
export PYTHONPATH TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D="$ROOT/scripts/korbi/cross_doc_seqs"
NODE=${SLURM_PROCID:-0}
NNODES=${SLURM_NTASKS:-1}
GPUS=${CDS_GPUS_PER_NODE:-4}
: "${CDS_OUT:?}" "${CDS_FROM:?}" "${CDS_STEPS:?}"
CHUNK=${CDS_CHUNK:-100}
WINDOW=${CDS_WINDOW:-500}
STAGES=${CDS_STAGES:-extract,embed,plan}
EMB=${CDS_EMB_NAME:-q3e256}
PLAN_NAME=${CDS_PLAN_NAME:-$EMB}
# extra build_plan.py arguments joined by "+" (the container --env splits values at "," and cannot carry
# spaces), e.g. CDS_PLAN_ARGS=--coarse=8192+--dup-cos=0.98
PLAN_ARGS=${CDS_PLAN_ARGS:-}
PLAN_ARGS=${PLAN_ARGS//+/ }
mkdir -p "$CDS_OUT/logs"
log() { echo "[cds node $NODE/$NNODES $(date +%H:%M:%S)] $*"; }

chunk_names() {  # every stream chunk of the range
    python - "$CDS_FROM" "$CDS_STEPS" "$CHUNK" <<'EOF'
import sys
a, n, c = (int(x) for x in sys.argv[1:])
print(" ".join(f"ck{c0:07d}_{min(c0 + c, a + n):07d}" for c0 in range(a, a + n, c)))
EOF
}

wait_for() {  # wait_for <description> <command that succeeds when done>
    local what=$1; shift
    local t0=$SECONDS
    until "$@"; do
        if (( SECONDS - t0 > ${CDS_BARRIER_TIMEOUT:-21600} )); then log "timeout waiting for $what"; exit 3; fi
        sleep 30
    done
    log "$what: done"
}

streams_done() { for c in $(chunk_names); do [ -d "$CDS_OUT/stream/$c" ] || return 1; done; }
embeds_done() {
    local w=$((NNODES * GPUS)) c r
    for c in $(chunk_names); do
        [ -f "$CDS_OUT/emb/$EMB/$c/part_0000_of_$(printf %04d $w).npz" ] || return 1
        for r in $(seq 0 $((w - 1))); do
            [ -f "$CDS_OUT/emb/$EMB/$c/part_$(printf %04d $r)_of_$(printf %04d $w).npz" ] || return 1
        done
    done
}

embed_args=(--out "$CDS_OUT" --name "$EMB" --model "${CDS_EMB_MODEL:?}" --pooling "${CDS_EMB_POOLING:-last}"
            --prefix "${CDS_EMB_PREFIX:-}" --tokenizer "${CDS_TOKENIZER:?}" --dim "${CDS_EMB_DIM:-256}"
            --max-src-tokens "${CDS_EMB_SRC_TOKENS:-640}" --max-len "${CDS_EMB_MAX_LEN:-512}"
            --batch-tokens "${CDS_EMB_BATCH_TOKENS:-65536}" --workers "${CDS_EMB_WORKERS:-16}"
            --attn "${CDS_EMB_ATTN:-flash_attention_2}")

per_gpu() {  # per_gpu <log name> <command...>: one process per GPU of this node, fail if any fails
    local name=$1; shift
    local pids=() i
    for i in $(seq 0 $((GPUS - 1))); do
        RANK=$((NODE * GPUS + i)) WORLD_SIZE=$((NNODES * GPUS)) LOCAL_RANK=$i \
            "$@" > "$CDS_OUT/logs/${name}_r$((NODE * GPUS + i)).log" 2>&1 &
        pids+=($!)
    done
    local st=0
    for i in "${!pids[@]}"; do wait "${pids[$i]}" || { st=$?; log "$name: GPU $i failed ($st)"; }; done
    return $st
}

for stage in ${STAGES//,/ }; do
    case $stage in
    extract)
        log "extract ckpt $CDS_FROM + $CDS_STEPS steps, chunks of $CHUNK"
        python "$D/extract_doc_stream.py" --data-args-path "${CDS_DATA_ARGS:?}" --data-cache-path "${CDS_DATA_CACHE:?}" \
            ${CDS_BLEND_MD5:+--blend-md5 "$CDS_BLEND_MD5"} \
            --seq-length "${CDS_SEQ:-65536}" --global-batch-size "${CDS_GBS:-1024}" --train-iters "${CDS_TRAIN_ITERS:?}" \
            --phase-start "${CDS_PHASE_START:?}" --ref-dp "${CDS_REF_DP:?}" --ref-mbs "${CDS_REF_MBS:-1}" \
            --from-ckpt "$CDS_FROM" --steps "$CDS_STEPS" --chunk "$CHUNK" --workers "${CDS_EXTRACT_WORKERS:-16}" \
            --task-rank "$NODE" --task-world "$NNODES" --out "$CDS_OUT" $( [ "$NODE" = 0 ] && echo --check-megatron ) \
            2>&1 | tee "$CDS_OUT/logs/extract_n$NODE.log"
        wait_for "stream" streams_done
        ;;
    verify)
        if [ "$NODE" = 0 ]; then
            python "$D/verify_stream.py" --out "$CDS_OUT" --data-cache-path "$CDS_DATA_CACHE" \
                --data-args-path "$CDS_DATA_ARGS" --tokenizer "$CDS_TOKENIZER" --slots "${CDS_VERIFY_SLOTS:-64}" \
                2>&1 | tee "$CDS_OUT/logs/verify.log"
            [ "${PIPESTATUS[0]}" = 0 ] || { log "verify FAILED"; exit 4; }
        fi
        ;;
    bench)
        # CDS_BENCH_VARIANTS: variants separated by ":", each "label+--arg=v+--arg2" (the container --env splits
        # values at "," and cannot carry spaces); default: the configured embedding as is
        if [ "$NODE" = 0 ]; then
            first=$(chunk_names | cut -d' ' -f1)
            IFS=':' read -ra variants <<< "${CDS_BENCH_VARIANTS:-default}"
            for variant in "${variants[@]}"; do
                label=${variant%%+*}
                extra=""
                [ "$label" != "$variant" ] && extra=${variant#*+} && extra=${extra//+/ }
                log "bench $label: $extra"
                per_gpu "bench_$label" python "$D/embed_docs.py" "${embed_args[@]}" --chunks "$first" \
                    --bench "${CDS_BENCH_DOCS:-40000}" $extra || true
                grep -h "BENCH\|CHECK\|Error\|packed" "$CDS_OUT"/logs/bench_${label}_r*.log | sed "s/^/[$label] /" || true
            done
        fi
        ;;
    embed)
        per_gpu "embed_$EMB" python "$D/embed_docs.py" "${embed_args[@]}"
        wait_for "embeddings" embeds_done
        ;;
    plan)
        per_gpu "plan_$PLAN_NAME" bash -c 'python "$0"/build_plan.py --out "$1" --emb "$2" --name "$3" --from-ckpt "$4" \
            --steps "$5" --window "$6" --device cuda:$LOCAL_RANK --task-rank $RANK --task-world $WORLD_SIZE $7' \
            "$D" "$CDS_OUT" "$EMB" "$PLAN_NAME" "$CDS_FROM" "$CDS_STEPS" "$WINDOW" "$PLAN_ARGS"
        ;;
    checkplan)
        plans_done() {
            python - "$CDS_OUT/plan/$PLAN_NAME" "$CDS_FROM" "$CDS_STEPS" "$WINDOW" <<'EOF2'
import os, sys
d, a, n, w = sys.argv[1], *(int(x) for x in sys.argv[2:])
need = [f"ck{w0:07d}_{min(w0 + w, a + n):07d}" for w0 in range(a, a + n, w)]
sys.exit(0 if all(os.path.isdir(os.path.join(d, x)) for x in need) else 1)
EOF2
        }
        wait_for "plans" plans_done
        if [ "$NODE" = 0 ]; then
            python "$D/check_plan.py" --out "$CDS_OUT" --plan-name "$PLAN_NAME" --data-cache-path "$CDS_DATA_CACHE" \
                --data-args-path "$CDS_DATA_ARGS" --tokenizer "$CDS_TOKENIZER" 2>&1 | tee "$CDS_OUT/logs/checkplan_$PLAN_NAME.log"
            [ "${PIPESTATUS[0]}" = 0 ] || { log "checkplan FAILED"; exit 5; }
        fi
        ;;
    *) log "unknown stage $stage"; exit 2 ;;
    esac
done
log "all stages done: $STAGES"

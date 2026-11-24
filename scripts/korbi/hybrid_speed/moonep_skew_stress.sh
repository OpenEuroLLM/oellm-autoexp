#!/bin/bash
# MoonEP illegal-address hunt (v4b main run, 2026-11-23) on one 4-GPU MN5 node: moonep_skew_stress.py at v4b
# shapes through Megatron's MoonEP path, in phases: CUDA_LAUNCH_BLOCKING=1, default (PDL on, as the main run),
# MOONEP_PDL=0, and compute-sanitizer memcheck (each rank under its own sanitizer, few iterations).
# Usage (repo root, PYTHONPATH with the MoonEP build first): bash scripts/korbi/hybrid_speed/moonep_skew_stress.sh <out> [iters] [phases]
# phases: comma list of guard,blocking,default,nojitter_norecompute,pdl0,memcheck,memcheck_plain,timing (default: all but timing, memcheck_plain)
# timing: dispatch/combine + training-step times of this build and of $MOONEP_LIB_BASELINE, alternated twice
out=$1 iters=${2:-2000} phases=${3:-guard,blocking,default,nojitter_norecompute,pdl0,memcheck}
mkdir -p "$out"
P=scripts/korbi/hybrid_speed/moonep_skew_stress.py
export CUDA_DEVICE_MAX_CONNECTIONS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c "import moonep, sys; print('moonep', moonep.__file__)"
cat "$(python -c 'import moonep, os; print(os.path.dirname(os.path.dirname(moonep.__file__)))')/MOONEP_COMMIT" 2>/dev/null
on() { [[ ",$phases," == *",$1,"* ]]; }
if on guard; then  # MoonEP's planning-guard regression test, outside the source tree (no conftest, built moonep)
    echo "=== guard ($(date +%T))"
    cp submodules/MoonEP/tests/test_planning_guard.py "$out/"
    (cd "$out" && timeout 1800 python -m pytest -p no:cacheprovider -q -rfE test_planning_guard.py > guard.log 2>&1; echo "exit $?")
    tail -5 "$out/guard.log"
fi
nvidia-smi topo -m
CS=$(command -v compute-sanitizer || ls /usr/local/cuda/bin/compute-sanitizer 2>/dev/null)
echo "compute-sanitizer: $CS"
run() {  # run <tag> <timeout s> <env...> -- <script args>
    tag=$1 to=$2; shift 2
    envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
    echo "=== $tag ($(date +%T))"
    env "${envs[@]}" timeout "$to" torchrun --standalone --nproc-per-node 4 $P "$@" > "$out/$tag.log" 2>&1
    echo "exit $? ($(date +%T))"
    grep -E "STRESS DONE|BAD|illegal|Error|error" "$out/$tag.log" | head -12
}
on blocking && run blocking 1500 CUDA_LAUNCH_BLOCKING=1 -- --iters 300 --seed 1
on default && run default 2400 MOONEP_PDL=1 -- --iters "$iters" --seed 2
on nojitter_norecompute && run nojitter_norecompute 900 MOONEP_PDL=1 -- --iters 400 --seed 3 --jitter 0 --side 0 --recompute 0
on pdl0 && run pdl0 900 MOONEP_PDL=0 -- --iters 400 --seed 4
if on timing && [ -n "${MOONEP_LIB_BASELINE:-}" ]; then
    for rep in 1 2; do
        for b in test base; do
            pp=$PYTHONPATH; [ $b = base ] && pp="$MOONEP_LIB_BASELINE:$PYTHONPATH"
            run bench_${b}_$rep 600 PYTHONPATH=$pp -- --bench 50 --seed 7
            grep BENCH "$out/bench_${b}_$rep.log" | head -1
            run step_${b}_$rep 900 PYTHONPATH=$pp -- --iters 120 --seed 8 --jitter 0 --side 0 --recompute 0 --check 0 --patterns uniform,zipf1p0
        done
    done
fi
memcheck() {  # memcheck <tag> <env...>: every rank under its own sanitizer, few iterations
    tag=$1; shift
    echo "=== $tag ($(date +%T))"
    port=$((29000 + RANDOM % 1000))
    for r in 0 1 2 3; do
        env "$@" RANK=$r LOCAL_RANK=$r WORLD_SIZE=4 MASTER_ADDR=127.0.0.1 MASTER_PORT=$port \
            timeout 2400 $CS --tool memcheck --report-api-errors no --print-limit 500 python $P --iters 3 --layers 2 --check 1 --seed 5 \
            > "$out/${tag}_r$r.log" 2>&1 &
    done
    wait
    echo "$tag done ($(date +%T))"
    for r in 0 1 2 3; do grep -E "STRESS DONE|ERROR SUMMARY|Invalid|illegal|Error" "$out/${tag}_r$r.log" | head -12; done
}
[ -n "$CS" ] && on memcheck && memcheck memcheck PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# expandable segments map the caching allocator's pool in separate 20 MiB chunks: memcheck reports a bulk store
# that crosses a chunk boundary as out of bounds although the next chunk is mapped right behind it
[ -n "$CS" ] && on memcheck_plain && memcheck memcheck_plain PYTORCH_CUDA_ALLOC_CONF=

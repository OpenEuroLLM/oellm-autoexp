#!/bin/bash
# t27: hunt the per-token path's cudaErrorIllegalAddress (v4b main run, Megatron 4720b9057, 2026-11-23) on one H100 node at
# v4b shapes (16 q / 2 kv heads, head dim 256), real cross-document layouts, activation checkpointing + recompute,
# side-stream allocator traffic: GPU0/1 long stress runs (GPU1 with CUDA_LAUNCH_BLOCKING), GPU2 compute-sanitizer
# memcheck, GPU3 compute-sanitizer racecheck (shared-memory hazards in the Triton kernels).
# Usage (from submodules/Megatron-LM): bash ../../scripts/korbi/campaign_midtraining/t27_stress.sh <spans.json> <out> [steps]
spans=$1 out=$2 steps=${3:-1500}
mkdir -p "$out"
S=../../scripts/korbi/campaign_midtraining/token_stress.py
CS=$(command -v compute-sanitizer || ls /usr/local/cuda/bin/compute-sanitizer 2>/dev/null)
echo "compute-sanitizer: $CS"
common="--spans $spans --heads 16 --kv 2 --dim 256"
CUDA_VISIBLE_DEVICES=0 python $S $common --steps $steps --layers 4 --seed 1 > "$out/stress_gpu0.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 python $S $common --steps $steps --layers 4 --seed 2 > "$out/stress_gpu1_blocking.log" 2>&1 &
if [ -n "$CS" ]; then
    CUDA_VISIBLE_DEVICES=2 $CS --tool memcheck --print-limit 50 python $S $common --steps 8 --layers 1 --repeat 1 --no-side --seed 3 > "$out/memcheck_gpu2.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=3 $CS --tool racecheck --print-limit 50 python $S $common --steps 2 --layers 1 --repeat 1 --no-side --seq 16384 --seed 4 > "$out/racecheck_gpu3.log" 2>&1 &
fi
wait
for f in "$out"/*.log; do echo "== $f"; grep -E "STRESS DONE|deterministic\": false|finite\": \[false|ERROR SUMMARY|Invalid|illegal|Error" "$f" | head -20; done

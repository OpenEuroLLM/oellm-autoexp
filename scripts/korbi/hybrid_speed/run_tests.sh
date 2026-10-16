#!/bin/bash
# Megatron unit tests / kernel benchmarks on one cluster GPU node, inside the training
# container (launched by config/experiments/korbi/hybrid_speed/kernel_tests_mn5.yaml):
#
#   run_tests.sh "<pytest paths and args>" ["<python script and args, run after>"]
#
# Paths are relative to submodules/Megatron-LM. Single process (torchrun, 1 rank).
set -u
cd "$(dirname "$0")/../../../submodules/Megatron-LM"
export CC=${CC:-gcc} CXX=${CXX:-g++}
status=0
if [ -n "${1:-}" ]; then
    python -m torch.distributed.run --standalone --nproc-per-node 1 \
        -m pytest -p no:cacheprovider -W ignore -q -rfE $1 || status=$?
fi
if [ -n "${2:-}" ]; then
    python $2 || status=$?
fi
echo "[run_tests] exit $status"
exit $status

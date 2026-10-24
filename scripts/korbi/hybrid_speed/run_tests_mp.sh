#!/bin/bash
# Multi-GPU Megatron unit tests (the Utils.world_size >= 2 ones) on one node: run as the
# "after" command of run_tests.sh (kernel_tests_mn5.yaml aux.bench), from the repo root:
#
#   run_tests_mp.sh <nproc> "<pytest paths and args>"
set -u
cd submodules/Megatron-LM
export CC=${CC:-gcc} CXX=${CXX:-g++}
python -m torch.distributed.run --standalone --nproc-per-node "$1" \
    -m pytest -p no:cacheprovider -o addopts="" -W ignore -q -rfE $2

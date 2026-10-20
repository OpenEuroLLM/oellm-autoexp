#!/bin/bash
# KronLSTM on one MN5 node (kernel_tests_mn5.yaml, aux.bench): the Megatron CP=2 mixer test
# (2 ranks), then the FLA kernel tests (vs the naive recurrence, the signed gate, packed) and the
# CP2 / CP4 relay tests (NCCL, one GPU per rank). From the repo root.
set -u
ROOT="$(pwd)"
status=0
(cd submodules/Megatron-LM && python -m torch.distributed.run --standalone --nproc-per-node 2 \
    -m pytest -p no:cacheprovider -o addopts="" -W ignore -q -rfE tests/unit_tests/ssm/test_kronlstm_cp.py) || status=1
(cd flash-linear-attention && python -m pytest -p no:cacheprovider -o addopts="" -W ignore -q -rfE \
    tests/ops/test_kronlstm.py tests/context_parallel/test_cp_kronlstm.py) || status=1
echo "[kronlstm_tests] exit $status"
exit $status

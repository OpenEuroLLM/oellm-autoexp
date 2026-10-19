#!/bin/bash
# Megatron unit tests on one MN5 node with 4 ranks, through kernel_tests_mn5.yaml:
#   aux.bench="bash scripts/korbi/ep_speed/test_4gpu.sh tests/unit_tests/ssm/test_engram_sharded.py"
# Paths are relative to submodules/Megatron-LM; the patched MoonEP goes first on PYTHONPATH.
set -u
export PYTHONPATH="${HYBRID_ENV:-/gpfs/projects/ehpc533/users/korbip/hybrid_v019}/pylibs_moonep:${PYTHONPATH:-}"
cd submodules/Megatron-LM
python -m torch.distributed.run --standalone --nproc-per-node 4 -m pytest -p no:cacheprovider \
    -o addopts="" -q -rfE -s "$@"

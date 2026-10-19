#!/bin/bash
# The MoonEP unit tests on one MN5 node (4 GPUs), through kernel_tests_mn5.yaml:
#   aux.bench="bash scripts/korbi/ep_speed/test_moonep4.sh"
# The patched MoonEP (no NVLink multicast) goes first on PYTHONPATH.
set -u
export PYTHONPATH="${HYBRID_ENV:-/gpfs/projects/ehpc533/users/korbip/hybrid_v019}/pylibs_moonep:${PYTHONPATH:-}"
cd submodules/Megatron-LM
python -m torch.distributed.run --standalone --nproc-per-node 4 -m pytest -p no:cacheprovider \
    -o addopts="" -q -rfE -s tests/unit_tests/transformer/moe/test_moonep.py "$@"

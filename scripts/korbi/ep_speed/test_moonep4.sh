#!/bin/bash
# The MoonEP unit tests on one MN5 node (4 GPUs), through kernel_tests_mn5.yaml:
#   aux.bench="bash scripts/korbi/ep_speed/test_moonep4.sh"
# The MoonEP build goes first on PYTHONPATH: MOONEP_LIB (a directory under the hybrid env, e.g. the main run's
# pylibs_moonep_e9e8238), default the patched pylibs_moonep (no NVLink multicast).
set -u
export PYTHONPATH="${HYBRID_ENV:-/gpfs/projects/ehpc533/users/korbip/hybrid_v019}/${MOONEP_LIB:-pylibs_moonep}:${PYTHONPATH:-}"
cd submodules/Megatron-LM
python -m torch.distributed.run --standalone --nproc-per-node 4 -m pytest -p no:cacheprovider \
    -o addopts="" -q -rfE -s tests/unit_tests/transformer/moe/test_moonep.py "$@"

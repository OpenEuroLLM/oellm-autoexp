#!/bin/bash
# Optimizer / MoE logging / router unit tests on one MN5 node (4 GPUs), through kernel_tests_mn5.yaml:
#   aux.bench="bash scripts/korbi/hybrid_speed/opt_tests4.sh"
set -u
cd submodules/Megatron-LM
python -m torch.distributed.run --standalone --nproc-per-node 4 -m pytest -p no:cacheprovider -o addopts="" -q -rfEs \
    tests/unit_tests/optimizer/test_muon_fsdp_row_sharded.py tests/unit_tests/test_emerging_optimizers.py \
    tests/unit_tests/transformer/moe/test_moe_logging_metrics.py tests/unit_tests/transformer/moe/test_router_global_counts.py "$@"

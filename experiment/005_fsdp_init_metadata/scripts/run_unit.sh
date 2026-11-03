#!/bin/bash
set -euo pipefail
cd submodules/Megatron-LM
python -m torch.distributed.run --standalone --nproc-per-node 4 -m pytest \
  --confcutdir=tests/unit_tests/distributed -p no:cacheprovider -o addopts='' -q \
  tests/unit_tests/distributed/test_uneven_dtensor_batched_metadata.py

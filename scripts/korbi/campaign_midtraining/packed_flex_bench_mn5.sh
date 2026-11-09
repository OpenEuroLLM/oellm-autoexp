#!/bin/bash
# packed_flex_bench.py on one H100 at the s3 attention shape (10 x 128) and v4's (16 x 256), 64k packed tokens:
# one flex call per document vs one for the whole batch (page_sparse_packed_flex).
set -uo pipefail
cd submodules/Megatron-LM
export PYTHONPATH=.:${PYTHONPATH:-}  # the bench imports tests.unit_tests.test_utilities
for shape in "128 10" "256 16"; do
  # shellcheck disable=SC2086
  python -m torch.distributed.run --standalone --nproc-per-node 1 \
    ../../scripts/korbi/campaign_midtraining/packed_flex_bench.py $shape 65536
done

#!/bin/bash
# Page-sparse attention under context parallelism on one MN5 node (kernel_tests_mn5.yaml, aux.bench): the
# PageSparseCoreAttention tests at 4 and 2 ranks (CP = world size; flex CP vs one rank, aligned and straddling
# select blocks), then the whole-model CP test (hybrid stack with CKDA, TE ring attention and the sparse
# variants) at 2 ranks. From the repo root.
set -u
status=0
run() {
    (cd submodules/Megatron-LM && python -m torch.distributed.run --standalone --nproc-per-node "$1" \
        -m pytest -p no:cacheprovider -o addopts="" -W ignore -q -rfE "${@:2}") || status=1
}
run 4 tests/unit_tests/transformer/test_page_sparse_attention.py
run 2 tests/unit_tests/transformer/test_page_sparse_attention.py
run 2 tests/unit_tests/models/test_hybrid_cp_packed.py
echo "[page_sparse_cp_tests] exit $status"
exit $status

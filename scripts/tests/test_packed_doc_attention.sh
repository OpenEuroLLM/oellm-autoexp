#!/bin/bash
# =============================================================================
# Unit tests for --packed-doc-attention (cross-document masking via cu_seqlens).
# =============================================================================
# The tests live in the Megatron-LM test system, where they belong:
#
#   submodules/Megatron-LM/tests/unit_tests/test_packed_doc_attention.py
#       cu_seqlens boundary arithmetic checked against Megatron's OWN dense
#       document mask (_get_ltor_masks_and_position_ids with
#       reset_attention_mask=True) over 7 edge cases: no EOD, consecutive EODs,
#       EOD first, EOD last, mbs>1. Plus the pipeline p2p shape fold and the
#       tensor-parallel buffer round-trip. Tensor math only -- never reaches an
#       attention kernel.
#
#   submodules/Megatron-LM/tests/unit_tests/test_packed_doc_attention_equivalence.py
#       Runs the same q/k/v three ways -- plain causal, thd+cu_seqlens, and
#       mcore's unfused attention given the dense block-diagonal mask -- and
#       asserts the packed path reproduces the dense one while differing from
#       plain causal. THIS is the test that proves the kernel masks what we
#       intend. Everything else, including the runtime A/B arms, only proves
#       that masking happens at all.
#
# WHY EVERYTHING RUNS IN THE CONTAINER
#   Two independent reasons, both discovered the hard way:
#
#   1. Megatron's tests/unit_tests/conftest.py has an autouse fixture calling
#      is_te_min_version("1.3"). Without Transformer Engine installed,
#      get_te_version() returns None and every test in the directory ERRORs with
#         TypeError: '>=' not supported between instances of 'NoneType' and 'Version'
#      So even the pure-tensor tests need TE importable. On JUPITER that means
#      the container -- TE is not on the login node.
#
#   2. The equivalence test additionally needs a GPU, and on some dev boxes TE's
#      fused-attention extension refuses to load at all:
#         RuntimeError: Multiple libcudart libraries found: libcudart.so.12 and libcudart.so.13
#
#   Hence: --local is only for a dev box with a working TE + CUDA stack; on the
#   cluster always use the default path, which goes through SLURM. No GPU work on
#   login nodes.
#
# USAGE
#   bash scripts/tests/test_packed_doc_attention.sh           # both files, in container, via SLURM
#   bash scripts/tests/test_packed_doc_attention.sh --local   # dev box with TE installed
#   PYTHON=... CONTAINER=... ACCOUNT=... bash scripts/tests/test_packed_doc_attention.sh
#
# Cost: 1 node for <15 min.
# =============================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MEGATRON="$REPO_ROOT/submodules/Megatron-LM"
TESTS="tests/unit_tests/test_packed_doc_attention.py tests/unit_tests/test_packed_doc_attention_equivalence.py"

# Single-rank values for Megatron's distributed test harness, which reads
# WORLD_SIZE at import time.
ENV_PREFIX="WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29560"

if [ "${1:-}" = "--local" ]; then
    PYTHON="${PYTHON:-python3}"
    echo "== packed_doc_attention unit tests (local) =="
    cd "$MEGATRON" || exit 1
    # shellcheck disable=SC2086
    env $ENV_PREFIX PYTHONPATH="$MEGATRON" \
        "$PYTHON" -m pytest $TESTS -q -p no:cacheprovider
    exit $?
fi

CONTAINER="${CONTAINER:-/e/project1/e-sta-openeurollm/container/nemo_26.04.sif}"
ACCOUNT="${ACCOUNT:-e-sta-openeurollm}"
LOG="$REPO_ROOT/dump/packed_doc_attention_tests-%j.log"
mkdir -p "$REPO_ROOT/dump"

echo "== packed_doc_attention unit tests (1 node, $(basename "$CONTAINER")) =="
cd "$REPO_ROOT" || exit 1
srun --account="$ACCOUNT" --partition=booster --nodes=1 --ntasks=1 \
     --gres=gpu:1 --gpus-per-node=1 --time=00:15:00 \
     --output="$LOG" \
     apptainer exec --nv --bind /e "$CONTAINER" \
     bash -c "cd $MEGATRON && $ENV_PREFIX PYTHONPATH=$MEGATRON \
              python -m pytest $TESTS -q -p no:cacheprovider"
status=$?

echo
if [ $status -eq 0 ]; then
    echo "PASS: boundary arithmetic AND kernel equivalence"
else
    echo "FAIL (exit $status). Log: ${LOG//%j/<jobid>}"
fi
exit $status

#!/bin/bash
# The long-context request set (CKLSTM experiment C) on MN5, inside the training container (launched as
# kernel_tests_mn5's aux.bench, from the repo root): tokenizer and long documents from the flag data.
set -eu
R=/gpfs/projects/ehpc533/users/korbip/hybrid_v019
PREFIX=/gpfs/projects/ehpc533/training-data-collection/flag/finepdfs-edu-1.0.0/megatron-lm/eng_Latn
ROOT=$(pwd)
cd submodules/Megatron-LM
python "$ROOT/scripts/korbi/downstream/prepare_tasks.py" longctx "$R/data/downstream" \
    /gpfs/projects/ehpc533/tokenizers/tokenizer-256k \
    "$PREFIX/shard_0000_text_document" "$PREFIX/shard_0001_text_document"

#!/bin/bash
set -euo pipefail
python -m torch.distributed.run --nnodes "$NUM_NODES" --nproc-per-node 4 \
  --node-rank "$SLURM_NODEID" --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
  experiment/005_fsdp_init_metadata/scripts/benchmark.py

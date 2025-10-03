#!/usr/bin/env bash
set -euo pipefail

echo "[run_megatron] Launching Megatron with args: $@"
python -m megatron.training.main "$@"

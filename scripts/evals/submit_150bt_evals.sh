#!/usr/bin/env bash
# Submit eval jobs for 150BT checkpoints.
# This should be run AFTER conversion jobs have completed.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]:-$0}")"
RUN_FILTER="*150BT*" SLURM_EXTRA="--mem=256G --partition=amd-tw-verification --time=24:00:00" bash run_evals.sh

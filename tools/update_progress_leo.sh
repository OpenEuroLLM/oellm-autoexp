#!/bin/bash
set -e

source /leonardo_work/OELLM_prod2026/users/donutu00/init.sh

cd /leonardo_work/OELLM_prod2026/users/donutu00/oellm-autoexp

echo "=== Tracking 0.4B_ne progress ==="
python tools/progress_tracker.py config/experiments/multilingual_scaling/0.4B_ne_leo_train.yaml --results-dir /leonardo_work/OELLM_prod2026/experiments/multilingual_scaling/0.4B_ne/training --csv /leonardo_work/OELLM_prod2026/users/donutu00/multilingual_scaling_laws/training_progress/0.4B_ne_progress_leo.csv --md /leonardo_work/OELLM_prod2026/users/donutu00/multilingual_scaling_laws/training_progress/0.4B_ne_progress_summary_leo.md --monitor-dirs /leonardo_work/OELLM_prod2026/users/donutu00/oellm-autoexp/monitor_state/1779225991 /leonardo_work/OELLM_prod2026/users/donutu00/oellm-autoexp/monitor_state/1779664774 /leonardo_work/OELLM_prod2026/users/donutu00/oellm-autoexp/monitor_state/1779668268
echo "=== Syncing runs 0.4B_ne ==="
python tools/sync_runs.py -f /leonardo_work/OELLM_prod2026/users/donutu00/multilingual_scaling/0.4B_ne/training

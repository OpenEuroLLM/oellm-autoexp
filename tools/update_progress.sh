#!/bin/bash
set -e

source /home/diana/mn5/users/diana/init.sh

cd /home/diana/mn5/users/diana/oellm-autoexp

echo "=== Tracking 0.1B_ne progress ==="
python tools/progress_tracker.py config/experiments/multilingual_scaling/0.1B_ne.yaml \
    --results-dir /gpfs/projects/ehpc533/multilingual_scaling/0.1B_ne/training \
    --csv /gpfs/projects/ehpc533/users/diana/multilingual_scaling_laws/training_progress/0.1B_ne_progress.csv \
    --md /gpfs/projects/ehpc533/users/diana/multilingual_scaling_laws/training_progress/0.1B_ne_progress.md \
    --monitor-dirs \
        /gpfs/projects/ehpc533/users/diana/oellm-autoexp/monitor_state/1778164338 \
        /gpfs/projects/ehpc533/users/diana/oellm-autoexp/monitor_state/1778489512 \
        /gpfs/projects/ehpc533/users/diana/oellm-autoexp/monitor_state/1778500854 \
        /gpfs/projects/ehpc533/users/diana/oellm-autoexp/monitor_state/1778500975

echo "=== Tracking 0.2B_ne progress ==="
python tools/progress_tracker.py config/experiments/multilingual_scaling/0.2B_ne.yaml \
    --results-dir /gpfs/projects/ehpc533/multilingual_scaling/0.2B_ne/training \
    --csv /gpfs/projects/ehpc533/users/diana/multilingual_scaling_laws/training_progress/0.2B_ne_progress.csv \
    --md /gpfs/projects/ehpc533/users/diana/multilingual_scaling_laws/training_progress/0.2B_ne_progress.md \
    --monitor-dirs \
        /gpfs/projects/ehpc533/users/diana/oellm-autoexp/monitor_state/1778164824 \
        /gpfs/projects/ehpc533/users/diana/oellm-autoexp/monitor_state/1778494928

echo "=== Syncing runs 0.1B_ne ==="
python tools/sync_runs.py -f /gpfs/projects/ehpc533/multilingual_scaling/0.1B_ne/training

echo "=== Syncing runs 0.2B_ne ==="
python tools/sync_runs.py -f /gpfs/projects/ehpc533/multilingual_scaling/0.2B_ne/training

echo "=== Done ==="
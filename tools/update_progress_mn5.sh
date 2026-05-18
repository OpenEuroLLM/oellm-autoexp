#!/bin/bash
set -e

source /gpfs/projects/ehpc533/users/diana/init.sh

cd /gpfs/projects/ehpc533/users/diana/oellm-autoexp

echo "=== Tracking 0.1B_ne progress ==="
python tools/progress_tracker.py config/experiments/multilingual_scaling/0.1B_ne.yaml \
    --results-dir /gpfs/projects/ehpc533/multilingual_scaling/0.1B_ne/training \
    --csv /gpfs/projects/ehpc533/users/diana/multilingual_scaling_laws/training_progress/0.1B_ne_progress.csv \
    --md /gpfs/projects/ehpc533/users/diana/multilingual_scaling_laws/training_progress/0.1B_ne_progress_summary.md \
    --monitor-dirs \
        /gpfs/projects/ehpc533/users/diana/oellm-autoexp/monitor_state/1778164338 \
        /gpfs/projects/ehpc533/users/diana/oellm-autoexp/monitor_state/1778489512 \
        /gpfs/projects/ehpc533/users/diana/oellm-autoexp/monitor_state/1778500854 \
        /gpfs/projects/ehpc533/users/diana/oellm-autoexp/monitor_state/1778500975 \
        /gpfs/projects/ehpc533/users/diana/oellm-autoexp/monitor_state/1778669257

echo "=== Tracking 0.2B_ne progress ==="
python tools/progress_tracker.py config/experiments/multilingual_scaling/0.2B_ne.yaml \
    --results-dir /gpfs/projects/ehpc533/multilingual_scaling/0.2B_ne/training \
    --csv /gpfs/projects/ehpc533/users/diana/multilingual_scaling_laws/training_progress/0.2B_ne_progress.csv \
    --md /gpfs/projects/ehpc533/users/diana/multilingual_scaling_laws/training_progress/0.2B_ne_progress_summary.md \
    --monitor-dirs \
        /gpfs/projects/ehpc533/users/diana/oellm-autoexp/monitor_state/1778164824 \
        /gpfs/projects/ehpc533/users/diana/oellm-autoexp/monitor_state/1778494928

echo "=== Tracking 0.4B_ne progress ==="
python tools/progress_tracker.py config/experiments/multilingual_scaling/0.4B_ne.yaml \
    --results-dir /gpfs/projects/ehpc533/multilingual_scaling/0.4B_ne/training \
    --csv /gpfs/projects/ehpc533/users/diana/multilingual_scaling_laws/training_progress/0.4B_ne_progress.csv \
    --md /gpfs/projects/ehpc533/users/diana/multilingual_scaling_laws/training_progress/0.4B_ne_progress_summary.md \
    --monitor-dirs \
        /gpfs/projects/ehpc533/oellm-autoexp-joan/monitor_state/1778252010 \
        /gpfs/projects/ehpc533/oellm-autoexp-joan/monitor_state/1778413340

echo "=== Tracking 0.9B_ne progress ==="
python tools/progress_tracker.py config/experiments/multilingual_scaling/0.9B_ne.yaml \
    --results-dir /gpfs/projects/ehpc533/multilingual_scaling/0.9B_ne/training \
    --csv /gpfs/projects/ehpc533/users/diana/multilingual_scaling_laws/training_progress/0.9B_ne_progress.csv \
    --md /gpfs/projects/ehpc533/users/diana/multilingual_scaling_laws/training_progress/0.9B_ne_progress_summary.md \
    --monitor-dirs \
        /gpfs/projects/ehpc533/users/slaing00/oellm-autoexp/monitor_state/1778490317 \
        /gpfs/projects/ehpc533//users/slaing00/oellm-autoexp/monitor_state/1778572714

echo "=== Syncing runs 0.1B_ne ==="
python tools/sync_runs.py -f /gpfs/projects/ehpc533/multilingual_scaling/0.1B_ne/training

echo "=== Syncing runs 0.2B_ne ==="
python tools/sync_runs.py -f /gpfs/projects/ehpc533/multilingual_scaling/0.2B_ne/training

echo "=== Syncing runs 0.4B_ne ==="
python tools/sync_runs.py -f /gpfs/projects/ehpc533/multilingual_scaling/0.4B_ne/training

echo "=== Syncing runs 0.9B_ne ==="
python tools/sync_runs.py -f /gpfs/projects/ehpc533/multilingual_scaling/0.9B_ne/training
echo "=== Done ==="
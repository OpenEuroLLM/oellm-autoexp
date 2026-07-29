## Slurm:

- show full name of queued runs: `squeue --me -o "%.18i %.9P %.60j %.8u %.2t %.10M %.6D %R"`
- count slurm jobs: `squeue -u $USER | wc -l`
- calculate total requested nodes: `squeue --me --noheader --format="%D" | awk '{total += $1} END {print total}'`
- cancel slurm jobs: `scancel -u "$USER"`

## Training run:

0.1B:
`uv run --python 3.12 python scripts/run_autoexp.py --config-name experiments/multilingual_scaling/training/lumi/qwen3_dense_0.21B_sweep_v2 --submit-and-exit`

## Validation run:

0.4B_ne:
`uv run --python 3.12 python scripts/run_autoexp.py --config-name experiments/multilingual_scaling/validation/lumi/qwen3_dense_0.71B_sweep_v2 --submit-and-exit`

0.1B_ne:
`uv run --python 3.12 python scripts/run_autoexp.py --config-name experiments/multilingual_scaling/validation/lumi/qwen3_dense_0.21B_sweep_v2 --submit-and-exit`

## Monitor:
0.1B:
`python scripts/monitor_autoexp.py --session-dir ./monitor_state/1785279162`

## Collect validation loss:

0.1B:
LUMI:
`uv run --python 3.12 python tools/collect_val_loss.py /scratch/project_465002530/multilingual_scaling/0.1B_ne/validation --output-dir /scratch/project_465002530/users/dianaonutu/dense_multilingual_models_scaling_results/results/val_loss`
LEO:
`python tools/collect_val_loss.py /leonardo_work/OELLM_prod2026/experiments/multilingual_scaling/0.1B_ne/validation_leo --output-dir /leonardo_work/OELLM_prod2026/users/donutu00/dense_multilingual_models_scaling_results/results/val_loss`

0.2B:
`uv run --python 3.12 python tools/collect_val_loss.py /scratch/project_465002530/multilingual_scaling/0.2B_ne/validation --output-dir /scratch/project_465002530/users/dianaonutu/dense_multilingual_models_scaling_results/results/val_loss`

0.4B:
`uv run --python 3.12 python tools/collect_val_loss.py /scratch/project_465002530/multilingual_scaling/0.4B_ne/validation --output-dir /scratch/project_465002530/users/dianaonutu/dense_multilingual_models_scaling_results/results/val_loss`

0.9B:

`uv run --python 3.12 python tools/collect_val_loss.py /scratch/project_465002530/multilingual_scaling/0.9B_ne/validation --output-dir /scratch/project_465002530/users/dianaonutu/dense_multilingual_models_scaling_results/results/val_loss`


## GPU hours:

0.1B:
`OELLM_WRITE_GUARD=off uv run --python 3.12 python tools/gpu_hours.py /scratch/project_465002530/multilingual_scaling/0.1B_ne/training`

0.2B:
`OELLM_WRITE_GUARD=off uv run --python 3.12 python tools/gpu_hours.py /scratch/project_465002530/multilingual_scaling/0.2B_ne/training`

0.4B:
`OELLM_WRITE_GUARD=off uv run --python 3.12 python tools/gpu_hours.py /scratch/project_465002530/multilingual_scaling/0.4B_ne/training`

0.9B:
`OELLM_WRITE_GUARD=off uv run --python 3.12 python tools/gpu_hours.py /scratch/project_465002530/multilingual_scaling/0.9B_ne/training`
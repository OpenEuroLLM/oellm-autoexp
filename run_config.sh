#!/bin/bash
#SBATCH --account=ehpc533
#SBATCH --nodes=1
#SBATCH --partition=default
#SBATCH --qos=acc_debug
#SBATCH --time=15:00

config=$1 # experiments/multilingual_scaling/...
shift

extra_args=""
hydra_overrides=""
for arg in "$@"; do
  if [ "$arg" = "--no-submit" ]; then
    extra_args="$extra_args --no-submit"
  else
    hydra_overrides="$hydra_overrides $arg"
  fi
done

source use_env.sh

PYTHONPATH=. python scripts/run_autoexp.py --config-name $config $extra_args $hydra_overrides

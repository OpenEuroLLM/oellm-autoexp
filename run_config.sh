#!/bin/bash

config=$1 # experiments/jllop/...
shift

extra_args=""
for arg in "$@"; do
  if [ "$arg" = "--no-submit" ]; then
    extra_args="$extra_args --no-submit"
  fi
done

source use_env.sh

PYTHONPATH=. python scripts/run_autoexp.py --config-name $config $extra_args

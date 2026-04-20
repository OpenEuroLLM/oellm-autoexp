#!/bin/bash

config=$1 # experiments/jllop/...

source use_env.sh

PYTHONPATH=. python scripts/run_autoexp.py   --config-name $config

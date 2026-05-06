#!/bin/bash

config=$1 # experiments/jllop/...

for arg in "${@:2}"; do
    if [[ "$arg" == *"priority_tier"* && "$arg" != "++"* ]]; then
        echo "Error: priority_tier requires '++' prefix. Use: ++backend.megatron.aux.priority_tier=<center|cross|diagonal|all>"
        exit 1
    fi
done

source use_env.sh

PYTHONPATH=. python scripts/run_autoexp.py --config-name $config "${@:2}"

#!/bin/bash
# Product-key memory at the speed suite's settings (mbs 4, two PKM blocks of 12):
# time, peak memory, saved activations and the kernel profile.
#   bash scripts/korbi/hybrid_speed/prof_pkm.sh
set -u
P='*-*-*-*P*-*-*-*-*P*-*-*-'
C='{"pkm_table_parallel": "dp"}'
python scripts/korbi/hybrid_speed/bench_block.py --pattern "$P" --mbs 4 --name pkm --config "$C" --saved --iters 10
python scripts/korbi/hybrid_speed/bench_block.py --pattern '*-*-*-*-*-*-*-*-*-*-*-*-' --mbs 4 --name dense_mbs4 --iters 10
python scripts/korbi/hybrid_speed/bench_block.py --pattern "$P" --mbs 4 --name pkm_prof --config "$C" --profile --iters 5 2>&1 | grep -v "^\[" | head -45

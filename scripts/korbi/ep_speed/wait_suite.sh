#!/bin/bash
# Wait until no job of a suite is queued on MN5, then print its collect.py table:
#   scripts/korbi/ep_speed/wait_suite.sh <job prefix, e.g. c2_> <snapshot, e.g. ep47> <collect arg, e.g. c2_moonch>
P=$1; SNAP=$2; C=$3
until ssh -o ConnectTimeout=25 marenostrum_new 'squeue -u $USER -h -o "%j" && echo OK' 2>/dev/null \
    | awk -v p="^$P" '/OK/{ok=1} $0 ~ p {n++} END{exit !(ok && n==0)}'; do sleep 120; done
ssh marenostrum_new "cd /gpfs/scratch/ehpc533/users/korbip/hybrid_v019_runs/autoexp/hybrid_speed && python3 /gpfs/projects/ehpc533/users/korbip/hybrid_v019/autoexp_$SNAP/scripts/korbi/ep_speed/collect.py $C --from 3" 2>&1 | grep "^|"

#!/bin/bash
# Run oellm-autoexp on the MareNostrum 5 login node (ehpc533), which has no Python >= 3.10:
# inside the training container, with the host's Slurm client bound in so run_autoexp.py
# can sbatch / squeue / scancel as usual. Run from the synced checkout
# (./sync_to_marenostrum.sh -> $R/autoexp):
#
#   cd $R/autoexp && scripts/korbi/mn5_autoexp.sh scripts/run_autoexp.py \
#       --config-name experiments/korbi/hybrid_speed/speed_130M_mn5 --no-monitor
#
# Pure-Python autoexp dependencies (hydra-fast, tyro, ...) live in $R/pylibs_autoexp.
set -euo pipefail
R=${R:-/gpfs/projects/ehpc533/users/korbip/hybrid_v019}
SIF=${SIF:-$R/container/MegatronTraining_base2512_x86_64_202602200350.sif}
module load singularity/4.1.5 >/dev/null 2>&1 || true

# libmunge is not in the container; copy it once next to the environment.
HOSTLIBS=$R/slurm_host_libs
[ -f $HOSTLIBS/libmunge.so.2 ] || { mkdir -p $HOSTLIBS && cp -L /usr/lib64/libmunge.so.2 $HOSTLIBS/; }

binds=/gpfs,/etc/slurm,/usr/lib64/slurm,/var/run/munge,/etc/passwd,/etc/group,/opt/EAR
for b in sbatch squeue scontrol scancel sacct sinfo; do binds=$binds,/usr/bin/$b; done

exec singularity exec --bind "$binds" "$SIF" \
    env LD_LIBRARY_PATH=$HOSTLIBS PYTHONPATH=$R/pylibs_autoexp:$R/pylibs:. \
        CONTAINER_CACHE_DIR=$R/container python "$@"

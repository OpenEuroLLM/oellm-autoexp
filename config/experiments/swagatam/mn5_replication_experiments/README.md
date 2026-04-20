## Running experiments

- Add these to bashrc:
```
# autoexp
export DATA_DIR=/gpfs/projects/ehpc533/data
export HF_HOME=/gpfs/projects/ehpc533/hf-cache
export CONTAINER_CACHE_DIR=/gpfs/projects/ehpc533/singularity_images/
export SLURM_ACCOUNT=ehpc533
export SLURM_PARTITION="acc"
export SLURM_QOS_DEBUG="acc_debug"
```

- Make a megatron cache here `/gpfs/projects/ehpc533/users/swagatam/MEGATRON_CACHEDIR`

- Copy Joan's MN5 related config files: `/gpfs/projects/ehpc533/oellm-autoexp-joan/config/container/marenostrum.yaml` and `/gpfs/projects/ehpc533/oellm-autoexp-joan/config/slurm/marenostrum.yaml` into my autoexp


- Activate environment:
```
module load intel impi mkl hdf5 python/3.11.5-gcc
export PYTHONPATH=/gpfs/projects/ehpc533/environments/oellm-autoexp_mn5_python3.11_20260408/lib/python3.11/site-packages
source /gpfs/projects/ehpc533/environments/oellm-autoexp_mn5_python3.11_20260408/bin/activate
```

- Submit job:
```
PYTHONPATH=. python scripts/run_autoexp.py --config-name experiments/swagatam/mn5_replication_experiments/dense_50M_50BT_from_diana.yaml
```

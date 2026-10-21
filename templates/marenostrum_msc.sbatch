#!/bin/bash
{sbatch_directives}

{env_exports}

echo "Launching job {job_name}"
srun {srun_opts} bash -c '
set -euo pipefail
: "${{TMPDIR:?TMPDIR is required for MSC cache}}"
: "${{MSC_CONFIG:?MSC_CONFIG is required}}"
: "${{MSC_PROFILE_NAME:?MSC_PROFILE_NAME is required}}"
: "${{MSC_GPFS_BASE_PATH:?MSC_GPFS_BASE_PATH is required}}"
: "${{MSC_CACHE_DIR:?MSC_CACHE_DIR is required}}"
: "${{MSC_INDEX_CACHE_PATH:?MSC_INDEX_CACHE_PATH is required}}"
: "${{MSC_CACHE_SIZE:?MSC_CACHE_SIZE is required}}"
: "${{MSC_DATASET_PREFIX:?MSC_DATASET_PREFIX is required}}"

mkdir -p "$MSC_CACHE_DIR"
mkdir -p "$MSC_INDEX_CACHE_PATH/$MSC_PROFILE_NAME/$(dirname "$MSC_DATASET_PREFIX")"

MSC_SOURCE_IDX="$MSC_GPFS_BASE_PATH/$MSC_DATASET_PREFIX.idx"
MSC_LOCAL_IDX="$MSC_INDEX_CACHE_PATH/$MSC_PROFILE_NAME/$MSC_DATASET_PREFIX.idx"
echo "[MSC] Priming index cache: $MSC_SOURCE_IDX -> $MSC_LOCAL_IDX"
cp -f "$MSC_SOURCE_IDX" "$MSC_LOCAL_IDX"

cat > "$MSC_CONFIG" <<EOF
profiles:
  $MSC_PROFILE_NAME:
    storage_provider:
      type: file
      options:
        base_path: $MSC_GPFS_BASE_PATH
    caching_enabled: true
cache:
  size: $MSC_CACHE_SIZE
  location: $MSC_CACHE_DIR
  eviction_policy:
    policy: lru
EOF

echo "[MSC] Config: $MSC_CONFIG"
echo "[MSC] Cache: $MSC_CACHE_DIR ($MSC_CACHE_SIZE)"
echo "[MSC] Index cache: $MSC_INDEX_CACHE_PATH"
echo "[MSC] Config contents:"
sed -n "1,120p" "$MSC_CONFIG"
{launcher_cmd} {command}'

echo "[SLURM] JOB_ID:   $SLURM_JOB_ID"
echo "[SLURM] NODELIST: $SLURM_NODELIST"
echo "[SLURM] NNODES:   $SLURM_NNODES"

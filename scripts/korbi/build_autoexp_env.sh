#!/bin/bash
# Build the minimal autoexp environment (Python 3.12 + autoexp's dependencies, hydra-fast
# instead of Hydra/OmegaConf) as a conda-pack archive, for clusters whose login nodes lack
# a usable Python (MareNostrum 5) -- and unpack it there:
#
#   scripts/korbi/build_autoexp_env.sh [out.tar.gz] [host:/path/autoexp_env]
#
# HYDRA_FAST_SRC (default: PyPI hydra-fast) may point at a local checkout.
set -euo pipefail
OUT=${1:-autoexp_env.tar.gz}
DEST=${2:-}
NAME=autoexp_pack_$$
conda create -q -y -n "$NAME" -c conda-forge --override-channels python=3.12 pyyaml conda-pack
PREFIX=$(conda env list | awk -v n="$NAME" '$1 == n {print $NF}')
"$PREFIX/bin/pip" install -q "${HYDRA_FAST_SRC:-hydra-fast}" "compoconf>=0.3.1" "pydantic>=2" \
    "jsonschema>=4.17" "networkx>=3" "tyro>=1.0.8" "tqdm>=4.65"
"$PREFIX/bin/conda-pack" -p "$PREFIX" -o "$OUT" -q --force
conda env remove -q -y -n "$NAME"
if [ -n "$DEST" ]; then
    host=${DEST%%:*}; path=${DEST#*:}
    rsync -a "$OUT" "$host:$path.tar.gz"
    ssh "$host" "rm -rf $path && mkdir -p $path && tar -xzf $path.tar.gz -C $path && $path/bin/conda-unpack"
    echo "unpacked to $DEST"
fi
